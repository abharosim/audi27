"""Play a short local sound when a wake word is heard.

This watches `utterance` events and does not import the router, the dispatcher,
or the recorder. An `ok` or `suspect` line that starts with a `[turns]` trigger
opens a turn, so the sound starts then — not after `wall_close_s`, and before
route or dispatch. Playback is a separate process (`afplay` on macOS, at most
one second) and this loop does not wait for it.

A missing file, a missing `afplay`, or any other OS is logged. The process
keeps running. A play appends `cue_played` (`ref` = the utterance id). That
window is the same kind of self-speech as a TTS read-back: a bare yes/no whose
audio starts inside it is ignored. The first start does not replay history
(`data/cue.cursor.json`). `[cue] enabled` only decides whether `audi27 supervise`
starts this loop; `cue run` plays either way.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

from .events import append, make_event, read_new, replay, snapshot_offsets
from .turns import match_trigger

log = logging.getLogger("audi27.cue")

CUE_SOURCE = "cue"
DEFAULT_SOUND = "/System/Library/Sounds/Tink.aiff"
# afplay is capped at this many seconds so a long file cannot fill the room.
PLAY_LIMIT_S = 1
# Playback cap plus the 0.5 s tail confirm already treats as echo after `say`.
SUPPRESS_S = 1.5
_CURSOR_NAME = "cue.cursor.json"
_REAP_S = 5.0


def afplay_player(*, popen=subprocess.Popen, which=shutil.which, is_file=None, platform: str | None = None):
    """`play(sound) -> {ok, backend, sound}`. Returns before `afplay` finishes.

    `ok` is false when this host cannot play (not macOS, no `afplay`, missing
    file, or the process could not be started). A non-zero exit after start is
    only logged: the caller has already moved on.
    """
    host = sys.platform if platform is None else platform
    exists = os.path.isfile if is_file is None else is_file

    def play(sound: str) -> dict:
        if not isinstance(sound, str):
            sound = "" if sound is None else str(sound)
        if host != "darwin" or which("afplay") is None:
            log.info("cue: afplay is not available; not playing %s", sound)
            return {"ok": False, "backend": "log", "sound": sound}
        if not sound or not exists(sound):
            log.warning("cue: sound file missing: %s", sound)
            return {"ok": False, "backend": "log", "sound": sound}
        argv = ["afplay", "-t", str(PLAY_LIMIT_S), sound]
        try:
            proc = popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            log.exception("cue: could not start afplay")
            return {"ok": False, "backend": "log", "sound": sound}

        def reap() -> None:
            try:
                code = proc.wait(timeout=_REAP_S)
            except Exception:
                log.exception("cue: afplay did not finish")
                try:
                    proc.kill()
                except Exception:
                    log.exception("cue: could not stop afplay")
                return
            if code not in (0, None):
                log.warning("cue: afplay exited %s", code)

        threading.Thread(target=reap, name="audi27-cue", daemon=True).start()
        return {"ok": True, "backend": "afplay", "sound": sound}

    return play


def build_player():
    """Real `afplay`. Tests pass their own `play(sound)`."""
    return afplay_player()


def _aware(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt


def playback_windows(events) -> list[tuple[datetime, datetime]]:
    """`(begin, end)` for cues that actually played. A logged miss is not a window.

    `end` is `data.until` when that timestamp parses, otherwise `ts` plus
    `SUPPRESS_S`. The end is exclusive.
    """
    windows = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") != "cue_played":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if data.get("ok") is not True:
            continue
        begin = _aware(ev.get("ts"))
        if begin is None:
            continue
        end = _aware(data.get("until"))
        if end is None:
            end = begin + timedelta(seconds=SUPPRESS_S)
        if end > begin:
            windows.append((begin, end))
    return windows


def wake_trigger(event: dict, triggers) -> str | None:
    """Surface form when `event` would open a turn, otherwise None.

    Same gate as the turn assembler: status `ok` or `suspect`, non-blank text,
    and `match_trigger` against the configured list (not the built-in defaults).
    """
    if not isinstance(event, dict) or event.get("type") != "utterance":
        return None
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    if data.get("status") not in ("ok", "suspect"):
        return None
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    ident = event.get("id")
    if not isinstance(ident, str) or not ident or any(c.isspace() for c in ident):
        return None
    return match_trigger(text, triggers)


def _outcome(player, sound: str) -> dict:
    try:
        result = player(sound)
    except Exception:
        log.exception("cue failed")
        result = None
    if not isinstance(result, dict):
        return {"ok": False, "backend": "log", "sound": sound}
    ok = bool(result.get("ok"))
    backend = result.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        backend = "afplay" if ok else "log"
    return {"ok": ok, "backend": backend.strip()[:40], "sound": sound}


def _already(data_dir, ident: str) -> bool:
    for ev in replay(data_dir, types="cue_played"):
        if ev.get("id") == ident:
            return True
    return False


def _flush(data_dir, pending: dict, ident: str) -> bool:
    """Write a cue that was already played. False leaves it pending for the next poll."""
    event = pending.get(ident)
    if event is None:
        return True
    try:
        append(data_dir, event, if_absent=True)
    except OSError:
        log.exception("cue event was not written for %s", event.get("ref"))
        return False
    pending.pop(ident, None)
    return True


def ack(data_dir, cfg, utterance: dict, *, player, pending: dict) -> bool:
    """Play the cue for one wake utterance and append `cue_played`.

    False only when that event still needs to be written, so the cursor stays
    and the next poll tries the write again. The sound is not played twice:
    `pending` remembers an id whose append failed. An id already stored is
    not played again. Anything that would not open a turn is ignored.
    """
    trigger = wake_trigger(utterance, cfg.turns.triggers)
    if trigger is None:
        return True
    utt_id = utterance["id"]
    ident = "cue-" + utt_id
    if ident in pending:
        return _flush(data_dir, pending, ident)
    if _already(data_dir, ident):
        return True
    sound = cfg.cue.sound
    # Stamp the window before starting playback so the beep itself falls inside it.
    started = datetime.now().astimezone()
    result = _outcome(player, sound)
    data = {
        "sound": sound,
        "backend": result["backend"],
        "ok": result["ok"],
        "trigger": trigger,
    }
    if result["ok"]:
        data["until"] = (started + timedelta(seconds=SUPPRESS_S)).isoformat(timespec="milliseconds")
    try:
        event = make_event("cue_played", CUE_SOURCE, data, ts=started, ref=utt_id, id=ident)
    except Exception:
        log.exception("cue event was not built for %s", utt_id)
        return True
    # Remember the play before the append. A failed write must not ding again.
    pending[ident] = event
    if result["ok"]:
        log.info("cue played for %s", utt_id)
    else:
        log.info("cue not played for %s", utt_id)
    return _flush(data_dir, pending, ident)


def play_once(cfg, *, player=None) -> dict:
    """Play the configured sound once. Does not write an event (`cue test`)."""
    player = build_player() if player is None else player
    return _outcome(player, cfg.cue.sound)


def _cursor_path(data_dir) -> Path:
    return Path(data_dir) / _CURSOR_NAME


def _save_cursor(data_dir, offsets: dict) -> None:
    path = _cursor_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"offsets": offsets}, ensure_ascii=False, sort_keys=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _load_cursor(data_dir) -> dict:
    path = _cursor_path(data_dir)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            offs = raw.get("offsets") if isinstance(raw, dict) else None
            if isinstance(offs, dict) and all(
                    isinstance(k, str) and type(v) is int and v >= 0 for k, v in offs.items()):
                return dict(offs)
        except (OSError, json.JSONDecodeError, UnicodeError):
            log.warning("ignoring unreadable cue cursor %s", path)
    offsets = snapshot_offsets(data_dir)
    try:
        _save_cursor(data_dir, offsets)
    except OSError:
        log.exception("cue cursor was not written")
    return offsets


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None, player=None) -> None:
    """Play a cue for each new wake-word utterance. The first start does not read history.

    Blocks until `stop`. An utterance whose `cue_played` event was not written
    is read again after the next poll; the sound is not played a second time.
    """
    player = build_player() if player is None else player
    stop = stop or threading.Event()
    offsets = _load_cursor(data_dir)
    durable = dict(offsets)
    pending: dict = {}

    def save(cursor: dict) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor)
        except OSError:
            log.exception("cue cursor was not written")
            return
        durable = dict(cursor)

    def cue_new(found) -> bool:
        ok = True
        for utterance in found:
            if not ack(data_dir, cfg, utterance, player=player, pending=pending):
                ok = False
        return ok

    log.info("cue watching %s (%s)", data_dir, cfg.cue.sound)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            held = offsets
            found, offsets = read_new(data_dir, offsets, types="utterance")
            if found:
                if cue_new(found):
                    save(offsets)
                else:
                    offsets = held
            elif offsets != durable:
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets, types="utterance")
            if found and cue_new(found):
                save(offsets)
        except Exception:
            log.exception("cue did not flush on the way out")
