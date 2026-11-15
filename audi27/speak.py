"""Speak `brief` events in the order they were written, and record a `spoken` event.

The voice is macOS `say` (`[speak] voice`, default Tingting, which is zh_CN).
On any other OS, or when `say` is not on PATH, the line is logged and the
spoken event still says it completed, with `backend` `log`. Tests pass a
`backend(text, voice, stop_check)`.

With `[presence] enabled` off (the default), interruption uses the recorder
heartbeat (`data/status.json`, field `speaking`), written about every 0.4 s.
Nothing is started while that flag is true. `say` is killed if the flag turns
true mid-sentence. `say` cannot report how many words were spoken; `text` is
the whole line that was handed to it.

With presence on, the speaker reads `data/presence.json` instead. It does not
start while that state is `speaking` or for `grace_s` after speech ends. The
file is updated on the transition and at least every `publish_s`, and this
loop polls quickly enough that `say` is killed within about 0.3 s of the file
saying `speaking`. The spoken event then has `interrupted` true. The queue
releases a new brief id for that item; this module does not speak a brief that
already has a spoken event. A missing or stale presence file does not block.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import presence as presence_mod
from .events import append, make_event, read_new, replay, snapshot_offsets
from .status import read_status

log = logging.getLogger("audi27.speak")

SPEAK_SOURCE = "speak"
_CURSOR_NAME = "speak.cursor.json"


def user_speaking(data_dir) -> bool:
    """True when a fresh recorder heartbeat says the user is mid-utterance.

    A missing or stale file means the recorder is not running, so it is not
    evidence that anyone is speaking.
    """
    snap = read_status(Path(data_dir))
    if not snap.get("running"):
        return False
    status = snap.get("status") or {}
    return bool(status.get("speaking"))


def user_busy(data_dir, cfg, now: float | None = None) -> bool:
    """True when the speaker should stay quiet.

    Presence off: the heartbeat above, unchanged. Presence on: the fast file.
    """
    presence = getattr(cfg, "presence", None)
    if presence is None or not presence.enabled:
        return user_speaking(data_dir)
    return presence_mod.busy(data_dir, presence, now=now)


def _wait_quiet(speaking, stop: threading.Event | None, poll_s: float) -> bool:
    """Block until `speaking()` is false. False when `stop` is set first."""
    while True:
        try:
            talking = bool(speaking())
        except Exception:
            log.exception("could not read whether the user is speaking")
            talking = False
        if not talking:
            return True
        if stop is not None and stop.is_set():
            return False
        if stop is not None:
            if stop.wait(poll_s):
                return False
        else:
            time.sleep(poll_s)


def say_backend(*, popen=subprocess.Popen, which=shutil.which, sleep=time.sleep, platform: str | None = None):
    """`backend(text, voice, stop_check) -> {text, completed, interrupted, backend}`.

    `stop_check()` is polled while `say` runs. True kills the process and the
    result is interrupted. A non-zero exit (unknown voice, say missing at
    runtime) is logged instead of retried, so a bad voice cannot spin the queue.
    """
    host = sys.platform if platform is None else platform

    def backend(text: str, voice: str, stop_check) -> dict:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if host != "darwin" or which("say") is None:
            log.info("speak: %s", text)
            return {"text": text, "completed": True, "interrupted": False, "backend": "log"}
        if not text.strip():
            return {"text": text, "completed": True, "interrupted": False, "backend": "log"}
        try:
            if stop_check():
                return {"text": text, "completed": False, "interrupted": True, "backend": "say"}
        except Exception:
            log.exception("could not read whether the user is speaking")
        proc = _spawn(popen, ["say", "-v", voice, "--", text])
        try:
            while True:
                try:
                    talking = bool(stop_check())
                except Exception:
                    log.exception("could not read whether the user is speaking")
                    talking = False
                if talking:
                    _stop_proc(proc)
                    return {"text": text, "completed": False, "interrupted": True, "backend": "say"}
                if proc.poll() is not None:
                    break
                sleep(0.1)
            if proc.returncode not in (0, None):
                log.warning("say exited %s; logging the line instead", proc.returncode)
                log.info("speak: %s", text)
                return {"text": text, "completed": True, "interrupted": False, "backend": "log"}
            return {"text": text, "completed": True, "interrupted": False, "backend": "say"}
        finally:
            # A signal, a timeout, or the caller giving up must not leave `say` running.
            if proc.poll() is None:
                _stop_proc(proc)

    return backend


def _spawn(popen, argv):
    """Start `say`. The real process is its own session so a child dies with it."""
    if popen is not subprocess.Popen:
        return popen(argv)
    proc = popen(argv, start_new_session=True)
    proc._audi27_group = True  # noqa: SLF001
    return proc


def _signal_proc(proc, sig: int, *, kill: bool) -> None:
    pid = getattr(proc, "pid", None)
    if getattr(proc, "_audi27_group", False) and isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, sig)
            return
        except OSError:
            pass
    try:
        if kill:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        log.exception("could not stop say")


def _stop_proc(proc) -> None:
    """Stop `say`. Terminate, then kill if it is still there, so it is not left running."""
    # SIGTERM immediately. If it is still alive, SIGKILL before the 0.3 s budget.
    _signal_proc(proc, signal.SIGTERM, kill=False)
    try:
        proc.wait(timeout=0.2)
        return
    except Exception:
        pass
    _signal_proc(proc, signal.SIGKILL, kill=True)
    try:
        proc.wait(timeout=0.2)
    except Exception:
        log.exception("could not stop say")


def build_backend(cfg=None):
    """Real `say`. With presence on, poll a little faster than the default 0.1 s.

    The file is written on the transition, so a 0.05 s poll keeps the kill
    inside about 0.3 s. The flag-off path keeps the 0.1 s poll.
    """
    if cfg is not None and cfg.presence.enabled:
        return say_backend(sleep=lambda seconds: time.sleep(0.05 if seconds > 0.05 else seconds))
    return say_backend()


def spoken_event(brief: dict, result: dict) -> dict:
    data = brief.get("data") if isinstance(brief.get("data"), dict) else {}
    text = result.get("text")
    if not isinstance(text, str):
        text = data.get("summary") if isinstance(data.get("summary"), str) else ""
    return make_event("spoken", SPEAK_SOURCE, {
        "text": text,
        "completed": bool(result.get("completed")),
        "interrupted": bool(result.get("interrupted")),
        "backend": result.get("backend") or "log",
        "brief_id": brief["id"],
    }, ref=brief["id"], id="spoken-" + brief["id"])


def unspoken(data_dir) -> list[dict]:
    """Briefs that do not yet have a spoken event, in log order."""
    acked = set()
    for ev in replay(data_dir, types="spoken"):
        if isinstance(ev.get("ref"), str):
            acked.add(ev["ref"])
    return [ev for ev in replay(data_dir, types="brief") if ev["id"] not in acked]


def _speak_one(brief: dict, cfg, backend, speaking) -> dict:
    data = brief.get("data") if isinstance(brief.get("data"), dict) else {}
    text = data.get("summary") if isinstance(data.get("summary"), str) else ""
    try:
        result = backend(text, cfg.speak.voice, speaking)
    except Exception:
        log.exception("speak failed; logging the line")
        result = {"text": text, "completed": True, "interrupted": False, "backend": "log"}
    if not isinstance(result, dict):
        result = {"text": text, "completed": True, "interrupted": False, "backend": "log"}
    result.setdefault("text", text)
    return result


def speak_pending(data_dir, cfg, *, backend=None, speaking=None, stop: threading.Event | None = None,
                  poll_s: float = 0.1) -> int:
    """Speak every brief that has no spoken event yet. Returns how many were spoken.

    Used by the end-to-end test. Live `serve` does not catch up: the first
    start snapshots the end so enabling the speaker does not read the backlog.
    """
    backend = build_backend(cfg) if backend is None else backend
    if speaking is None:
        speaking = lambda: user_busy(data_dir, cfg)  # noqa: E731
    n = 0
    for brief in unspoken(data_dir):
        if not _wait_quiet(speaking, stop, poll_s):
            break

        def stop_check():
            if stop is not None and stop.is_set():
                return True
            try:
                return bool(speaking())
            except Exception:
                log.exception("could not read whether the user is speaking")
                return False

        result = _speak_one(brief, cfg, backend, stop_check)
        append(data_dir, spoken_event(brief, result), if_absent=True)
        n += 1
    return n


def speak_text(text: str, cfg, *, backend=None, speaking=None, poll_s: float = 0.1) -> dict:
    """Speak one line and return the result. Does not write an event (`speak test`)."""
    backend = build_backend(cfg) if backend is None else backend
    if speaking is None:
        speaking = lambda: False  # noqa: E731
    if not _wait_quiet(speaking, None, poll_s):
        return {"text": text, "completed": False, "interrupted": False, "backend": "log"}
    try:
        result = backend(text, cfg.speak.voice, speaking)
    except Exception:
        log.exception("speak failed")
        result = {"text": text, "completed": True, "interrupted": False, "backend": "log"}
    if not isinstance(result, dict):
        result = {"text": text, "completed": True, "interrupted": False, "backend": "log"}
    result.setdefault("text", text)
    return result


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
            log.warning("ignoring unreadable speak cursor %s", path)
    offsets = snapshot_offsets(data_dir)
    try:
        _save_cursor(data_dir, offsets)
    except OSError:
        log.exception("speak cursor was not written")
    return offsets


def _already_spoken(data_dir, brief_id: str) -> bool:
    prefix = "spoken-" + brief_id
    for ev in replay(data_dir, types="spoken"):
        if ev.get("id") == prefix or ev.get("ref") == brief_id:
            return True
    return False


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None, backend=None, speaking=None) -> None:
    """Speak new briefs. The first start does not read history. Blocks until `stop`."""
    backend = build_backend(cfg) if backend is None else backend
    if speaking is None:
        speaking = lambda: user_busy(data_dir, cfg)  # noqa: E731
    stop = stop or threading.Event()
    # While waiting out speech or the grace period, recheck about as often as
    # the file is published. The idle poll between briefs stays `poll_s`.
    # Flag off keeps a single interval, the one the caller passed.
    wait_s = min(poll_s, cfg.presence.publish_s) if cfg.presence.enabled else poll_s
    offsets = _load_cursor(data_dir)
    durable = dict(offsets)

    def save(cursor: dict) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor)
        except OSError:
            log.exception("speak cursor was not written")
            return
        durable = dict(cursor)

    log.info("speaker watching %s (voice %s)", data_dir, cfg.speak.voice)
    if ready is not None:
        ready.set()

    def speak_new(found) -> bool:
        """Speak `found`. False when a brief was not recorded, so the cursor must stay."""
        for brief in found:
            if stop.is_set():
                return False
            if _already_spoken(data_dir, brief["id"]):
                continue
            if not _wait_quiet(speaking, stop, wait_s):
                return False

            def stop_check():
                if stop.is_set():
                    return True
                try:
                    return bool(speaking())
                except Exception:
                    log.exception("could not read whether the user is speaking")
                    return False

            result = _speak_one(brief, cfg, backend, stop_check)
            try:
                append(data_dir, spoken_event(brief, result), if_absent=True)
            except OSError:
                log.exception("spoken event was not written for %s", brief.get("id"))
                return False
        return True

    try:
        while not stop.is_set():
            pending = offsets
            found, offsets = read_new(data_dir, offsets, types="brief")
            if found:
                if speak_new(found):
                    save(offsets)
                else:
                    # Do not keep a cursor that already stepped past a brief we
                    # did not record. The next poll reads the batch again.
                    offsets = pending
            elif offsets != durable:
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets, types="brief")
            if found and not stop.is_set() and speak_new(found):
                save(offsets)
        except Exception:
            log.exception("speaker did not flush on the way out")
