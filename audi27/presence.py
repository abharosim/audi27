"""Fast "is the user talking" signal for the speaker.

The barge-in path cannot wait on the event log. The recorder already decides
speech or silence per frame with the energy VAD in the segmenter. While
`[presence] enabled` is on, it publishes that decision to `data/presence.json`
by atomic replace (write a temp file, then `os.replace`), the same way
`status.json` is published. A change is written on the spot, and the file is
refreshed at least every `publish_s` (default 0.1 s) so a long utterance does
not look dead. That is the whole IPC: one writer, readers on the same machine,
and the file's age is the timeout when capture dies. A Unix datagram would
save little time and would leave a socket behind.

Only a debounced transition is appended as a `presence` event (`data.state`
`speaking` or `quiet`). Short pauses stay inside the hangover and do not flap
the log. The speaker reads the file, not those events.

A missing, unreadable, stale, or future-dated file is unknown. Unknown does
not block, so a dead recorder cannot silence the speaker forever. A backwards
clock jump restarts the debounce windows and rewrites the file with the new
time; a forwards jump simply ages the file out.

An open 达达 turn is not tracked here. Turn events are written when the turn
closes, so there is no open-turn record to read without a second assembler.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from .events import append, make_event

log = logging.getLogger("audi27.presence")

PRESENCE_SOURCE = "capture"
FILENAME = "presence.json"
# A timestamp this far in the future is a clock jump, not a fresh sample.
_SKEW_S = 0.25


class Tracker:
    """Debounce raw speech bits into `speaking` / `quiet`.

    `onset_s` of continuous speech opens `speaking`. `hangover_s` of continuous
    silence closes it. A gap shorter than the hangover keeps the current state,
    which is what stops a breath in the middle of a sentence from flapping.
    """

    def __init__(self, onset_s: float, hangover_s: float):
        self.onset_s = onset_s
        self.hangover_s = hangover_s
        self.state = "quiet"
        self._speech_from: float | None = None
        self._silence_from: float | None = None
        self._last_now: float | None = None

    def observe(self, speech: bool, now: float) -> str | None:
        """Return the new state when it changes, else None."""
        if self._last_now is not None and now < self._last_now - _SKEW_S:
            self._speech_from = now if speech else None
            self._silence_from = now if (self.state == "speaking" and not speech) else None
        self._last_now = now
        if speech:
            self._silence_from = None
            if self._speech_from is None:
                self._speech_from = now
            if self.state != "speaking" and now - self._speech_from >= self.onset_s:
                self.state = "speaking"
                self._speech_from = None
                return "speaking"
            return None
        self._speech_from = None
        if self.state != "speaking":
            return None
        if self._silence_from is None:
            self._silence_from = now
        if now - self._silence_from >= self.hangover_s:
            self.state = "quiet"
            self._silence_from = None
            return "quiet"
        return None


def write_presence(data_dir, payload: dict) -> None:
    """Atomically replace `presence.json`. Readers see the old file or the new one.

    The temp file sits in the same directory as the destination, so `os.replace`
    is a rename and stays atomic on macOS (a cross-device rename is not). There
    is no fsync: a reader on this machine sees the page cache, and fsync at the
    publish rate would stall the recorder. A failed replace deletes the temp
    file; the previous `presence.json` is left in place.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (FILENAME + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, root / FILENAME)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_presence(data_dir) -> dict | None:
    """The raw file, or None when it is missing or not a JSON object.

    One retry covers a reader that lands between the replace steps on a
    filesystem where that is observable. A failure is unknown, not speaking.
    """
    path = Path(data_dir) / FILENAME
    for _ in range(2):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def fresh(data: dict | None, now: float, stale_s: float) -> bool:
    """True when `updated` is a believable wall-clock timestamp within `stale_s`.

    `updated` and `now` are both `time.time()`. A monotonic clock cannot be
    compared with a timestamp another process wrote.
    """
    if not isinstance(data, dict):
        return False
    updated = _number(data.get("updated"))
    if updated is None:
        return False
    age = now - updated
    return -_SKEW_S <= age <= stale_s


def busy(data_dir, cfg, now: float | None = None) -> bool:
    """True when the speaker must not be talking.

    `speaking`, or quiet but still inside `grace_s` of the transition that
    ended speech. Unknown (missing, stale, future, garbage) is not busy.
    """
    now = time.time() if now is None else now
    data = load_presence(data_dir)
    if not isinstance(data, dict) or not fresh(data, now, cfg.stale_s):
        return False
    if data.get("state") == "speaking":
        return True
    if data.get("state") != "quiet":
        return False
    since = _number(data.get("since"))
    if since is None:
        return False
    gap = now - since
    if gap < -_SKEW_S:
        return False
    return gap < cfg.grace_s


class PresencePublisher:
    """Turns per-frame speech bits into the file and, on change, one event.

    The file is replaced on this thread, before the event is appended. The
    event log takes an exclusive lock and fsyncs; with `defer_log` that work
    runs on a side thread so a slow disk cannot stall the frame loop or delay
    the file the speaker reads. `tick` and `close` swallow errors: a presence
    failure must not kill the recorder.
    """

    def __init__(self, data_dir, cfg, *, clock=None, append_fn=None, defer_log=False):
        self.root = Path(data_dir)
        self.cfg = cfg
        self.clock = clock or time.time
        self.append_fn = append_fn or append
        self.tracker = Tracker(cfg.onset_ms / 1000.0, cfg.hangover_ms / 1000.0)
        self._last_write: float | None = None
        self._since: float | None = None
        self._raw = False
        self._io_error = False
        self._q: queue.Queue | None = None
        self._log_thread: threading.Thread | None = None
        if defer_log:
            self._q = queue.Queue()
            self._log_thread = threading.Thread(
                target=self._drain_log, name="audi27-presence-log", daemon=True)
            self._log_thread.start()

    def tick(self, speech: bool, now: float | None = None):
        """Observe one frame. Returns the presence event when the state changed."""
        try:
            return self._tick(bool(speech), now)
        except Exception:
            self._log_io("presence update failed")
            return None

    def _tick(self, speech: bool, now: float | None):
        now = self.clock() if now is None else float(now)
        prev = self.tracker._last_now
        jumped = prev is not None and now < prev - _SKEW_S
        changed = self.tracker.observe(speech, now)
        self._raw = speech
        if jumped and self._since is not None and self._since > now + _SKEW_S:
            # The old `since` is in the future relative to the new clock.
            self._since = now if self.tracker.state == "speaking" else None
        if changed is not None:
            self._since = now
        due = self._last_write is None or now - self._last_write >= self.cfg.publish_s
        if changed is not None or due or jumped:
            self._write(now)
        if changed is None:
            return None
        return self._emit(changed, now)

    def close(self, now: float | None = None):
        """If speech was open, record quiet. A clean stop should not stick on speaking."""
        event = None
        try:
            now = self.clock() if now is None else float(now)
            if self.tracker.state == "speaking":
                self.tracker.state = "quiet"
                self._raw = False
                self._since = now
                self._write(now)
                event = self._emit("quiet", now)
        except Exception:
            self._log_io("presence was not closed")
            event = None
        self.flush()
        self._stop_log()
        return event

    def flush(self, timeout: float = 2.0) -> None:
        """Wait until queued presence events have been appended, or `timeout` elapses."""
        if self._q is None:
            return
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)

    def _stop_log(self) -> None:
        thread = self._log_thread
        if thread is None or self._q is None:
            return
        self._log_thread = None
        self._q.put(None)
        thread.join(timeout=1.0)

    def _drain_log(self) -> None:
        assert self._q is not None
        while True:
            event = self._q.get()
            try:
                if event is None:
                    return
                try:
                    self.append_fn(self.root, event)
                except Exception:
                    self._log_io("presence event was not written")
                else:
                    self._io_error = False
            finally:
                self._q.task_done()

    def _emit(self, state: str, now: float):
        try:
            ts = datetime.fromtimestamp(now).astimezone()
        except (OverflowError, OSError, ValueError):
            ts = None
        try:
            event = make_event("presence", PRESENCE_SOURCE, {"state": state}, ts=ts)
        except Exception:
            self._log_io("presence event was not written")
            return None
        if self._q is not None:
            self._q.put(event)
            return event
        try:
            self.append_fn(self.root, event)
        except Exception:
            self._log_io("presence event was not written")
            return None
        self._io_error = False
        return event

    def _write(self, now: float) -> None:
        payload = {
            "state": self.tracker.state,
            "updated": now,
            "since": self._since,
            "raw": self._raw,
        }
        try:
            write_presence(self.root, payload)
        except Exception:
            self._log_io("presence file was not written")
            return
        self._io_error = False
        self._last_write = now

    def _log_io(self, message: str) -> None:
        if self._io_error:
            return
        self._io_error = True
        log.exception(message)
