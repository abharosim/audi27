"""Assemble utterance events that start with the trigger into one `turn`.

The segmenter cuts on silence, so one spoken command arrives as several utterances.
This module is the only thing that sticks them back together. It reads `utterance`
events and, in live mode, writes `turn` events. It does not import the recorder,
the corrector, or the router.

A turn starts only when the utterance text, after leading whitespace, begins with
a trigger. 大大 is a normal word, so a match in the middle of a sentence is not a
trigger; Latin variants also need a word boundary, so "dadapple" does not match.
The default spellings are 达达, 哒哒, 大大, DaDa, dada, and Dada. Latin triggers are
case-insensitive. Do not "fix" 大大 via the context.md glossary: that replacement
runs on the whole sentence, not only at the start.

`ok` and `suspect` both count. A suspect line is real speech the validator was
unsure about; dropping it would miss commands, so it is merged and the turn gets
`has_suspect`. `rejected` (and any other status) is dropped and does not move the
gap. Blank text is dropped the same way.

Following utterances join the open turn until the audio gap — next start minus
previous end — is greater than `gap_s` (equal still joins), the turn's audio
span reaches `max_duration_s` (the utterance that crosses the cap is kept), or
another trigger arrives. A trigger after a long gap closes the old turn as
`gap`; a trigger inside the gap closes it as `new_trigger`.

Replay (`feed` / `assemble_replay`) uses only those utterance timestamps, then
closes a leftover turn as `eof`. It never reads the wall clock, so the same
events always produce the same turns. Live mode also closes a turn that nothing
has joined for `wall_close_s` after the utterance was *seen* (`timeout`). The
wait is measured from receipt, not from the audio end: the event is already late
by the time transcription writes it, and the next piece of the same command may
still be in the transcriber. Merging itself stays on the audio gap.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .events import append, follow, make_event, replay

log = logging.getLogger("audi27.turns")

TURN_SOURCE = "turns"
DEFAULT_TRIGGERS = ("达达", "哒哒", "大大", "DaDa", "dada", "Dada")
# Apostrophes keep "dada's" from matching; a real command is "dada" then a break.
_LATIN_STUCK = "'’"


@dataclass
class Turn:
    utterance_ids: list[str]
    text: str
    raw_text: str
    start: str
    end: str
    reason: str
    has_suspect: bool
    trigger: str

    def data(self) -> dict:
        return {
            "utterance_ids": list(self.utterance_ids),
            "text": self.text,
            "raw_text": self.raw_text,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
            "trigger": self.trigger,
            "has_suspect": self.has_suspect,
        }


@dataclass
class _Pending:
    ids: list[str]
    parts: list[str]
    raw_parts: list[str]
    start: datetime
    end: datetime
    start_s: str
    end_s: str
    trigger: str
    has_suspect: bool
    wall: float


@dataclass
class _Utt:
    id: str
    text: str
    status: str
    start: datetime
    end: datetime
    start_s: str
    end_s: str


def _latin_trigger(trig: str) -> bool:
    return trig.isascii() and any(c.isalpha() for c in trig)


def _stuck_to_latin(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in _LATIN_STUCK)


def _is_sep(ch: str) -> bool:
    if ch.isspace():
        return True
    return unicodedata.category(ch)[:1] in "PS"


def match_trigger(text: str, triggers) -> str | None:
    """Surface form of the trigger at the start of `text`, or None.

    Leading whitespace is ignored. The returned string is the characters as
    written (so "DADA" stays "DADA"), not the configured spelling.
    """
    if not isinstance(text, str) or not text:
        return None
    body = text.lstrip()
    best = None
    best_len = -1
    for trig in triggers:
        if not isinstance(trig, str):
            continue
        trig = trig.strip()
        if not trig or len(body) < len(trig):
            continue
        head = body[:len(trig)]
        if _latin_trigger(trig):
            if head.lower() != trig.lower():
                continue
            nxt = body[len(trig):len(trig) + 1]
            if nxt and _stuck_to_latin(nxt):
                continue
        elif head != trig:
            continue
        if len(head) > best_len:
            best, best_len = head, len(head)
    return best


def _strip_trigger(text: str, surface: str) -> str:
    rest = text.lstrip()[len(surface):]
    i = 0
    while i < len(rest) and _is_sep(rest[i]):
        i += 1
    return rest[i:].strip()


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _join(parts: list[str]) -> str:
    return " ".join(p for p in parts if p)


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt


def _as_record(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "utterance" and isinstance(obj.get("data"), dict):
        rec = dict(obj["data"])
        if not rec.get("id") and obj.get("id"):
            rec["id"] = obj["id"]
        return rec
    if "type" in obj:
        return None
    if "start" in obj and "text" in obj:
        return obj
    return None


def _parse(obj) -> _Utt | None:
    rec = _as_record(obj)
    if rec is None:
        return None
    ident = rec.get("id")
    if not isinstance(ident, str) or not ident or any(c.isspace() for c in ident):
        return None
    text = rec.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    status = rec.get("status")
    if status is None or status == "":
        status = "ok"
    if status not in ("ok", "suspect"):
        return None
    start_s = rec.get("start")
    end_s = rec.get("end") or start_s
    start, end = _parse_ts(start_s), _parse_ts(end_s)
    if start is None or end is None or not isinstance(start_s, str) or not isinstance(end_s, str):
        return None
    if end < start:
        end, end_s = start, start_s
    return _Utt(ident, text, status, start, end, start_s, end_s)


class TurnAssembler:
    """Pure fold over a sequence of utterance records or `utterance` events.

    `push` returns turns closed by this record. `poll` applies the live idle
    timeout and is not used by `feed`. `finish` closes a leftover turn as `eof`.
    """

    def __init__(self, triggers=None, *, gap_s: float = 2.0, max_duration_s: float = 120.0,
                 wall_close_s: float = 8.0, clock=None):
        source = DEFAULT_TRIGGERS if triggers is None else triggers
        self.triggers = [t.strip() for t in source if isinstance(t, str) and t.strip()]
        self.gap_s = float(gap_s)
        self.max_duration_s = float(max_duration_s)
        self.wall_close_s = float(wall_close_s)
        self._clock = clock or time.monotonic
        self._pending: _Pending | None = None

    @classmethod
    def from_config(cls, cfg, clock=None) -> "TurnAssembler":
        t = cfg.turns
        return cls(t.triggers, gap_s=t.gap_s, max_duration_s=t.max_duration_s,
                   wall_close_s=t.wall_close_s, clock=clock)

    def push(self, obj) -> list[Turn]:
        utt = _parse(obj)
        if utt is None:
            return []
        closed: list[Turn] = []
        surface = match_trigger(utt.text, self.triggers)
        if surface is not None:
            if self._pending is not None:
                gap = (utt.start - self._pending.end).total_seconds()
                closed.append(self._close("gap" if gap > self.gap_s else "new_trigger"))
            self._open(utt, surface)
            if self._duration() >= self.max_duration_s:
                closed.append(self._close("max"))
            return closed
        if self._pending is None:
            return []
        gap = (utt.start - self._pending.end).total_seconds()
        if gap > self.gap_s:
            closed.append(self._close("gap"))
            return closed
        self._add(utt)
        if self._duration() >= self.max_duration_s:
            closed.append(self._close("max"))
        return closed

    def poll(self, now: float | None = None) -> list[Turn]:
        if self._pending is None:
            return []
        if now is None:
            now = self._clock()
        if now - self._pending.wall >= self.wall_close_s:
            return [self._close("timeout")]
        return []

    def finish(self) -> list[Turn]:
        if self._pending is None:
            return []
        return [self._close("eof")]

    def feed(self, records) -> list[Turn]:
        out: list[Turn] = []
        for rec in records:
            out.extend(self.push(rec))
        out.extend(self.finish())
        return out

    def _open(self, utt: _Utt, surface: str) -> None:
        self._pending = _Pending(
            ids=[utt.id],
            parts=[_strip_trigger(utt.text, surface)],
            raw_parts=[_collapse(utt.text)],
            start=utt.start,
            end=utt.end,
            start_s=utt.start_s,
            end_s=utt.end_s,
            trigger=surface,
            has_suspect=utt.status == "suspect",
            wall=self._clock(),
        )

    def _add(self, utt: _Utt) -> None:
        pending = self._pending
        assert pending is not None
        pending.ids.append(utt.id)
        pending.parts.append(_collapse(utt.text))
        pending.raw_parts.append(_collapse(utt.text))
        if utt.end >= pending.end:
            pending.end = utt.end
            pending.end_s = utt.end_s
        if utt.status == "suspect":
            pending.has_suspect = True
        pending.wall = self._clock()

    def _duration(self) -> float:
        pending = self._pending
        assert pending is not None
        return (pending.end - pending.start).total_seconds()

    def _close(self, reason: str) -> Turn:
        pending = self._pending
        assert pending is not None
        self._pending = None
        return Turn(
            utterance_ids=list(pending.ids),
            text=_join(pending.parts),
            raw_text=_join(pending.raw_parts),
            start=pending.start_s,
            end=pending.end_s,
            reason=reason,
            has_suspect=pending.has_suspect,
            trigger=pending.trigger,
        )


def turn_id(turn: Turn) -> str:
    return "turn-" + turn.utterance_ids[0]


def turn_event(turn: Turn) -> dict:
    """One `turn` event. The id is fixed from the first utterance, so a replay can be repeated."""
    return make_event("turn", TURN_SOURCE, turn.data(), ts=turn.end,
                      ref=turn.utterance_ids[0], id=turn_id(turn))


def format_turn(turn: Turn) -> str:
    try:
        stamp = datetime.fromisoformat(turn.start).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        stamp = str(turn.start)
    mark = "?" if turn.has_suspect else " "
    return f"{stamp} {mark} {turn.reason:<12} {len(turn.utterance_ids)}utt  [{turn.trigger}] {turn.text}"


def assemble_replay(data_dir, cfg, *, since=None, until=None) -> tuple[list[Turn], dict]:
    """Fold utterance events in `[since, until)` into turns. Does not write anything."""
    asm = TurnAssembler.from_config(cfg)
    turns: list[Turn] = []
    seen = 0
    for ev in replay(data_dir, since=since, until=until, types="utterance"):
        seen += 1
        turns.extend(asm.push(ev))
    turns.extend(asm.finish())
    stats = {
        "turns": len(turns),
        "utterances": sum(len(t.utterance_ids) for t in turns),
        "seen": seen,
    }
    return turns, stats


def write_turns(data_dir, turns) -> dict:
    """Append turn events. An id already in that day's file is skipped."""
    wrote = skipped = 0
    for turn in turns:
        stored = append(data_dir, turn_event(turn), if_absent=True)
        if stored is None:
            skipped += 1
        else:
            wrote += 1
    return {"wrote": wrote, "skipped": skipped}


def write_turn_file(path, turns) -> None:
    """Replace `path` with the turn events as JSON lines. A second run writes the same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(turn_event(t), ensure_ascii=False) + "\n" for t in turns)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


_STOP = object()


def _emit(data_dir, turn: Turn) -> None:
    try:
        stored = append(data_dir, turn_event(turn), if_absent=True)
    except Exception:
        log.exception("turn event was not written for %s", turn.utterance_ids[0])
        return
    if stored is not None:
        log.info("[%s] %s", turn.reason, turn.text[:100] or "-")


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None) -> None:
    """Follow new utterance events and emit turns. Blocks until `stop` or Ctrl-C.

    Starts at the current end of the stream, like `audi27 events tail`. Utterances
    from before this process started are left for `turns replay`.
    """
    asm = TurnAssembler.from_config(cfg)
    stop = stop or threading.Event()
    box: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for ev in follow(data_dir, types="utterance", poll=poll_s, stop=stop, ready=ready):
                box.put(ev)
        except Exception:
            log.exception("turn follower stopped")
        finally:
            box.put(_STOP)

    def flush() -> None:
        """Apply utterances already queued, then close a leftover turn as eof."""
        while True:
            try:
                queued = box.get_nowait()
            except queue.Empty:
                break
            if queued is _STOP or queued is None:
                continue
            for turn in asm.push(queued):
                _emit(data_dir, turn)
        for turn in asm.finish():
            _emit(data_dir, turn)

    log.info("turn assembler watching %s (gap %.1fs, wall %.1fs, max %.0fs)",
             data_dir, asm.gap_s, asm.wall_close_s, asm.max_duration_s)
    worker = threading.Thread(target=pump, name="audi27-turns", daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = box.get(timeout=poll_s)
            except queue.Empty:
                item = None
            if item is _STOP:
                flush()
                return
            if item is not None:
                for turn in asm.push(item):
                    _emit(data_dir, turn)
            for turn in asm.poll():
                _emit(data_dir, turn)
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
        worker.join(timeout=2)
        flush()
