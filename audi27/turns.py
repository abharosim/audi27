"""Assemble utterance events that start with the trigger into one `turn`.

The segmenter cuts on silence, so one spoken command arrives as several utterances.
This module is the only thing that sticks them back together. It reads `utterance`
events and, in live mode, writes `turn` events. It does not import the recorder,
the corrector, or the router.

A turn starts only when the utterance text, after leading whitespace and
invisible format characters, begins with a trigger. 大大 is a normal word, so a
match in the middle of a sentence is not a trigger, and neither is 大大 as the
head of an AABB word (大大小小, 大大方方). Latin variants also need a word
boundary, so "dadapple" and "dada's" (straight or curly apostrophe, fullwidth
digit) do not match. Fullwidth Latin is matched as its halfwidth form. The
default spellings are 达达, 哒哒, 大大, DaDa, dada, and Dada. Latin triggers are
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

`feed` / `assemble_replay` order utterances by audio start (file order is not
trusted: a retried segment or a backfill can land later in the file), then use
only those timestamps, then close a leftover turn as `eof`. They never read the
wall clock, so the same events always produce the same turns.

Live mode holds what it has seen until the stream has been idle for
`wall_close_s` (`timeout` when nothing follows; `gap` / `new_trigger` / `max`
when the ordered audio says so). The wait is measured from receipt, not from
the audio end: transcription is already late, and a later line in the file can
be an earlier segment. The first start snapshots the end of the stream and does
not backfill; the cursor is `turns.cursor.json`, so a restart continues there
and does not emit a turn id that was already written.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .events import append, make_event, read_new, replay, snapshot_offsets

log = logging.getLogger("audi27.turns")

TURN_SOURCE = "turns"
DEFAULT_TRIGGERS = ("达达", "哒哒", "大大", "DaDa", "dada", "Dada")
# Apostrophes keep "dada's" from matching; a real command is "dada" then a break.
# Curly and fullwidth quotes are what a transcript actually contains; the ASCII
# apostrophe alone is not enough (see _stuck_to_latin).
_LATIN_STUCK = frozenset("'’‘＇ʼ")
_CURSOR_NAME = "turns.cursor.json"
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


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


def _fold(text: str) -> str:
    """Halfwidth, case-insensitive form. Fullwidth ＤＡＤＡ and DADA compare equal."""
    return unicodedata.normalize("NFKC", text).casefold()


def _latin_trigger(trig: str) -> bool:
    folded = unicodedata.normalize("NFKC", trig)
    return folded.isascii() and any(c.isalpha() for c in folded)


def _stuck_to_latin(ch: str) -> bool:
    """True when `ch` continues a Latin token (dadapple, dada's, dada２)."""
    if ch in _LATIN_STUCK:
        return True
    folded = unicodedata.normalize("NFKC", ch)
    return any(c in _LATIN_STUCK or (c.isascii() and c.isalnum()) for c in folded)


def _is_sep(ch: str) -> bool:
    if ch.isspace():
        return True
    return unicodedata.category(ch)[:1] in "PS"


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2A6DF


def _strip_leading_ignorable(text: str) -> str:
    """Drop leading whitespace and format characters (BOM, zero-width space)."""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or unicodedata.category(ch) == "Cf":
            i += 1
            continue
        break
    return text[i:]


def _aabb_after_dada(body: str, head: str) -> bool:
    """大大方方 / 大大小小 is one word. 大大，… and 大大的… stay a wake word."""
    if head != "大大":
        return False
    rest = body[len(head):]
    return len(rest) >= 2 and rest[0] == rest[1] and _is_cjk(rest[0])


def match_trigger(text: str, triggers) -> str | None:
    """Surface form of the trigger at the start of `text`, or None.

    Leading whitespace and format characters are ignored. The returned string
    is the characters as written (so "DADA" stays "DADA", fullwidth stays
    fullwidth), not the configured spelling.
    """
    if not isinstance(text, str) or not text:
        return None
    body = _strip_leading_ignorable(text)
    best = None
    best_len = -1
    for trig in triggers:
        if not isinstance(trig, str):
            continue
        trig = trig.strip()
        if not trig or len(body) < len(trig):
            continue
        head = body[:len(trig)]
        if len(head) != len(trig) or _fold(head) != _fold(trig):
            continue
        if _latin_trigger(trig):
            nxt = body[len(trig):len(trig) + 1]
            if nxt and _stuck_to_latin(nxt):
                continue
        elif _aabb_after_dada(body, head):
            continue
        if len(head) > best_len:
            best, best_len = head, len(head)
    return best


def _strip_trigger(text: str, surface: str) -> str:
    rest = _strip_leading_ignorable(text)[len(surface):]
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
        self._seen: set[str] = set()

    @classmethod
    def from_config(cls, cfg, clock=None) -> "TurnAssembler":
        t = cfg.turns
        return cls(t.triggers, gap_s=t.gap_s, max_duration_s=t.max_duration_s,
                   wall_close_s=t.wall_close_s, clock=clock)

    def push(self, obj) -> list[Turn]:
        utt = _parse(obj)
        if utt is None:
            return []
        if utt.id in self._seen:
            return []
        self._seen.add(utt.id)
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
        """Fold `records` in audio-time order, then close a leftover turn as `eof`.

        Arrival order is not used. A later line with an earlier timestamp is a
        continuation or an earlier command, not "the next utterance".
        """
        out: list[Turn] = []
        for rec in _order_records(list(records)):
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


def _order_records(records: list) -> list:
    """Audio start, then input order. The same utterance id is kept once.

    Records that do not parse sort first and are left for `push` to skip, so one
    bad line cannot hide a later command or change the gap around it.
    """
    keyed = []
    for index, rec in enumerate(records):
        utt = _parse(rec)
        when = utt.start if utt is not None else _EARLIEST
        keyed.append((when, index, rec, None if utt is None else utt.id))
    keyed.sort(key=lambda item: (item[0], item[1]))
    seen: set[str] = set()
    ordered = []
    for _, _, rec, ident in keyed:
        if ident is not None:
            if ident in seen:
                continue
            seen.add(ident)
        ordered.append(rec)
    return ordered


def assemble_replay(data_dir, cfg, *, since=None, until=None) -> tuple[list[Turn], dict]:
    """Fold utterance events in `[since, until)` into turns. Does not write anything.

    Events are ordered by audio start before the fold. Two replays of the same
    bytes produce the same turns even when a line was appended out of order.
    """
    asm = TurnAssembler.from_config(cfg)
    events = list(replay(data_dir, since=since, until=until, types="utterance"))
    turns: list[Turn] = []
    for ev in _order_records(events):
        turns.extend(asm.push(ev))
    turns.extend(asm.finish())
    seen = len(events)
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


def _cursor_path(data_dir) -> Path:
    return Path(data_dir) / _CURSOR_NAME


def _save_cursor(data_dir, offsets: dict[str, int]) -> None:
    path = _cursor_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"offsets": offsets}, ensure_ascii=False, sort_keys=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _load_cursor(data_dir) -> dict[str, int]:
    """Resume offset, or the current end when this stream has no cursor yet.

    The first start snapshots the end and saves it, so history is not backfilled
    and a crash before the next save still resumes from that end.
    """
    path = _cursor_path(data_dir)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            offs = raw.get("offsets") if isinstance(raw, dict) else None
            if isinstance(offs, dict) and all(
                    isinstance(k, str) and type(v) is int and v >= 0 for k, v in offs.items()):
                return dict(offs)
        except (OSError, json.JSONDecodeError, UnicodeError):
            log.warning("ignoring unreadable turn cursor %s", path)
    offsets = snapshot_offsets(data_dir)
    try:
        _save_cursor(data_dir, offsets)
    except OSError:
        log.exception("turn cursor was not written")
    return offsets


def _emit(data_dir, turn: Turn) -> bool:
    """Append the turn. False only when the write failed; an id already stored is success."""
    try:
        stored = append(data_dir, turn_event(turn), if_absent=True)
    except Exception:
        log.exception("turn event was not written for %s", turn.utterance_ids[0])
        return False
    if stored is not None:
        log.info("[%s] %s", turn.reason, turn.text[:100] or "-")
    return True


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None) -> None:
    """Follow new utterance events and emit turns. Blocks until `stop` or Ctrl-C.

    The first start snapshots the end of the stream and does not backfill.
    Later starts resume from `turns.cursor.json`. Utterances are held until the
    stream has been idle for `wall_close_s`, then folded in audio-time order, so
    a segment that lands late is not treated as the next thing that was said.
    The cursor advances only after that fold, so a crash replays those lines
    instead of dropping them. Turn ids already in the stream are not written again.
    """
    asm = TurnAssembler.from_config(cfg)
    stop = stop or threading.Event()
    offsets = _load_cursor(data_dir)
    durable = dict(offsets)
    inbox: list = []
    last_rx: float | None = None
    cursor_ok = True

    def fold(closing: bool) -> bool:
        """Apply the held utterances in audio order. `closing` ends the process (`eof`).

        False means a turn was not written. The cursor must stay put so a restart
        reads those utterances again.
        """
        nonlocal inbox, last_rx
        ordered = _order_records(inbox)
        inbox = []
        stamp = last_rx
        last_rx = None
        previous = asm._clock
        # Receipt time, not "now": the idle wait already elapsed, and poll must see it.
        if stamp is not None and not closing:
            asm._clock = lambda: stamp
        ok = True
        try:
            for ev in ordered:
                for turn in asm.push(ev):
                    ok = _emit(data_dir, turn) and ok
            if closing:
                closed = asm.finish()
            elif stamp is not None:
                closed = asm.poll(time.monotonic())
            else:
                closed = []
            for turn in closed:
                ok = _emit(data_dir, turn) and ok
        finally:
            asm._clock = previous
        return ok

    def save(cursor: dict[str, int]) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor)
        except OSError:
            log.exception("turn cursor was not written")
            return
        durable = dict(cursor)

    log.info("turn assembler watching %s (gap %.1fs, wall %.1fs, max %.0fs)",
             data_dir, asm.gap_s, asm.wall_close_s, asm.max_duration_s)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            found, offsets = read_new(data_dir, offsets, types="utterance")
            if found:
                inbox.extend(found)
                last_rx = time.monotonic()
            now = time.monotonic()
            if inbox and last_rx is not None and now - last_rx >= asm.wall_close_s:
                cursor_ok = fold(False) and cursor_ok
                if cursor_ok:
                    save(offsets)
            elif not inbox and cursor_ok:
                # Also move past turn events this process just appended.
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets, types="utterance")
            if found:
                inbox.extend(found)
                last_rx = time.monotonic()
            if cursor_ok and fold(True):
                save(offsets)
        except Exception:
            log.exception("turn assembler did not flush on the way out")
