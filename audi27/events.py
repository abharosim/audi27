"""Append-only event stream under data/events/YYYY-MM-DD.jsonl (local date).

Modules talk only through these files. Each line is one JSON object:

    {"id", "ts", "type", "source", "ref", "data"}

`ts` is ISO 8601 with a timezone offset, stored in the machine's local offset.
`ref` is the id of the event that caused this one, or null. `data` is whatever
that type needs. A type this process has never heard of is stored and replayed
unchanged; do not reject unknown types.

Several processes may append at once. Each append takes an exclusive flock,
closes a torn trailing line left by a crash, writes one full line, and fsyncs
before unlocking. Readers take a shared lock and still skip a partial last line
and any line that is not a valid event. Lines are split on LF only.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

import fcntl

from .store import clean_json, iter_records

# Documented types from the voice-secretary design. Not a whitelist.
KNOWN_TYPES = frozenset({
    "utterance", "screen", "presence", "turn", "route", "task", "reply", "brief", "spoken",
    "confirm",
})
UTTERANCE_SOURCE = "transcribe"
_FIELDS = ("id", "ts", "type", "source", "ref", "data")


class EventError(ValueError):
    pass


def events_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "events"


def _parse_ts(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as e:
            raise EventError(f"ts is not ISO 8601: {value}") from e
    else:
        raise EventError("ts must be an ISO 8601 string or datetime")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise EventError("ts must include a timezone offset")
    return dt


def _token(value, field: str) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise EventError(f"{field} must be a non-empty string without whitespace")
    return value


def _json_data(data) -> dict:
    if not isinstance(data, dict):
        raise EventError("data must be a dict")
    data = clean_json(data)
    try:
        json.dumps(data, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise EventError(f"data is not JSON: {e}") from e
    return data


def make_event(type: str, source: str, data: dict, *, ts: datetime | str | None = None,
               ref: str | None = None, id: str | None = None) -> dict:
    """Build one event. Fills `id` and `ts` (local now) when omitted. Does not write."""
    if ts is None:
        ts = datetime.now().astimezone()
    return {
        "id": uuid.uuid4().hex if id is None else _token(id, "id"),
        "ts": _parse_ts(ts).astimezone().isoformat(timespec="milliseconds"),
        "type": _token(type, "type"),
        "source": _token(source, "source"),
        "ref": None if ref is None else _token(ref, "ref"),
        "data": _json_data(data),
    }


def _canonical(event: dict) -> dict:
    if not isinstance(event, dict):
        raise EventError("event must be a dict")
    missing = [name for name in _FIELDS if name not in event]
    if missing:
        raise EventError("missing fields: " + ", ".join(missing))
    return make_event(event["type"], event["source"], event["data"],
                      ts=event["ts"], ref=event["ref"], id=event["id"])


def _day_path(data_dir: str | Path, ts: str) -> Path:
    day = _parse_ts(ts).astimezone().strftime("%Y-%m-%d")
    return events_dir(data_dir) / f"{day}.jsonl"


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write appending an event")
        view = view[n:]


def _read_fd(fd: int) -> bytes:
    chunks = []
    while True:
        blob = os.read(fd, 1 << 20)
        if not blob:
            break
        chunks.append(blob)
    return b"".join(chunks)


def _close_torn_tail(fd: int) -> bool:
    """End a crash-torn last line so the next event is not glued onto it. True if a newline was added."""
    size = os.lseek(fd, 0, os.SEEK_END)
    if size <= 0:
        return False
    os.lseek(fd, size - 1, os.SEEK_SET)
    if os.read(fd, 1) != b"\n":
        _write_all(fd, b"\n")
        return True
    return False


def _json_lines(blob: bytes) -> list[str]:
    """LF-delimited records. U+2028 / U+2029 / U+0085 can sit inside a JSON string; they are not boundaries."""
    return [line for line in blob.decode("utf-8", errors="replace").split("\n") if line.strip()]


def _ids_in(fd: int) -> set[str]:
    os.lseek(fd, 0, os.SEEK_SET)
    found = set()
    for line in _json_lines(_read_fd(fd)):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            found.add(obj["id"])
    return found


def _exclusive(path: Path, body):
    """Run `body(fd)` while holding an exclusive lock on `path` (created if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            return body(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# json.dumps leaves these raw when ensure_ascii is false, but they are line breaks for
# str.splitlines and for some other languages' "split on newlines". Escape them so a record
# stays one LF-terminated line no matter who reads it.
_EXTRA_LINE_BREAKS = (
    ("\u2028", "\\u2028"),
    ("\u2029", "\\u2029"),
    ("\u0085", "\\u0085"),
)


def _line(ev: dict) -> bytes:
    text = json.dumps(ev, ensure_ascii=False, allow_nan=False)
    for ch, esc in _EXTRA_LINE_BREAKS:
        if ch in text:
            text = text.replace(ch, esc)
    return (text + "\n").encode("utf-8")


def append(data_dir: str | Path, event: dict, *, if_absent: bool = False) -> dict | None:
    """Validate and append one event. Returns the stored event, or None when `if_absent` and the id is already there.

    A torn tail is closed before the id check. Otherwise a crash that wrote a whole JSON value
    but not its newline would look present and stay invisible to readers forever.
    """
    ev = _canonical(event)
    path = _day_path(data_dir, ev["ts"])

    def body(fd: int):
        repaired = _close_torn_tail(fd)
        if if_absent and ev["id"] in _ids_in(fd):
            if repaired:
                os.fsync(fd)
            return None
        _write_all(fd, _line(ev))
        os.fsync(fd)
        return ev

    return _exclusive(path, body)


def emit(data_dir: str | Path, type: str, source: str, data: dict, *,
         ts: datetime | str | None = None, ref: str | None = None, id: str | None = None) -> dict:
    """Build an event and append it."""
    stored = append(data_dir, make_event(type, source, data, ts=ts, ref=ref, id=id))
    if stored is None:  # append skips a write only when if_absent hits an existing id
        raise EventError("event was not written")
    return stored


def utterance_event(record: dict) -> dict:
    """One `utterance` for a transcript record. The event id is the transcript id; `data` is the record."""
    if not isinstance(record, dict) or not record.get("id") or not record.get("start"):
        raise EventError("a transcript record needs id and start")
    return make_event("utterance", UTTERANCE_SOURCE, dict(record), ts=record["start"], id=str(record["id"]))


def emit_utterance(data_dir: str | Path, record: dict) -> dict | None:
    """Append the utterance for `record` unless that id is already in the stream."""
    return append(data_dir, utterance_event(record), if_absent=True)


def _as_set(values: str | list[str] | tuple | set | frozenset | None) -> frozenset | None:
    if values is None:
        return None
    if isinstance(values, str):
        return frozenset({values})
    return frozenset(values)


def _accept(ev: dict, since, until, types: frozenset | None, sources: frozenset | None) -> bool:
    if types is not None and ev["type"] not in types:
        return False
    if sources is not None and ev["source"] not in sources:
        return False
    if since is None and until is None:
        return True
    ts = _parse_ts(ev["ts"])
    if since is not None and ts < _parse_ts(since):
        return False
    if until is not None and ts >= _parse_ts(until):
        return False
    return True


def _coerce(obj) -> dict | None:
    """Return a well-formed event, or None when the line should be ignored."""
    if not isinstance(obj, dict):
        return None
    try:
        _parse_ts(obj.get("ts"))
        event_id = _token(obj.get("id"), "id")
        event_type = _token(obj.get("type"), "type")
        source = _token(obj.get("source"), "source")
    except EventError:
        return None
    data = obj.get("data")
    if not isinstance(data, dict):
        return None
    ref = obj.get("ref", None)
    if ref is not None:
        try:
            ref = _token(ref, "ref")
        except EventError:
            return None
    out = {"id": event_id, "ts": obj["ts"], "type": event_type, "source": source, "ref": ref, "data": data}
    return out


def _read_complete(path: Path, offset: int) -> tuple[list[dict], int]:
    """Events in `path` after byte `offset`. A trailing partial line is left for the next read."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return [], offset
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            size = os.lseek(fd, 0, os.SEEK_END)
            if size < offset:
                offset = 0
            if size == offset:
                return [], offset
            os.lseek(fd, offset, os.SEEK_SET)
            blob = _read_fd(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    if not blob:
        return [], offset
    if not blob.endswith(b"\n"):
        cut = blob.rfind(b"\n")
        if cut < 0:
            return [], offset
        blob = blob[:cut + 1]
    events = []
    for line in _json_lines(blob):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = _coerce(obj)
        if ev is not None:
            events.append(ev)
    return events, offset + len(blob)


def _files(data_dir: str | Path, since=None, until=None) -> list[Path]:
    root = events_dir(data_dir)
    if not root.is_dir():
        return []
    paths = sorted(p for p in root.glob("*.jsonl") if p.is_file())
    if since is None and until is None:
        return paths
    start = _parse_ts(since).astimezone().strftime("%Y-%m-%d") if since is not None else None
    end = _parse_ts(until).astimezone().strftime("%Y-%m-%d") if until is not None else None
    kept = []
    for path in paths:
        day = path.stem
        if len(day) == 10 and day[4] == "-" and day[7] == "-":
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
        kept.append(path)
    return kept


def snapshot_offsets(data_dir: str | Path) -> dict[str, int]:
    """Byte offset at the end of each day file. A trailing partial line is not consumed."""
    offsets: dict[str, int] = {}
    for path in _files(data_dir):
        _, offsets[path.name] = _read_complete(path, 0)
    return offsets


def read_new(data_dir: str | Path, offsets: dict[str, int], *, types=None,
             sources=None) -> tuple[list[dict], dict[str, int]]:
    """Events appended after `offsets` (day-file name -> byte offset).

    A day file that is not in the map is read from byte 0, so a file created later
    (the next local day) is not skipped. A trailing partial line stays unconsumed.
    The input map is not modified; the returned map is the cursor to keep.
    """
    want_types, want_sources = _as_set(types), _as_set(sources)
    cursor: dict[str, int] = {}
    for key, value in offsets.items():
        if isinstance(key, str) and type(value) is int and value >= 0:
            cursor[key] = value
    found: list[dict] = []
    for path in _files(data_dir):
        events, cursor[path.name] = _read_complete(path, cursor.get(path.name, 0))
        for ev in events:
            if _accept(ev, None, None, want_types, want_sources):
                found.append(ev)
    return found, cursor


def replay(data_dir: str | Path, *, since=None, until=None, types=None, sources=None) -> Iterator[dict]:
    """Yield stored events in file order. `since` is inclusive, `until` is exclusive.

    `types` and `sources` are one name or a collection; omit them to accept every value,
    including types that are not in `KNOWN_TYPES`.
    """
    want_types, want_sources = _as_set(types), _as_set(sources)
    for path in _files(data_dir, since, until):
        events, _ = _read_complete(path, 0)
        for ev in events:
            if _accept(ev, since, until, want_types, want_sources):
                yield ev


def follow(data_dir: str | Path, *, since=None, until=None, types=None, sources=None,
           poll: float = 0.1, stop: threading.Event | None = None,
           ready: threading.Event | None = None) -> Iterator[dict]:
    """Poll for events. With `since`, yield that history first; otherwise start at the current end.

    `stop`, when set, ends the loop. `ready` is set once the start position is known
    (and, when replaying history, after that history has been read). Latency is one poll.
    """
    want_types, want_sources = _as_set(types), _as_set(sources)
    offsets: dict[Path, int] = {}

    def stopped() -> bool:
        return stop is not None and stop.is_set()

    if since is None:
        for path in _files(data_dir):
            _, offsets[path] = _read_complete(path, 0)
        if ready is not None:
            ready.set()

    while not stopped():
        for path in _files(data_dir):
            found, offsets[path] = _read_complete(path, offsets.get(path, 0))
            for ev in found:
                if _accept(ev, since, until, want_types, want_sources):
                    yield ev
                if stopped():
                    if ready is not None:
                        ready.set()
                    return
        if ready is not None and not ready.is_set():
            ready.set()
        if stop is not None:
            if stop.wait(poll):
                return
        else:
            time.sleep(poll)


def ref_chain(data_dir: str | Path, event_id: str) -> list[dict]:
    """Events from the origin up to `event_id`, following `ref` backward.

    The last item is `event_id`. The first is the furthest event still in the stream
    (its `ref` may point at an id that was never written). A cycle stops the walk.
    """
    by_id: dict[str, dict] = {}
    for ev in replay(data_dir):
        by_id.setdefault(ev["id"], ev)
    if event_id not in by_id:
        raise EventError(f"unknown event {event_id}")
    chain: list[dict] = []
    seen: set[str] = set()
    current: str | None = event_id
    while current and current not in seen:
        ev = by_id.get(current)
        if ev is None:
            break
        seen.add(current)
        chain.append(ev)
        current = ev.get("ref")
    chain.reverse()
    return chain


def _append_missing(path: Path, batch: list[dict]) -> tuple[int, int]:
    """Append events whose ids are not already in `path`. One lock, one fsync.

    Closes a torn tail even when every id is already there, so a retried backfill makes
    a crash-cut last line visible instead of counting it as skipped and leaving it unread.
    """
    def body(fd: int) -> tuple[int, int]:
        repaired = _close_torn_tail(fd)
        have = _ids_in(fd)
        fresh = []
        wrote = skipped = 0
        for ev in batch:
            if ev["id"] in have:
                skipped += 1
                continue
            have.add(ev["id"])
            fresh.append(_line(ev))
            wrote += 1
        if fresh:
            _write_all(fd, b"".join(fresh))
            os.fsync(fd)
        elif repaired:
            os.fsync(fd)
        return wrote, skipped

    return _exclusive(path, body)


def backfill_utterances(data_dir: str | Path, *, since=None, until=None) -> dict:
    """Copy transcript records into `utterance` events. Safe to repeat: an id already present is skipped.

    Returns `{"wrote", "skipped"}`. Transcripts are not modified.

    `since` / `until` accept the same values as `replay`: an aware datetime, or an ISO 8601
    string with an offset. `since` is inclusive and `until` is exclusive.
    """
    since_dt = None if since is None else _parse_ts(since)
    until_dt = None if until is None else _parse_ts(until)
    grouped: dict[Path, list[dict]] = {}
    for rec in iter_records(data_dir, since_dt, until_dt):
        if not rec.get("id") or not rec.get("start"):
            continue
        ev = utterance_event(rec)
        grouped.setdefault(_day_path(data_dir, ev["ts"]), []).append(ev)
    wrote = skipped = 0
    for path, batch in grouped.items():
        w, s = _append_missing(path, batch)
        wrote += w
        skipped += s
    return {"wrote": wrote, "skipped": skipped}
