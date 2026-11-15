"""Report queue. Pure fold over events: replies and failures in, briefs out.

Priority, highest first: decision (the reply asks for a call), error (a failed
task), done (a normal reply), info (a queue command such as "what's left", or
a reply that says so). Several items that arrive together are ordered before
the next one is released. The one already handed to the speaker is not
preempted; a later, more urgent reply waits.

One brief is released at a time. The next waits for a `spoken` event, so the
speaker and the queue only meet through the log. State is whatever `push_all`
builds from the log, so a restart replays instead of reading a side file.

Queue commands are `route` events with target `queue`:

- details: speak the full text of the current item, or the last one
- skip: drop the current item
- later: put the current item at the back (behind everything queued now)
- pending: one short line listing what is left
- close: drop everything still queued

`summarize_fn` is the hook for a later LLM summarizer. It is not called unless
passed in. The default summary is the first sentence, truncated.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .events import append, make_event, read_new, replay, snapshot_offsets

log = logging.getLogger("audi27.queue")

QUEUE_SOURCE = "queue"
PRIORITIES = ("decision", "error", "done", "info")
_RANK = {name: i for i, name in enumerate(PRIORITIES)}
COMMANDS = ("details", "skip", "later", "pending", "close")
_CURSOR_NAME = "queue.cursor.json"
# A "." ends a sentence only when it is not the dot in a decimal (3.14).
_SENTENCE_END = "。！？!?"


@dataclass
class Item:
    id: str
    priority: str
    summary: str
    text: str
    text_ref: str
    seq: int
    ts: str
    deferred: bool = False
    attempts: int = 0


@dataclass
class _Followup:
    kind: str
    route_id: str
    ts: str
    priority: str = "info"
    text: str = ""
    text_ref: str = ""
    item_id: str | None = None


@dataclass
class ReportQueue:
    """In-memory fold. Create a new one and `push_all` the log to rebuild."""

    summary_chars: int = 80
    summarize_fn: object | None = None
    items: list[Item] = field(default_factory=list)
    inflight: Item | None = None
    last: Item | None = None
    holding: str | None = None
    disposition: str | None = None  # None, "drop" (skip/close), "back" (later)
    followups: list[_Followup] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    seq: int = 0

    def push_all(self, events) -> list[dict]:
        """Briefs newly decided by `events`, in the order they should be spoken.

        Replies and failed tasks that sit next to each other are ordered by
        priority before one of them is released. A `spoken` or a queue command
        is applied in place, after any replies that preceded it.
        """
        out: list[dict] = []
        batch: list[dict] = []

        def flush() -> None:
            # An empty flush must not release. Control events drain themselves,
            # and a second drain here would speak a "later" item immediately.
            if not batch:
                return
            for ev in batch:
                self._ingest_work(ev)
            batch.clear()
            out.extend(self._drain(allow_deferred=True))

        for ev in events:
            if not isinstance(ev, dict):
                continue
            if ev.get("type") in ("reply", "task"):
                batch.append(ev)
                continue
            flush()
            out.extend(self._ingest_control(ev))
        flush()
        return out

    def _ingest_work(self, ev: dict) -> None:
        ident = ev.get("id")
        if not isinstance(ident, str) or ident in self.seen:
            return
        self.seen.add(ident)
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        ts = ev.get("ts") if isinstance(ev.get("ts"), str) else ""
        if ev.get("type") == "reply":
            text = data.get("text") if isinstance(data.get("text"), str) else ""
            self._add(ident, _reply_priority(data), text, ts)
            return
        if ev.get("type") == "task" and data.get("status") == "failed":
            error = data.get("error") if isinstance(data.get("error"), str) else ""
            text = error or (data.get("text") if isinstance(data.get("text"), str) else "") or "失败"
            self._add(ident, "error", text, ts)

    def _add(self, ident: str, priority: str, text: str, ts: str) -> None:
        self.seq += 1
        self.items.append(Item(
            id=ident,
            priority=priority if priority in _RANK else "info",
            summary=summarize(text, limit=self.summary_chars, summarize_fn=self.summarize_fn),
            text=text,
            text_ref=ident,
            seq=self.seq,
            ts=ts,
        ))
        self._sort()

    def _sort(self) -> None:
        # Deferred ("later") sits behind every item that was not deferred,
        # including lower priorities. Within a group, priority then arrival.
        self.items.sort(key=lambda it: (it.deferred, _RANK.get(it.priority, 9), it.seq))

    def _ingest_control(self, ev: dict) -> list[dict]:
        if ev.get("type") == "spoken":
            return self._on_spoken(ev)
        if ev.get("type") == "route":
            return self._on_route(ev)
        return []

    def _on_route(self, ev: dict) -> list[dict]:
        ident = ev.get("id")
        if not isinstance(ident, str) or ident in self.seen:
            return []
        self.seen.add(ident)
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if data.get("target") != "queue":
            return []
        command = data.get("command")
        if command not in COMMANDS:
            return []
        if command == "skip":
            return self._skip()
        if command == "later":
            return self._later()
        if command == "close":
            return self._close()
        if command == "details":
            return self._details(ev)
        if command == "pending":
            return self._pending(ev)
        return []

    def _skip(self) -> list[dict]:
        if self.inflight is not None:
            self.disposition = "drop"
            return []
        # A details/pending line is what is being spoken. Skipping it must not
        # discard the next report, which is still queued behind that line.
        if self.holding:
            return []
        if self.items:
            self.items.pop(0)
        return self._drain(allow_deferred=True)

    def _later(self) -> list[dict]:
        if self.inflight is not None:
            self.disposition = "back"
            return []
        if self.holding:
            return []
        if not self.items:
            return []
        self._defer(self.items.pop(0))
        # The item just moved to the back must not be picked again immediately
        # when it is the only one left.
        return self._drain(allow_deferred=False)

    def _close(self) -> list[dict]:
        self.items.clear()
        self.followups.clear()
        if self.inflight is not None:
            self.disposition = "drop"
        return []

    def _details(self, ev: dict) -> list[dict]:
        item = self.inflight or self.last or (self.items[0] if self.items else None)
        if item is None:
            return []
        follow = _Followup(
            "details", ev["id"], ev.get("ts") if isinstance(ev.get("ts"), str) else "",
            item.priority, item.text or item.summary, item.text_ref, item.id,
        )
        if self.holding:
            self.followups.append(follow)
            return []
        return [self._emit_followup_now(follow)]

    def _pending(self, ev: dict) -> list[dict]:
        follow = _Followup("pending", ev["id"], ev.get("ts") if isinstance(ev.get("ts"), str) else "")
        if self.holding:
            self.followups.append(follow)
            return []
        return [self._emit_followup_now(follow)]

    def _on_spoken(self, ev: dict) -> list[dict]:
        ident = ev.get("id")
        if isinstance(ident, str):
            if ident in self.seen:
                return []
            self.seen.add(ident)
        if self.holding is None or ev.get("ref") != self.holding:
            return []
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        interrupted = bool(data.get("interrupted")) or data.get("completed") is False
        item = self.inflight
        disposition = self.disposition
        self.holding = None
        self.inflight = None
        self.disposition = None
        parked = False
        if item is not None and disposition == "drop":
            self.last = item
        elif item is not None and disposition == "back":
            self._defer(item)
            self.last = item
            parked = not any(not it.deferred for it in self.items)
        elif item is not None and interrupted:
            item.deferred = False
            self.items.append(item)
            self._sort()
            self.last = item
        elif item is not None:
            self.last = item
        return self._drain(allow_deferred=not parked)

    def _defer(self, item: Item) -> None:
        item.deferred = True
        self.seq += 1
        item.seq = self.seq
        self.items.append(item)
        self._sort()

    def _drain(self, *, allow_deferred: bool) -> list[dict]:
        if self.holding:
            return []
        if self.followups:
            return [self._emit_followup_now(self.followups.pop(0))]
        return self._release(allow_deferred=allow_deferred)

    def _release(self, *, allow_deferred: bool) -> list[dict]:
        if self.holding or not self.items:
            return []
        idx = next((i for i, it in enumerate(self.items) if not it.deferred), None)
        if idx is None:
            if not allow_deferred:
                return []
            idx = 0
        item = self.items.pop(idx)
        item.attempts += 1
        self.inflight = item
        self.last = item
        brief_id = f"brief-{item.id}" if item.attempts == 1 else f"brief-{item.id}-{item.attempts}"
        brief = _brief(
            id=brief_id, ref=item.id, priority=item.priority, summary=item.summary,
            text_ref=item.text_ref, kind="report", item_id=item.id, ts=item.ts,
        )
        self.holding = brief["id"]
        return [brief]

    def _emit_followup_now(self, follow: _Followup) -> dict:
        if follow.kind == "details":
            brief = _brief(
                id=f"brief-details-{follow.route_id}", ref=follow.route_id,
                priority=follow.priority, summary=follow.text, text_ref=follow.text_ref,
                kind="details", item_id=follow.item_id, ts=follow.ts,
            )
        else:
            brief = _brief(
                id=f"brief-pending-{follow.route_id}", ref=follow.route_id,
                priority="info", summary=self._pending_text(), text_ref=follow.route_id,
                kind="pending", item_id=None, ts=follow.ts,
            )
        self.holding = brief["id"]
        # The report item, if any, was already resolved. This utterance is not one.
        self.inflight = None
        return brief

    def _pending_text(self) -> str:
        rows = list(self.items)
        if self.inflight is not None and self.disposition != "drop":
            rows = [self.inflight] + rows
        if not rows:
            return "没有待汇报的。"
        shown = [it.summary.rstrip("。！？!?… ") for it in rows[:5]]
        text = f"还有 {len(rows)} 条：" + "；".join(shown) + "。"
        if len(rows) > 5:
            text += f"另有 {len(rows) - 5} 条。"
        return text


def _reply_priority(data: dict) -> str:
    if data.get("needs_decision"):
        return "decision"
    pri = data.get("priority")
    if pri in ("decision", "done", "info"):
        return pri
    return "done"


def _brief(*, id: str, ref: str, priority: str, summary: str, text_ref: str,
           kind: str, item_id: str | None, ts: str) -> dict:
    return make_event("brief", QUEUE_SOURCE, {
        "priority": priority,
        "summary": summary,
        "text_ref": text_ref,
        "kind": kind,
        "item_id": item_id,
    }, ts=ts or None, ref=ref, id=id)


def first_sentence(text: str, limit: int) -> str:
    """First sentence, or the whole line when it has no sentence end. Truncate to `limit`."""
    text = " ".join(text.split())
    if not text:
        return ""
    end = len(text)
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END:
            end = i + 1
            break
        if ch == "." and (i == 0 or not text[i - 1].isdigit()) and (i + 1 == len(text) or text[i + 1].isspace()):
            end = i + 1
            break
    sentence = text[:end].strip()
    if limit > 0 and len(sentence) > limit:
        if limit == 1:
            return "…"
        return sentence[:limit - 1].rstrip() + "…"
    return sentence


def summarize(text: str, *, limit: int = 80, summarize_fn=None) -> str:
    """One-sentence summary. `summarize_fn(text)` may return a shorter line later.

    An LLM summarizer plugs in here. This module does not call a model. A hook
    that raises, or returns a blank, falls back to the first sentence.
    """
    if summarize_fn is not None:
        try:
            out = summarize_fn(text)
        except Exception:
            log.warning("summarizer failed; using the first sentence", exc_info=True)
            out = None
        if isinstance(out, str) and out.strip():
            return " ".join(out.split())
    return first_sentence(text if isinstance(text, str) else "", limit)


def load_state(data_dir, cfg, *, summarize_fn=None) -> ReportQueue:
    """Rebuild the queue from the event log. Does not write."""
    q = ReportQueue(summary_chars=cfg.queue.summary_chars, summarize_fn=summarize_fn)
    q.push_all(replay(data_dir))
    return q


def write_briefs(data_dir, briefs) -> dict:
    wrote = skipped = 0
    for event in briefs:
        if append(data_dir, event, if_absent=True) is None:
            skipped += 1
        else:
            wrote += 1
    return {"wrote": wrote, "skipped": skipped}


def sync(data_dir, cfg, *, summarize_fn=None) -> tuple[ReportQueue, dict]:
    """Rebuild from the log and append briefs that are not stored yet."""
    queue = ReportQueue(summary_chars=cfg.queue.summary_chars, summarize_fn=summarize_fn)
    briefs = queue.push_all(list(replay(data_dir)))
    return queue, write_briefs(data_dir, briefs)


def format_queue(queue: ReportQueue) -> str:
    lines = []
    if queue.inflight is not None:
        lines.append(f"current  {queue.inflight.priority:<10} {queue.inflight.summary}")
    if not queue.items:
        lines.append("pending  (empty)")
    for item in queue.items:
        label = "later" if item.deferred else "pending"
        lines.append(f"{label:<8} {item.priority:<10} {item.summary}")
    return "\n".join(lines)


def _cursor_path(data_dir) -> Path:
    return Path(data_dir) / _CURSOR_NAME


def _valid_offsets(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in value.items()):
        return None
    return dict(value)


def _load_book(data_dir) -> tuple[dict, dict] | None:
    """`(epoch, offsets)` or None when this process has never started.

    `epoch` is the end of the log at the first start. A restart rebuilds state
    from events after that point only, so turning the queue on does not brief
    the backlog, and a later restart still sees what arrived while it ran.
    """
    path = _cursor_path(data_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        log.warning("ignoring unreadable queue cursor %s", path)
        return None
    if not isinstance(raw, dict):
        return None
    epoch = _valid_offsets(raw.get("epoch"))
    offsets = _valid_offsets(raw.get("offsets"))
    if epoch is None or offsets is None:
        log.warning("ignoring unreadable queue cursor %s", path)
        return None
    return epoch, offsets


def _save_cursor(data_dir, offsets: dict, epoch: dict) -> None:
    path = _cursor_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"epoch": epoch, "offsets": offsets}, ensure_ascii=False, sort_keys=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None, summarize_fn=None) -> None:
    """Follow the log and append briefs. Blocks until `stop`.

    State is the fold of every event since the first start (`epoch` in the
    cursor). A restart replays that range, so pending items come back, and
    brief ids already stored are not written again. `queue show` is different:
    it folds the whole log, which is the right view when you are inspecting.
    """
    stop = stop or threading.Event()
    queue = ReportQueue(summary_chars=cfg.queue.summary_chars, summarize_fn=summarize_fn)
    saved = _load_book(data_dir)
    if saved is None:
        epoch = snapshot_offsets(data_dir)
        offsets = dict(epoch)
    else:
        epoch = saved[0]
        found, offsets = read_new(data_dir, epoch)
        write_briefs(data_dir, queue.push_all(found))
    try:
        _save_cursor(data_dir, offsets, epoch)
    except OSError:
        log.exception("queue cursor was not written")
    durable = dict(offsets)

    def save(cursor: dict) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor, epoch)
        except OSError:
            log.exception("queue cursor was not written")
            return
        durable = dict(cursor)

    log.info("report queue watching %s", data_dir)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            found, offsets = read_new(data_dir, offsets)
            if found:
                write_briefs(data_dir, queue.push_all(found))
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets)
            if found:
                write_briefs(data_dir, queue.push_all(found))
                save(offsets)
        except Exception:
            log.exception("report queue did not flush on the way out")
