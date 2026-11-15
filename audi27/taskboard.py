"""Read-only task status for the dashboard.

Three lists, folded from the event log (and the grokbot quarantine directory):

- running: a task was dispatched (`sent` / `running` / `done`) and no reply
  has arrived, and it has not failed
- waiting: a reply is still in the report queue, so it has not been spoken
- recent: the last few that finished — spoken, failed, a spoken
  `needs_decision`, or a file left in `inbox/grokbot/quarantine/`

Nothing here writes. Dispatch, the queue, and the inbox are unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .events import replay
from .grokbot import inbox_dir
from .queue import ReportQueue

_OPEN = frozenset({"sent", "running", "done"})
_MAX_QUARANTINE_BYTES = 1024 * 1024
_TEXT_LIMIT = 80
_RECENT_LIMIT = 10


def derive(events, *, now: datetime, quarantine=None, text_limit: int = _TEXT_LIMIT,
           recent_limit: int = _RECENT_LIMIT) -> dict:
    """Fold `events` into ``{"running", "waiting", "recent"}``.

    `quarantine` is a list of dicts (`name`, `at`, `task_id`, `route_id`, `text`)
    already read from disk. This function does not touch the filesystem.
    """
    now = _aware(now)
    events = [ev for ev in events if isinstance(ev, dict)]
    routes: dict[str, dict] = {}
    by_task: dict[str, str] = {}
    briefs: dict[str, str] = {}

    for ev in events:
        kind = ev.get("type")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if kind == "brief":
            item = data.get("item_id") or data.get("text_ref") or ev.get("ref")
            if isinstance(ev.get("id"), str) and isinstance(item, str) and item:
                briefs[ev["id"]] = item
            continue
        if kind == "task":
            _note_task(routes, by_task, ev, data)
        elif kind == "reply":
            _note_reply(routes, ev, data)

    queued = _queued_ids(events)
    reply_ids = {rec["reply_id"] for rec in routes.values() if rec.get("reply_id")}
    spoken_at: dict[str, datetime] = {}
    for ev in events:
        if ev.get("type") != "spoken":
            continue
        item = _spoken_item(ev.get("ref"), briefs, reply_ids)
        when = _parse(ev.get("ts"))
        if item and when is not None:
            prev = spoken_at.get(item)
            if prev is None or when >= prev:
                spoken_at[item] = when

    running: list[dict] = []
    waiting: list[dict] = []
    recent: list[dict] = []
    for route_id, rec in routes.items():
        reply_id = rec.get("reply_id")
        if reply_id and reply_id in queued:
            status = "needs_decision" if rec.get("needs_decision") else "queued"
            waiting.append(_listed(
                rec, id=route_id, status=status, origin=rec.get("reply_at"),
                since="reply", now=now, text_limit=text_limit, finish=rec.get("reply_at"),
            ))
            continue
        if reply_id and reply_id in spoken_at:
            status = "needs_decision" if rec.get("needs_decision") else "spoken"
            recent.append(_listed(
                rec, id=route_id, status=status, origin=rec.get("reply_at"),
                since="reply", now=now, text_limit=text_limit, finish=spoken_at[reply_id],
            ))
            continue
        if reply_id:
            # The reply was skipped or closed before it was spoken.
            continue
        if rec.get("status") == "failed":
            recent.append(_listed(
                rec, id=route_id, status="failed", origin=rec.get("dispatch_at"),
                since="dispatch", now=now, text_limit=text_limit, finish=rec.get("failed_at"),
            ))
            continue
        if rec.get("status") in _OPEN:
            running.append(_listed(
                rec, id=route_id, status=rec["status"], origin=rec.get("dispatch_at"),
                since="dispatch", now=now, text_limit=text_limit, finish=rec.get("dispatch_at"),
            ))

    for item in quarantine or []:
        if isinstance(item, dict):
            recent.append(_quarantine_row(item, routes, by_task, now=now, text_limit=text_limit))

    running.sort(key=lambda row: (row["_key"], row["id"]))
    waiting.sort(key=lambda row: (row["_key"], row["id"]))
    recent.sort(key=lambda row: (row["_key"], row["id"]), reverse=True)
    if recent_limit < 0:
        recent = []
    else:
        recent = recent[:recent_limit]
    for row in running + waiting + recent:
        row.pop("_key", None)
    return {"running": running, "waiting": waiting, "recent": recent}


def snapshot(data_dir, *, now: datetime | None = None, text_limit: int = _TEXT_LIMIT,
             recent_limit: int = _RECENT_LIMIT) -> dict:
    """Replay the log and read `inbox/grokbot/quarantine/`. Does not write."""
    if now is None:
        now = datetime.now().astimezone()
    events = list(replay(
        data_dir, types=("task", "reply", "brief", "spoken", "route"),
    ))
    return derive(
        events, now=now, quarantine=_quarantine(data_dir),
        text_limit=text_limit, recent_limit=recent_limit,
    )


def _note_task(routes: dict, by_task: dict, ev: dict, data: dict) -> None:
    route_id = data.get("route_id")
    if not isinstance(route_id, str) or not route_id:
        return
    rec = routes.setdefault(route_id, _blank())
    ident = ev.get("id")
    if isinstance(ident, str):
        by_task[ident] = route_id
    status = data.get("status") if isinstance(data.get("status"), str) else ""
    when = _parse(ev.get("ts"))
    text = data.get("text") if isinstance(data.get("text"), str) else ""
    if text.strip() and not rec["text"]:
        rec["text"] = text
    _fill_who(rec, data)
    if status == "sent":
        if not rec["saw_sent"]:
            rec["dispatch_at"] = when
            rec["saw_sent"] = True
    elif rec["dispatch_at"] is None:
        rec["dispatch_at"] = when
    if status == "failed":
        rec["failed_at"] = when
    if status:
        rec["status"] = status


def _note_reply(routes: dict, ev: dict, data: dict) -> None:
    route_id = data.get("route_id")
    if not isinstance(route_id, str) or not route_id:
        return
    rec = routes.setdefault(route_id, _blank())
    ident = ev.get("id")
    if isinstance(ident, str):
        rec["reply_id"] = ident
    rec["reply_at"] = _parse(ev.get("ts"))
    rec["needs_decision"] = bool(data.get("needs_decision"))
    _fill_who(rec, data)
    text = data.get("text") if isinstance(data.get("text"), str) else ""
    if text.strip() and not rec["text"]:
        rec["text"] = text


def _blank() -> dict:
    return {
        "text": "", "target": "", "adapter": "", "status": "",
        "dispatch_at": None, "reply_at": None, "failed_at": None,
        "reply_id": None, "needs_decision": False, "saw_sent": False,
    }


def _fill_who(rec: dict, data: dict) -> None:
    target = data.get("target")
    adapter = data.get("adapter")
    if isinstance(target, str) and target:
        rec["target"] = target
    if isinstance(adapter, str) and adapter:
        rec["adapter"] = adapter


def _queued_ids(events: list) -> set[str]:
    """Reply (and failed-task) ids the report queue has not spoken yet."""
    queue = ReportQueue()
    queue.push_all(events)
    queued = {item.id for item in queue.items}
    if queue.inflight is not None:
        queued.add(queue.inflight.id)
    return queued


def _spoken_item(ref, briefs: dict, reply_ids: set[str]) -> str | None:
    if not isinstance(ref, str) or not ref:
        return None
    if ref in briefs:
        return briefs[ref]
    if not ref.startswith("brief-"):
        return None
    rest = ref[len("brief-"):]
    if rest in reply_ids:
        return rest
    head, sep, tail = rest.rpartition("-")
    if sep and tail.isdigit() and head in reply_ids:
        return head
    return None


def _listed(rec: dict, *, id: str, status: str, origin, since: str, now: datetime,
            text_limit: int, finish) -> dict:
    row = _row(
        id=id, text=_clip(rec.get("text") or "", text_limit),
        target=rec.get("target") or "", adapter=rec.get("adapter") or "",
        status=status, origin=origin, since=since, now=now,
    )
    row["_key"] = finish or datetime.min.replace(tzinfo=timezone.utc)
    return row


def _quarantine_row(item: dict, routes: dict, by_task: dict, *, now: datetime, text_limit: int) -> dict:
    name = item.get("name") if isinstance(item.get("name"), str) and item.get("name") else "quarantine"
    matched = _match(item, routes, by_task)
    if matched is not None and matched.get("text"):
        text = matched["text"]
        target = matched.get("target") or ""
        adapter = matched.get("adapter") or ""
    else:
        text = item.get("text") if isinstance(item.get("text"), str) else ""
        text = text or name
        target = ""
        adapter = ""
    origin = _parse(item.get("at"))
    row = _row(
        id=f"quarantine:{name}", text=_clip(text, text_limit),
        target=target, adapter=adapter, status="quarantined",
        origin=origin, since="file", now=now,
    )
    row["_key"] = origin or datetime.min.replace(tzinfo=timezone.utc)
    return row


def _match(item: dict, routes: dict, by_task: dict) -> dict | None:
    task_id = item.get("task_id") if isinstance(item.get("task_id"), str) else ""
    if task_id and task_id in by_task:
        return routes.get(by_task[task_id])
    name = item.get("name") if isinstance(item.get("name"), str) else ""
    stem = name[:-5] if name.endswith(".json") else name
    if stem and stem in by_task:
        return routes.get(by_task[stem])
    route_id = item.get("route_id") if isinstance(item.get("route_id"), str) else ""
    if route_id and route_id in routes:
        return routes[route_id]
    return None


def _row(*, id: str, text: str, target: str, adapter: str, status: str, origin, since: str,
         now: datetime) -> dict:
    return {
        "id": id,
        "text": text,
        "target": target,
        "adapter": adapter,
        "status": status,
        "elapsed_s": None if origin is None else round((now - origin).total_seconds(), 1),
        "since": since,
        "at": None if origin is None else origin.astimezone().isoformat(timespec="milliseconds"),
    }


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if limit <= 0 or len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[:limit - 1].rstrip() + "…"


def _parse(value) -> datetime | None:
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


def _aware(now: datetime) -> datetime:
    if not isinstance(now, datetime):
        raise TypeError("now must be a datetime")
    if now.tzinfo is None or now.utcoffset() is None:
        return now.astimezone()
    return now


def _quarantine(data_dir) -> list[dict]:
    """JSON files parked by grokbot. Symlinks are not followed."""
    root = inbox_dir(data_dir) / "quarantine"
    if root.is_symlink() or not root.is_dir():
        return []
    found = []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        item = {
            "name": path.name,
            "at": datetime.fromtimestamp(st.st_mtime).astimezone(),
            "task_id": None,
            "route_id": None,
            "text": "",
        }
        if st.st_size <= _MAX_QUARANTINE_BYTES:
            parsed = _read_quarantine(path)
            if parsed is not None:
                item.update(parsed)
        found.append(item)
    return found


def _read_quarantine(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    task_id = obj.get("task_id") if isinstance(obj.get("task_id"), str) else ""
    route_id = obj.get("route_id") if isinstance(obj.get("route_id"), str) else ""
    text = obj.get("text") if isinstance(obj.get("text"), str) else ""
    summary = obj.get("summary") if isinstance(obj.get("summary"), str) else ""
    return {
        "task_id": task_id.strip() or None,
        "route_id": route_id.strip() or None,
        "text": text.strip() or summary.strip(),
    }
