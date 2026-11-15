"""Hand `route` events to the adapter configured for their target.

This is the execution switch the router deliberately does not flip. It reads
`route` events and writes `task` / `reply` events. It never imports a queue or
a speaker. Routes with target `none` or `queue` are left alone (the queue
module consumes the latter). A route with `needs_confirmation` is not sent:
one `task` with status `awaiting_confirmation` is written. With `[confirm]`
off that status stays forever. With it on, a later `confirm` event is what
releases the original route (`approve`, exactly once) or finishes it
(`reject` -> `cancelled`). Past `[confirm] ttl_s` the task is `expired` and
is never sent. A late approve is not sent; one brief asks the speaker to say so.

`[dispatch] enabled` only decides whether `audi27 supervise` starts this
process. `audi27 dispatch run` always executes, same as `route run`. The first
live start snapshots the end of the stream and does not replay history.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent import Agent, build_agent, task_event
from .confirm import classify_answer, overlaps_playback
from .events import append, make_event, read_new, replay, snapshot_offsets
from .route import AGENT_TARGETS

log = logging.getLogger("audi27.dispatch")

_CURSOR_NAME = "dispatch.cursor.json"
# A route is handed to an adapter at most once. `sent` / `running` / `done`
# mean the call already happened; calling `handle` again would repeat a real
# agent's side effects. `failed`, `cancelled`, `expired`, and `ignored` are
# finished and have no reply. `awaiting_confirmation` is only a hold. A reply
# is finished too. An interrupted chain (handed off, process died before a
# reply or `failed`) is closed with one `failed` task instead of being sent again.
_HANDED = frozenset({"sent", "running", "done"})
# `awaiting_confirmation` is a hold, not an ending: approve moves it forward.
# `ignored` is a confirmation answer that must not be sent to an agent.
_STICKY = frozenset({"failed", "cancelled", "expired", "ignored", "reply"})
_RANK = {"awaiting_confirmation": 0, "sent": 1, "running": 2, "done": 3}
_BLOCK = _STICKY | frozenset(_RANK) | {"silent"}
_INTERRUPTED = "执行中断，没有重新发送。"
_ADAPTER_ERROR = "适配器出错，没有重新发送。"
_NO_REPLY = "适配器没有回复，没有重新发送。"
_EXPIRED_NOTE = "这条确认已经过期，没有执行。"
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


def adapter_name(cfg, target: str) -> str:
    """Per-target name from `[dispatch.adapters]`, otherwise `fake`."""
    adapters = cfg.dispatch.adapters if isinstance(cfg.dispatch.adapters, dict) else {}
    name = adapters.get(target)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "fake"


def scan_route_state(data_dir) -> dict[str, str]:
    """route id -> `reply`, or the task status that wins so far.

    `failed`, `cancelled`, `expired`, `ignored`, and `reply` stick. Progress
    only moves forward through `awaiting_confirmation`, `sent`, `running`, `done`.
    """
    state: dict[str, str] = {}
    for ev in replay(data_dir, types=("task", "reply")):
        _note(state, ev)
    return state


def _note(state: dict, event: dict) -> None:
    """Update `state` from one event this process just emitted or replayed."""
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    route_id = data.get("route_id")
    if not isinstance(route_id, str):
        return
    current = state.get(route_id)
    if event.get("type") == "reply":
        # A reply finishes the route. cancelled / expired / ignored stay as they are:
        # those are confirmation outcomes, and a stray reply must not reopen them.
        if current not in ("cancelled", "expired", "ignored"):
            state[route_id] = "reply"
        return
    if event.get("type") != "task":
        return
    status = data.get("status")
    if not isinstance(status, str) or current in _STICKY:
        return
    if status in _STICKY:
        state[route_id] = status
    elif status in _RANK and (current not in _RANK or _RANK[status] >= _RANK[current]):
        state[route_id] = status


def _grokbot_routes(data_dir) -> set[str]:
    """Routes already handed to Grok Bot. The reply arrives as a file, so a restart waits."""
    found = set()
    for ev in replay(data_dir, types="task"):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if data.get("adapter") == "grokbot" and isinstance(data.get("route_id"), str):
            found.add(data["route_id"])
    return found


def close_interrupted(data_dir, emit, state: dict) -> None:
    """Write one `failed` task for every route an adapter already took and did not finish.

    Does not call the adapter. A restart must not send the same work again.
    A grokbot route is left alone: the inbox watcher owns its timeout, and a
    late reply file is still applied. It is not sent a second time.
    """
    routes = {}
    for ev in replay(data_dir, types="route"):
        if isinstance(ev.get("id"), str):
            routes[ev["id"]] = ev
    waiting = _grokbot_routes(data_dir)
    for route_id, status in list(state.items()):
        if status not in _HANDED:
            continue
        if route_id in waiting:
            continue
        route = routes.get(route_id)
        if route is None:
            continue
        emit(task_event(route, status="failed", ref=route["id"], adapter=None, error=_INTERRUPTED))
        state[route_id] = "failed"


def dispatch_one(route: dict, cfg, emit, *, agents: dict | None = None, data_dir=None,
                 force: bool = False) -> None:
    """Emit the events for one route. `emit` may be called zero times.

    `agents` maps an adapter name to an `Agent`. Tests pass this; the live
    process uses `build_agent`. `force` is an approve: the route was already
    held for confirmation, and this call is the one adapter invocation.
    """
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    target = data.get("target")
    if target not in AGENT_TARGETS:
        return
    if data.get("needs_confirmation") and not force:
        emit(task_event(route, status="awaiting_confirmation", ref=route["id"], adapter=None))
        return
    name = adapter_name(cfg, target)
    agent = None if agents is None else agents.get(name)
    if agent is None and (agents is None or name not in agents):
        agent = build_agent(name, cfg)
    if not isinstance(agent, Agent):
        emit(task_event(route, status="failed", ref=route["id"], adapter=name,
                        error=f"没有适配器 {name}"))
        return
    if data_dir is not None and hasattr(agent, "data_dir"):
        agent.data_dir = str(data_dir)
    agent.handle(route, emit)


def _invoke(route: dict, cfg, emit, agents, state: dict, data_dir=None) -> None:
    """Hand `route` to an adapter at most once. `emit` records into `state`.

    `OSError` propagates so the caller can leave the cursor and retry the
    write. Any other exception becomes one `failed` task and is not retried:
    the adapter may already have started the work.
    """
    route_id = route.get("id")
    if not isinstance(route_id, str) or state.get(route_id) in _BLOCK:
        return
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    if data.get("target") not in AGENT_TARGETS:
        state[route_id] = "silent"
        return
    if _is_confirmation_answer(route, cfg, state):
        # Persist it. A later restart must not send "确认" to the agent after
        # the pending task has already been approved or cancelled.
        emit(task_event(route, status="ignored", ref=route["id"], adapter=None))
        return
    _hand_off(route, cfg, emit, agents, state, data_dir, force=False)


def _hand_off(route, cfg, emit, agents, state, data_dir, *, force: bool) -> None:
    """Call the adapter. `OSError` propagates. Any other error is one `failed` task."""
    route_id = route.get("id")
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    try:
        dispatch_one(route, cfg, emit, agents=agents, data_dir=data_dir, force=force)
    except OSError:
        raise
    except Exception:
        log.exception("adapter failed for %s", route_id)
        if state.get(route_id) not in _STICKY:
            emit(task_event(route, status="failed", ref=route["id"], adapter=adapter_name(cfg, data.get("target")),
                            error=_ADAPTER_ERROR))
        return
    if state.get(route_id) not in _STICKY | _HANDED:
        # Nothing was emitted. There is no evidence of a side effect, but
        # there is also nothing for the queue to say. Close it visibly.
        # A hold (`awaiting_confirmation`) is the confirmation path, not a miss.
        if state.get(route_id) == "awaiting_confirmation":
            return
        emit(task_event(route, status="failed", ref=route["id"], adapter=adapter_name(cfg, data.get("target")),
                        error=_NO_REPLY))


def _is_confirmation_answer(route: dict, cfg, state: dict) -> bool:
    """True when this route is only a yes/no/unclear reply to a pending confirmation."""
    if not cfg.confirm.enabled:
        return False
    if not any(status == "awaiting_confirmation" for status in state.values()):
        return False
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    if data.get("needs_confirmation"):
        return False
    text = data.get("text") if isinstance(data.get("text"), str) else ""
    if not text:
        text = data.get("turn_text") if isinstance(data.get("turn_text"), str) else ""
    return classify_answer(text, cfg.turns.triggers) is not None


def _parse_ts(value):
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


def _confirm_key(ev: dict):
    """Reject wins a same-timestamp tie, so one clock tick cannot approve and cancel."""
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    rank = 0 if data.get("decision") == "reject" else 1
    return (_parse_ts(ev.get("ts")) or _EARLIEST, rank, ev.get("id") or "")


def _index(data_dir):
    """Routes, held tasks, read-backs, spoken ends, and confirm events in the log."""
    routes, tasks, briefs, origins, confirms = {}, {}, {}, {}, []
    spoken_at: dict[str, datetime] = {}
    for ev in replay(data_dir):
        if isinstance(ev.get("id"), str):
            origins[ev["id"]] = ev
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        kind = ev.get("type")
        if kind == "route" and isinstance(ev.get("id"), str):
            routes[ev["id"]] = ev
        elif kind == "task" and data.get("status") == "awaiting_confirmation" and isinstance(ev.get("id"), str):
            tasks[ev["id"]] = {
                "id": ev["id"],
                "route_id": data.get("route_id") if isinstance(data.get("route_id"), str) else "",
                "ts": ev.get("ts") or "",
            }
        elif kind == "brief" and data.get("kind") == "confirm" and isinstance(data.get("task_id"), str):
            when = _parse_ts(ev.get("ts"))
            if when is not None and isinstance(ev.get("id"), str):
                briefs.setdefault(data["task_id"], []).append((when, ev["id"]))
        elif kind == "spoken" and isinstance(ev.get("ref"), str):
            when = _parse_ts(ev.get("ts"))
            if when is not None:
                prev = spoken_at.get(ev["ref"])
                if prev is None or when < prev:
                    spoken_at[ev["ref"]] = when
        elif kind == "confirm":
            confirms.append(ev)
    confirms.sort(key=_confirm_key)
    return routes, tasks, briefs, origins, confirms, spoken_at


def _expire(tasks, routes, state, emit, now, cfg) -> None:
    ttl = timedelta(seconds=float(cfg.confirm.ttl_s))
    for info in tasks.values():
        route_id = info["route_id"]
        if state.get(route_id) != "awaiting_confirmation":
            continue
        started = _parse_ts(info["ts"])
        if started is None or now < started + ttl:
            continue
        route = routes.get(route_id)
        if route is None:
            continue
        emit(task_event(route, status="expired", ref=info["id"], adapter=None, ts=now))


def _effective(ev, task_id, briefs, origins, spoken_at):
    """(counts, answer time). A voice answer before the read-back does not count.

    A bare utterance that overlaps the read-back playback does not count either:
    that is the speaker coming back through the microphone. The caller also
    drops an answer timestamped before the task itself existed.
    """
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    source = data.get("source")
    if source == "dashboard":
        when = _parse_ts(ev.get("ts"))
        return when is not None, when
    origin = origins.get(source) if isinstance(source, str) else None
    if origin is None:
        return False, None
    body = origin.get("data") if isinstance(origin.get("data"), dict) else {}
    answer = _parse_ts(body.get("start") or origin.get("ts"))
    if answer is None:
        return False, None
    times = [when for when, _bid in (briefs.get(task_id) or [])]
    if not times or answer < min(times):
        return False, answer
    if origin.get("type") == "utterance" and overlaps_playback(answer, briefs, spoken_at):
        return False, answer
    return True, answer


def _note_expired(ev, task_id, emit) -> None:
    emit(make_event("brief", "dispatch", {
        "priority": "info",
        "summary": _EXPIRED_NOTE,
        "text_ref": ev.get("id"),
        "kind": "note",
        "item_id": None,
        "task_id": task_id,
    }, ts=ev.get("ts"), ref=ev.get("id"), id=f"brief-expired-{ev.get('id')}"))


def _apply_confirm(ev, tasks, routes, briefs, origins, spoken_at, state, cfg, emit, agents, data_dir) -> None:
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    decision = data.get("decision")
    task_id = data.get("task_id")
    if decision not in ("approve", "reject") or not isinstance(task_id, str):
        return
    info = tasks.get(task_id)
    if info is None:
        return
    route = routes.get(info["route_id"])
    if route is None:
        return
    route_id = info["route_id"]
    ok, answer_time = _effective(ev, task_id, briefs, origins, spoken_at)
    if not ok or answer_time is None:
        return
    started = _parse_ts(info["ts"])
    # A confirm written before this task existed must not release it later.
    if started is None or answer_time < started:
        return
    deadline = started + timedelta(seconds=float(cfg.confirm.ttl_s)) if started is not None else None
    if (state.get(route_id) == "awaiting_confirmation" and answer_time is not None
            and deadline is not None and answer_time >= deadline):
        emit(task_event(route, status="expired", ref=info["id"], adapter=None, ts=answer_time))
    if decision == "approve":
        if state.get(route_id) == "expired":
            _note_expired(ev, task_id, emit)
            return
        if state.get(route_id) != "awaiting_confirmation":
            return
        _hand_off(route, cfg, emit, agents, state, data_dir, force=True)
        return
    if decision == "reject" and state.get(route_id) == "awaiting_confirmation":
        emit(task_event(route, status="cancelled", ref=ev.get("id"), adapter=None, ts=answer_time))


def settle(data_dir, cfg, emit, state, agents, now) -> None:
    """Apply confirmations and expiry already in the log. Idempotent.

    Does nothing when `[confirm]` is off, so a task stays `awaiting_confirmation`.
    """
    if not cfg.confirm.enabled:
        return
    routes, tasks, briefs, origins, confirms, spoken_at = _index(data_dir)
    for ev in confirms:
        when = _parse_ts(ev.get("ts")) or now
        _expire(tasks, routes, state, emit, when, cfg)
        _apply_confirm(ev, tasks, routes, briefs, origins, spoken_at, state, cfg, emit, agents, data_dir)
    _expire(tasks, routes, state, emit, now, cfg)


def dispatch_all(data_dir, cfg, *, agents: dict | None = None, now=None) -> dict:
    """Dispatch every route that has not been handed to an adapter yet.

    A route that already has a task is not sent again. One that was handed
    off and never finished is closed with `failed`. Live `serve` does not
    call this: turning the process on must not send the backlog. Returns
    `{"wrote": n}`.
    """
    state = scan_route_state(data_dir)
    wrote = 0

    def emit(event: dict) -> None:
        nonlocal wrote
        if append(data_dir, event, if_absent=True) is not None:
            wrote += 1
        _note(state, event)

    close_interrupted(data_dir, emit, state)
    for route in replay(data_dir, types="route"):
        _invoke(route, cfg, emit, agents, state, data_dir)
    settle(data_dir, cfg, emit, state, agents, now or datetime.now().astimezone())
    return {"wrote": wrote}


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
    """Resume offset, or the current end when this stream has no cursor yet."""
    path = _cursor_path(data_dir)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            offs = raw.get("offsets") if isinstance(raw, dict) else None
            if isinstance(offs, dict) and all(
                    isinstance(k, str) and type(v) is int and v >= 0 for k, v in offs.items()):
                return dict(offs)
        except (OSError, json.JSONDecodeError, UnicodeError):
            log.warning("ignoring unreadable dispatch cursor %s", path)
    offsets = snapshot_offsets(data_dir)
    try:
        _save_cursor(data_dir, offsets)
    except OSError:
        log.exception("dispatch cursor was not written")
    return offsets


def _process(data_dir, route: dict, cfg, agents, state: dict) -> bool:
    """False only when a write failed, so the cursor stays and a restart retries."""
    def emit(event: dict) -> None:
        append(data_dir, event, if_absent=True)
        _note(state, event)

    try:
        _invoke(route, cfg, emit, agents, state, data_dir)
    except OSError:
        log.exception("dispatch event was not written for %s", route.get("id"))
        return False
    return True


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None, agents: dict | None = None) -> None:
    """Follow new route events and dispatch them. Blocks until `stop` or Ctrl-C.

    The first start snapshots the end of the stream and does not backfill.
    Later starts resume from `dispatch.cursor.json`.
    """
    stop = stop or threading.Event()
    offsets = _load_cursor(data_dir)
    durable = dict(offsets)
    state = scan_route_state(data_dir)

    def emit(event: dict) -> None:
        append(data_dir, event, if_absent=True)
        _note(state, event)

    def grokbot_tick() -> None:
        # Reply files and the reply timeout live with the dispatcher so
        # start.sh does not grow another process. A failure here must not
        # stop route delivery.
        try:
            from .grokbot import tick as grokbot_tick
            grokbot_tick(data_dir, cfg, emit)
        except OSError:
            log.exception("grokbot inbox was not processed")
        except Exception:
            log.exception("grokbot inbox failed")

    def confirm_tick() -> None:
        if not cfg.confirm.enabled:
            return
        try:
            settle(data_dir, cfg, emit, state, agents, datetime.now().astimezone())
        except OSError:
            log.exception("confirmations were not applied")

    try:
        grokbot_tick()  # a reply that landed while we were down, before other adapters are closed
        close_interrupted(data_dir, emit, state)
        confirm_tick()
    except OSError:
        log.exception("could not close an interrupted route")
    cursor_ok = True

    def save(cursor: dict) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor)
        except OSError:
            log.exception("dispatch cursor was not written")
            return
        durable = dict(cursor)

    log.info("dispatcher watching %s", data_dir)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            found, offsets = read_new(data_dir, offsets, types=("route", "confirm"))
            for event in found:
                if event.get("type") == "route":
                    cursor_ok = _process(data_dir, event, cfg, agents, state) and cursor_ok
            grokbot_tick()
            confirm_tick()
            if cursor_ok:
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets, types=("route", "confirm"))
            for event in found:
                if event.get("type") == "route":
                    cursor_ok = _process(data_dir, event, cfg, agents, state) and cursor_ok
            grokbot_tick()
            confirm_tick()
            if cursor_ok:
                save(offsets)
        except Exception:
            log.exception("dispatcher did not flush on the way out")
