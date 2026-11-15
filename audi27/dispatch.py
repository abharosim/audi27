"""Hand `route` events to the adapter configured for their target.

This is the execution switch the router deliberately does not flip. It reads
`route` events and writes `task` / `reply` events. It never imports a queue or
a speaker. Routes with target `none` or `queue` are left alone (the queue
module consumes the latter). A route with `needs_confirmation` is not sent:
one `task` with status `awaiting_confirmation` is written so it can be shown,
and the voice confirmation itself is a later step. Each route is handed to
`handle` at most once.

`[dispatch] enabled` only decides whether `audi27 supervise` starts this
process. `audi27 dispatch run` always executes, same as `route run`. The first
live start snapshots the end of the stream and does not replay history.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from .agent import Agent, build_agent, task_event
from .events import append, read_new, replay, snapshot_offsets
from .route import AGENT_TARGETS

log = logging.getLogger("audi27.dispatch")

_CURSOR_NAME = "dispatch.cursor.json"
# A route is handed to an adapter at most once. `sent` / `running` / `done`
# mean the call already happened; calling `handle` again would repeat a real
# agent's side effects. `failed` and `awaiting_confirmation` are finished and
# have no reply. A reply is finished too. An interrupted chain (handed off,
# process died before a reply or `failed`) is closed with one `failed` task
# instead of being sent again.
_HANDED = frozenset({"sent", "running", "done"})
_TERMINAL = frozenset({"failed", "awaiting_confirmation", "reply"})
_INTERRUPTED = "执行中断，没有重新发送。"
_ADAPTER_ERROR = "适配器出错，没有重新发送。"
_NO_REPLY = "适配器没有回复，没有重新发送。"


def adapter_name(cfg, target: str) -> str:
    """Per-target name from `[dispatch.adapters]`, otherwise `fake`."""
    adapters = cfg.dispatch.adapters if isinstance(cfg.dispatch.adapters, dict) else {}
    name = adapters.get(target)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "fake"


def scan_route_state(data_dir) -> dict[str, str]:
    """route id -> `reply`, or the task status that wins so far.

    A reply, `failed`, or `awaiting_confirmation` sticks. Progress only moves
    forward through `sent`, `running`, `done`.
    """
    state: dict[str, str] = {}
    rank = {"sent": 1, "running": 2, "done": 3}
    for ev in replay(data_dir, types=("task", "reply")):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        route_id = data.get("route_id")
        if not isinstance(route_id, str):
            continue
        if ev.get("type") == "reply":
            state[route_id] = "reply"
            continue
        status = data.get("status")
        current = state.get(route_id)
        if current in _TERMINAL:
            continue
        if status in ("failed", "awaiting_confirmation"):
            state[route_id] = status
            continue
        if status in rank and (current not in rank or rank[status] >= rank[current]):
            state[route_id] = status
    return state


def _note(state: dict, event: dict) -> None:
    """Update `state` from one event this process just emitted."""
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    route_id = data.get("route_id")
    if not isinstance(route_id, str):
        return
    if event.get("type") == "reply":
        state[route_id] = "reply"
        return
    if event.get("type") != "task":
        return
    status = data.get("status")
    if not isinstance(status, str) or state.get(route_id) in _TERMINAL:
        return
    rank = {"sent": 1, "running": 2, "done": 3}
    current = state.get(route_id)
    if status in ("failed", "awaiting_confirmation"):
        state[route_id] = status
    elif status in rank and (current not in rank or rank[status] >= rank[current]):
        state[route_id] = status


def close_interrupted(data_dir, emit, state: dict) -> None:
    """Write one `failed` task for every route an adapter already took and did not finish.

    Does not call the adapter. A restart must not send the same work again.
    """
    routes = {}
    for ev in replay(data_dir, types="route"):
        if isinstance(ev.get("id"), str):
            routes[ev["id"]] = ev
    for route_id, status in list(state.items()):
        if status not in _HANDED:
            continue
        route = routes.get(route_id)
        if route is None:
            continue
        emit(task_event(route, status="failed", ref=route["id"], adapter=None, error=_INTERRUPTED))
        state[route_id] = "failed"


def dispatch_one(route: dict, cfg, emit, *, agents: dict | None = None) -> None:
    """Emit the events for one route. `emit` may be called zero times.

    `agents` maps an adapter name to an `Agent`. Tests pass this; the live
    process uses `build_agent`.
    """
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    target = data.get("target")
    if target not in AGENT_TARGETS:
        return
    if data.get("needs_confirmation"):
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
    agent.handle(route, emit)


def _invoke(route: dict, cfg, emit, agents, state: dict) -> None:
    """Hand `route` to an adapter at most once. `emit` records into `state`.

    `OSError` propagates so the caller can leave the cursor and retry the
    write. Any other exception becomes one `failed` task and is not retried:
    the adapter may already have started the work.
    """
    route_id = route.get("id")
    if not isinstance(route_id, str) or state.get(route_id) in _TERMINAL | _HANDED or state.get(route_id) == "silent":
        return
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    if data.get("target") not in AGENT_TARGETS:
        state[route_id] = "silent"
        return
    try:
        dispatch_one(route, cfg, emit, agents=agents)
    except OSError:
        raise
    except Exception:
        log.exception("adapter failed for %s", route_id)
        if state.get(route_id) not in _TERMINAL:
            emit(task_event(route, status="failed", ref=route["id"], adapter=adapter_name(cfg, data.get("target")),
                            error=_ADAPTER_ERROR))
        return
    if state.get(route_id) not in _TERMINAL | _HANDED:
        # Nothing was emitted. There is no evidence of a side effect, but
        # there is also nothing for the queue to say. Close it visibly.
        emit(task_event(route, status="failed", ref=route["id"], adapter=adapter_name(cfg, data.get("target")),
                        error=_NO_REPLY))


def dispatch_all(data_dir, cfg, *, agents: dict | None = None) -> dict:
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
        _invoke(route, cfg, emit, agents, state)
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
        _invoke(route, cfg, emit, agents, state)
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

    try:
        close_interrupted(data_dir, emit, state)
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
            found, offsets = read_new(data_dir, offsets, types="route")
            for event in found:
                cursor_ok = _process(data_dir, event, cfg, agents, state) and cursor_ok
            if cursor_ok:
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets, types="route")
            for event in found:
                cursor_ok = _process(data_dir, event, cfg, agents, state) and cursor_ok
            if cursor_ok:
                save(offsets)
        except Exception:
            log.exception("dispatcher did not flush on the way out")
