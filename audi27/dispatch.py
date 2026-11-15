"""Hand `route` events to the adapter configured for their target.

This is the execution switch the router deliberately does not flip. It reads
`route` events and writes `task` / `reply` events. It never imports a queue or
a speaker. Routes with target `none` or `queue` are left alone (the queue
module consumes the latter). A route with `needs_confirmation` is not sent:
one `task` with status `awaiting_confirmation` is written so it can be shown,
and the voice confirmation itself is a later step.

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
# `done` is not enough to skip a route: the reply is a separate write, and a
# crash between them must run the adapter again. Ids are stable, so sent /
# running / done already on disk are not duplicated. `failed` and
# `awaiting_confirmation` have no reply.
_FINISHED_TASK = frozenset({"failed", "awaiting_confirmation"})


def adapter_name(cfg, target: str) -> str:
    """Per-target name from `[dispatch.adapters]`, otherwise `fake`."""
    adapters = cfg.dispatch.adapters if isinstance(cfg.dispatch.adapters, dict) else {}
    name = adapters.get(target)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "fake"


def terminal_routes(data_dir) -> set[str]:
    """Routes that must not be sent again: a reply, a failure, or a held confirmation."""
    done = set()
    for ev in replay(data_dir, types=("task", "reply")):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        route_id = data.get("route_id")
        if not isinstance(route_id, str):
            continue
        if ev.get("type") == "reply" or data.get("status") in _FINISHED_TASK:
            done.add(route_id)
    return done


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


def dispatch_all(data_dir, cfg, *, agents: dict | None = None) -> dict:
    """Dispatch every route that has no reply, failure, or held confirmation yet. Idempotent.

    Live `serve` does not call this: turning the process on must not send the
    backlog. Tests and a manual catch-up use it. Returns `{"wrote": n}`.
    """
    done = terminal_routes(data_dir)
    wrote = 0

    def emit(event: dict) -> None:
        nonlocal wrote
        if append(data_dir, event, if_absent=True) is not None:
            wrote += 1

    for route in replay(data_dir, types="route"):
        if route["id"] in done:
            continue
        dispatch_one(route, cfg, emit, agents=agents)
        done.add(route["id"])
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


def _process(data_dir, route: dict, cfg, agents, done: set[str]) -> bool:
    """False only when a write failed, so the cursor stays and a restart retries."""
    if route.get("id") in done:
        return True

    def emit(event: dict) -> None:
        stored = append(data_dir, event, if_absent=True)
        if stored is None:
            return
        data = event.get("data") or {}
        route_id = data.get("route_id")
        if not isinstance(route_id, str):
            return
        if event.get("type") == "reply" or (
                event.get("type") == "task" and data.get("status") in _FINISHED_TASK):
            done.add(route_id)

    try:
        dispatch_one(route, cfg, emit, agents=agents)
    except OSError:
        log.exception("dispatch event was not written for %s", route.get("id"))
        return False
    except Exception:
        log.exception("dispatch skipped %s", route.get("id"))
        return True
    done.add(route.get("id"))
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
    done = terminal_routes(data_dir)
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
                cursor_ok = _process(data_dir, event, cfg, agents, done) and cursor_ok
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
                cursor_ok = _process(data_dir, event, cfg, agents, done) and cursor_ok
            if cursor_ok:
                save(offsets)
        except Exception:
            log.exception("dispatcher did not flush on the way out")
