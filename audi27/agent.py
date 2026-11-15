"""Agent adapter contract, and the only adapter implemented so far: `fake`.

A real adapter (Claude Code next, then Grok, Muse, the local model) implements
`Agent.handle`. Given one `route` event it emits `task` events and, when the
work succeeds, one `reply`. It does not read or write the event log; the
dispatcher does that. `ref` links each event to the one before it, so the reply
chains back to the route.

`fake` echoes the route text, waits `delay_s` (0 in tests), and can fail or
ask for a decision when the text contains the configured marker. Nothing here
imports another secretary module.
"""
from __future__ import annotations

import time

from .events import make_event

DISPATCH_SOURCE = "dispatch"


class Agent:
    """Minimal adapter. Subclasses set `name` and implement `handle`."""

    name = "agent"

    def handle(self, route: dict, emit) -> None:
        """Emit tasks, then a reply. `emit(event)` is called once per event, in order.

        On success the statuses are `sent`, `running`, `done`, then a `reply`
        whose `ref` is the done task. On failure the last task is `failed` and
        there is no reply. `awaiting_confirmation` is emitted by the dispatcher,
        not by an adapter: a route that needs confirmation is never handed here.
        """
        raise NotImplementedError


def route_text(route: dict) -> str:
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    turn = data.get("turn_text")
    return turn.strip() if isinstance(turn, str) else ""


def route_target(route: dict) -> str:
    data = route.get("data") if isinstance(route.get("data"), dict) else {}
    target = data.get("target")
    return target if isinstance(target, str) else ""


def task_event(route: dict, *, status: str, ref: str, adapter: str | None, error: str | None = None) -> dict:
    data = {
        "route_id": route["id"],
        "target": route_target(route),
        "adapter": adapter,
        "status": status,
        "text": route_text(route),
    }
    if error:
        data["error"] = error
    return make_event("task", DISPATCH_SOURCE, data, ts=route.get("ts"), ref=ref,
                      id=f"task-{route['id']}-{status}")


def reply_event(route: dict, *, ref: str, text: str, needs_decision: bool, adapter: str) -> dict:
    return make_event("reply", DISPATCH_SOURCE, {
        "route_id": route["id"],
        "target": route_target(route),
        "adapter": adapter,
        "text": text,
        "needs_decision": needs_decision,
    }, ts=route.get("ts"), ref=ref, id=f"reply-{route['id']}")


def _marked(text: str, marker: str) -> bool:
    marker = marker.strip() if isinstance(marker, str) else ""
    return bool(marker) and marker in text


class FakeAgent(Agent):
    """Echo the route text. `delay_s` sleeps between `running` and the result.

    The text is a failure when it contains `fail_marker` (no reply). It asks
    for a decision when it contains `decision_marker`. A text that contains
    both fails: there is nothing to decide if the call did not succeed.
    `sleep` is injectable so tests do not wait.
    """

    name = "fake"

    def __init__(self, *, delay_s: float = 0.0, fail_marker: str = "模拟失败",
                 decision_marker: str = "需要你决定", sleep=None):
        self.delay_s = float(delay_s)
        self.fail_marker = fail_marker
        self.decision_marker = decision_marker
        self.sleep = time.sleep if sleep is None else sleep

    def handle(self, route: dict, emit) -> None:
        text = route_text(route)
        sent = task_event(route, status="sent", ref=route["id"], adapter=self.name)
        running = task_event(route, status="running", ref=sent["id"], adapter=self.name)
        emit(sent)
        emit(running)
        if self.delay_s > 0:
            self.sleep(self.delay_s)
        if _marked(text, self.fail_marker):
            emit(task_event(route, status="failed", ref=running["id"], adapter=self.name,
                            error=f"做不到：{text}" if text else "做不到。"))
            return
        done = task_event(route, status="done", ref=running["id"], adapter=self.name)
        emit(done)
        if _marked(text, self.decision_marker):
            body = f"需要你决定：{text}" if text else "需要你决定。"
            needs = True
        else:
            body = f"收到：{text}" if text else "收到。"
            needs = False
        emit(reply_event(route, ref=done["id"], text=body, needs_decision=needs, adapter=self.name))


def build_agent(name: str, cfg) -> Agent | None:
    """The adapter named in config, or None when this build does not know it.

    Only `fake` exists. Claude Code and the others will be added behind this
    same function; callers already treat None as "emit a failed task".
    """
    if name != "fake":
        return None
    fake = cfg.fake
    return FakeAgent(delay_s=fake.delay_s, fail_marker=fake.fail_marker,
                     decision_marker=fake.decision_marker)
