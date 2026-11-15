"""Dispatcher: route in, task and reply out. none/queue/confirmation are not sent."""
import json
import threading
import time
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.agent import Agent, FakeAgent, reply_event, task_event
from audi27.dispatch import dispatch_all, dispatch_one, serve
from audi27.events import append, make_event, replay, snapshot_offsets
from audi27.supervisor import child_specs

TS = "2026-11-15T12:00:00+00:00"


def route(ident, text, *, target="grok", confirm=False, command=None):
    return make_event("route", "route", {
        "target": target, "text": text, "turn_text": text, "rule": "default", "reason": "默认",
        "method": "default", "needs_confirmation": confirm, "command": command,
    }, ts=TS, ref="turn-1", id=ident)


def test_fake_agent_emits_a_task_chain_and_an_echo():
    seen = []
    FakeAgent().handle(route("route-turn-1", "今天天气怎么样"), seen.append)
    assert [e["type"] for e in seen] == ["task", "task", "task", "reply"]
    assert [e["data"]["status"] for e in seen[:3]] == ["sent", "running", "done"]
    assert seen[0]["ref"] == "route-turn-1"
    assert seen[1]["ref"] == seen[0]["id"]
    assert seen[2]["ref"] == seen[1]["id"]
    assert seen[3]["ref"] == seen[2]["id"]
    assert seen[3]["data"]["text"] == "收到：今天天气怎么样"
    assert seen[3]["data"]["needs_decision"] is False
    assert [e["id"] for e in seen] == [
        "task-route-turn-1-sent", "task-route-turn-1-running",
        "task-route-turn-1-done", "reply-route-turn-1",
    ]


def test_fake_delay_happens_after_sent_and_can_fail_or_ask():
    order = []

    def emit(ev):
        order.append(ev["data"].get("status") or "reply")

    def sleep(seconds):
        order.append(seconds)

    FakeAgent(delay_s=1.5, sleep=sleep).handle(route("route-1", "你好"), emit)
    assert order == ["sent", "running", 1.5, "done", "reply"]

    failed = []
    FakeAgent().handle(route("route-2", "请模拟失败一下"), failed.append)
    assert [e["type"] for e in failed] == ["task", "task", "task"]
    assert failed[-1]["data"]["status"] == "failed"
    assert failed[-1]["data"]["error"].startswith("做不到：")
    assert not any(e["type"] == "reply" for e in failed)

    asked = []
    FakeAgent().handle(route("route-3", "这件事需要你决定"), asked.append)
    assert asked[-1]["type"] == "reply" and asked[-1]["data"]["needs_decision"] is True
    assert asked[-1]["data"]["text"].startswith("需要你决定：")

    both = []
    FakeAgent().handle(route("route-4", "模拟失败，而且需要你决定"), both.append)
    assert both[-1]["data"]["status"] == "failed"
    assert not any(e["type"] == "reply" for e in both)


def test_dispatcher_skips_none_and_queue_and_holds_confirmation(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-none", "算了", target="none"))
    append(data, route("route-q", "跳过", target="queue", command="skip"))
    append(data, route("route-pay", "帮我发邮件给张三", confirm=True))
    append(data, route("route-ok", "今天天气怎么样"))
    assert dispatch_all(data, C.Config())["wrote"] == 5  # 1 confirmation task + sent/running/done/reply
    tasks = list(replay(data, types="task"))
    by_route = {}
    for ev in tasks:
        by_route.setdefault(ev["data"]["route_id"], []).append(ev["data"]["status"])
    assert "route-none" not in by_route and "route-q" not in by_route
    assert by_route["route-pay"] == ["awaiting_confirmation"]
    assert by_route["route-ok"] == ["sent", "running", "done"]
    replies = list(replay(data, types="reply"))
    assert [e["data"]["route_id"] for e in replies] == ["route-ok"]
    assert replies[0]["ref"] == "task-route-ok-done"

    again = dispatch_all(data, C.Config())
    assert again["wrote"] == 0
    assert len(list(replay(data, types="task"))) == 4


class _Counting(Agent):
    def __init__(self):
        self.calls = []

    def handle(self, route, emit):
        self.calls.append(route["id"])


def _rewind_dispatch(data):
    offs = {name: 0 for name in snapshot_offsets(data)}
    (Path(data) / "dispatch.cursor.json").write_text(
        json.dumps({"offsets": offs}), encoding="utf-8")


def test_a_done_task_without_a_reply_is_not_sent_again(tmp_path):
    """The adapter already ran. A missing reply must not run it a second time."""
    data = tmp_path / "data"
    append(data, route("route-1", "今天天气怎么样"))
    for status, ref in (("sent", "route-1"), ("running", "task-route-1-sent"), ("done", "task-route-1-running")):
        append(data, make_event("task", "dispatch", {
            "route_id": "route-1", "target": "grok", "adapter": "fake",
            "status": status, "text": "今天天气怎么样",
        }, ts=TS, ref=ref, id=f"task-route-1-{status}"))
    agent = _Counting()
    assert dispatch_all(data, C.Config(), agents={"fake": agent})["wrote"] == 1
    assert agent.calls == []
    tasks = list(replay(data, types="task"))
    assert [ev["data"]["status"] for ev in tasks] == ["sent", "running", "done", "failed"]
    assert tasks[-1]["data"]["error"]
    assert list(replay(data, types="reply")) == []
    assert dispatch_all(data, C.Config(), agents={"fake": agent})["wrote"] == 0
    assert agent.calls == []


def test_a_raising_adapter_fails_once_and_is_not_retried(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-1", "你好"))

    class Boom(_Counting):
        def handle(self, route, emit):
            super().handle(route, emit)
            emit(task_event(route, status="sent", ref=route["id"], adapter="fake"))
            raise RuntimeError("timeout")

    agent = Boom()
    assert dispatch_all(data, C.Config(), agents={"fake": agent})["wrote"] == 2
    assert agent.calls == ["route-1"]
    tasks = list(replay(data, types="task"))
    assert [ev["data"]["status"] for ev in tasks] == ["sent", "failed"]
    assert list(replay(data, types="reply")) == []
    assert dispatch_all(data, C.Config(), agents={"fake": agent})["wrote"] == 0
    assert agent.calls == ["route-1"]


def test_a_reply_may_arrive_after_handle_returns(tmp_path):
    """A slow adapter can emit the reply after handle returns, without a second call."""
    data = tmp_path / "data"
    append(data, route("route-1", "今天天气怎么样"))
    held = {}

    class Slow(_Counting):
        def handle(self, route, emit):
            super().handle(route, emit)
            sent = task_event(route, status="sent", ref=route["id"], adapter="fake")
            emit(sent)
            emit(task_event(route, status="running", ref=sent["id"], adapter="fake"))
            held["emit"] = emit
            held["route"] = route

    agent = Slow()
    assert dispatch_all(data, C.Config(), agents={"fake": agent})["wrote"] == 2
    assert agent.calls == ["route-1"]
    assert list(replay(data, types="reply")) == []
    assert not any(ev["data"]["status"] == "failed" for ev in replay(data, types="task"))
    done = task_event(held["route"], status="done", ref="task-route-1-running", adapter="fake")
    held["emit"](done)
    held["emit"](reply_event(held["route"], ref=done["id"], text="收到：今天天气怎么样",
                             needs_decision=False, adapter="fake"))
    assert [ev["data"]["text"] for ev in replay(data, types="reply")] == ["收到：今天天气怎么样"]
    assert not any(ev["data"]["status"] == "failed" for ev in replay(data, types="task"))

    class Never(_Counting):
        def handle(self, route, emit):
            super().handle(route, emit)
            raise AssertionError("dispatched twice")

    assert dispatch_all(data, C.Config(), agents={"fake": Never()})["wrote"] == 0


def test_restart_before_the_reply_does_not_call_the_adapter(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-1", "你好"))

    class Slow(_Counting):
        def handle(self, route, emit):
            super().handle(route, emit)
            emit(task_event(route, status="sent", ref=route["id"], adapter="fake"))
            emit(task_event(route, status="running", ref="task-route-1-sent", adapter="fake"))

    first = Slow()
    dispatch_all(data, C.Config(), agents={"fake": first})
    assert first.calls == ["route-1"]
    second = Slow()
    assert dispatch_all(data, C.Config(), agents={"fake": second})["wrote"] == 1
    assert second.calls == []
    failed = [ev for ev in replay(data, types="task") if ev["data"]["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["data"]["error"]
    assert list(replay(data, types="reply")) == []
    assert dispatch_all(data, C.Config(), agents={"fake": second})["wrote"] == 0


def test_serve_restart_closes_an_interrupted_route_without_calling_again(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-1", "你好"))
    _rewind_dispatch(data)
    calls = []

    class Once(_Counting):
        def handle(self, route, emit):
            super().handle(route, emit)
            calls.append("handle")
            emit(task_event(route, status="sent", ref=route["id"], adapter="fake"))
            emit(task_event(route, status="running", ref="task-route-1-sent", adapter="fake"))

    cfg = C.Config()
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "agents": {"fake": Once()}},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    deadline = time.time() + 3
    while time.time() < deadline and "handle" not in calls:
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert calls == ["handle"]
    assert not thread.is_alive()

    class Never(_Counting):
        def handle(self, route, emit):
            super().handle(route, emit)
            calls.append("again")

    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "agents": {"fake": Never()}},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    deadline = time.time() + 3
    while time.time() < deadline and not any(
            ev["data"]["status"] == "failed" for ev in replay(data, types="task")):
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert calls == ["handle"]
    assert any(ev["data"]["status"] == "failed" for ev in replay(data, types="task"))
    assert list(replay(data, types="reply")) == []


def test_unknown_adapter_fails_the_task_instead_of_raising(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-1", "你好"))
    cfg = C.Config()
    cfg.dispatch.adapters["grok"] = "claude"
    assert dispatch_all(data, cfg)["wrote"] == 1
    tasks = list(replay(data, types="task"))
    assert tasks[0]["data"]["status"] == "failed"
    assert "claude" in tasks[0]["data"]["error"]
    assert list(replay(data, types="reply")) == []


def test_dispatch_one_can_be_given_an_agent_directly():
    seen = []
    agent = FakeAgent()
    dispatch_one(route("route-1", "你好"), C.Config(), seen.append, agents={"fake": agent})
    assert seen[-1]["type"] == "reply"


def test_live_dispatch_does_not_backfill_and_does_not_duplicate(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-old", "旧的"))
    cfg = C.Config()
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.25)
    assert list(replay(data, types="task")) == []
    append(data, route("route-new", "新的"))
    found = []
    deadline = time.time() + 3
    while time.time() < deadline:
        found = list(replay(data, types="reply"))
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert [e["data"]["route_id"] for e in found] == ["route-new"]
    assert not thread.is_alive()

    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.3)
    stop.set()
    thread.join(2)
    assert [e["id"] for e in replay(data, types="reply")] == ["reply-route-new"]


def test_dispatch_queue_speak_config_defaults_off_and_supervisor_flags(tmp_path):
    cfg = C.Config()
    assert cfg.dispatch.enabled is False and cfg.queue.enabled is False and cfg.speak.enabled is False
    assert cfg.fake.delay_s == 0.0 and cfg.speak.voice == "Tingting"
    assert cfg.dispatch.adapters == {}
    example = C.load(Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.dispatch.enabled is False and example.queue.enabled is False
    assert example.speak.enabled is False and example.speak.voice == "Tingting"
    assert example.dispatch.adapters["claude_code"] == "fake"
    path = tmp_path / "c.toml"
    path.write_text(
        '[dispatch]\nenabled = true\n[dispatch.adapters]\ngrok = "fake"\n'
        '[fake]\ndelay_s = 1\n[queue]\nenabled = true\nsummary_chars = 40\n'
        '[speak]\nenabled = true\nvoice = " Meijia "\n')
    loaded = C.load(path)
    assert loaded.dispatch.enabled is True and loaded.dispatch.adapters == {"grok": "fake"}
    assert loaded.fake.delay_s == 1.0 and loaded.queue.summary_chars == 40
    assert loaded.speak.voice == "Meijia"
    for bad in ("[dispatch]\nenabled = \"yes\"\n", "[dispatch.adapters]\nnope = \"fake\"\n",
                "[fake]\ndelay_s = -1\n", "[queue]\nsummary_chars = 0\n", "[speak]\nvoice = \"\"\n"):
        path.write_text(bad)
        with pytest.raises(C.ConfigError):
            C.load(path)
    off = child_specs(None, "127.0.0.1", 8727)
    on = child_specs("/tmp/audi27.toml", "127.0.0.1", 8727, dispatch=True, queue=True, speak=True)
    assert "dispatch" not in off and "queue" not in off and "speak" not in off
    assert on["dispatch"][-2:] == ["dispatch", "run"]
    assert on["queue"][-2:] == ["queue", "run"]
    assert on["speak"][-2:] == ["speak", "run"]
