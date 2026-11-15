"""Dispatcher: route in, task and reply out. none/queue/confirmation are not sent."""
import threading
import time
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.agent import FakeAgent
from audi27.dispatch import dispatch_all, dispatch_one, serve
from audi27.events import append, make_event, replay
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


def test_a_done_task_without_a_reply_is_sent_again(tmp_path):
    """A crash between `done` and the reply must not drop the reply on restart."""
    data = tmp_path / "data"
    event = route("route-1", "今天天气怎么样")
    append(data, event)
    for status, ref in (("sent", "route-1"), ("running", "task-route-1-sent"), ("done", "task-route-1-running")):
        append(data, make_event("task", "dispatch", {
            "route_id": "route-1", "target": "grok", "adapter": "fake",
            "status": status, "text": "今天天气怎么样",
        }, ts=TS, ref=ref, id=f"task-route-1-{status}"))
    assert dispatch_all(data, C.Config())["wrote"] == 1
    replies = list(replay(data, types="reply"))
    assert len(replies) == 1 and replies[0]["ref"] == "task-route-1-done"
    assert dispatch_all(data, C.Config())["wrote"] == 0


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
