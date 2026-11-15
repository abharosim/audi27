"""Confirmation gate: read it back, wait for an explicit yes or no, then dispatch once.

The adapter is never called from a needs_confirmation route unless a `confirm`
event says approve, the answer is after the read-back, and the task is still
pending. Everything here is the event log; no audio device.
"""
import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from audi27 import config as C
from audi27.agent import FakeAgent
from audi27.confirm import classify_answer, plan, record_decision, serve, sync
from audi27.dispatch import dispatch_all
from audi27.events import append, make_event, replay
from audi27.queue import ReportQueue
from audi27.queue import sync as sync_queue
from audi27.supervisor import child_specs

TS = "2026-11-15T12:00:00+00:00"
BRIEF_TS = "2026-11-15T12:01:00+00:00"
NOW = datetime(2026, 11, 15, 12, 5, tzinfo=timezone.utc)
CLOCK = datetime(2026, 11, 15, 12, 1, tzinfo=timezone.utc)


def cfg_on(**over):
    cfg = C.Config()
    cfg.confirm.enabled = True
    for key, value in over.items():
        setattr(cfg.confirm, key, value)
    return cfg


def route(ident, text, *, target="grok", confirm=False, ts=TS):
    return make_event("route", "route", {
        "target": target, "text": text, "turn_text": text, "rule": "default", "reason": "",
        "method": "default", "needs_confirmation": confirm, "command": None,
    }, ts=ts, ref="turn-src", id=ident)


def awaiting(route_id, text, *, target="grok", ts=TS):
    return make_event("task", "dispatch", {
        "route_id": route_id, "target": target, "adapter": None,
        "status": "awaiting_confirmation", "text": text,
    }, ts=ts, ref=route_id, id=f"task-{route_id}-awaiting_confirmation")


def readback(task_id, summary, ts=BRIEF_TS):
    return make_event("brief", "queue", {
        "priority": "decision", "summary": summary, "text_ref": task_id,
        "kind": "confirm", "item_id": task_id, "task_id": task_id,
    }, ts=ts, ref=task_id, id=f"brief-{task_id}")


def turn(ident, text, start):
    return make_event("turn", "turns", {
        "utterance_ids": [ident + "-u"], "text": text, "raw_text": "达达 " + text,
        "start": start, "end": start, "reason": "gap", "trigger": "达达", "has_suspect": False,
    }, ts=start, ref=ident + "-u", id=ident)


def utterance(ident, text, start, *, status="ok"):
    return make_event("utterance", "transcribe", {
        "id": ident, "text": text, "status": status, "start": start, "end": start,
    }, ts=start, id=ident)


class CountFake(FakeAgent):
    def __init__(self):
        super().__init__()
        self.calls = []

    def handle(self, route, emit):
        self.calls.append(route["id"])
        super().handle(route, emit)


def statuses(data):
    return [ev["data"]["status"] for ev in replay(data, types="task")]


def confirms(data):
    return list(replay(data, types="confirm"))


def dump(data):
    return [(ev["id"], ev["type"], ev["data"]) for ev in replay(data)]


# --- words -------------------------------------------------------------------


def test_confirm_and_cancel_words_and_everything_else():
    for word in ("确认", "好的", "可以", "发吧", "执行", "yes", "ok", "OK", "Yes"):
        hit = classify_answer(f"达达，{word}")
        assert hit is not None and hit.decision == "approve" and hit.hint == "", word
    for word in ("取消", "算了", "不要", "别", "no", "No"):
        hit = classify_answer(word)
        assert hit is not None and hit.decision == "reject" and hit.hint == "", word
    both = classify_answer("确认取消")
    assert both is not None and both.decision == "unclear"
    assert classify_answer("今天天气怎么样") is None
    assert classify_answer("别人").decision != "reject"
    named = classify_answer("确认发给 Grok")
    assert named.decision == "approve" and "grok" in named.hint


def test_flag_defaults_off_and_example_config_stays_off():
    cfg = C.Config()
    assert cfg.confirm.enabled is False
    assert cfg.confirm.ttl_s == 600
    assert cfg.confirm.window_s == 20
    example = C.load(__import__("pathlib").Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.confirm.enabled is False
    assert example.confirm.ttl_s == 600 and example.confirm.window_s == 20
    off = child_specs(None, "127.0.0.1", 8727)
    on = child_specs("/tmp/audi27.toml", "127.0.0.1", 8727, confirm=True)
    assert "confirm" not in off
    assert on["confirm"][-2:] == ["confirm", "run"]


# --- read-back ---------------------------------------------------------------


def test_pending_task_is_read_back_only_when_the_flag_is_on():
    task = awaiting("route-pay", "帮我发邮件给张三")
    assert ReportQueue().push_all([task]) == []
    briefs = ReportQueue(confirm=True, clock=lambda: CLOCK).push_all([task])
    assert len(briefs) == 1
    data = briefs[0]["data"]
    assert data["priority"] == "decision" and data["kind"] == "confirm"
    assert data["task_id"] == task["id"]
    assert "Grok" in data["summary"] and "张三" in data["summary"]
    assert "确认还是取消？" in data["summary"]
    # The question is the read-back, not a sentence that gets cut off before it.
    assert data["summary"].endswith("确认还是取消？")


def test_a_confirmation_outranks_an_ordinary_reply():
    task = awaiting("route-pay", "帮我发邮件给张三", target="muse")
    reply = make_event("reply", "dispatch", {
        "text": "天气不错。", "target": "grok", "adapter": "fake",
        "needs_decision": False, "route_id": "route-ok",
    }, ts=TS, ref="t", id="reply-1")
    briefs = ReportQueue(confirm=True, clock=lambda: CLOCK).push_all([reply, task])
    assert briefs[0]["data"]["task_id"] == task["id"]
    assert "Muse" in briefs[0]["data"]["summary"]


# --- voice decisions ---------------------------------------------------------


def _pending_events(start_answer, text, *, bare=False, task_text="帮我发邮件给张三", target="grok"):
    task = awaiting("route-pay", task_text, target=target)
    brief = readback(task["id"], "发给 Grok：发邮件。确认还是取消？")
    if bare:
        answer = utterance("u-ans", text, start_answer)
    else:
        answer = turn("turn-ans", text, start_answer)
    return [task, brief, answer], task


def test_a_dada_turn_after_the_read_back_approves_and_one_before_does_not():
    early, task = _pending_events("2026-11-15T12:00:30+00:00", "确认")
    assert plan(early) == []
    late, _ = _pending_events("2026-11-15T12:02:00+00:00", "确认")
    out = plan(late)
    assert [ev["type"] for ev in out] == ["confirm"]
    assert out[0]["data"] == {"task_id": task["id"], "decision": "approve", "source": "turn-ans"}
    assert out[0]["ref"] == task["id"]
    assert plan(late) == out


def test_bare_answer_only_inside_the_window_a_dada_turn_also_later():
    inside, task = _pending_events("2026-11-15T12:01:20+00:00", "好的", bare=True)
    out = plan(inside, window_s=20)
    assert [ev["data"]["decision"] for ev in out] == ["approve"]
    assert out[0]["data"]["source"] == "u-ans"

    outside, _ = _pending_events("2026-11-15T12:01:21+00:00", "好的", bare=True)
    assert plan(outside, window_s=20) == []

    # A wake-word turn is not limited to that window.
    later = outside + [turn("turn-later", "确认", "2026-11-15T12:01:21+00:00")]
    decided = plan(later, window_s=20)
    assert [ev["data"]["source"] for ev in decided] == ["turn-later"]

    # The utterance that will become a turn is not also a bare yes.
    covered = inside + [turn("turn-ans", "确认", "2026-11-15T12:01:20+00:00")]
    covered[2] = utterance("turn-ans-u", "确认", "2026-11-15T12:01:20+00:00")
    covered[3]["data"]["utterance_ids"] = ["turn-ans-u"]
    sources = [ev["data"]["source"] for ev in plan(covered, window_s=20)]
    assert sources == ["turn-ans"]

    # A wake word with no closed turn yet is not a bare answer.
    waiting = [awaiting("route-pay", "帮我发邮件给张三"),
               readback("task-route-pay-awaiting_confirmation", "问。确认还是取消？"),
               utterance("u-wake", "达达，确认", "2026-11-15T12:01:05+00:00")]
    assert plan(waiting, window_s=20) == []


def test_only_the_latest_pending_task_unless_the_user_names_one():
    mail = awaiting("route-mail", "帮我发邮件给张三", target="grok", ts=TS)
    post = awaiting("route-post", "帮我发帖介绍项目", target="muse",
                    ts="2026-11-15T12:00:30+00:00")
    events = [
        mail, post,
        readback(mail["id"], "发给 Grok：发邮件。确认还是取消？", ts=BRIEF_TS),
        readback(post["id"], "发给 Muse：发帖。确认还是取消？",
                 ts="2026-11-15T12:02:00+00:00"),
        turn("turn-yes", "确认", "2026-11-15T12:03:00+00:00"),
    ]
    out = plan(events)
    assert out[0]["data"]["task_id"] == post["id"] and out[0]["data"]["decision"] == "approve"

    events[-1] = turn("turn-yes", "确认邮件", "2026-11-15T12:03:00+00:00")
    named = plan(events)
    assert named[0]["data"]["task_id"] == mail["id"]

    events[-1] = turn("turn-yes", "取消发给 Muse 的", "2026-11-15T12:03:00+00:00")
    rejected = plan(events)
    assert rejected[0]["data"] == {"task_id": post["id"], "decision": "reject", "source": "turn-yes"}


def test_ambiguous_asks_once_then_a_clear_answer_still_works():
    task = awaiting("route-pay", "帮我发邮件给张三")
    brief = readback(task["id"], "发给 Grok：发邮件。确认还是取消？")
    first = turn("turn-1", "嗯", "2026-11-15T12:02:00+00:00")
    second = turn("turn-2", "也许", "2026-11-15T12:03:00+00:00")
    third = turn("turn-3", "好的", "2026-11-15T12:04:00+00:00")
    once = plan([task, brief, first])
    assert [ev["type"] for ev in once] == ["brief"]
    assert once[0]["id"] == f"brief-reask-{task['id']}"
    assert "没听清" in once[0]["data"]["summary"] and "确认还是取消？" in once[0]["data"]["summary"]
    twice = plan([task, brief, first, second])
    assert [ev["id"] for ev in twice] == [once[0]["id"]]
    cleared = plan([task, brief, first, second, third])
    assert [ev["id"] for ev in cleared] == [once[0]["id"], f"confirm-{task['id']}-approve-turn-3"]
    assert cleared[-1]["data"]["decision"] == "approve"


def test_a_new_command_while_something_is_pending_is_not_an_answer():
    task = awaiting("route-pay", "帮我发邮件给张三")
    brief = readback(task["id"], "发给 Grok：发邮件。确认还是取消？")
    weather = turn("turn-w", "今天天气怎么样", "2026-11-15T12:02:00+00:00")
    assert plan([task, brief, weather]) == []


# --- dispatcher safety -------------------------------------------------------


def _open(data, text="帮我发邮件给张三", *, target="grok", route_id="route-pay",
         when=None, now=None, brief_at=None):
    when = when or TS
    now = now or NOW
    append(data, route(route_id, text, confirm=True, target=target, ts=when))
    assert dispatch_all(data, cfg_on(), now=now)["wrote"] == 1
    task_id = f"task-{route_id}-awaiting_confirmation"
    append(data, readback(task_id, f"发给 {target}：{text}。确认还是取消？",
                          ts=brief_at or BRIEF_TS))
    return task_id


def test_no_adapter_call_without_an_approve_event(tmp_path):
    data = tmp_path / "data"
    agent = CountFake()
    _open(data)
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 0
    assert agent.calls == []
    assert statuses(data) == ["awaiting_confirmation"]


def test_approve_dispatches_once_and_a_word_from_before_the_read_back_cannot(tmp_path):
    data = tmp_path / "data"
    task_id = _open(data)
    append(data, turn("turn-early", "确认", "2026-11-15T12:00:30+00:00"))
    assert sync(data, cfg_on())["wrote"] == 0
    agent = CountFake()
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 0
    assert agent.calls == []

    # A confirm event that points at that early turn is still not an approve.
    forged = make_event("confirm", "confirm", {
        "task_id": task_id, "decision": "approve", "source": "turn-early",
    }, ts="2026-11-15T12:00:30+00:00", ref=task_id, id="confirm-forged")
    append(data, forged)
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 0
    assert agent.calls == []

    append(data, turn("turn-yes", "确认", "2026-11-15T12:02:00+00:00"))
    assert sync(data, cfg_on())["wrote"] == 1
    assert confirms(data)[-1]["data"]["decision"] == "approve"
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 4
    assert agent.calls == ["route-pay"]
    assert "sent" in statuses(data) and "done" in statuses(data)
    reply = list(replay(data, types="reply"))
    assert len(reply) == 1 and reply[0]["data"]["text"].startswith("收到：")

    again = dump(data)
    assert sync(data, cfg_on())["wrote"] == 0
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 0
    assert agent.calls == ["route-pay"]
    assert dump(data) == again


def test_double_approve_does_not_dispatch_twice(tmp_path):
    data = tmp_path / "data"
    task_id = _open(data)
    append(data, turn("turn-a", "yes", "2026-11-15T12:02:00+00:00"))
    append(data, turn("turn-b", "ok", "2026-11-15T12:03:00+00:00"))
    written = sync(data, cfg_on())
    assert written["wrote"] == 2
    agent = CountFake()
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)
    assert agent.calls == ["route-pay"]
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)
    assert agent.calls == ["route-pay"]
    assert [ev["data"]["task_id"] for ev in confirms(data)] == [task_id, task_id]


def test_reject_is_terminal(tmp_path):
    data = tmp_path / "data"
    _open(data)
    append(data, turn("turn-no", "算了", "2026-11-15T12:02:00+00:00"))
    append(data, turn("turn-yes", "确认", "2026-11-15T12:03:00+00:00"))
    sync(data, cfg_on())
    agent = CountFake()
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)
    assert agent.calls == []
    assert "cancelled" in statuses(data)
    assert "sent" not in statuses(data)
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 0


def test_expire_is_terminal_and_a_late_approve_is_a_spoken_note(tmp_path):
    data = tmp_path / "data"
    _open(data)
    late_now = datetime(2026, 11, 15, 12, 11, tzinfo=timezone.utc)  # task 12:00 + ttl 600s
    agent = CountFake()
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=late_now)["wrote"] == 1
    assert statuses(data)[-1] == "expired"
    assert agent.calls == []

    append(data, turn("turn-late", "确认", "2026-11-15T12:11:00+00:00"))
    sync(data, cfg_on())
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=late_now)["wrote"] >= 1
    assert agent.calls == []
    assert statuses(data).count("expired") == 1
    notes = [ev for ev in replay(data, types="brief") if "过期" in (ev["data"].get("summary") or "")]
    assert len(notes) == 1
    assert "没有执行" in notes[0]["data"]["summary"]

    from audi27.speak import speak_pending

    spoken = []

    def backend(text, voice, stop_check):
        spoken.append(text)
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    assert speak_pending(data, cfg_on(), backend=backend, speaking=lambda: False) >= 1
    assert any("过期" in line and "没有执行" in line for line in spoken)
    # Saying it twice does not dispatch, and the note is not written again.
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=late_now)["wrote"] == 0
    assert agent.calls == []
    assert sum(1 for ev in replay(data, types="brief") if "过期" in (ev["data"].get("summary") or "")) == 1


def test_flag_off_keeps_today_behaviour_even_if_someone_already_said_yes(tmp_path):
    data = tmp_path / "data"
    append(data, route("route-pay", "帮我发邮件给张三", confirm=True))
    append(data, route("route-yes", "确认", confirm=False, ts="2026-11-15T12:02:00+00:00"))
    task_id = "task-route-pay-awaiting_confirmation"
    append(data, make_event("confirm", "confirm", {
        "task_id": task_id, "decision": "approve", "source": "dashboard",
    }, ts="2026-11-15T12:03:00+00:00", ref=task_id, id="confirm-dash"))
    agent = CountFake()
    off = C.Config()
    dispatch_all(data, off, agents={"fake": agent}, now=NOW)
    assert agent.calls == ["route-yes"]  # an ordinary route still goes out
    assert statuses(data)[0] == "awaiting_confirmation"
    assert "expired" not in statuses(data) and "cancelled" not in statuses(data)
    assert not any(ev["data"]["route_id"] == "route-pay" and ev["data"]["status"] == "sent"
                   for ev in replay(data, types="task"))


def test_a_yes_route_is_not_sent_to_the_agent_while_a_confirmation_is_pending(tmp_path):
    data = tmp_path / "data"
    _open(data)
    append(data, route("route-yes", "确认", confirm=False, ts="2026-11-15T12:02:00+00:00"))
    agent = CountFake()
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)
    assert agent.calls == []
    assert list(replay(data, types="reply")) == []


def test_rebuild_from_the_log_matches_after_a_restart(tmp_path):
    data = tmp_path / "data"
    on = cfg_on()
    append(data, route("route-pay", "帮我发邮件给张三", confirm=True))
    agent = CountFake()
    dispatch_all(data, on, agents={"fake": agent}, now=NOW)
    sync_queue(data, on, clock=lambda: CLOCK)
    append(data, turn("turn-yes", "发吧", "2026-11-15T12:02:00+00:00"))
    sync(data, on)
    dispatch_all(data, on, agents={"fake": agent}, now=NOW)
    assert agent.calls == ["route-pay"]
    snapshot = dump(data)
    # A restarted dispatcher, queue, and confirm step read the same log.
    dispatch_all(data, on, agents={"fake": agent}, now=NOW)
    sync_queue(data, on, clock=lambda: CLOCK)
    sync(data, on)
    assert agent.calls == ["route-pay"]
    assert dump(data) == snapshot


def test_live_confirm_does_not_backfill_and_follows_new_turns(tmp_path):
    data = tmp_path / "data"
    on = cfg_on()
    task = awaiting("route-pay", "帮我发邮件给张三")
    append(data, task)
    append(data, readback(task["id"], "发给 Grok：发邮件。确认还是取消？"))
    append(data, turn("turn-old", "确认", "2026-11-15T12:02:00+00:00"))
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, on),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    import time
    time.sleep(0.25)
    assert confirms(data) == []
    append(data, turn("turn-new", "确认", "2026-11-15T12:03:00+00:00"))
    found = []
    deadline = time.time() + 3
    while time.time() < deadline:
        found = confirms(data)
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert [ev["data"]["source"] for ev in found] == ["turn-new"]
    assert not thread.is_alive()


# --- dashboard ---------------------------------------------------------------


def _post(port, path, body, headers=None):
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=raw, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_dashboard_approve_writes_the_same_confirm_event_and_dispatches(tmp_path):
    from audi27.web import make_server

    data = tmp_path / "data"
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(f'[store]\ndata_dir = "{data}"\n[confirm]\nenabled = true\n', encoding="utf-8")
    # The click is stamped with the wall clock, so the task has to be recent.
    opened = datetime.now(timezone.utc) - timedelta(seconds=20)
    task_id = _open(
        data, when=opened, now=datetime.now(timezone.utc),
        brief_at=opened + timedelta(seconds=1))
    server = make_server(data, port=0, config_path=str(cfg_path))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read()
        assert "确认".encode() in page and b"/api/confirm" in page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/loop") as response:
            body = json.loads(response.read())
        pending = [row for row in body["events"] if row.get("status") == "awaiting_confirmation"]
        assert pending and pending[0]["task_id"] == task_id
        code, saved = _post(port, "/api/confirm", {"task_id": task_id, "decision": "approve"})
        assert code == 200
        assert saved["data"]["decision"] == "approve" and saved["data"]["source"] == "dashboard"
        assert saved["data"]["task_id"] == task_id and saved["ref"] == task_id
        code, _ = _post(port, "/api/confirm", {"task_id": task_id, "decision": "approve"})
        assert code == 200
        assert len(confirms(data)) == 1
        assert _post(port, "/api/confirm", {"task_id": task_id, "decision": "maybe"})[0] == 400
    finally:
        server.shutdown()
        server.server_close()

    agent = CountFake()
    later = datetime.now(timezone.utc) + timedelta(seconds=5)
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=later)
    assert agent.calls == ["route-pay"]
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=later)
    assert agent.calls == ["route-pay"]


def test_record_decision_rejects_a_bad_choice():
    try:
        record_decision("/tmp/does-not-matter", "task-1", "maybe")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _decision(events, phrase, *, bare=False):
    """The confirm decision `phrase` produces against `events`, or None."""
    start = "2026-11-15T12:01:10+00:00"
    if bare:
        events = events + [utterance("u-probe", phrase, start)]
    else:
        events = events + [turn("turn-probe", phrase, start)]
    for ev in plan(events, window_s=20):
        if ev["type"] == "confirm":
            return ev["data"]["decision"]
    return None


def test_questions_negation_and_buried_yes_do_not_approve():
    task = awaiting("route-pay", "帮我发邮件给张三")
    brief = readback(task["id"], "发给 Grok：帮我发邮件给张三。确认还是取消？")
    base = [task, brief]
    # The whole line is the yes-word: still an approval.
    assert _decision(base, "确认") == "approve"
    assert _decision(base, "可以", bare=True) == "approve"
    assert _decision(base, "book") is None
    assert _decision(base, "bookmark") is None
    for phrase in (
        "不确认", "别确认", "不可以", "不是ok", "不要确认",
        "可以吗", "可以吗？", "可以?", "可以？", "好的？", "确认？", "ok?", "yes?", "好呢", "可以呢",
        "好的我想想", "好的我再想想", "好吧我看看",
        "别取消", "算了吧还是发",
        "那就好", "那就行", "这个好", "一个好", "可以的话", "帮我确认一下",
        "好了", "可以了", "行了",
        "好不好", "行不行",
    ):
        assert _decision(base, phrase) != "approve", phrase
        assert _decision(base, phrase, bare=True) != "approve", phrase
    # Negating the yes is a no, and it does cancel. It must not approve.
    assert _decision(base, "不确认") == "reject"
    assert _decision(base, "不可以") == "reject"
    # Hesitation that happens to sit inside the task text is still not a yes.
    think = awaiting("route-think", "帮我想想接下来怎么做")
    think_brief = readback(think["id"], "发给 Grok：帮我想想。确认还是取消？")
    assert _decision([think, think_brief], "好的我想想") != "approve"
    assert _decision([think, think_brief], "好的我想想", bare=True) != "approve"


def test_a_name_that_matches_two_tasks_does_not_approve_either():
    mail = awaiting("route-mail", "发邮件给张三", target="grok", ts=TS)
    other = awaiting("route-other", "把邮件转给李四", target="muse",
                     ts="2026-11-15T12:00:30+00:00")
    events = [
        mail, other,
        readback(mail["id"], "发给 Grok：发邮件。确认还是取消？", ts=BRIEF_TS),
        readback(other["id"], "发给 Muse：转邮件。确认还是取消？",
                 ts="2026-11-15T12:01:30+00:00"),
        turn("turn-yes", "确认邮件", "2026-11-15T12:02:00+00:00"),
    ]
    decided = [ev for ev in plan(events) if ev["type"] == "confirm"]
    assert decided == []
    # One distinctive name still selects that task.
    events[-1] = turn("turn-yes", "确认发给张三", "2026-11-15T12:02:00+00:00")
    named = plan(events)
    assert named[0]["data"]["task_id"] == mail["id"] and named[0]["data"]["decision"] == "approve"


def test_an_answer_before_the_second_read_back_stays_on_the_first():
    first = awaiting("route-mail", "发邮件给张三", ts=TS)
    second = awaiting("route-post", "帮我发帖", target="muse", ts="2026-11-15T12:00:30+00:00")
    events = [
        first, second,
        readback(first["id"], "发给 Grok：发邮件。确认还是取消？", ts=BRIEF_TS),
        turn("turn-yes", "确认", "2026-11-15T12:01:20+00:00"),
        readback(second["id"], "发给 Muse：发帖。确认还是取消？",
                 ts="2026-11-15T12:01:40+00:00"),
    ]
    out = [ev for ev in plan(events) if ev["type"] == "confirm"]
    assert len(out) == 1 and out[0]["data"]["task_id"] == first["id"]


def test_echo_during_playback_cannot_approve_and_a_line_after_it_can():
    task = awaiting("route-pay", "帮我发邮件给张三")
    brief = readback(task["id"], "发给 Grok：帮我发邮件给张三。确认还是取消？")
    spoken = make_event("spoken", "speak", {
        "text": brief["data"]["summary"], "completed": True, "interrupted": False, "backend": "say",
        "brief_id": brief["id"],
    }, ts="2026-11-15T12:01:04+00:00", ref=brief["id"], id="spoken-" + brief["id"])
    during = utterance("u-echo", "确认", "2026-11-15T12:01:03+00:00")
    tail = utterance("u-tail", "确认", "2026-11-15T12:01:04.200+00:00")
    after = utterance("u-yes", "好的", "2026-11-15T12:01:06+00:00")
    assert plan([task, brief, spoken, during], window_s=20) == []
    assert plan([task, brief, spoken, tail], window_s=20) == []
    out = plan([task, brief, spoken, after], window_s=20)
    assert [ev["data"]["decision"] for ev in out] == ["approve"]
    # The full read-back, question and all, is not a yes even with no spoken event.
    echo_line = utterance("u-full", "发给 Grok：帮我发邮件给张三。确认还是取消？",
                          "2026-11-15T12:01:05+00:00")
    assert plan([task, brief, echo_line], window_s=20) == []


def test_playback_echo_does_not_dispatch_and_a_later_yes_does(tmp_path):
    data = tmp_path / "data"
    task_id = _open(data)
    brief_id = f"brief-{task_id}"
    append(data, make_event("spoken", "speak", {
        "text": "发给 Grok：帮我发邮件给张三。确认还是取消？",
        "completed": True, "interrupted": False, "backend": "say", "brief_id": brief_id,
    }, ts="2026-11-15T12:01:04+00:00", ref=brief_id, id="spoken-" + brief_id))
    append(data, utterance("u-echo", "确认", "2026-11-15T12:01:02+00:00"))
    assert sync(data, cfg_on())["wrote"] == 0
    # A confirm event that points at the echo is still not an approval.
    append(data, make_event("confirm", "confirm", {
        "task_id": task_id, "decision": "approve", "source": "u-echo",
    }, ts="2026-11-15T12:01:02+00:00", ref=task_id, id="confirm-echo"))
    agent = CountFake()
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] == 0
    assert agent.calls == []
    append(data, utterance("u-yes", "确认", "2026-11-15T12:01:06+00:00"))
    assert sync(data, cfg_on())["wrote"] == 1
    assert dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)["wrote"] >= 1
    assert agent.calls == ["route-pay"]


def test_same_timestamp_reject_wins_and_a_confirm_from_before_the_task_does_not(tmp_path):
    data = tmp_path / "data"
    task_id = _open(data)
    when = "2026-11-15T12:02:00+00:00"
    append(data, turn("turn-no", "取消", when))
    append(data, turn("turn-yes", "确认", when))
    sync(data, cfg_on())
    agent = CountFake()
    dispatch_all(data, cfg_on(), agents={"fake": agent}, now=NOW)
    assert agent.calls == []
    assert "cancelled" in statuses(data)
    assert "sent" not in statuses(data)

    data2 = tmp_path / "early"
    early_task = _open(data2)
    append(data2, make_event("confirm", "confirm", {
        "task_id": early_task, "decision": "approve", "source": "dashboard",
    }, ts="2026-11-15T11:00:00+00:00", ref=early_task, id="confirm-early"))
    agent2 = CountFake()
    dispatch_all(data2, cfg_on(), agents={"fake": agent2}, now=NOW)
    assert agent2.calls == []
    assert statuses(data2) == ["awaiting_confirmation"]


def test_cross_day_answer_uses_the_timestamp_not_the_file_order():
    # 00:10 +08 on the 16th is 16:10 UTC on the 15th, after the task. The
    # calendar date changes; the comparison is the instant, not the filename.
    task = awaiting("route-pay", "帮我发邮件给张三", ts="2026-11-15T16:00:00+00:00")
    brief = readback(task["id"], "发给 Grok：发邮件。确认还是取消？",
                     ts="2026-11-16T00:10:00+08:00")
    before = turn("turn-before", "确认", "2026-11-15T23:00:00+08:00")
    after = turn("turn-after", "确认", "2026-11-16T00:20:00+08:00")
    assert plan([task, brief, before]) == []
    out = plan([task, brief, after])
    assert [ev["data"]["decision"] for ev in out if ev["type"] == "confirm"] == ["approve"]


def test_confirm_allows_only_a_loopback_host():
    from audi27.web import _loopback_host

    assert _loopback_host("127.0.0.1") and _loopback_host("127.0.0.1:8727")
    assert _loopback_host("localhost") and _loopback_host("[::1]:8727")
    assert not _loopback_host("0.0.0.0") and not _loopback_host("0.0.0.0:8727")
    assert not _loopback_host("evil.example") and not _loopback_host(None)
    assert not _loopback_host("127.0.0.1.evil.example")


def test_confirm_endpoint_refuses_cross_site_get_and_a_gate_that_is_off(tmp_path):
    from audi27.web import make_server

    data = tmp_path / "data"
    off = tmp_path / "off.toml"
    off.write_text(f'[store]\ndata_dir = "{data}"\n[confirm]\nenabled = false\n', encoding="utf-8")
    on = tmp_path / "on.toml"
    on.write_text(f'[store]\ndata_dir = "{data}"\n[confirm]\nenabled = true\n', encoding="utf-8")
    opened = datetime.now(timezone.utc) - timedelta(seconds=30)
    task_id = _open(data, when=opened, now=datetime.now(timezone.utc),
                    brief_at=opened + timedelta(seconds=1))

    def serve(cfg):
        server = make_server(data, port=0, config_path=str(cfg))
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port

    server, port = serve(off)
    try:
        code, body = _post(port, "/api/confirm", {"task_id": task_id, "decision": "approve"})
        assert code == 403 and "off" in body["error"]
        assert confirms(data) == []
        # GET must not approve, even if the query string looks like a decision.
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/confirm?task_id={task_id}&decision=approve")
            raise AssertionError("GET /api/confirm was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()
    assert confirms(data) == []

    server, port = serve(on)
    try:
        # A form body, and a content type that only contains the letters application/json.
        raw = json.dumps({"task_id": task_id, "decision": "approve"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/confirm", data=raw, method="POST",
            headers={"Content-Type": "text/plain;charset=application/json"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("form-like content type was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 415
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/confirm", data=raw, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("form post was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 415
        code, body = _post(port, "/api/confirm", {"task_id": task_id, "decision": "approve"},
                           {"Origin": "http://evil.example", "Host": "evil.example"})
        # urllib's Host may be overwritten by the URL host. Origin mismatch is enough.
        assert code == 403
        code, body = _post(port, "/api/confirm", {"task_id": task_id, "decision": "approve"},
                           {"Sec-Fetch-Site": "cross-site"})
        assert code == 403
        code, body = _post(port, "/api/confirm", {"task_id": "task-not-real", "decision": "approve"})
        assert code == 400
        code, body = _post(port, "/api/confirm", {"task_id": 12, "decision": "approve"})
        assert code == 400
        assert confirms(data) == []
        code, saved = _post(port, "/api/confirm", {"task_id": task_id, "decision": "approve"})
        assert code == 200 and saved["data"]["decision"] == "approve"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/loop") as response:
            payload = json.loads(response.read())
        assert payload["confirm"] is True
    finally:
        server.shutdown()
        server.server_close()
