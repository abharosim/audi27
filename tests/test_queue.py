"""Report queue: replies and failures in, one brief at a time, commands, replay."""
from audi27.events import make_event
from audi27.queue import ReportQueue, first_sentence, summarize

TS = "2026-11-15T12:00:00+00:00"


def reply(ident, text, **extra):
    data = {"text": text, "target": "grok", "adapter": "fake", "needs_decision": False, "route_id": "route-1"}
    data.update(extra)
    return make_event("reply", "dispatch", data, ts=TS, ref="task-1", id=ident)


def failed(ident, error):
    return make_event("task", "dispatch", {
        "route_id": "route-x", "target": "grok", "adapter": "fake",
        "status": "failed", "text": "job", "error": error,
    }, ts=TS, ref="task-run", id=ident)


def task(status):
    return make_event("task", "dispatch", {
        "route_id": "route-1", "target": "grok", "adapter": "fake", "status": status, "text": "x",
    }, ts=TS, ref="route-1", id=f"task-{status}")


def command(ident, name):
    return make_event("route", "route", {
        "target": "queue", "command": name, "text": name, "turn_text": name,
        "rule": f"queue:{name}", "reason": "", "method": "rule", "needs_confirmation": False,
    }, ts=TS, ref="turn-c", id=ident)


def spoken(brief_id, *, interrupted=False, completed=True):
    return make_event("spoken", "speak", {
        "text": "x", "completed": completed, "interrupted": interrupted,
        "backend": "mock", "brief_id": brief_id,
    }, ts=TS, ref=brief_id, id="spoken-" + brief_id)


def test_a_reply_becomes_one_brief_with_the_first_sentence():
    briefs = ReportQueue().push_all([reply("r1", "今天天气不错。下午再看")])
    assert len(briefs) == 1
    data = briefs[0]["data"]
    assert briefs[0]["id"] == "brief-r1" and briefs[0]["ref"] == "r1"
    assert data["priority"] == "done" and data["kind"] == "report"
    assert data["summary"] == "今天天气不错。"
    assert data["text_ref"] == "r1" and data["item_id"] == "r1"


def test_summary_truncates_and_the_hook_can_replace_it():
    long = "这" * 100
    brief = ReportQueue(summary_chars=80).push_all([reply("r1", long)])[0]
    assert brief["data"]["summary"].endswith("…") and len(brief["data"]["summary"]) == 80

    hooked = ReportQueue(summarize_fn=lambda _text: "一句话")
    assert hooked.push_all([reply("r1", "第一句。第二句")])[0]["data"]["summary"] == "一句话"

    def boom(_text):
        raise RuntimeError("down")

    fallen = ReportQueue(summarize_fn=boom)
    assert fallen.push_all([reply("r1", "第一句。第二句")])[0]["data"]["summary"] == "第一句。"
    assert summarize("  第二句没了。还有", limit=80) == "第二句没了。"
    assert first_sentence("Version 3.14 is out. More later", 80) == "Version 3.14 is out."


def test_waiting_items_come_out_by_priority_and_do_not_preempt():
    queue = ReportQueue()
    first = queue.push_all([
        reply("r-done", "完成了甲。还有细节"),
        reply("r-info", "随便说说。忽略", priority="info"),
        failed("t-fail", "连接失败。超时了"),
        reply("r-dec", "要不要发出去。请选择", needs_decision=True),
    ])
    assert [b["id"] for b in first] == ["brief-r-dec"]
    assert first[0]["data"]["priority"] == "decision"
    assert first[0]["data"]["summary"] == "要不要发出去。"

    # A decision that arrives while one is already out waits.
    held = ReportQueue()
    opened = held.push_all([reply("r1", "先说这个。")])
    assert held.push_all([reply("r2", "要决定。", needs_decision=True)]) == []
    nxt = held.push_all([spoken(opened[0]["id"])])
    assert nxt[0]["data"]["text_ref"] == "r2" and nxt[0]["data"]["priority"] == "decision"

    error = queue.push_all([spoken(first[0]["id"])])
    assert error[0]["data"]["priority"] == "error" and error[0]["data"]["summary"] == "连接失败。"
    done = queue.push_all([spoken(error[0]["id"])])
    assert done[0]["data"]["priority"] == "done" and done[0]["data"]["summary"] == "完成了甲。"
    info = queue.push_all([spoken(done[0]["id"])])
    assert info[0]["data"]["priority"] == "info" and info[0]["data"]["summary"] == "随便说说。"
    assert queue.push_all([spoken(info[0]["id"])]) == []


def test_needs_decision_wins_over_an_explicit_info_priority():
    brief = ReportQueue().push_all([
        reply("r1", "请拍板。其余不重要", needs_decision=True, priority="info"),
    ])[0]
    assert brief["data"]["priority"] == "decision"


def test_only_a_failed_task_joins_the_queue():
    queue = ReportQueue()
    assert queue.push_all([task(status) for status in
                           ("sent", "running", "done", "awaiting_confirmation")]) == []
    assert queue.items == []


def test_the_same_reply_is_not_queued_twice():
    queue = ReportQueue()
    event = reply("r1", "甲。")
    assert len(queue.push_all([event])) == 1
    assert queue.push_all([event]) == []


def test_agent_routes_are_not_queue_commands():
    event = make_event("route", "route", {
        "target": "grok", "command": "skip", "text": "跳过", "turn_text": "跳过",
        "rule": "default", "reason": "", "method": "default", "needs_confirmation": False,
    }, ts=TS, ref="t", id="route-1")
    assert ReportQueue().push_all([event]) == []


def test_skip_drops_the_current_item_and_speaks_the_next():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。"), reply("b", "乙。")])
    out = queue.push_all([command("c1", "skip"), spoken("brief-a")])
    assert [b["data"]["text_ref"] for b in out] == ["b"]
    assert queue.push_all([command("c2", "skip"), spoken("brief-b")]) == []
    assert queue.items == [] and queue.inflight is None


def test_later_moves_the_current_item_behind_the_others():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。"), reply("b", "乙。")])
    out = queue.push_all([command("c", "later"), spoken("brief-a")])
    assert out[0]["data"]["text_ref"] == "b"
    out = queue.push_all([spoken("brief-b")])
    assert out[0]["data"]["text_ref"] == "a" and out[0]["id"] == "brief-a-2"


def test_later_on_the_only_item_does_not_repeat_it_immediately():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。")])
    assert queue.push_all([command("c", "later"), spoken("brief-a")]) == []
    assert [it.id for it in queue.items] == ["a"] and queue.items[0].deferred
    out = queue.push_all([reply("b", "乙。")])
    assert out[0]["data"]["text_ref"] == "b"
    out = queue.push_all([spoken("brief-b")])
    assert out[0]["data"]["text_ref"] == "a"


def test_pending_lists_what_is_left_without_dropping_it():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲事项。"), reply("b", "乙事项。")])
    out = queue.push_all([command("c", "pending"), spoken("brief-a")])
    assert out[0]["data"]["kind"] == "pending" and out[0]["data"]["priority"] == "info"
    assert "乙事项" in out[0]["data"]["summary"] and "1 条" in out[0]["data"]["summary"]
    assert "甲事项" not in out[0]["data"]["summary"]
    assert out[0]["ref"] == "c"
    nxt = queue.push_all([spoken(out[0]["id"])])
    assert nxt[0]["data"]["text_ref"] == "b"


def test_pending_when_nothing_is_queued():
    out = ReportQueue().push_all([command("c", "pending")])
    assert out[0]["data"]["summary"] == "没有待汇报的。"


def test_details_speaks_the_full_text_of_the_current_item():
    text = "第一句。第二句也要能听到。"
    queue = ReportQueue()
    first = queue.push_all([reply("a", text)])
    assert first[0]["data"]["summary"] == "第一句。"
    out = queue.push_all([command("c", "details"), spoken(first[0]["id"])])
    assert out[0]["data"]["kind"] == "details"
    assert out[0]["data"]["summary"] == text
    assert out[0]["data"]["text_ref"] == "a" and out[0]["ref"] == "c"


def test_skip_during_details_does_not_drop_the_next_report():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。全文"), reply("b", "乙。")])
    details = queue.push_all([command("d", "details"), spoken("brief-a")])
    assert details[0]["data"]["kind"] == "details"
    assert queue.inflight is None and queue.holding == details[0]["id"]
    assert queue.push_all([command("s", "skip")]) == []
    assert [it.id for it in queue.items] == ["b"]
    nxt = queue.push_all([spoken(details[0]["id"])])
    assert nxt[0]["data"]["text_ref"] == "b"


def test_later_during_pending_does_not_push_back_the_next_report():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。"), reply("b", "乙。")])
    pending = queue.push_all([command("p", "pending"), spoken("brief-a")])
    assert pending[0]["data"]["kind"] == "pending"
    assert queue.push_all([command("c", "later")]) == []
    assert [it.id for it in queue.items] == ["b"] and queue.items[0].deferred is False
    nxt = queue.push_all([spoken(pending[0]["id"])])
    assert nxt[0]["data"]["text_ref"] == "b"


def test_details_uses_the_last_item_when_nothing_is_in_flight():
    text = "第一句。第二句全文。"
    queue = ReportQueue()
    first = queue.push_all([reply("a", text)])
    assert queue.push_all([spoken(first[0]["id"])]) == []
    assert queue.inflight is None and queue.last.id == "a"
    out = queue.push_all([command("c", "details")])
    assert out[0]["data"]["summary"] == text and out[0]["data"]["kind"] == "details"


def test_close_discards_everything_still_queued():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。"), reply("b", "乙。"), reply("c", "丙。")])
    assert queue.push_all([command("x", "close"), spoken("brief-a")]) == []
    assert queue.items == [] and queue.inflight is None


def test_an_interrupted_brief_is_spoken_again():
    queue = ReportQueue()
    queue.push_all([reply("a", "甲。")])
    sp = spoken("brief-a", interrupted=True, completed=False)
    out = queue.push_all([sp])
    assert out[0]["id"] == "brief-a-2" and out[0]["data"]["text_ref"] == "a"
    # The same spoken event must not mint another copy. While the user is still
    # talking there is no further spoken event, so the re-offer stays in flight.
    assert queue.push_all([sp]) == []
    assert queue.holding == "brief-a-2" and queue.items == []
    rebuilt = ReportQueue()
    again = rebuilt.push_all([reply("a", "甲。"), sp])
    assert [b["id"] for b in again] == ["brief-a", "brief-a-2"]
    assert rebuilt.holding == queue.holding and rebuilt.items == []


def test_skip_and_later_and_close_beat_barge_in():
    skipped = ReportQueue()
    skipped.push_all([reply("a", "甲。"), reply("b", "乙。")])
    skipped.push_all([command("s", "skip")])
    out = skipped.push_all([spoken("brief-a", interrupted=True, completed=False)])
    assert [b["data"]["text_ref"] for b in out] == ["b"]
    assert [it.id for it in skipped.items] == []

    later = ReportQueue()
    later.push_all([reply("a", "甲。"), reply("b", "乙。")])
    later.push_all([command("l", "later")])
    out = later.push_all([spoken("brief-a", interrupted=True, completed=False)])
    assert [b["data"]["text_ref"] for b in out] == ["b"]
    assert [it.id for it in later.items] == ["a"] and later.items[0].deferred
    # The only item, deferred, is not said again until something else arrives.
    only = ReportQueue()
    only.push_all([reply("a", "甲。")])
    only.push_all([command("l", "later")])
    assert only.push_all([spoken("brief-a", interrupted=True, completed=False)]) == []
    assert [it.id for it in only.items] == ["a"] and only.items[0].deferred

    closed = ReportQueue()
    closed.push_all([reply("a", "甲。"), reply("b", "乙。")])
    closed.push_all([command("c", "close")])
    assert closed.push_all([spoken("brief-a", interrupted=True, completed=False)]) == []
    assert closed.items == [] and closed.holding is None


def test_an_interrupted_details_or_pending_line_is_reoffered_once():
    text = "第一句。第二句全文。"
    queue = ReportQueue()
    first = queue.push_all([reply("a", text)])
    details = queue.push_all([command("d", "details"), spoken(first[0]["id"])])
    assert details[0]["id"] == "brief-details-d" and details[0]["data"]["summary"] == text
    sp = spoken(details[0]["id"], interrupted=True, completed=False)
    out = queue.push_all([sp])
    assert out[0]["id"] == "brief-details-d-2" and out[0]["data"]["summary"] == text
    assert queue.push_all([sp]) == []
    assert queue.push_all([spoken(out[0]["id"])]) == []
    assert queue.holding is None and queue.items == []

    pending_q = ReportQueue()
    pending_q.push_all([reply("a", "甲。"), reply("b", "乙。")])
    pending = pending_q.push_all([command("p", "pending"), spoken("brief-a")])
    again = pending_q.push_all([spoken(pending[0]["id"], interrupted=True, completed=False)])
    assert again[0]["id"] == "brief-pending-p-2" and again[0]["data"]["kind"] == "pending"
    nxt = pending_q.push_all([spoken(again[0]["id"])])
    assert nxt[0]["data"]["text_ref"] == "b"

    # close during the details line drops it, even if that line is then interrupted.
    dropped = ReportQueue()
    first = dropped.push_all([reply("a", text)])
    details = dropped.push_all([command("d", "details"), spoken(first[0]["id"])])
    assert dropped.push_all([command("c", "close")]) == []
    assert dropped.push_all([spoken(details[0]["id"], interrupted=True, completed=False)]) == []
    assert dropped.holding is None and dropped.items == []


def test_replaying_the_same_events_rebuilds_the_same_briefs():
    events = [reply("a", "甲。细节"), reply("b", "要不要。", needs_decision=True)]
    queue = ReportQueue()
    first = queue.push_all(events)
    more = queue.push_all([spoken(first[0]["id"])])
    produced = first + more
    rebuilt = ReportQueue()
    again = rebuilt.push_all(events + produced + [spoken(first[0]["id"])])
    assert [b["id"] for b in again] == [b["id"] for b in produced]
    assert [it.id for it in rebuilt.items] == [it.id for it in queue.items]
    assert rebuilt.holding == queue.holding
