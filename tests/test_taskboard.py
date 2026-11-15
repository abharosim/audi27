"""Task-status fold: running, waiting to be spoken, and recently finished.

Derived from the event log and the quarantine directory. Nothing here writes.
"""
import json
import threading
from datetime import datetime, timedelta, timezone

from audi27.events import append, make_event
from audi27.taskboard import derive, snapshot

NOW = datetime(2026, 11, 15, 12, 2, 0, tzinfo=timezone.utc)
SENT = "2026-11-15T12:00:00+00:00"
RUNNING_AT = "2026-11-15T12:00:02+00:00"
REPLY_AT = "2026-11-15T12:01:00+00:00"
SPOKEN_AT = "2026-11-15T12:01:30+00:00"


def task(route, status, text, ts, *, adapter="grokbot", target="grok", ident=None, error=None):
    data = {
        "route_id": route, "target": target, "adapter": adapter,
        "status": status, "text": text,
    }
    if error:
        data["error"] = error
    return make_event(
        "task", "dispatch", data, ts=ts, ref=route,
        id=ident or f"task-{route}-{status}",
    )


def reply(route, text, ts, *, needs_decision=False, ident=None, adapter="grokbot", target="grok"):
    return make_event("reply", "dispatch", {
        "route_id": route, "target": target, "adapter": adapter,
        "text": text, "needs_decision": needs_decision,
    }, ts=ts, ref=f"task-{route}-done", id=ident or f"reply-{route}")


def spoken(brief_id, ts):
    return make_event("spoken", "speak", {
        "text": "x", "completed": True, "interrupted": False, "backend": "mock",
    }, ts=ts, ref=brief_id, id="spoken-" + brief_id)


def close(ident="route-close"):
    return make_event("route", "route", {
        "target": "queue", "command": "close", "text": "关掉", "turn_text": "关掉",
        "rule": "queue:close", "reason": "", "method": "rule", "needs_confirmation": False,
    }, ts=SPOKEN_AT, ref="turn-c", id=ident)


def test_empty_log_has_three_empty_lists():
    board = derive([], now=NOW)
    assert board == {"running": [], "waiting": [], "recent": []}


def test_dispatched_task_without_a_reply_is_running():
    text = "今天天气怎么样" + "呀" * 80
    events = [
        task("route-1", "sent", text, SENT),
        task("route-1", "running", text, RUNNING_AT),
    ]
    board = derive(events, now=NOW, text_limit=80)
    assert board["waiting"] == [] and board["recent"] == []
    assert len(board["running"]) == 1
    row = board["running"][0]
    assert row["id"] == "route-1"
    assert row["status"] == "running"
    assert row["adapter"] == "grokbot" and row["target"] == "grok"
    assert row["since"] == "dispatch"
    assert row["elapsed_s"] == 120.0
    assert row["text"].endswith("…") and len(row["text"]) == 80
    assert row["text"].startswith("今天天气怎么样")


def test_older_running_task_is_listed_first():
    events = [
        task("route-new", "sent", "新的", "2026-11-15T12:01:00+00:00"),
        task("route-old", "running", "旧的", "2026-11-15T12:00:00+00:00"),
    ]
    board = derive(events, now=NOW)
    assert [row["id"] for row in board["running"]] == ["route-old", "route-new"]
    assert board["running"][0]["elapsed_s"] == 120.0
    assert board["running"][1]["elapsed_s"] == 60.0


def test_unsent_tasks_are_not_running():
    events = [
        task("route-wait", "awaiting_confirmation", "发邮件给王老师", SENT),
        task("route-no", "cancelled", "算了", SENT),
        task("route-old", "expired", "过期的", SENT),
        task("route-skip", "ignored", "不是指令", SENT),
    ]
    assert derive(events, now=NOW) == {"running": [], "waiting": [], "recent": []}


def test_reply_still_in_the_queue_is_waiting_since_the_reply():
    events = [
        task("route-1", "sent", "今天天气怎么样", SENT),
        task("route-1", "running", "今天天气怎么样", RUNNING_AT),
        task("route-1", "done", "今天天气怎么样", REPLY_AT),
        reply("route-1", "收到：今天天气怎么样", REPLY_AT),
    ]
    board = derive(events, now=NOW)
    assert board["running"] == [] and board["recent"] == []
    row = board["waiting"][0]
    assert row["id"] == "route-1" and row["status"] == "queued"
    assert row["text"] == "今天天气怎么样"
    assert row["adapter"] == "grokbot" and row["target"] == "grok"
    assert row["since"] == "reply" and row["elapsed_s"] == 60.0


def test_needs_decision_waits_until_it_is_spoken_then_finishes_as_needs_decision():
    events = [
        task("route-1", "sent", "要不要发出去", SENT),
        reply("route-1", "需要你决定：要不要发出去", REPLY_AT, needs_decision=True),
    ]
    waiting = derive(events, now=NOW)
    assert [row["status"] for row in waiting["waiting"]] == ["needs_decision"]
    assert waiting["recent"] == [] and waiting["running"] == []

    events.append(spoken("brief-reply-route-1", SPOKEN_AT))
    done = derive(events, now=NOW)
    assert done["waiting"] == [] and done["running"] == []
    assert len(done["recent"]) == 1
    row = done["recent"][0]
    assert row["status"] == "needs_decision" and row["since"] == "reply"
    assert row["elapsed_s"] == 60.0
    assert row["text"] == "要不要发出去"


def test_spoken_reply_leaves_the_queue_and_is_recent():
    events = [
        task("route-1", "sent", "天气", SENT, adapter="fake"),
        reply("route-1", "收到：天气", REPLY_AT, adapter="fake"),
        spoken("brief-reply-route-1", SPOKEN_AT),
    ]
    board = derive(events, now=NOW)
    assert board["running"] == [] and board["waiting"] == []
    assert board["recent"][0]["status"] == "spoken"
    assert board["recent"][0]["adapter"] == "fake"
    assert board["recent"][0]["text"] == "天气"


def test_interrupted_speech_puts_the_reply_back_to_waiting():
    events = [
        task("route-1", "sent", "天气", SENT),
        reply("route-1", "收到：天气", REPLY_AT),
        make_event("spoken", "speak", {
            "text": "收到：天气", "completed": False, "interrupted": True, "backend": "mock",
        }, ts=SPOKEN_AT, ref="brief-reply-route-1", id="spoken-cut"),
    ]
    board = derive(events, now=NOW)
    assert [row["status"] for row in board["waiting"]] == ["queued"]
    assert board["recent"] == []


def test_close_drops_replies_that_were_not_released():
    events = [
        task("route-a", "sent", "甲", SENT),
        reply("route-a", "收到：甲", REPLY_AT, ident="reply-a"),
        task("route-b", "sent", "乙", SENT),
        reply("route-b", "收到：乙", REPLY_AT, ident="reply-b"),
        close(),
    ]
    board = derive(events, now=NOW)
    assert [row["id"] for row in board["waiting"]] == ["route-a"]
    assert board["recent"] == [] and board["running"] == []


def test_failed_task_waits_in_the_queue_until_the_error_is_spoken():
    failed = task("route-1", "failed", "查一下账单", "2026-11-15T12:01:40+00:00", error="Grok Bot 没有回复")
    events = [
        task("route-1", "sent", "查一下账单", SENT),
        task("route-1", "running", "查一下账单", RUNNING_AT),
        failed,
    ]
    board = derive(events, now=NOW)
    assert board["running"] == [] and board["recent"] == []
    row = board["waiting"][0]
    assert row["status"] == "failed" and row["text"] == "查一下账单"
    assert row["since"] == "dispatch" and row["elapsed_s"] == 120.0

    events.append(spoken("brief-" + failed["id"], "2026-11-15T12:01:50+00:00"))
    done = derive(events, now=NOW)
    assert done["waiting"] == [] and done["running"] == []
    assert done["recent"][0]["status"] == "failed"
    assert done["recent"][0]["id"] == "route-1"


def test_failure_dropped_by_close_leaves_the_queue_and_is_recent():
    events = [
        task("route-a", "failed", "甲", SENT, ident="task-a-failed"),
        task("route-b", "failed", "乙", REPLY_AT, ident="task-b-failed"),
        close(),
    ]
    board = derive(events, now=NOW)
    assert [(row["id"], row["status"]) for row in board["waiting"]] == [("route-a", "failed")]
    assert [(row["id"], row["status"]) for row in board["recent"]] == [("route-b", "failed")]


def test_recent_keeps_the_newest_ten():
    events = []
    for i in range(12):
        route = f"route-{i:02d}"
        sent = (datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()
        failed_at = (datetime(2026, 11, 15, 11, 30, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()
        events.append(task(route, "sent", f"第{i}件", sent))
        events.append(task(route, "failed", f"第{i}件", failed_at, error="失败"))
    events.append(close())
    board = derive(events, now=NOW)
    assert [row["id"] for row in board["waiting"]] == ["route-00"]
    assert [row["id"] for row in board["recent"]] == [f"route-{i:02d}" for i in range(11, 1, -1)]


def test_quarantine_file_is_recent_and_uses_the_original_text_when_the_task_is_known():
    events = [task("route-1", "sent", "原来的那句话", SENT, ident="task-route-1-sent")]
    quarantine = [{
        "name": "task-route-1-sent.json",
        "at": "2026-11-15T12:01:30+00:00",
        "task_id": "task-route-1-sent",
        "route_id": "route-1",
        "text": "这不是原话，是坏掉的回复",
    }]
    board = derive(events, now=NOW, quarantine=quarantine)
    assert [row["id"] for row in board["running"]] == ["route-1"]
    row = board["recent"][0]
    assert row["status"] == "quarantined"
    assert row["text"] == "原来的那句话"
    assert row["adapter"] == "grokbot" and row["target"] == "grok"
    assert row["since"] == "file" and row["elapsed_s"] == 30.0
    assert row["id"] == "quarantine:task-route-1-sent.json"


def test_unknown_quarantine_file_uses_its_own_text():
    quarantine = [{
        "name": "not-a-task.json",
        "at": "2026-11-15T12:01:00+00:00",
        "task_id": None,
        "route_id": None,
        "text": "乱七八糟的内容",
    }]
    row = derive([], now=NOW, quarantine=quarantine)["recent"][0]
    assert row["status"] == "quarantined" and row["text"] == "乱七八糟的内容"
    assert row["adapter"] == "" and row["target"] == ""


def test_snapshot_reads_the_log_and_quarantine_and_skips_symlinks(tmp_path):
    data = tmp_path / "data"
    append(data, task("route-1", "sent", "还在跑", SENT))
    root = data / "inbox" / "grokbot" / "quarantine"
    root.mkdir(parents=True)
    (root / "bad.json").write_text("not json", encoding="utf-8")
    (root / "note.txt").write_text("ignore", encoding="utf-8")
    outside = tmp_path / "secret.json"
    outside.write_text(json.dumps({"text": "secret"}), encoding="utf-8")
    (root / "link.json").symlink_to(outside)
    board = snapshot(data, now=NOW)
    assert [row["text"] for row in board["running"]] == ["还在跑"]
    assert [row["id"] for row in board["recent"]] == ["quarantine:bad.json"]
    assert board["recent"][0]["text"] == "bad.json"


def test_confirmation_readback_waits_while_the_gate_is_on():
    events = [task("route-1", "awaiting_confirmation", "发邮件给王老师", SENT)]
    assert derive(events, now=NOW) == {"running": [], "waiting": [], "recent": []}
    board = derive(events, now=NOW, confirm=True)
    assert board["running"] == [] and board["recent"] == []
    row = board["waiting"][0]
    assert row["status"] == "awaiting_confirmation"
    assert row["text"] == "发邮件给王老师" and row["target"] == "grok"


def test_snapshot_skips_events_older_than_yesterday(tmp_path):
    now = datetime.now().astimezone().replace(hour=15, minute=0, second=0, microsecond=0)
    data = tmp_path / "data"

    def one(route, when):
        append(data, task(route, "sent", route, when.isoformat()))

    one("route-old", now - timedelta(days=3))
    one("route-yest", now - timedelta(days=1))
    one("route-today", now - timedelta(hours=1))
    ids = {row["id"] for row in snapshot(data, now=now)["running"]}
    assert {"route-today", "route-yest"} <= ids
    assert "route-old" not in ids


def test_tasks_and_the_loop_share_one_read_of_that_window(tmp_path, monkeypatch):
    import audi27.taskboard as tb
    from audi27.web import list_loop

    now = datetime.now().astimezone()
    data = tmp_path / "data"
    append(data, task("route-1", "running", "还在", now.isoformat()))
    calls = []
    real = tb.replay

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get("since"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tb, "replay", wrapped)
    assert snapshot(data, now=now)["running"][0]["text"] == "还在"
    assert snapshot(data, now=now)["running"][0]["text"] == "还在"
    assert [row["type"] for row in list_loop(data)["events"]] == ["task"]
    assert len(calls) == 1 and calls[0] is not None


def test_dashboard_serves_the_task_section(tmp_path):
    import urllib.request

    from audi27.web import make_server

    data = tmp_path / "data"
    append(data, task("route-1", "running", "帮我记一下", RUNNING_AT, adapter="fake", target="muse"))
    server = make_server(data, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read()
        assert b'id="taskBoard"' in page and b"/api/tasks" in page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tasks") as response:
            body = json.loads(response.read())
        assert body["running"][0]["text"] == "帮我记一下"
        assert body["running"][0]["target"] == "muse" and body["running"][0]["adapter"] == "fake"
        assert body["waiting"] == [] and body["recent"] == []
    finally:
        server.shutdown()
        server.server_close()
