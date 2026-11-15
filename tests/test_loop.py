"""The whole loop, through the event files only: utterance -> spoken, refs intact."""
import json
import threading
from datetime import datetime

from audi27 import config as C
from audi27.cli import main
from audi27.dispatch import dispatch_all
from audi27.events import emit_utterance, ref_chain, replay
from audi27.queue import sync
from audi27.route import replay_routes, write_routes
from audi27.speak import speak_pending
from audi27.turns import assemble_replay, write_turns


def test_spoken_chains_back_to_the_utterance(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    emit_utterance(data, {
        "id": "u1",
        "start": "2026-11-15T14:00:00.000+08:00",
        "end": "2026-11-15T14:00:01.200+08:00",
        "text": "达达，今天天气怎么样",
        "status": "ok",
    })
    turns, stats = assemble_replay(data, cfg)
    assert stats["turns"] == 1 and turns[0].text == "今天天气怎么样"
    assert write_turns(data, turns)["wrote"] == 1
    routes = replay_routes(data, cfg)
    assert routes[0]["data"]["target"] == "grok" and routes[0]["data"]["needs_confirmation"] is False
    assert write_routes(data, routes)["wrote"] == 1
    assert dispatch_all(data, cfg)["wrote"] == 4
    _queue, written = sync(data, cfg)
    assert written["wrote"] == 1
    calls = []

    def backend(text, voice, stop_check):
        calls.append(text)
        assert voice == "Tingting" and stop_check() is False
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    assert speak_pending(data, cfg, backend=backend, speaking=lambda: False) == 1
    assert calls == ["收到：今天天气怎么样"]
    spoken = list(replay(data, types="spoken"))
    assert len(spoken) == 1 and spoken[0]["data"]["text"] == "收到：今天天气怎么样"
    chain = ref_chain(data, spoken[0]["id"])
    assert [ev["type"] for ev in chain] == [
        "utterance", "turn", "route", "task", "task", "task", "reply", "brief", "spoken",
    ]
    assert [ev["data"].get("status") for ev in chain if ev["type"] == "task"] == ["sent", "running", "done"]
    for prev, ev in zip(chain, chain[1:]):
        assert ev["ref"] == prev["id"]
    assert chain[0]["id"] == "u1"
    # Rebuilding the queue from the log, briefs included, does not invent a second one.
    again = sync(data, cfg)[1]
    assert again["wrote"] == 0 and again["skipped"] == 1


def test_queue_show_rebuilds_without_writing(tmp_path, capsys):
    from audi27.events import append, make_event

    data = tmp_path / "data"
    ts = "2026-11-15T12:00:00+00:00"
    for ident, text in (("r1", "甲。细节"), ("r2", "乙。细节")):
        append(data, make_event("reply", "dispatch", {
            "text": text, "target": "grok", "adapter": "fake", "needs_decision": False, "route_id": "route-1",
        }, ts=ts, ref="t", id=ident))
    assert main(["queue", "show", "--data-dir", str(data)]) == 0
    out = capsys.readouterr().out
    assert "甲" in out and "乙" in out
    assert list(replay(data, types="brief")) == []


def test_queue_run_does_not_brief_history(tmp_path):
    from audi27.events import append, make_event
    from audi27.queue import serve

    data = tmp_path / "data"
    ts = "2026-11-15T12:00:00+00:00"

    def one(ident, text):
        return make_event("reply", "dispatch", {
            "text": text, "target": "grok", "adapter": "fake", "needs_decision": False, "route_id": "route-1",
        }, ts=ts, ref="t", id=ident)

    append(data, one("old", "旧的一句。"))
    cfg = C.Config()
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    import time
    time.sleep(0.25)
    assert list(replay(data, types="brief")) == []
    append(data, one("new", "新的一句。"))
    found = []
    deadline = time.time() + 3
    while time.time() < deadline:
        found = list(replay(data, types="brief"))
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert [e["data"]["text_ref"] for e in found] == ["new"]

    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.3)
    stop.set()
    thread.join(2)
    briefs = list(replay(data, types="brief"))
    assert [e["data"]["text_ref"] for e in briefs] == ["new"]


def test_dashboard_lists_tasks_replies_and_briefs(tmp_path):
    import urllib.request

    from audi27.events import append, make_event
    from audi27.web import make_server

    data = tmp_path / "data"
    ts = datetime.now().astimezone().isoformat()
    append(data, make_event("task", "dispatch", {
        "route_id": "route-1", "target": "grok", "adapter": "fake", "status": "done", "text": "天气",
    }, ts=ts, ref="route-1", id="task-1"))
    append(data, make_event("reply", "dispatch", {
        "text": "收到：天气", "target": "grok", "adapter": "fake", "needs_decision": False, "route_id": "route-1",
    }, ts=ts, ref="task-1", id="reply-1"))
    append(data, make_event("brief", "queue", {
        "priority": "done", "summary": "收到：天气", "text_ref": "reply-1", "kind": "report", "item_id": "reply-1",
    }, ts=ts, ref="reply-1", id="brief-1"))
    server = make_server(data, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read()
        assert b'id="loopList"' in page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/loop") as response:
            body = json.loads(response.read())
        assert [row["type"] for row in body["events"]] == ["brief", "reply", "task"]
        assert body["events"][0]["summary"].startswith("[done]")
    finally:
        server.shutdown()
        server.server_close()
