"""Grok Bot webhook adapter: POST a route, pick the reply file up later."""
import json
import os
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.agent import Agent, build_agent, task_event
from audi27.cli import main
from audi27.dispatch import dispatch_all, dispatch_one, serve
from audi27.events import append, make_event, ref_chain, replay
from audi27.grokbot import TIMEOUT_REASON, GrokbotAgent, poll_inbox, tick
from audi27.queue import ReportQueue

TS = "2026-11-15T12:00:00+00:00"
OLD = "2026-11-15T00:00:00+00:00"


def route(ident, text="今天天气怎么样", *, target="grok", confirm=False, command=None):
    return make_event("route", "route", {
        "target": target, "text": text, "turn_text": text, "rule": "default", "reason": "默认",
        "method": "default", "needs_confirmation": confirm, "command": command,
    }, ts=TS, ref="turn-1", id=ident)


def utterance(ident, text, *, status="ok"):
    return make_event("utterance", "transcribe", {
        "id": ident, "text": text, "status": status, "start": TS,
    }, ts=TS, id=ident)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = self.server
        server.hits += 1
        if server.drop_remaining:
            server.drop_remaining -= 1
            self.connection.close()
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        server.requests.append({
            "raw": raw,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "path": self.path,
        })
        # The POST must not leave the client until `sent` is already recorded.
        server.order.append("post")
        body = server.body
        self.send_response(server.status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class Webhook:
    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.hits = 0
        self.httpd.drop_remaining = 0
        self.httpd.requests = []
        self.httpd.order = []
        self.httpd.status = 200
        self.httpd.body = b"{}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/hook"

    def stop(self):
        self.httpd.shutdown()
        self.thread.join(2)
        self.httpd.server_close()

    @property
    def requests(self):
        return self.httpd.requests

    @property
    def order(self):
        return self.httpd.order


@pytest.fixture
def webhook():
    box = Webhook()
    url = box.start()
    try:
        yield box, url
    finally:
        box.stop()


def grok_cfg(data, monkeypatch, *, url, key="sekret"):
    cfg = C.Config()
    cfg.store.data_dir = str(data)
    cfg.dispatch.adapters = {"grok": "grokbot"}
    cfg.grokbot.timeout_s = 3
    monkeypatch.setenv("AUDI27_GROKBOT_WEBHOOK_URL", url)
    monkeypatch.setenv("AUDI27_GROKBOT_WEBHOOK_KEY", key)
    return cfg


def appending(data, order=None):
    def emit(ev):
        append(data, ev, if_absent=True)
        if order is not None and ev.get("type") == "task":
            order.append(ev["data"]["status"])
        elif order is not None and ev.get("type") == "reply":
            order.append("reply")
    return emit


def reply_file(data, task_id, route_id, *, status="done", summary="晴", text="今天晴。", name=None):
    folder = Path(data) / "inbox" / "grokbot"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name or f"{task_id}.json")
    path.write_text(json.dumps({
        "task_id": task_id, "route_id": route_id, "status": status, "summary": summary, "text": text,
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _priorities(data):
    """What the queue would say, including items still waiting behind the first."""
    queue = ReportQueue()
    briefs = queue.push_all(list(replay(data)))
    return [ev["data"]["priority"] for ev in briefs] + [item.priority for item in queue.items]


def test_happy_path_posts_after_sent_and_a_reply_file_chains_back(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    order = box.order
    append(data, route("route-1"))
    dispatch_one(route("route-1"), cfg, appending(data, order), data_dir=data)
    assert order == ["sent", "post", "running"]
    assert box.httpd.hits == 1
    body = json.loads(box.requests[0]["raw"].decode())
    assert body["route_id"] == "route-1"
    assert body["text"] == "今天天气怎么样"
    assert body["task_id"] == "task-route-1-sent"
    assert "context" not in body
    assert box.requests[0]["headers"]["authorization"] == "Bearer sekret"
    assert "sekret" not in Path(list((data / "events").glob("*.jsonl"))[0]).read_text(encoding="utf-8")

    reply_file(data, body["task_id"], "route-1", summary="晴", text="今天晴。")
    tick(data, cfg, appending(data))
    replies = list(replay(data, types="reply"))
    assert len(replies) == 1
    assert replies[0]["data"]["text"] == "今天晴。"
    assert replies[0]["data"]["needs_decision"] is False
    assert replies[0]["ref"] == "task-route-1-done"
    chain = [ev["id"] for ev in ref_chain(data, replies[0]["id"])]
    assert chain[0] == "route-1" and chain[-1] == replies[0]["id"]
    assert _priorities(data) == ["done"]
    assert not (data / "inbox" / "grokbot" / f"{body['task_id']}.json").exists()
    assert (data / "inbox" / "grokbot" / "processed" / f"{body['task_id']}.json").is_file()


def test_needs_decision_and_failed_files_set_queue_priority(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    append(data, route("route-ask", "要不要订周五的票"))
    append(data, route("route-bad", "查一下账单"))
    dispatch_one(route("route-ask", "要不要订周五的票"), cfg, appending(data), data_dir=data)
    dispatch_one(route("route-bad", "查一下账单"), cfg, appending(data), data_dir=data)
    reply_file(data, "task-route-ask-sent", "route-ask", status="needs_decision",
               summary="要你决定", text="周五有两班，选上午还是下午？")
    reply_file(data, "task-route-bad-sent", "route-bad", status="failed",
               summary="登录过期", text="打开账单需要重新登录。")
    tick(data, cfg, appending(data))
    replies = {ev["data"]["route_id"]: ev for ev in replay(data, types="reply")}
    assert set(replies) == {"route-ask"}
    assert replies["route-ask"]["data"]["needs_decision"] is True
    failed = [ev for ev in replay(data, types="task")
              if ev["data"]["route_id"] == "route-bad" and ev["data"]["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["data"]["error"] == "登录过期"
    assert _priorities(data) == ["decision", "error"]


def test_non_2xx_fails_the_task_without_a_retry(tmp_path, webhook, monkeypatch):
    box, url = webhook
    box.httpd.status = 503
    box.httpd.body = b"nope"
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    dispatch_one(route("route-1"), cfg, appending(data), data_dir=data)
    assert box.httpd.hits == 1
    tasks = list(replay(data, types="task"))
    assert [ev["data"]["status"] for ev in tasks] == ["sent", "failed"]
    assert "503" in tasks[-1]["data"]["error"]
    assert list(replay(data, types="reply")) == []


def test_network_error_is_retried_once(tmp_path, webhook, monkeypatch):
    box, url = webhook
    box.httpd.drop_remaining = 1
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    dispatch_one(route("route-1"), cfg, appending(data), data_dir=data)
    assert box.httpd.hits == 2
    assert [ev["data"]["status"] for ev in replay(data, types="task")] == ["sent", "running"]
    assert box.requests and json.loads(box.requests[0]["raw"].decode())["text"] == "今天天气怎么样"

    box.httpd.drop_remaining = 5
    box.httpd.hits = 0
    dispatch_one(route("route-2", "再试一次"), cfg, appending(data), data_dir=data)
    assert box.httpd.hits == 2
    failed = [ev for ev in replay(data, types="task")
              if ev["data"]["route_id"] == "route-2" and ev["data"]["status"] == "failed"]
    assert len(failed) == 1 and "连不上" in failed[0]["data"]["error"]


def test_header_template_is_configurable(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, monkeypatch, url=url, key="abc")
    cfg.grokbot.header_name = "X-Grok-Key"
    cfg.grokbot.header_value = "token {key}"
    dispatch_one(route("route-1"), cfg, appending(data), data_dir=data)
    headers = box.requests[0]["headers"]
    assert headers["x-grok-key"] == "token abc"
    assert "authorization" not in headers


def test_context_is_the_last_few_utterances(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    cfg.grokbot.context_utterances = 2
    cfg.store.data_dir = str(tmp_path / "elsewhere")
    append(data, utterance("u1", "第一句"))
    append(data, utterance("u2", "丢掉", status="rejected"))
    append(data, utterance("u3", "第二句"))
    append(data, utterance("u4", "第三句"))
    dispatch_one(route("route-1"), cfg, appending(data), data_dir=data)
    body = json.loads(box.requests[0]["raw"].decode())
    assert body["context"] == ["第二句", "第三句"]


def test_duplicate_reply_file_emits_once(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    reply_file(data, "task-route-1-sent", "route-1", text="一次", name="a.json")
    reply_file(data, "task-route-1-sent", "route-1", text="一次", name="b.json")
    tick(data, cfg, appending(data))
    tick(data, cfg, appending(data))
    assert [ev["data"]["text"] for ev in replay(data, types="reply")] == ["一次"]
    assert list((data / "inbox" / "grokbot").glob("*.json")) == []
    assert len(list((data / "inbox" / "grokbot" / "processed").glob("*.json"))) == 2


def test_malformed_file_is_quarantined(tmp_path, caplog):
    data = tmp_path / "data"
    cfg = C.Config()
    inbox = data / "inbox" / "grokbot"
    inbox.mkdir(parents=True)
    (inbox / "bad.json").write_text(json.dumps({"task_id": "task-route-1-sent"}), encoding="utf-8")
    (inbox / "note.txt").write_text("ignore me", encoding="utf-8")
    with caplog.at_level("ERROR"):
        poll_inbox(data, cfg, appending(data))
    assert (inbox / "quarantine" / "bad.json").is_file()
    assert (inbox / "note.txt").is_file()
    assert list(replay(data, types=("task", "reply"))) == []
    assert any("bad.json" in rec.message for rec in caplog.records)


def test_unknown_task_id_is_quarantined_not_emitted(tmp_path, caplog):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    reply_file(data, "task-does-not-exist", "route-1", text="不该出现")
    with caplog.at_level("ERROR"):
        tick(data, cfg, appending(data))
    assert list(replay(data, types="reply")) == []
    assert (data / "inbox" / "grokbot" / "quarantine" / "task-does-not-exist.json").is_file()
    assert any("task-does-not-exist" in rec.message for rec in caplog.records)


def test_partial_json_is_retried_then_quarantined(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    cfg.grokbot.parse_retries = 2
    inbox = data / "inbox" / "grokbot"
    inbox.mkdir(parents=True)
    (inbox / "partial.json").write_text("{", encoding="utf-8")
    poll_inbox(data, cfg, appending(data))
    assert (inbox / "partial.json").is_file()
    poll_inbox(data, cfg, appending(data))
    assert not (inbox / "partial.json").exists()
    assert (inbox / "quarantine" / "partial.json").is_file()
    assert list(replay(data)) == []


def test_timeout_then_a_late_reply_still_emits(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    cfg.grokbot.reply_timeout_s = 1800
    append(data, route("route-1"))
    sent = task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot",
                      ts=datetime.fromisoformat(OLD))
    append(data, sent)
    append(data, task_event(route("route-old"), status="sent", ref="route-old", adapter="fake",
                            ts=datetime.fromisoformat(OLD)))
    base = datetime.fromisoformat(OLD)
    tick(data, cfg, appending(data), now=base + timedelta(seconds=1799))
    assert not any(ev["data"]["status"] == "failed" for ev in replay(data, types="task"))
    tick(data, cfg, appending(data), now=base + timedelta(seconds=1800))
    failed = [ev for ev in replay(data, types="task") if ev["data"]["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["data"]["route_id"] == "route-1"
    assert failed[0]["data"]["error"] == TIMEOUT_REASON == "Grok Bot 没有回复"
    tick(data, cfg, appending(data), now=base + timedelta(seconds=1800))
    assert len([ev for ev in replay(data, types="task") if ev["data"]["status"] == "failed"]) == 1

    reply_file(data, "task-route-1-sent", "route-1", text="晚点到了也算数")
    tick(data, cfg, appending(data), now=base + timedelta(seconds=4000))
    replies = list(replay(data, types="reply"))
    assert [ev["data"]["text"] for ev in replies] == ["晚点到了也算数"]
    assert replies[0]["ref"] == "task-route-1-done"


def test_restart_does_not_emit_a_reply_twice_or_drop_a_waiting_task(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    cfg.dispatch.adapters = {"grok": "grokbot"}
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    reply_file(data, "task-route-1-sent", "route-1", text="只一次")
    tick(data, cfg, appending(data))
    # A new process: no in-memory state, and the file is put back as if the move was lost.
    reply_file(data, "task-route-1-sent", "route-1", text="只一次", name="again.json")
    tick(data, cfg, appending(data))
    assert [ev["data"]["text"] for ev in replay(data, types="reply")] == ["只一次"]

    append(data, route("route-wait", "还在等"))
    append(data, task_event(route("route-wait", "还在等"), status="running",
                            ref="task-route-wait-sent", adapter="grokbot"))

    class Boom(Agent):
        def handle(self, route, emit):
            raise AssertionError("sent again")

    assert dispatch_all(data, cfg, agents={"grokbot": Boom()})["wrote"] == 0
    assert not any(ev["data"]["route_id"] == "route-wait" and ev["data"]["status"] == "failed"
                   for ev in replay(data, types="task"))


def test_missing_webhook_env_fails_without_a_post(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDI27_GROKBOT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AUDI27_GROKBOT_WEBHOOK_KEY", raising=False)
    data = tmp_path / "data"
    cfg = C.Config()
    cfg.dispatch.adapters = {"grok": "grokbot"}
    agent = build_agent("grokbot", cfg)
    assert isinstance(agent, GrokbotAgent)
    assert build_agent("fake", cfg).name == "fake"
    assert build_agent("claude", cfg) is None
    seen = []
    agent.handle(route("route-1"), seen.append)
    assert [ev["data"]["status"] for ev in seen] == ["failed"]
    assert "AUDI27_GROKBOT_WEBHOOK_URL" in seen[0]["data"]["error"]


def test_default_adapter_stays_fake_and_guards_still_hold(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    append(data, route("route-default"))
    assert dispatch_all(data, C.Config())["wrote"] == 4
    assert list(replay(data, types="reply"))[0]["data"]["text"].startswith("收到：")
    assert box.httpd.hits == 0

    guarded = tmp_path / "guarded"
    cfg = grok_cfg(guarded, url=url, monkeypatch=monkeypatch)
    append(guarded, route("route-none", "算了", target="none"))
    append(guarded, route("route-q", "跳过", target="queue", command="skip"))
    append(guarded, route("route-pay", "帮我发邮件", confirm=True))
    append(guarded, route("route-ok", "你好"))
    dispatch_all(guarded, cfg)
    assert box.httpd.hits == 1
    by_route = {}
    for ev in replay(guarded, types="task"):
        by_route.setdefault(ev["data"]["route_id"], []).append(ev["data"]["status"])
    assert "route-none" not in by_route and "route-q" not in by_route
    assert by_route["route-pay"] == ["awaiting_confirmation"]
    assert by_route["route-ok"] == ["sent", "running"]


def test_dispatch_run_loop_ingests_the_reply_file(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    try:
        append(data, route("route-live"))
        deadline = datetime.now().timestamp() + 3
        while datetime.now().timestamp() < deadline and not box.requests:
            threading.Event().wait(0.05)
        assert box.requests, "dispatch run did not post"
        reply_file(data, "task-route-live-sent", "route-live", text="回路通了")
        found = []
        deadline = datetime.now().timestamp() + 3
        while datetime.now().timestamp() < deadline:
            found = list(replay(data, types="reply"))
            if found:
                break
            threading.Event().wait(0.05)
        assert [ev["data"]["text"] for ev in found] == ["回路通了"]
    finally:
        stop.set()
        thread.join(2)
    assert not thread.is_alive()


def test_cli_test_prints_status_and_dotenv_supplies_the_key(tmp_path, webhook, monkeypatch, capsys):
    box, url = webhook
    monkeypatch.delenv("AUDI27_GROKBOT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AUDI27_GROKBOT_WEBHOOK_KEY", raising=False)
    dotenv = Path.cwd() / ".env"
    dotenv.write_text(
        f"AUDI27_GROKBOT_WEBHOOK_URL={url}\nAUDI27_GROKBOT_WEBHOOK_KEY=from-dotenv\n",
        encoding="utf-8")
    try:
        code = main(["grokbot", "test", "你好"])
    finally:
        os.environ.pop("AUDI27_GROKBOT_WEBHOOK_URL", None)
        os.environ.pop("AUDI27_GROKBOT_WEBHOOK_KEY", None)
        C._FROM_DOTENV.discard("AUDI27_GROKBOT_WEBHOOK_URL")
        C._FROM_DOTENV.discard("AUDI27_GROKBOT_WEBHOOK_KEY")
    assert code == 0
    assert capsys.readouterr().out.strip() == "200"
    body = json.loads(box.requests[0]["raw"].decode())
    assert body["text"] == "你好"
    assert box.requests[0]["headers"]["authorization"] == "Bearer from-dotenv"


def test_grokbot_config_defaults_and_rejects_secrets_and_bad_values(tmp_path):
    cfg = C.Config()
    assert cfg.grokbot.header_name == "Authorization"
    assert cfg.grokbot.header_value == "Bearer {key}"
    assert cfg.grokbot.timeout_s == 10
    assert cfg.grokbot.context_utterances == 0
    assert cfg.grokbot.reply_timeout_s == 1800
    assert cfg.grokbot.parse_retries == 3
    example = C.load(Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.dispatch.adapters["grok"] == "fake"
    assert example.grokbot.header_value == "Bearer {key}"
    assert example.grokbot.reply_timeout_s == 1800
    path = tmp_path / "c.toml"
    path.write_text('[grokbot]\nwebhook_url = "http://x"\n', encoding="utf-8")
    with pytest.raises(C.ConfigError):
        C.load(path)
    path.write_text('[grokbot]\ncontext_utterances = 21\n', encoding="utf-8")
    with pytest.raises(C.ConfigError):
        C.load(path)
    path.write_text('[grokbot]\nheader_name = "Not a header"\n', encoding="utf-8")
    with pytest.raises(C.ConfigError):
        C.load(path)
