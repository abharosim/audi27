"""Grok Bot webhook adapter: POST a route, pick the reply file up later."""
import json
import os
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.agent import Agent, build_agent, task_event
from audi27.cli import main
from audi27.dispatch import dispatch_all, dispatch_one, serve
from audi27.events import append, make_event, ref_chain, replay
from audi27.grokbot import TIMEOUT_REASON, GrokbotAgent, poll_inbox, post_webhook, serve_inbox, tick
from audi27.screen import prepare
from audi27.queue import ReportQueue
from audi27.route import decide, load_rules, route_event

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


def _with_screen(event, title="main.py", *, redacted=False):
    event["data"]["screen"] = {
        "app": "Cursor", "bundle_id": "com.cursor", "title": title, "title_redacted": redacted,
    }
    return event


def test_screen_stays_off_the_post_unless_include_screen(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, url=url, monkeypatch=monkeypatch)
    hidden = _with_screen(route("route-off"), "secret-title")
    dispatch_one(hidden, cfg, appending(data), data_dir=data)
    raw = box.requests[0]["raw"].decode()
    body = json.loads(raw)
    assert "context" not in body
    assert "secret-title" not in raw

    cfg.grokbot.include_screen = True
    dispatch_one(_with_screen(route("route-on")), cfg, appending(data), data_dir=data)
    body = json.loads(box.requests[1]["raw"].decode())
    assert body["context"] == {
        "screen": {
            "app": "Cursor", "bundle_id": "com.cursor", "title": "main.py", "title_redacted": False,
        },
    }

    cfg.grokbot.context_utterances = 1
    append(data, utterance("u1", "旁边一句"))
    dispatch_one(_with_screen(route("route-both"), "both.py"), cfg, appending(data), data_dir=data)
    body = json.loads(box.requests[2]["raw"].decode())
    assert body["context"]["utterances"] == ["旁边一句"]
    assert body["context"]["screen"]["title"] == "both.py"

    redacted = _with_screen(route("route-red"), "hunter2", redacted=True)
    dispatch_one(redacted, cfg, appending(data), data_dir=data)
    raw = box.requests[3]["raw"].decode()
    assert "hunter2" not in raw
    assert json.loads(raw)["context"]["screen"]["title"] is None
    assert json.loads(raw)["context"]["screen"]["title_redacted"] is True

    # An excluded app's title is dropped again at the POST, even if a route
    # stored it without the redaction flag.
    secret = "vault-master-password"
    leaked = route("route-vault")
    leaked["data"]["screen"] = {
        "app": "钥匙串访问",
        "bundle_id": "com.apple.keychainaccess",
        "title": secret,
        "title_redacted": False,
    }
    dispatch_one(leaked, cfg, appending(data), data_dir=data)
    raw = box.requests[4]["raw"].decode()
    assert secret not in raw
    posted = json.loads(raw)["context"]["screen"]
    assert posted["title"] is None and posted["title_redacted"] is True
    assert posted["app"] == "钥匙串访问"

    prepared = prepare(
        {"app": "密码", "bundle_id": "com.agilebits.onepassword7", "title": secret},
        C.Config().screen,
    )
    assert secret not in json.dumps(prepared)
    shaped = route("route-1p")
    shaped["data"]["screen"] = prepared
    dispatch_one(shaped, cfg, appending(data), data_dir=data)
    raw = box.requests[5]["raw"].decode()
    assert secret not in raw
    assert json.loads(raw)["context"]["screen"]["title"] is None


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
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDI27_GROKBOT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AUDI27_GROKBOT_WEBHOOK_KEY", raising=False)
    (tmp_path / ".env").write_text(
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
    assert cfg.grokbot.include_screen is False
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


def _logged(data) -> str:
    root = Path(data) / "events"
    if not root.is_dir():
        return ""
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.glob("*.jsonl")))


def _assert_secret_absent(blob: str, *secrets: str) -> None:
    for secret in secrets:
        assert secret not in blob, secret


class _StatusHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = self.server
        server.hits += 1
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(length)
        body = server.body
        self.send_response(server.status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class _TrickleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = self.server
        server.hits += 1
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        try:
            while not getattr(server, "stop", False):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.05)
        except Exception:
            return

    def log_message(self, fmt, *args):
        return


class _HangHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.hits += 1
        time.sleep(5)

    def log_message(self, fmt, *args):
        return


def _background_server(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    httpd.hits = 0
    httpd.status = 500
    httpd.body = b""
    httpd.stop = False
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/hook"
    return httpd, url


def test_secrets_are_not_printed_logged_or_written_to_events(tmp_path, monkeypatch, capsys, caplog):
    secret = "SUPERSECRETKEY"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDI27_GROKBOT_WEBHOOK_URL", f"{secret}://not-a-url")
    monkeypatch.setenv("AUDI27_GROKBOT_WEBHOOK_KEY", secret)
    with caplog.at_level("ERROR"):
        code = main(["grokbot", "test", "你好"])
    captured = capsys.readouterr()
    assert code == 1
    _assert_secret_absent(captured.out + captured.err, secret)
    assert "Traceback" not in captured.err
    _assert_secret_absent("\n".join(rec.getMessage() for rec in caplog.records), secret)

    httpd, url = _background_server(_StatusHandler)
    httpd.body = f"echo {secret}".encode()
    try:
        data = tmp_path / "data"
        cfg = grok_cfg(data, monkeypatch, url=url, key=secret)
        agent = GrokbotAgent(
            data_dir=data, url=url, key="sekret密钥", header_name="Authorization",
            header_value="Bearer {key}", timeout_s=1, context_utterances=0)
        seen = []
        with caplog.at_level("ERROR"):
            agent.handle(route("route-key"), seen.append)
        blob = json.dumps(seen, ensure_ascii=False) + _logged(data)
        messages = "\n".join(rec.getMessage() for rec in caplog.records)
        _assert_secret_absent(blob + messages, "sekret密钥", "密钥")
        assert [ev["data"]["status"] for ev in seen] == ["sent", "failed"]

        cfg.grokbot.timeout_s = 1
        dispatch_one(route("route-echo"), cfg, appending(data), data_dir=data)
        log_text = _logged(data) + "\n".join(rec.getMessage() for rec in caplog.records)
        _assert_secret_absent(log_text, secret)
        failed = [ev for ev in replay(data, types="task") if ev["data"]["status"] == "failed"]
        assert failed and "500" in failed[-1]["data"]["error"]
        assert httpd.hits == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dotenv_quotes_comments_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    C.load_dotenv(tmp_path / "missing.env")
    for name in ("AUDI27_GROKBOT_WEBHOOK_URL", "AUDI27_GROKBOT_WEBHOOK_KEY", "AUDI27_OTHER"):
        monkeypatch.delenv(name, raising=False)
        C._FROM_DOTENV.discard(name)
    (tmp_path / ".env").write_text(
        "# comment\n"
        "export AUDI27_GROKBOT_WEBHOOK_URL=\"http://127.0.0.1:9/hook\" # inline\n"
        "AUDI27_GROKBOT_WEBHOOK_KEY='sek\"ret' # trailing\n"
        "AUDI27_OTHER=plain # comment\n"
        "broken line\n",
        encoding="utf-8")
    C.load_dotenv(tmp_path / ".env")
    assert os.environ["AUDI27_GROKBOT_WEBHOOK_URL"] == "http://127.0.0.1:9/hook"
    assert os.environ["AUDI27_GROKBOT_WEBHOOK_KEY"] == 'sek"ret'
    assert os.environ["AUDI27_OTHER"] == "plain"


def test_llm_routes_post_only_when_the_adapter_is_configured(tmp_path, webhook, monkeypatch):
    box, url = webhook
    rules = load_rules(Path(__file__).resolve().parents[1] / "routes.example.toml")

    def routed(ident, text):
        decision = decide(text, rules, classify=lambda _text: "grok")
        turn = make_event("turn", "turns", {"text": text}, ts=TS, id=ident)
        return route_event(turn, decision), decision

    weather, weather_d = routed("turn-weather", "今天天气怎么样")
    mail, mail_d = routed("turn-mail", "帮我发一封邮件")
    negated, negated_d = routed("turn-no", "别用 Grok 今天天气怎么样")
    assert (weather_d.method, weather_d.target, weather_d.needs_confirmation) == ("llm", "grok", False)
    assert mail_d.needs_confirmation is True and mail_d.target == "grok" and mail_d.method == "llm"
    assert negated_d.target == "none" and negated_d.method == "rule"

    plain = tmp_path / "plain"
    append(plain, weather)
    assert dispatch_all(plain, C.Config())["wrote"] == 4
    assert box.httpd.hits == 0
    assert list(replay(plain, types="reply"))[0]["data"]["text"].startswith("收到：")

    data = tmp_path / "data"
    cfg = grok_cfg(data, monkeypatch, url=url)
    append(data, weather)
    append(data, mail)
    append(data, negated)
    dispatch_all(data, cfg)
    assert box.httpd.hits == 1
    body = json.loads(box.requests[0]["raw"].decode())
    assert body["route_id"] == weather["id"]
    by_route = {}
    for ev in replay(data, types="task"):
        by_route.setdefault(ev["data"]["route_id"], []).append(ev["data"]["status"])
    assert by_route[weather["id"]] == ["sent", "running"]
    assert by_route[mail["id"]] == ["awaiting_confirmation"]
    assert negated["id"] not in by_route
    dispatch_all(data, cfg)
    assert box.httpd.hits == 1


def test_restart_does_not_repost_and_timeout_comes_from_the_log(tmp_path, webhook, monkeypatch):
    box, url = webhook
    data = tmp_path / "data"
    cfg = grok_cfg(data, monkeypatch, url=url)
    fresh = datetime.now().astimezone()
    append(data, route("route-old"))
    append(data, task_event(route("route-old"), status="sent", ref="route-old", adapter="grokbot", ts=fresh))
    (data).mkdir(parents=True, exist_ok=True)
    (Path(data) / "dispatch.cursor.json").write_text(json.dumps({"offsets": {}}), encoding="utf-8")
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    try:
        threading.Event().wait(0.3)
        assert box.httpd.hits == 0
        assert not any(ev["data"]["status"] == "failed" for ev in replay(data, types="task"))
    finally:
        stop.set()
        thread.join(2)

    expired = tmp_path / "expired"
    cfg.store.data_dir = str(expired)
    cfg.grokbot.reply_timeout_s = 1800
    old = datetime.now().astimezone() - timedelta(days=2)
    append(expired, route("route-aged"))
    append(expired, task_event(route("route-aged"), status="running", ref="task-route-aged-sent",
                               adapter="grokbot", ts=old))
    append(expired, task_event(route("route-aged"), status="sent", ref="route-aged",
                               adapter="grokbot", ts=old))
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(expired, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    try:
        failed = [ev for ev in replay(expired, types="task") if ev["data"]["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["data"]["error"] == TIMEOUT_REASON
        assert failed[0]["data"]["route_id"] == "route-aged"
    finally:
        stop.set()
        thread.join(2)
    assert box.httpd.hits == 0


def test_reply_for_a_task_from_another_day_is_applied(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    old = datetime.now().astimezone() - timedelta(days=2)
    append(data, route("route-yday"))
    sent = task_event(route("route-yday"), status="sent", ref="route-yday", adapter="grokbot", ts=old)
    append(data, sent)
    days = {path.name for path in (data / "events").glob("*.jsonl")}
    assert len(days) == 2 or (datetime.now().astimezone() - old).days >= 1
    reply_file(data, sent["id"], "route-yday", text="跨天也算")
    tick(data, cfg, appending(data))
    replies = list(replay(data, types="reply"))
    assert [ev["data"]["text"] for ev in replies] == ["跨天也算"]
    assert not any(ev["data"]["status"] == "failed" for ev in replay(data, types="task"))


def test_symlink_and_path_tricks_stay_inside_the_inbox(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({
        "task_id": "task-route-1-sent", "route_id": "route-1", "status": "done",
        "summary": "不该读", "text": "链接外面的回复",
    }, ensure_ascii=False), encoding="utf-8")
    inbox = data / "inbox" / "grokbot"
    inbox.mkdir(parents=True)
    link = inbox / "task-route-1-sent.json"
    link.symlink_to(outside)
    tick(data, cfg, appending(data))
    assert list(replay(data, types="reply")) == []
    assert outside.is_file() and "链接外面的回复" in outside.read_text(encoding="utf-8")
    assert not link.exists()
    quarantined = list((inbox / "quarantine").iterdir())
    assert len(quarantined) == 1 and quarantined[0].is_symlink()

    escaped = tmp_path / "escaped"
    escaped.mkdir()
    reply_file(data, "../../escaped/pwned", "route-1", text="不该落地", name="nested-id.json")
    tick(data, cfg, appending(data))
    assert list(escaped.iterdir()) == []
    assert list(replay(data, types="reply")) == []
    assert (inbox / "quarantine" / "nested-id.json").is_file()

    real = tmp_path / "real-inbox"
    real.mkdir()
    (real / "task-route-1-sent.json").write_text(outside.read_text(encoding="utf-8"), encoding="utf-8")
    linked_root = tmp_path / "linked" / "inbox"
    linked_root.parent.mkdir()
    (linked_root).mkdir()
    (linked_root / "grokbot").symlink_to(real)
    poll_inbox(tmp_path / "linked", cfg, appending(tmp_path / "linked"))
    assert list(replay(tmp_path / "linked", types="reply")) == []
    assert (real / "task-route-1-sent.json").is_file()


def test_oversize_reply_is_quarantined_without_a_body(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    inbox = data / "inbox" / "grokbot"
    inbox.mkdir(parents=True)
    huge = inbox / "task-route-1-sent.json"
    payload = {"task_id": "task-route-1-sent", "route_id": "route-1", "status": "done",
               "summary": "太大", "text": "x"}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    huge.write_bytes(raw + b" " * (1024 * 1024))
    poll_inbox(data, cfg, appending(data))
    assert list(replay(data, types="reply")) == []
    assert not huge.exists()
    assert (inbox / "quarantine" / "task-route-1-sent.json").is_file()


def test_growing_file_is_left_until_it_settles(tmp_path, monkeypatch):
    data = tmp_path / "data"
    cfg = C.Config()
    cfg.grokbot.parse_retries = 1
    inbox = data / "inbox" / "grokbot"
    inbox.mkdir(parents=True)
    path = inbox / "partial.json"
    path.write_bytes(b"{" + b" " * 80)
    real_read = os.read
    grown = {"done": False}

    def grow(fd, n):
        blob = real_read(fd, n)
        if not grown["done"] and n >= 80:
            grown["done"] = True
            with open(path, "ab") as handle:
                handle.write(b" ")
        return blob

    monkeypatch.setattr(os, "read", grow)
    poll_inbox(data, cfg, appending(data))
    assert path.is_file()
    assert not (inbox / "quarantine").exists() or list((inbox / "quarantine").glob("partial.json")) == []


def test_processed_symlink_is_not_followed(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    outside = tmp_path / "outside"
    outside.mkdir()
    inbox = data / "inbox" / "grokbot"
    inbox.mkdir(parents=True)
    (inbox / "processed").symlink_to(outside)
    reply_file(data, "task-route-1-sent", "route-1", text="留在里面")
    tick(data, cfg, appending(data))
    tick(data, cfg, appending(data))
    assert list(outside.iterdir()) == []
    assert [ev["data"]["text"] for ev in replay(data, types="reply")] == ["留在里面"]


def test_move_failure_does_not_emit_twice(tmp_path, monkeypatch):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    reply_file(data, "task-route-1-sent", "route-1", text="只一次")
    real_replace = os.replace
    failed = {"once": False}

    def flaky(src, dst):
        if not failed["once"] and "processed" in str(dst):
            failed["once"] = True
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    tick(data, cfg, appending(data))
    monkeypatch.setattr(os, "replace", real_replace)
    tick(data, cfg, appending(data))
    assert [ev["data"]["text"] for ev in replay(data, types="reply")] == ["只一次"]
    assert (data / "inbox" / "grokbot" / "processed" / "task-route-1-sent.json").is_file()


def test_reply_text_is_capped_and_safe_to_speak(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    append(data, route("route-1"))
    append(data, task_event(route("route-1"), status="sent", ref="route-1", adapter="grokbot"))
    nasty = "好\x00的\x1b[31m红\r\n第二句。" + ("长" * 5000) + "\u202eEND\ud800"
    folder = Path(data) / "inbox" / "grokbot"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task-route-1-sent.json").write_bytes(json.dumps({
        "task_id": "task-route-1-sent", "route_id": "route-1", "status": "done",
        "summary": "摘\x00要", "text": nasty,
    }).encode("utf-8"))
    tick(data, cfg, appending(data))
    reply = list(replay(data, types="reply"))[0]
    text = reply["data"]["text"]
    summary = reply["data"]["summary"]
    assert "\x00" not in text and "\x1b" not in text and "\r" not in text and "\n" not in text
    assert "\u202e" not in text and "\ud800" not in text
    assert "好的" in text and "第二句" in text
    assert len(text) <= 4000
    assert text.endswith("…")
    assert summary == "摘要"
    queue = ReportQueue()
    first = queue.push_all(list(replay(data)))
    spoken = make_event("spoken", "speak", {
        "text": first[0]["data"]["summary"], "completed": True, "interrupted": False,
        "brief_id": first[0]["id"],
    }, ref=first[0]["id"], id="spoken-" + first[0]["id"])
    details = queue.push_all([
        make_event("route", "route", {
            "target": "queue", "command": "details", "text": "详细说说", "turn_text": "详细说说",
            "rule": "queue:details", "reason": "队列指令", "method": "rule",
            "needs_confirmation": False,
        }, ts=TS, id="route-details"),
        spoken,
    ])
    spoken_text = details[0]["data"]["summary"]
    assert details[0]["data"]["kind"] == "details"
    assert spoken_text == text
    assert len(spoken_text) <= 4000


def test_one_bad_file_does_not_block_the_rest(tmp_path):
    data = tmp_path / "data"
    cfg = C.Config()
    for ident in ("route-bad", "route-good"):
        append(data, route(ident))
        append(data, task_event(route(ident), status="sent", ref=ident, adapter="grokbot"))
    reply_file(data, "task-route-bad-sent", "route-bad", text="坏的", name="a.json")
    reply_file(data, "task-route-good-sent", "route-good", text="好的", name="b.json")

    def emit(ev):
        if ev.get("type") == "reply" and ev["data"].get("text") == "坏的":
            raise RuntimeError("boom")
        append(data, ev, if_absent=True)

    tick(data, cfg, emit)
    assert [ev["data"]["text"] for ev in replay(data, types="reply")] == ["好的"]


def test_inbox_loop_survives_a_tick_error(tmp_path, monkeypatch):
    data = tmp_path / "data"
    cfg = C.Config()
    calls = {"n": 0}
    stop = threading.Event()

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("inbox down")
        stop.set()

    monkeypatch.setattr("audi27.grokbot.tick", boom)
    serve_inbox(data, cfg, poll_s=0.01, stop=stop)
    assert calls["n"] >= 2


def test_slow_body_does_not_stall_other_routes(tmp_path, monkeypatch):
    httpd, url = _background_server(_TrickleHandler)
    try:
        data = tmp_path / "data"
        cfg = grok_cfg(data, monkeypatch, url=url)
        cfg.grokbot.timeout_s = 0.4
        append(data, route("route-a", "第一件"))
        append(data, route("route-b", "第二件"))
        done = threading.Event()

        def run():
            try:
                dispatch_all(data, cfg)
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        started = time.monotonic()
        thread.start()
        assert done.wait(3), "a trickling webhook body stalled dispatch"
        assert time.monotonic() - started < 3
        assert httpd.hits == 2
        by_route = {}
        for ev in replay(data, types="task"):
            by_route.setdefault(ev["data"]["route_id"], []).append(ev["data"]["status"])
        assert by_route["route-a"] == ["sent", "running"]
        assert by_route["route-b"] == ["sent", "running"]
    finally:
        httpd.stop = True
        httpd.shutdown()
        httpd.server_close()


def test_timeout_before_a_response_is_bounded_and_retried_once(tmp_path, monkeypatch):
    httpd, url = _background_server(_HangHandler)
    try:
        started = time.monotonic()
        with pytest.raises(Exception):
            post_webhook(url, {"task_id": "t", "text": "hi"}, header_name="Authorization",
                         header_value="Bearer {key}", key="sekret", timeout_s=0.3)
        elapsed = time.monotonic() - started
        assert httpd.hits == 2
        assert elapsed < 2
    finally:
        httpd.shutdown()
        httpd.server_close()
