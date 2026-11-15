"""Intent router: one turn in, one route out. No model, no microphone, nothing sent."""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.cli import main
from audi27.events import append, make_event, replay
from audi27.route import (
    RulesError, decide, get_rules, load_rules, load_rules_text, make_classifier,
    route_event, serve,
)
from audi27.supervisor import child_specs

EXAMPLE = Path(__file__).resolve().parents[1] / "routes.example.toml"
RULES = load_rules(EXAMPLE)
TS = "2026-11-15T12:00:00+00:00"

# text, target, command, needs_confirmation, method
CASES = [
    ("帮我问一下 Muse 下个月面试怎么准备", "muse", None, False, "rule"),
    ("发给 Muse，看看职业规划", "muse", None, False, "rule"),
    ("问一下 Grok 今天的新闻", "grok", None, False, "rule"),
    ("问一下 Grok 面试怎么准备", "grok", None, False, "rule"),
    ("ask grok what time it is", "grok", None, False, "rule"),
    ("grog，今天有什么会", "grok", None, False, "rule"),
    ("发给 Grokbot 看一下 inbox", "grok", None, False, "rule"),
    ("groggy morning", "grok", None, False, "default"),
    ("让 Claude Code 把测试修一下", "claude_code", None, False, "rule"),
    ("让 Cloud Code 把测试跑一下", "claude_code", None, False, "rule"),
    ("用 Crowed Code 看这个报错", "claude_code", None, False, "rule"),
    ("CLOUD CODE 看一下", "claude_code", None, False, "rule"),
    ("cloudcode跑一下", "claude_code", None, False, "rule"),
    ("让 Claude 看一下", "grok", None, False, "default"),
    ("fix the code", "grok", None, False, "default"),
    ("用本地模型总结一下", "local", None, False, "rule"),
    ("用 local model 总结一下这段", "local", None, False, "rule"),
    ("locally something", "grok", None, False, "default"),
    ("museum tickets", "grok", None, False, "default"),
    ("下个月面试怎么准备", "muse", None, False, "rule"),
    ("帮我看看简历", "muse", None, False, "rule"),
    ("what's a good career plan", "muse", None, False, "rule"),
    ("我想跳槽", "muse", None, False, "rule"),
    ("prepare for the interview next month", "muse", None, False, "rule"),
    ("the offer is low", "muse", None, False, "rule"),
    ("offering help", "grok", None, False, "default"),
    ("详细说说职业规划", "muse", None, False, "rule"),
    ("deploy the fetcher", "grok", None, False, "default"),
    ("今天天气怎么样", "grok", None, False, "default"),
    ("帮我改一下这个函数", "grok", None, False, "default"),
    ("打开 Kindle Dashboard", "grok", None, False, "default"),
    ("等一下帮我查天气", "grok", None, False, "default"),
    ("下一个面试", "muse", None, False, "rule"),
    ("详细说说", "queue", "details", False, "rule"),
    ("详细说说这个", "queue", "details", False, "rule"),
    ("展开说说", "queue", "details", False, "rule"),
    ("details", "queue", "details", False, "rule"),
    ("跳过", "queue", "skip", False, "rule"),
    ("跳过吧", "queue", "skip", False, "rule"),
    ("跳过这个", "queue", "skip", False, "rule"),
    ("请跳过", "queue", "skip", False, "rule"),
    ("请帮我跳过", "queue", "skip", False, "rule"),
    ("skip", "queue", "skip", False, "rule"),
    ("帮我跳过这个文件", "grok", None, False, "default"),
    ("稍后", "queue", "later", False, "rule"),
    ("等一下", "queue", "later", False, "rule"),
    ("稍后再说", "queue", "later", False, "rule"),
    ("later", "queue", "later", False, "rule"),
    ("还有什么没处理", "queue", "pending", False, "rule"),
    ("还有什么没处理的吗", "queue", "pending", False, "rule"),
    ("还有什么", "queue", "pending", False, "rule"),
    ("还有什么问题", "grok", None, False, "default"),
    ("关掉", "queue", "close", False, "rule"),
    ("别说了", "queue", "close", False, "rule"),
    ("不用念了", "queue", "close", False, "rule"),
    ("close", "queue", "close", False, "rule"),
    ("关闭这个窗口", "grok", None, False, "default"),
    ("算了", "none", None, False, "rule"),
    ("算了吧", "none", None, False, "rule"),
    ("不用了", "none", None, False, "rule"),
    ("算了下个月的账", "grok", None, False, "default"),
    ("没事找事", "grok", None, False, "default"),
    ("", "none", None, False, "rule"),
    ("   ", "none", None, False, "rule"),
    ("？？", "none", None, False, "rule"),
    ("帮我发邮件给张三", "grok", None, True, "default"),
    ("让 Muse 发一封邮件给 hr", "muse", None, True, "rule"),
    ("帮我买杯咖啡", "grok", None, True, "default"),
    ("把那个文件删了", "grok", None, True, "default"),
    ("git push 一下", "grok", None, True, "default"),
    ("帮我 post 到论坛", "grok", None, True, "default"),
    ("poster design", "grok", None, False, "default"),
    ("notification 推送", "grok", None, False, "default"),
    ("问一下 Cloud Code，push 一下代码", "claude_code", None, True, "rule"),
    ("帮我把代码 push 上去", "grok", None, True, "default"),
    ("Muse 帮我看看 offer", "muse", None, False, "rule"),
]


def _turn(text, id="turn-1", ts=TS, ref="u1"):
    return make_event("turn", "turns", {
        "utterance_ids": [ref], "text": text, "raw_text": text,
        "start": ts, "end": ts, "reason": "gap", "trigger": "达达", "has_suspect": False,
    }, ts=ts, ref=ref, id=id)


@pytest.mark.parametrize(
    "text,target,command,confirm,method", CASES,
    ids=[f"{i}:{(text or 'empty')[:24]}" for i, (text, *_) in enumerate(CASES)])
def test_sentence_routes(text, target, command, confirm, method):
    decision = decide(text, RULES)
    assert decision.target == target
    assert decision.command == command
    assert decision.needs_confirmation is confirm
    assert decision.method == method
    assert decision.turn_text == text.strip()


@pytest.mark.parametrize("text,message", [
    ("帮我问一下 Muse 下个月面试怎么准备", "下个月面试怎么准备"),
    ("发给 Muse，看下日程", "看下日程"),
    ("问一下，Muse 看日程", "看日程"),
    ("发给muse", ""),
    ("让 Cloud Code 把测试跑一下", "把测试跑一下"),
    ("请发给 Claude Code，修一下这个测试", "修一下这个测试"),
    ("用本地模型总结一下", "总结一下"),
    ("使用本地模型总结一下", "总结一下"),
    ("用 local model 总结一下这段", "总结一下这段"),
    ("ask Muse how to prepare", "how to prepare"),
    ("tell claude code to fix the test", "to fix the test"),
    ("问一下 Grok", ""),
    ("下个月面试怎么准备", "下个月面试怎么准备"),
    ("Muse 帮我看看 offer", "Muse 帮我看看 offer"),
    ("问一下Ｍｕｓｅ明天的会", "明天的会"),
])
def test_addressing_is_removed_from_the_message(text, message):
    assert decide(text, RULES).text == message


def test_explicit_name_is_the_rule_even_when_a_keyword_also_matches():
    decision = decide("问一下 Grok 面试怎么准备", RULES)
    assert decision.rule == "explicit:grok"
    assert decide("下个月面试怎么准备", RULES).rule == "keyword:career"
    assert decide("跳过", RULES).rule == "queue:skip"
    assert decide("算了", RULES).rule == "none:not_a_command"
    assert decide("", RULES).rule == "empty"
    assert decide("今天天气怎么样", RULES).rule == "default"


def test_classifier_only_sees_leftovers_and_cannot_name_claude():
    seen = []

    def classify(text):
        seen.append(text)
        return "none"

    for text in ("跳过", "下个月面试怎么准备", "问一下 Muse 面试", "算了", ""):
        assert decide(text, RULES, classify=classify).method != "llm"
    assert seen == []
    decision = decide("今天天气怎么样", RULES, classify=classify)
    assert (decision.target, decision.method, decision.rule) == ("none", "llm", "llm")
    assert seen == ["今天天气怎么样"]
    assert decide("今天天气怎么样", RULES, classify=lambda _t: "claude_code").method == "default"
    assert decide("今天天气怎么样", RULES, classify=lambda _t: "  Muse\n").target == "muse"

    def boom(_text):
        raise RuntimeError("down")

    assert decide("今天天气怎么样", RULES, classify=boom).method == "default"


def test_classifier_timeout_falls_back_to_the_default():
    def slow(_text):
        time.sleep(0.3)
        return "muse"

    decision = decide("今天天气怎么样", RULES, classify=slow, timeout_s=0.05)
    assert decision.target == "grok" and decision.method == "default"


def test_file_default_is_used_when_nothing_matches():
    rules = load_rules_text('default = "muse"\n')
    decision = decide("今天天气怎么样", rules)
    assert decision.target == "muse" and decision.method == "default"


def test_rules_file_rejects_typos_and_keyword_targets_that_must_be_named():
    with pytest.raises(RulesError):
        load_rules_text('default = "grok"\nnope = 1\n')
    with pytest.raises(RulesError):
        load_rules_text('default = "claude_code"\n')
    with pytest.raises(RulesError):
        load_rules_text('default = "grok"\n[[keyword]]\nname = "x"\ntarget = "claude_code"\nwords = ["代码"]\n')
    with pytest.raises(RulesError):
        load_rules_text('default = "grok"\n[[keyword]]\nname = "x"\ntarget = "local"\nwords = ["代码"]\n')
    with pytest.raises(RulesError):
        load_rules_text("not toml [[[\n")


def test_rules_are_reread_when_the_file_changes(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text('default = "grok"\n', encoding="utf-8")
    assert decide("吃个苹果", get_rules(str(path))).target == "grok"
    path.write_text(
        'default = "grok"\n[[keyword]]\nname = "fruit"\ntarget = "muse"\nwords = ["苹果"]\n',
        encoding="utf-8")
    os_utime = time.time() + 5
    os.utime(path, (os_utime, os_utime))
    assert decide("吃个苹果", get_rules(str(path))).target == "muse"
    assert decide("吃个苹果", get_rules(str(path))).rule == "keyword:fruit"
    path.write_text("not toml [[[\n", encoding="utf-8")
    os.utime(path, (os_utime + 5, os_utime + 5))
    assert decide("吃个苹果", get_rules(str(path))).target == "muse"  # last good copy
    with pytest.raises(RulesError):
        get_rules(str(path), strict=True)


def test_missing_routes_toml_uses_the_example():
    rules = get_rules("routes.toml")
    assert rules.path.endswith("routes.example.toml")
    assert decide("下个月面试怎么准备", rules).target == "muse"
    assert decide("跳过", rules).command == "skip"


def test_route_event_id_is_stable_and_a_second_write_is_skipped(tmp_path):
    data = tmp_path / "data"
    turn = _turn("跳过")
    event = route_event(turn, decide("跳过", RULES))
    assert event["id"] == "route-turn-1"
    assert event["ref"] == "turn-1"
    assert event["type"] == "route" and event["source"] == "route"
    assert event["ts"] == turn["ts"]
    assert event["data"]["command"] == "skip"
    assert append(data, event) is not None
    assert append(data, event, if_absent=True) is None
    assert [item["id"] for item in replay(data, types="route")] == ["route-turn-1"]


def test_cli_test_and_replay_print_only_until_write(tmp_path, capsys):
    assert main(["route", "test", "跳过"]) == 0
    out = capsys.readouterr().out
    assert "target: queue" in out and "command: skip" in out
    assert main(["route", "test", "--json", "帮我问一下 Muse 下个月面试怎么准备"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["target"] == "muse" and body["text"] == "下个月面试怎么准备"
    assert body["needs_confirmation"] is False and body["method"] == "rule"

    data = tmp_path / "data"
    append(data, _turn("跳过"))
    args = ["route", "replay", "--data-dir", str(data),
            "--since", "2026-11-15T00:00:00+00:00", "--until", "2026-11-16T00:00:00+00:00"]
    assert main(args + ["--json"]) == 0
    first = capsys.readouterr().out
    assert main(args + ["--json"]) == 0
    assert capsys.readouterr().out == first
    event = json.loads(first.strip().splitlines()[0])
    assert event["id"] == "route-turn-1" and event["ref"] == "turn-1"
    assert event["data"]["target"] == "queue" and event["data"]["command"] == "skip"
    assert list(replay(data, types="route")) == []

    assert main(args + ["--write"]) == 0
    assert "wrote 1" in capsys.readouterr().err
    assert main(args + ["--write"]) == 0
    assert "already present 1" in capsys.readouterr().err
    stored = list(replay(data, types="route"))
    assert len(stored) == 1 and stored[0]["id"] == "route-turn-1"
    assert main(["route", "replay", "--since", "not-a-time"]) == 2


def test_live_run_does_not_backfill_and_does_not_duplicate(tmp_path):
    data = tmp_path / "data"
    append(data, _turn("旧的一句", id="turn-old"))
    cfg = C.Config()
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.25)
    assert list(replay(data, types="route")) == []
    append(data, _turn("跳过", id="turn-new", ts="2026-11-15T12:05:00+00:00", ref="u2"))
    found = []
    deadline = time.time() + 3
    while time.time() < deadline:
        found = list(replay(data, types="route"))
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert [item["id"] for item in found] == ["route-turn-new"]
    assert found[0]["data"]["command"] == "skip" and found[0]["ref"] == "turn-new"
    assert not thread.is_alive()

    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.3)
    stop.set()
    thread.join(2)
    assert [item["id"] for item in replay(data, types="route")] == ["route-turn-new"]
    assert not thread.is_alive()


def test_route_config_and_supervisor(tmp_path):
    cfg = C.Config()
    assert cfg.route.enabled is False and cfg.route.llm is False
    assert cfg.route.rules == "routes.toml" and cfg.route.timeout_s == 8.0
    assert make_classifier(cfg) is None
    example = C.load(Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.route.enabled is False and example.route.llm is False
    path = tmp_path / "c.toml"
    path.write_text("[route]\nenabled = true\nllm = false\ntimeout_s = 3\nrules = ' mine.toml '\n")
    loaded = C.load(path)
    assert loaded.route.enabled is True and loaded.route.timeout_s == 3.0
    assert loaded.route.rules == "mine.toml"
    for bad in ("[route]\nllm = \"yes\"\n", "[route]\ntimeout_s = 0\n", "[route]\nrules = \"\"\n",
                "[route]\nnope = 1\n"):
        path.write_text(bad)
        with pytest.raises(C.ConfigError):
            C.load(path)
    off = child_specs(None, "127.0.0.1", 8727)
    on = child_specs("/tmp/audi27.toml", "127.0.0.1", 8727, turns=True, route=True)
    assert "route" not in off and set(off) == {"run", "web"}
    assert on["route"][-2:] == ["route", "run"]
    assert on["turns"][-2:] == ["turns", "run"]
    assert "-c" in on["route"] and "/tmp/audi27.toml" in on["route"]


def _get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(port, path, body, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_dashboard_lists_routes_and_keeps_a_running_accuracy(tmp_path):
    from audi27.web import make_server

    data = tmp_path / "data"
    append(data, route_event(_turn("下个月面试怎么准备", id="turn-a"), decide("下个月面试怎么准备", RULES)))
    append(data, route_event(_turn("跳过", id="turn-b", ts="2026-11-15T12:05:00+00:00"), decide("跳过", RULES)))
    server = make_server(data, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, page = _get(port, "/")
        assert code == 200 and "路由".encode() in page and b'id="viewR"' in page and b'id="tabR"' in page
        body = json.loads(_get(port, "/api/routes")[1])
        assert [row["id"] for row in body["routes"]] == ["route-turn-b", "route-turn-a"]
        assert body["routes"][0]["command"] == "skip"
        assert body["routes"][1]["target"] == "muse" and body["routes"][1]["turn_text"] == "下个月面试怎么准备"
        assert body["routes"][1]["verdict"] is None
        assert body["accuracy"] == {"correct": 0, "judged": 0, "rate": None}

        assert _post(port, "/api/routes/verdict", {"id": "route-turn-a", "correct": True})[0] == 200
        assert _post(port, "/api/routes/verdict", {"id": "route-turn-b", "correct": False})[0] == 200
        body = json.loads(_get(port, "/api/routes")[1])
        marks = {row["id"]: row["verdict"] for row in body["routes"]}
        assert marks == {"route-turn-a": True, "route-turn-b": False}
        assert body["accuracy"] == {"correct": 1, "judged": 2, "rate": 0.5}

        assert _post(port, "/api/routes/verdict", {"id": "route-turn-b", "correct": True})[0] == 200
        body = json.loads(_get(port, "/api/routes")[1])
        assert body["accuracy"] == {"correct": 2, "judged": 2, "rate": 1.0}

        assert _post(port, "/api/routes/verdict", {"id": "route-turn-b", "cleared": True})[0] == 200
        body = json.loads(_get(port, "/api/routes")[1])
        assert {row["id"]: row["verdict"] for row in body["routes"]}["route-turn-b"] is None
        assert body["accuracy"] == {"correct": 1, "judged": 1, "rate": 1.0}

        assert _post(port, "/api/routes/verdict", {"id": "../x", "correct": True})[0] == 400
        assert _post(port, "/api/routes/verdict", {"id": "route-turn-a", "correct": "yes"})[0] == 400
        assert _post(port, "/api/routes/verdict", {"id": "route-turn-a", "correct": True},
                     {"Origin": "http://evil.example"})[0] == 403
        assert _get(port, "/api/routes?limit=0")[0] == 400
        assert not (data / "verdicts.jsonl").exists()
        assert (data / "route_verdicts.jsonl").is_file()
    finally:
        server.shutdown()
        server.server_close()
