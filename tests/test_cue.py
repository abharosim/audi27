"""Wake-word cue: a short local sound when the wake word is heard, before any agent runs.

The player is injected. Nothing here opens a sound device.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from audi27 import config as C
from audi27.cli import main
from audi27.confirm import plan
from audi27.dispatch import dispatch_all
from audi27.events import KNOWN_TYPES, append, make_event, replay, snapshot_offsets
from audi27.supervisor import child_specs

TS = "2026-11-15T12:00:00+00:00"
BRIEF_TS = "2026-11-15T12:01:00+00:00"
NOW = datetime(2026, 11, 15, 12, 5, tzinfo=timezone.utc)
SOUND = "/System/Library/Sounds/Tink.aiff"


def turn(ident="turn-1", trigger="达达", text="查天气", start=TS):
    return make_event("turn", "turns", {
        "utterance_ids": [ident + "-u"], "text": text, "raw_text": f"{trigger} {text}",
        "start": start, "end": start, "reason": "timeout", "trigger": trigger, "has_suspect": False,
    }, ts=start, ref=ident + "-u", id=ident)


def utterance(ident, text, start, *, status="ok"):
    return make_event("utterance", "transcribe", {
        "id": ident, "text": text, "status": status, "start": start, "end": start,
    }, ts=start, id=ident)


def cue_played(ident, start, until, *, ok=True, ref="turn-1"):
    return make_event("cue_played", "cue", {
        "sound": SOUND, "backend": "afplay" if ok else "log", "ok": ok,
        "trigger": "达达", "until": until,
    }, ts=start, ref=ref, id=ident)


class _Proc:
    def __init__(self, release, state):
        self.release = release
        self.state = state
        self.killed = False

    def wait(self, timeout=None):
        self.state["entered"] = True
        limit = 0.4 if timeout is None else min(float(timeout), 0.4)
        self.release.wait(limit)
        self.state["finished"] = True
        return 0

    def kill(self):
        self.killed = True
        self.release.set()


def _player_factory(*, platform="darwin", which=True, is_file=True, popen=None, release=None, state=None):
    from audi27.cue import afplay_player

    def default_popen(argv, **_kwargs):
        state["argv"] = list(argv)
        return _Proc(release or threading.Event(), state)

    return afplay_player(
        popen=popen or default_popen,
        which=(lambda _name: "/usr/bin/afplay") if which else (lambda _name: None),
        is_file=lambda _path: is_file,
        platform=platform,
    )


# --- player ------------------------------------------------------------------


def test_afplay_starts_and_returns_before_the_sound_finishes():
    release = threading.Event()
    state = {"entered": False, "finished": False}
    play = _player_factory(release=release, state=state)
    started = time.monotonic()
    result = play(SOUND)
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert result["ok"] is True and result["backend"] == "afplay" and result["sound"] == SOUND
    assert state["argv"] == ["afplay", "-t", "1", SOUND]
    assert state["entered"] is False or state["finished"] is False
    release.set()
    deadline = time.time() + 1
    while not state["finished"] and time.time() < deadline:
        time.sleep(0.01)
    assert state["finished"] is True


def test_missing_file_command_or_other_os_logs_and_does_not_play(caplog):
    state = {}

    def boom(_argv, **_kwargs):
        raise AssertionError("afplay should not start")

    caplog.set_level(logging.INFO)
    missing = _player_factory(is_file=False, popen=boom, state=state)
    assert missing(SOUND)["ok"] is False and missing(SOUND)["backend"] == "log"
    assert "missing" in caplog.text

    caplog.clear()
    absent = _player_factory(which=False, popen=boom, state=state)
    assert absent(SOUND) == {"ok": False, "backend": "log", "sound": SOUND}
    assert "afplay" in caplog.text

    caplog.clear()
    linux = _player_factory(platform="linux", popen=boom, state=state)
    assert linux(SOUND)["backend"] == "log" and linux(SOUND)["ok"] is False
    assert "not playing" in caplog.text

    def explode(_argv, **_kwargs):
        raise OSError("device busy")

    caplog.clear()
    failed = _player_factory(popen=explode, state=state)
    assert failed(SOUND)["ok"] is False and failed(SOUND)["backend"] == "log"
    assert "could not start" in caplog.text


def test_a_late_afplay_failure_is_logged_and_does_not_raise(caplog):
    from audi27.cue import afplay_player

    class Bad:
        def wait(self, timeout=None):
            return 1

        def kill(self):
            raise AssertionError("a finished process is not killed")

    caplog.set_level(logging.WARNING)
    play = afplay_player(
        popen=lambda _argv, **_kwargs: Bad(),
        which=lambda _name: "/usr/bin/afplay",
        is_file=lambda _path: True,
        platform="darwin",
    )
    assert play(SOUND)["ok"] is True
    deadline = time.time() + 1
    while "exited" not in caplog.text and time.time() < deadline:
        time.sleep(0.01)
    assert "exited" in caplog.text


# --- events ------------------------------------------------------------------


def _serve(data, cfg, player):
    from audi27.cue import serve

    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "player": player},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    return thread, stop


def _wait_cues(data, n, timeout=3):
    deadline = time.time() + timeout
    found = []
    while time.time() < deadline:
        found = list(replay(data, types="cue_played"))
        if len(found) >= n:
            break
        time.sleep(0.05)
    return found


def _player(calls, *, ok=True):
    def player(sound):
        calls.append(sound)
        return {"ok": ok, "backend": "mock" if ok else "log", "sound": sound}

    return player


def test_trigger_in_text_finds_a_wake_word_that_is_not_at_the_start():
    from audi27.cue import trigger_in_text

    assert trigger_in_text("今天达达，记一下", ["达达", "大大"]) == ("达达", False)
    assert trigger_in_text("达达，记一下", ["达达"]) == ("达达", True)
    assert trigger_in_text("  dada now", ["dada"]) == ("dada", True)
    assert trigger_in_text("say dada now", ["dada"]) == ("dada", False)
    assert trigger_in_text("大大小小", ["大大", "达达"]) == (None, False)
    assert trigger_in_text("dadapple", ["dada"]) == (None, False)
    assert trigger_in_text("明天下雨", ["达达"]) == (None, False)


def test_a_trigger_that_is_not_at_the_start_is_a_skipped_decision(tmp_path, caplog):
    from audi27.cue import ack

    data = tmp_path / "data"
    calls = []
    caplog.set_level(logging.INFO)
    assert ack(data, C.Config(), utterance("u-mid", "今天达达天气", TS), player=_player(calls), pending={}) is True
    assert calls == []
    assert list(replay(data, types="cue_played")) == []
    rows = list(replay(data, types="cue_decision"))
    assert len(rows) == 1
    assert rows[0]["ref"] == "u-mid"
    assert rows[0]["data"]["action"] == "skipped"
    assert rows[0]["data"]["reason"] == "trigger not at start so no turn"
    assert rows[0]["data"]["trigger"] == "达达"
    assert "cue_decision" in KNOWN_TYPES
    assert "u-mid" in caplog.text and "trigger not at start" in caplog.text


def test_an_utterance_with_no_trigger_writes_no_decision(tmp_path, caplog):
    from audi27.cue import ack

    data = tmp_path / "data"
    calls = []
    caplog.set_level(logging.INFO)
    said = utterance("u-ambient", "明天下雨", TS)
    assert ack(data, C.Config(), said, player=_player(calls), pending={}) is True
    assert calls == []
    assert list(replay(data, types="cue_decision")) == []
    assert list(replay(data, types="cue_played")) == []
    assert "cue decision" not in caplog.text


def test_a_rejected_trigger_is_skipped_for_status(tmp_path):
    from audi27.cue import ack

    data = tmp_path / "data"
    calls = []
    said = utterance("u-no", "达达，噪声", TS, status="rejected")
    assert ack(data, C.Config(), said, player=_player(calls), pending={}) is True
    assert calls == []
    row = list(replay(data, types="cue_decision"))[0]["data"]
    assert row["action"] == "skipped"
    assert row["reason"] == "status not ok (rejected)"
    assert row["trigger"] == "达达"


def test_a_played_cue_and_a_player_error_are_decisions(tmp_path):
    from audi27.cue import ack

    data = tmp_path / "data"
    calls = []
    cfg = C.Config()
    assert ack(data, cfg, utterance("u-ok", "达达，今天", TS), player=_player(calls), pending={}) is True
    assert ack(data, cfg, utterance("u-bad", "哒哒，第二句", "2026-11-15T12:00:05+00:00"),
               player=_player(calls, ok=False), pending={}) is True
    rows = list(replay(data, types="cue_decision"))
    assert [(row["ref"], row["data"]["action"], row["data"]["reason"]) for row in rows] == [
        ("u-ok", "played", "played"),
        ("u-bad", "skipped", "player error"),
    ]
    assert [ev["ref"] for ev in replay(data, types="cue_played")] == ["u-ok", "u-bad"]
    assert calls == [cfg.cue.sound, cfg.cue.sound]


def test_acking_the_same_utterance_again_is_dedup(tmp_path):
    from audi27.cue import ack

    data = tmp_path / "data"
    calls = []
    cfg = C.Config()
    said = utterance("u1", "达达，记一下", TS)
    assert ack(data, cfg, said, player=_player(calls), pending={}) is True
    assert ack(data, cfg, said, player=_player(calls), pending={}) is True
    assert calls == [cfg.cue.sound]
    rows = list(replay(data, types="cue_decision"))
    assert [row["data"]["reason"] for row in rows] == ["played", "dedup"]


def test_a_failed_cue_write_does_not_emit_a_second_decision(tmp_path, monkeypatch):
    import audi27.cue as cue_mod

    data = tmp_path / "data"
    calls = []
    real = cue_mod.append
    hits = {"n": 0}

    def flaky(data_dir, event, if_absent=False):
        if event.get("type") == "cue_played":
            hits["n"] += 1
            if hits["n"] == 1:
                raise OSError("no space")
        return real(data_dir, event, if_absent=if_absent)

    monkeypatch.setattr(cue_mod, "append", flaky)
    from audi27.cue import ack
    said = utterance("u1", "达达，记一下", TS)
    pending = {}
    assert ack(data, C.Config(), said, player=_player(calls), pending=pending) is False
    assert ack(data, C.Config(), said, player=_player(calls), pending=pending) is True
    assert calls == [SOUND]
    assert [row["data"]["reason"] for row in replay(data, types="cue_decision")] == ["played"]


def test_a_wake_utterance_dings_and_a_continuation_does_not(tmp_path):
    from audi27.cue import SUPPRESS_S

    data = tmp_path / "data"
    append(data, utterance("u-old", "达达，旧的", TS))
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    cfg = C.Config()
    cfg.cue.sound = SOUND
    cfg.turns.triggers = ["达达", "答答", "搭搭"]
    thread, stop = _serve(data, cfg, player)
    time.sleep(0.2)
    assert calls == [] and list(replay(data, types="cue_played")) == []

    append(data, utterance("u-wake", "达达，今天天气", "2026-11-15T12:00:01+00:00"))
    append(data, utterance("u-more", "明天下雨", "2026-11-15T12:00:02+00:00"))
    append(data, utterance("u-reject", "达达，噪声", "2026-11-15T12:00:03+00:00", status="rejected"))
    append(data, utterance("u-blank", "   ", "2026-11-15T12:00:04+00:00"))
    append(data, utterance("u-suspect", "答答？", "2026-11-15T12:00:05+00:00", status="suspect"))
    append(data, turn("turn-late", text="这条 turn 不该再响"))
    found = _wait_cues(data, 2)
    time.sleep(0.25)
    stop.set()
    thread.join(2)
    assert calls == [SOUND, SOUND]
    assert [ev["id"] for ev in found] == ["cue-u-wake", "cue-u-suspect"]
    assert [ev["ref"] for ev in found] == ["u-wake", "u-suspect"]
    assert found[0]["data"]["trigger"] == "达达" and found[1]["data"]["trigger"] == "答答"
    assert found[0]["data"]["ok"] is True and found[0]["data"]["backend"] == "mock"
    assert found[0]["data"]["sound"] == SOUND
    delta = datetime.fromisoformat(found[0]["data"]["until"]) - datetime.fromisoformat(found[0]["ts"])
    assert abs(delta.total_seconds() - SUPPRESS_S) < 0.05
    assert "cue_played" in KNOWN_TYPES
    assert not thread.is_alive()

    # A rewound cursor plays the pre-start line (it never got a cue) and skips the ones that did.
    offs = {name: 0 for name in snapshot_offsets(data)}
    (data / "cue.cursor.json").write_text(json.dumps({"offsets": offs}), encoding="utf-8")
    again, stop2 = _serve(data, cfg, player)
    _wait_cues(data, 3)
    time.sleep(0.2)
    stop2.set()
    again.join(2)
    assert calls == [SOUND, SOUND, SOUND]
    assert [ev["id"] for ev in replay(data, types="cue_played")] == [
        "cue-u-wake", "cue-u-suspect", "cue-u-old",
    ]


def test_a_second_trigger_dings_again(tmp_path):
    data = tmp_path / "data"
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    cfg = C.Config()
    thread, stop = _serve(data, cfg, player)
    append(data, utterance("u1", "达达，第一件", TS))
    append(data, utterance("u2", "继续说", "2026-11-15T12:00:01+00:00"))
    append(data, utterance("u3", "dada, next", "2026-11-15T12:00:02+00:00"))
    found = _wait_cues(data, 2)
    stop.set()
    thread.join(2)
    assert calls == [SOUND, SOUND]
    assert [ev["ref"] for ev in found] == ["u1", "u3"]
    assert [ev["data"]["trigger"] for ev in found] == ["达达", "dada"]


def test_a_player_exception_still_acks_the_utterance_and_the_next_one(tmp_path, caplog):
    data = tmp_path / "data"
    calls = []

    def player(sound):
        calls.append(sound)
        if len(calls) == 1:
            raise RuntimeError("no device")
        return {"ok": True, "backend": "mock", "sound": sound}

    caplog.set_level(logging.ERROR)
    thread, stop = _serve(data, C.Config(), player)
    append(data, utterance("u-a", "达达，第一句", TS))
    append(data, utterance("u-b", "达达，第二句", "2026-11-15T12:00:01+00:00"))
    deadline = time.time() + 3
    while time.time() < deadline and [ev["ref"] for ev in replay(data, types="cue_played")] != ["u-a", "u-b"]:
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    rows = list(replay(data, types="cue_played"))
    assert [ev["ref"] for ev in rows] == ["u-a", "u-b"]
    assert rows[0]["data"]["ok"] is False and rows[0]["data"]["backend"] == "log"
    assert rows[1]["data"]["ok"] is True
    assert "no device" in caplog.text
    assert not thread.is_alive()


def test_a_failed_cue_write_does_not_play_again(tmp_path, monkeypatch):
    import audi27.cue as cue_mod

    data = tmp_path / "data"
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    real = cue_mod.append
    hits = {"n": 0}

    def flaky(data_dir, event, if_absent=False):
        if event.get("type") == "cue_played":
            hits["n"] += 1
            if hits["n"] == 1:
                raise OSError("no space")
        return real(data_dir, event, if_absent=if_absent)

    monkeypatch.setattr(cue_mod, "append", flaky)
    thread, stop = _serve(data, C.Config(), player)
    append(data, utterance("u1", "达达，记一下", TS))
    deadline = time.time() + 3
    while time.time() < deadline and not list(replay(data, types="cue_played")):
        time.sleep(0.05)
    time.sleep(0.25)
    stop.set()
    thread.join(2)
    assert calls == [SOUND]
    rows = list(replay(data, types="cue_played"))
    assert [ev["id"] for ev in rows] == ["cue-u1"]
    assert hits["n"] >= 2


def test_cue_test_prints_a_result_and_writes_no_event(capsys, monkeypatch):
    def player(sound):
        return {"ok": True, "backend": "mock", "sound": sound}

    def boom(**_kwargs):
        raise AssertionError("afplay should not be built")

    monkeypatch.setattr("audi27.cue.build_player", lambda: player)
    monkeypatch.setattr("audi27.cue.afplay_player", boom)
    assert main(["cue", "test"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == f"mock: {SOUND}"
    assert list(replay("data", types="cue_played")) == []


# --- config / supervisor -----------------------------------------------------


def test_config_defaults_off_and_example_documents_the_sound():
    cfg = C.Config()
    assert cfg.cue.enabled is False
    assert cfg.cue.sound == SOUND
    example = C.load(Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.cue.enabled is False
    assert example.cue.sound == SOUND
    text = Path(__file__).resolve().parents[1].joinpath("config.example.toml").read_text(encoding="utf-8")
    assert "[cue]" in text and "enabled = false" in text.split("[cue]", 1)[1].split("[", 1)[0]


def test_config_rejects_a_bad_cue_section(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[cue]\nenabled = true\nsound = "/tmp/ding.aiff"\n', encoding="utf-8")
    loaded = C.load(path)
    assert loaded.cue.enabled is True and loaded.cue.sound == "/tmp/ding.aiff"
    for bad in (
        "[cue]\nenabled = \"yes\"\n",
        "[cue]\nsound = \"\"\n",
        "[cue]\nvolume = 1\n",
    ):
        path.write_text(bad, encoding="utf-8")
        try:
            C.load(path)
        except C.ConfigError:
            continue
        raise AssertionError(bad)


def test_supervisor_starts_cue_only_when_asked():
    assert "cue" not in child_specs(None, "127.0.0.1", 8727)
    assert "cue" not in child_specs(None, "127.0.0.1", 8727, turns=True, speak=True)
    on = child_specs("/tmp/audi27.toml", "127.0.0.1", 8727, cue=True)
    assert on["cue"][-2:] == ["cue", "run"]
    assert on["cue"][1:3] == ["-m", "audi27.cli"]
    assert "-c" in on["cue"] and "/tmp/audi27.toml" in on["cue"]


# --- echo suppression, same rule as a TTS read-back --------------------------


def _confirm_world():
    task = make_event("task", "dispatch", {
        "route_id": "route-pay", "target": "grok", "adapter": None,
        "status": "awaiting_confirmation", "text": "帮我发邮件给张三",
    }, ts=TS, ref="route-pay", id="task-route-pay-awaiting_confirmation")
    brief = make_event("brief", "queue", {
        "priority": "decision", "summary": "发给 Grok：帮我发邮件给张三。确认还是取消？",
        "text_ref": task["id"], "kind": "confirm", "item_id": task["id"], "task_id": task["id"],
    }, ts=BRIEF_TS, ref=task["id"], id=f"brief-{task['id']}")
    return task, brief


def test_a_bare_yes_during_the_cue_is_not_an_approval_and_one_after_it_is():
    task, brief = _confirm_world()
    played = cue_played(
        "cue-turn-1", "2026-11-15T12:01:02+00:00", "2026-11-15T12:01:03.500+00:00")
    during = utterance("u-echo", "确认", "2026-11-15T12:01:02.200+00:00")
    after = utterance("u-yes", "好的", "2026-11-15T12:01:04+00:00")
    assert plan([task, brief, played, during]) == []
    out = plan([task, brief, played, after])
    assert [ev["data"]["decision"] for ev in out] == ["approve"]
    # A wake-word turn during the ding is the person, same as talking over `say`.
    named = turn("turn-yes", text="确认", start="2026-11-15T12:01:02.200+00:00")
    decided = [ev for ev in plan([task, brief, played, named]) if ev["type"] == "confirm"]
    assert len(decided) == 1 and decided[0]["data"]["decision"] == "approve"
    # A cue that never played does not swallow a real answer.
    missed = cue_played(
        "cue-miss", "2026-11-15T12:01:02+00:00", "2026-11-15T12:01:03.500+00:00", ok=False)
    heard = plan([task, brief, missed, during])
    assert [ev["data"]["decision"] for ev in heard] == ["approve"]


def test_dispatch_ignores_a_cue_echo_and_sends_a_later_yes(tmp_path):
    from audi27.agent import FakeAgent
    from audi27.confirm import sync

    data = tmp_path / "data"
    cfg = C.Config()
    cfg.confirm.enabled = True
    append(data, make_event("route", "route", {
        "target": "grok", "text": "帮我发邮件给张三", "turn_text": "帮我发邮件给张三",
        "rule": "default", "reason": "", "method": "default",
        "needs_confirmation": True, "command": None,
    }, ts=TS, ref="turn-src", id="route-pay"))
    assert dispatch_all(data, cfg, now=NOW)["wrote"] == 1
    task_id = "task-route-pay-awaiting_confirmation"
    append(data, make_event("brief", "queue", {
        "priority": "decision", "summary": "发给 Grok：帮我发邮件给张三。确认还是取消？",
        "text_ref": task_id, "kind": "confirm", "item_id": task_id, "task_id": task_id,
    }, ts=BRIEF_TS, ref=task_id, id=f"brief-{task_id}"))
    append(data, cue_played(
        "cue-turn-1", "2026-11-15T12:01:02+00:00", "2026-11-15T12:01:03.500+00:00"))
    append(data, utterance("u-echo", "确认", "2026-11-15T12:01:02.200+00:00"))
    assert sync(data, cfg)["wrote"] == 0
    # A confirm that names the echo is still not an approval.
    append(data, make_event("confirm", "confirm", {
        "task_id": task_id, "decision": "approve", "source": "u-echo",
    }, ts="2026-11-15T12:01:02.200+00:00", ref=task_id, id="confirm-echo"))

    class Count(FakeAgent):
        def __init__(self):
            super().__init__()
            self.calls = []

        def handle(self, route, emit):
            self.calls.append(route["id"])
            super().handle(route, emit)

    agent = Count()
    assert dispatch_all(data, cfg, agents={"fake": agent}, now=NOW)["wrote"] == 0
    assert agent.calls == []
    append(data, utterance("u-yes", "确认", "2026-11-15T12:01:05+00:00"))
    assert sync(data, cfg)["wrote"] == 1
    assert dispatch_all(data, cfg, agents={"fake": agent}, now=NOW)["wrote"] >= 1
    assert agent.calls == ["route-pay"]
