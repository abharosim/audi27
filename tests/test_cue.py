"""Wake-word cue: a short local sound when a turn is detected, before any agent runs.

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


def utterance(ident, text, start):
    return make_event("utterance", "transcribe", {
        "id": ident, "text": text, "status": "ok", "start": start, "end": start,
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


def test_a_new_turn_plays_once_and_records_cue_played(tmp_path):
    from audi27.cue import SUPPRESS_S, serve

    data = tmp_path / "data"
    append(data, turn("turn-old"))
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    cfg = C.Config()
    cfg.cue.sound = SOUND
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "player": player},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.2)
    assert calls == [] and list(replay(data, types="cue_played")) == []

    append(data, turn("turn-new", text="今天天气"))
    deadline = time.time() + 3
    found = []
    while time.time() < deadline:
        found = list(replay(data, types="cue_played"))
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert calls == [SOUND]
    assert len(found) == 1
    ev = found[0]
    assert ev["type"] == "cue_played" and ev["source"] == "cue" and ev["ref"] == "turn-new"
    assert ev["id"] == "cue-turn-new"
    assert ev["data"]["ok"] is True and ev["data"]["backend"] == "mock"
    assert ev["data"]["sound"] == SOUND and ev["data"]["trigger"] == "达达"
    delta = datetime.fromisoformat(ev["data"]["until"]) - datetime.fromisoformat(ev["ts"])
    assert abs(delta.total_seconds() - SUPPRESS_S) < 0.05
    assert "cue_played" in KNOWN_TYPES
    assert not thread.is_alive()

    # Rewinding the cursor replays turns that never got a cue, and skips one that did.
    offs = {name: 0 for name in snapshot_offsets(data)}
    (data / "cue.cursor.json").write_text(json.dumps({"offsets": offs}), encoding="utf-8")
    ready2, stop2 = threading.Event(), threading.Event()
    again = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop2, "ready": ready2, "player": player},
        daemon=True)
    again.start()
    assert ready2.wait(2)
    deadline = time.time() + 3
    while time.time() < deadline and len(calls) < 2:
        time.sleep(0.05)
    time.sleep(0.2)
    stop2.set()
    again.join(2)
    assert calls == [SOUND, SOUND]
    assert sorted(ev["id"] for ev in replay(data, types="cue_played")) == ["cue-turn-new", "cue-turn-old"]
    assert not again.is_alive()


def test_a_player_exception_still_acks_the_turn_and_the_next_one(tmp_path, caplog):
    from audi27.cue import serve

    data = tmp_path / "data"
    calls = []

    def player(sound):
        calls.append(sound)
        if len(calls) == 1:
            raise RuntimeError("no device")
        return {"ok": True, "backend": "mock", "sound": sound}

    caplog.set_level(logging.ERROR)
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, C.Config()),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "player": player},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    append(data, turn("turn-a"))
    append(data, turn("turn-b", text="第二句"))
    deadline = time.time() + 3
    while time.time() < deadline and [ev["ref"] for ev in replay(data, types="cue_played")] != ["turn-a", "turn-b"]:
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    rows = list(replay(data, types="cue_played"))
    assert [ev["ref"] for ev in rows] == ["turn-a", "turn-b"]
    assert rows[0]["data"]["ok"] is False and rows[0]["data"]["backend"] == "log"
    assert rows[1]["data"]["ok"] is True
    assert "no device" in caplog.text
    assert not thread.is_alive()


def test_cue_test_prints_a_result_and_writes_no_event(capsys):
    assert main(["cue", "test"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith(("log:", "afplay:")) and "Tink.aiff" in out
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
