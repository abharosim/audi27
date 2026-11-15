"""Speaker: briefs in, spoken events out. The say process is mocked."""
import json
import threading
import time

import pytest

from audi27 import config as C
from audi27.cli import main
from audi27.events import append, make_event, replay, snapshot_offsets
from audi27.speak import say_backend, serve, speak_pending
from audi27.status import write_status

TS = "2026-11-15T12:00:00+00:00"


def brief(ident, summary):
    return make_event("brief", "queue", {
        "priority": "done", "summary": summary, "text_ref": "r1", "kind": "report", "item_id": "r1",
    }, ts=TS, ref="r1", id=ident)


class FakeProc:
    def __init__(self, code=None):
        self.returncode = code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_pending_briefs_are_spoken_in_order(tmp_path):
    data = tmp_path / "data"
    append(data, brief("brief-1", "第一句。"))
    append(data, brief("brief-2", "第二句。"))
    calls = []

    def backend(text, voice, stop_check):
        calls.append((text, voice, stop_check()))
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    assert speak_pending(data, C.Config(), backend=backend, speaking=lambda: False) == 2
    assert calls == [("第一句。", "Tingting", False), ("第二句。", "Tingting", False)]
    spoken = list(replay(data, types="spoken"))
    assert [e["ref"] for e in spoken] == ["brief-1", "brief-2"]
    assert spoken[0]["data"]["completed"] is True and spoken[0]["data"]["interrupted"] is False
    assert spoken[0]["data"]["text"] == "第一句。"
    assert speak_pending(data, C.Config(), backend=backend, speaking=lambda: False) == 0


def test_speaker_does_not_start_while_the_user_is_speaking(tmp_path):
    data = tmp_path / "data"
    append(data, brief("brief-1", "等等。"))
    seen = []

    def speaking():
        seen.append("check")
        return len(seen) < 3

    def backend(text, voice, stop_check):
        seen.append("speak")
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    assert speak_pending(data, C.Config(), backend=backend, speaking=speaking, poll_s=0.01) == 1
    assert seen == ["check", "check", "check", "speak"]


def test_say_is_killed_if_speaking_is_abandoned():
    proc = FakeProc()

    def sleep(_seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        say_backend(popen=lambda _cmd: proc, which=lambda _name: "/usr/bin/say",
                    sleep=sleep, platform="darwin")("你好", "Tingting", lambda: False)
    assert proc.terminated
    assert proc.poll() is not None


def test_say_stops_when_the_user_starts_and_logs_when_say_is_missing():
    proc = FakeProc()
    calls = []

    def popen(cmd):
        calls.append(cmd)
        return proc

    checks = {"n": 0}

    def stop_check():
        checks["n"] += 1
        return checks["n"] >= 2

    result = say_backend(popen=popen, which=lambda _name: "/usr/bin/say", sleep=lambda _s: None,
                         platform="darwin")("你好", "Tingting", stop_check)
    assert result["interrupted"] is True and result["completed"] is False
    assert proc.terminated and calls[0][:4] == ["say", "-v", "Tingting", "--"]

    def boom(_cmd):
        raise AssertionError("say should not be started")

    logged = say_backend(popen=boom, which=lambda _name: None, platform="darwin")("你好", "Tingting", lambda: False)
    assert logged["backend"] == "log" and logged["completed"] is True
    linux = say_backend(popen=boom, which=lambda _name: "/usr/bin/say", platform="linux")(
        "你好", "Tingting", lambda: False)
    assert linux["backend"] == "log"

    bad = FakeProc(code=1)
    failed = say_backend(popen=lambda _cmd: bad, which=lambda _name: "/usr/bin/say",
                         sleep=lambda _s: None, platform="darwin")("你好", "Tingting", lambda: False)
    assert failed["backend"] == "log" and failed["completed"] is True and failed["interrupted"] is False


def test_live_speaker_does_not_read_history(tmp_path):
    data = tmp_path / "data"
    append(data, brief("brief-old", "旧的"))
    calls = []

    def backend(text, voice, stop_check):
        calls.append(text)
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    cfg = C.Config()
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "backend": backend,
                "speaking": lambda: False},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    time.sleep(0.25)
    assert calls == [] and list(replay(data, types="spoken")) == []
    append(data, brief("brief-new", "新的一句"))
    deadline = time.time() + 3
    while time.time() < deadline and not calls:
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert calls == ["新的一句"]
    assert [e["ref"] for e in replay(data, types="spoken")] == ["brief-new"]
    assert not thread.is_alive()


def test_speak_test_prints_the_line_without_say(capsys):
    assert main(["speak", "test", "你好"]) == 0
    out = capsys.readouterr().out
    assert "你好" in out and out.startswith("log:")


def _rewind_speak(data):
    offs = {name: 0 for name in snapshot_offsets(data)}
    (data / "speak.cursor.json").write_text(json.dumps({"offsets": offs}), encoding="utf-8")


def test_the_backend_sees_speaker_stop_while_say_would_still_be_running(tmp_path):
    data = tmp_path / "data"
    append(data, brief("brief-1", "一句话"))
    _rewind_speak(data)
    seen = {}
    started = threading.Event()

    def backend(text, voice, stop_check):
        started.set()
        while not stop_check():
            time.sleep(0.01)
        seen["stopped"] = True
        return {"text": text, "completed": False, "interrupted": True, "backend": "mock"}

    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, C.Config()),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "backend": backend,
                "speaking": lambda: False},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    assert started.wait(2)
    stop.set()
    thread.join(2)
    assert seen.get("stopped") is True
    assert not thread.is_alive()


def test_a_failed_spoken_write_is_retried_and_does_not_skip_the_next_brief(tmp_path, monkeypatch):
    data = tmp_path / "data"
    append(data, brief("brief-1", "一"))
    append(data, brief("brief-2", "二"))
    _rewind_speak(data)
    calls = []

    def backend(text, voice, stop_check):
        calls.append(text)
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    import audi27.speak as speak_mod

    real = speak_mod.append
    hits = {"n": 0}

    def flaky(data_dir, event, if_absent=False):
        if event.get("type") == "spoken":
            hits["n"] += 1
            if hits["n"] == 1:
                raise OSError("no space")
        return real(data_dir, event, if_absent=if_absent)

    monkeypatch.setattr(speak_mod, "append", flaky)
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, C.Config()),
        kwargs={"poll_s": 0.05, "stop": stop, "ready": ready, "backend": backend,
                "speaking": lambda: False},
        daemon=True)
    thread.start()
    assert ready.wait(2)
    deadline = time.time() + 3
    while time.time() < deadline and [ev["ref"] for ev in replay(data, types="spoken")] != ["brief-1", "brief-2"]:
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert calls == ["一", "一", "二"]
    assert [ev["ref"] for ev in replay(data, types="spoken")] == ["brief-1", "brief-2"]


def test_a_fresh_heartbeat_counts_as_speaking(tmp_path):
    from audi27.speak import user_speaking

    assert user_speaking(tmp_path) is False
    write_status(tmp_path, {"updated": time.time(), "speaking": True})
    assert user_speaking(tmp_path) is True
    write_status(tmp_path, {"updated": time.time() - 30, "speaking": True})
    assert user_speaking(tmp_path) is False
    write_status(tmp_path, {"updated": time.time(), "speaking": False})
    assert user_speaking(tmp_path) is False
