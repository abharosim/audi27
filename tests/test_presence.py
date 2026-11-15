"""Presence: debounced speech signal, fast file, speaker barge-in. No microphone."""
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from audi27 import config as C
from audi27.events import append, make_event, replay
from audi27.presence import PresencePublisher, busy, load_presence, write_presence
from audi27.queue import sync
from audi27.segmenter import Segmenter
from audi27.speak import say_backend, speak_pending, unspoken, user_busy, user_speaking
from audi27.status import write_status
from audi27.supervisor import child_specs

TS = "2026-11-15T12:00:00+00:00"


def reply(ident, text):
    return make_event("reply", "dispatch", {
        "text": text, "target": "grok", "adapter": "fake", "needs_decision": False, "route_id": "route-1",
    }, ts=TS, ref="task-1", id=ident)


def brief(ident, summary):
    return make_event("brief", "queue", {
        "priority": "done", "summary": summary, "text_ref": "r1", "kind": "report", "item_id": "r1",
    }, ts=TS, ref="r1", id=ident)


def cfg_on(**kw):
    cfg = C.Config()
    cfg.presence.enabled = True
    for key, value in kw.items():
        setattr(cfg.presence, key, value)
    return cfg


def file(state, updated, since=None, raw=False):
    return {"state": state, "updated": updated, "since": since, "raw": raw}


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


def _states(events):
    return [ev["data"]["state"] for ev in events]


def test_short_speech_and_a_short_pause_do_not_flap(tmp_path):
    cfg = cfg_on(onset_ms=100, hangover_ms=300, publish_s=0.1)
    pub = PresencePublisher(tmp_path, cfg.presence)
    events = []

    def tick(speech, now):
        ev = pub.tick(speech, now)
        if ev is not None:
            events.append(ev)

    tick(True, 0.0)
    tick(True, 0.05)          # still inside the onset
    tick(False, 0.08)         # died before it counted
    assert events == [] and pub.tracker.state == "quiet"

    tick(True, 1.0)
    tick(True, 1.25)          # onset met
    tick(False, 1.5)
    tick(False, 1.7)          # pause shorter than the hangover
    tick(True, 1.8)           # still the same stretch of speech
    tick(False, 2.0)
    tick(False, 3.0)          # hangover met
    assert _states(events) == ["speaking", "quiet"]
    assert [ev["type"] for ev in events] == ["presence", "presence"]
    assert [ev["source"] for ev in events] == ["capture", "capture"]
    # The file kept moving across the whole utterance; the log did not.
    assert load_presence(tmp_path)["updated"] == 3.0
    assert load_presence(tmp_path)["state"] == "quiet"


def test_a_long_utterance_publishes_within_0_2s_and_only_one_event(tmp_path):
    cfg = cfg_on(onset_ms=50, publish_s=0.1, stale_s=1.0)
    pub = PresencePublisher(tmp_path, cfg.presence)
    assert pub.tick(True, 10.0) is None
    opened = pub.tick(True, 10.15)
    assert opened["data"]["state"] == "speaking"
    snap = load_presence(tmp_path)
    assert snap["state"] == "speaking" and snap["raw"] is True
    assert snap["updated"] - 10.0 <= 0.2
    # Heartbeats refresh `updated` so a long turn does not go stale and unblock the speaker.
    assert pub.tick(True, 12.0) is None
    assert load_presence(tmp_path)["updated"] == 12.0
    assert list(replay(tmp_path, types="presence")) == [opened]
    assert busy(tmp_path, cfg.presence, now=12.0) is True


def test_grace_does_not_grow_when_the_file_is_only_refreshed(tmp_path):
    cfg = cfg_on(onset_ms=0, hangover_ms=100, grace_s=0.3, publish_s=0.1)
    pub = PresencePublisher(tmp_path, cfg.presence)
    assert pub.tick(True, 5.0)["data"]["state"] == "speaking"
    assert pub.tick(False, 6.0) is None  # silence just started; hangover has not elapsed
    assert pub.tick(False, 7.0)["data"]["state"] == "quiet"
    assert load_presence(tmp_path)["since"] == 7.0
    assert busy(tmp_path, cfg.presence, now=7.2) is True
    assert busy(tmp_path, cfg.presence, now=7.4) is False
    assert pub.tick(False, 8.0) is None  # heartbeat, not a new transition
    assert load_presence(tmp_path)["updated"] == 8.0
    assert load_presence(tmp_path)["since"] == 7.0  # a refresh does not extend the grace period
    assert busy(tmp_path, cfg.presence, now=8.0) is False
    assert len(list(replay(tmp_path, types="presence"))) == 2


def test_clock_jump_back_rewrites_the_file_and_a_jump_forward_does_not_block(tmp_path):
    cfg = cfg_on(onset_ms=60, hangover_ms=400, grace_s=0.3, stale_s=1.0)
    pub = PresencePublisher(tmp_path, cfg.presence)
    pub.tick(True, 100.0)
    assert pub.tick(True, 100.1)["data"]["state"] == "speaking"
    assert pub.tick(True, 10.0) is None  # clock stepped back; still the same state
    snap = load_presence(tmp_path)
    assert snap["updated"] == 10.0 and snap["state"] == "speaking"
    assert busy(tmp_path, cfg.presence, now=10.0) is True
    assert pub.tick(False, 10.0) is None
    assert pub.tick(False, 10.5)["data"]["state"] == "quiet"
    assert busy(tmp_path, cfg.presence, now=10.5) is True
    assert busy(tmp_path, cfg.presence, now=10.9) is False

    # Capture kept writing "speaking", then the reader's clock leapt past stale_s.
    pub.tick(True, 30.0)
    pub.tick(True, 30.1)
    assert busy(tmp_path, cfg.presence, now=30.1) is True
    assert busy(tmp_path, cfg.presence, now=30.1 + 10) is False
    # The next sample, on the new clock, is trusted again.
    assert pub.tick(True, 40.1) is None
    assert busy(tmp_path, cfg.presence, now=40.1) is True


def test_a_dead_or_garbled_file_does_not_block(tmp_path):
    cfg = cfg_on(stale_s=1.0, grace_s=0.3)
    now = 1_000.0
    assert busy(tmp_path, cfg.presence, now=now) is False
    write_presence(tmp_path, file("speaking", now - 30, now - 30, True))
    assert busy(tmp_path, cfg.presence, now=now) is False
    write_presence(tmp_path, file("speaking", now + 30, now + 30, True))
    assert busy(tmp_path, cfg.presence, now=now) is False
    (tmp_path / "presence.json").write_text("{", encoding="utf-8")
    assert busy(tmp_path, cfg.presence, now=now) is False
    write_presence(tmp_path, {"state": "speaking", "updated": True, "since": now})
    assert busy(tmp_path, cfg.presence, now=now) is False
    write_presence(tmp_path, file("nope", now, now))
    assert busy(tmp_path, cfg.presence, now=now) is False
    # Known quiet, with the grace already over, is not busy either.
    write_presence(tmp_path, file("quiet", now, now - 5))
    assert busy(tmp_path, cfg.presence, now=now) is False


def test_close_records_quiet_once(tmp_path):
    cfg = cfg_on(onset_ms=0)
    pub = PresencePublisher(tmp_path, cfg.presence)
    assert pub.tick(False, 0.0) is None
    assert pub.close(1.0) is None
    assert pub.tick(True, 2.0)["data"]["state"] == "speaking"
    closed = pub.close(3.0)
    assert closed["data"]["state"] == "quiet"
    assert pub.close(4.0) is None
    assert _states(replay(tmp_path, types="presence")) == ["speaking", "quiet"]
    assert load_presence(tmp_path)["state"] == "quiet"


def test_a_failed_event_write_still_publishes_the_file(tmp_path):
    cfg = cfg_on(onset_ms=0)

    def boom(_root, _event):
        raise OSError("disk full")

    pub = PresencePublisher(tmp_path, cfg.presence, append_fn=boom)
    assert pub.tick(True, 1.0) is None
    assert load_presence(tmp_path)["state"] == "speaking"
    assert list(replay(tmp_path, types="presence")) == []


def test_segmenter_exposes_the_frame_speech_bit():
    seg = Segmenter(C.SegmentCfg(), 16000, 30)
    assert seg.last_speech is False
    seg.feed(np.zeros(480, np.int16), 0.0)
    assert seg.last_speech is False
    seg.feed(np.full(480, 8000, np.int16), 0.03)
    assert seg.last_speech is True


def test_recorder_writes_transitions_only_when_enabled(tmp_path, monkeypatch):
    import audi27.capture as capture
    from audi27.pipeline import Pipeline
    from audi27.status import read_status
    from audi27.store import Store
    from audi27.transcribers import Piece, Transcriber, Transcript

    class FakeT(Transcriber):
        name = "fake"
        model = "m"

        def transcribe(self, wav, language=None, prompt=None):
            return Transcript([Piece(" hello from presence", 0, 2, -0.2, 0.01, 1.2)], "en", 0.99, "fake", "m")

    quiet = np.zeros(480, np.int16)
    loud = np.full(480, 8000, np.int16)
    frames = [quiet] * 40 + [loud] * 60 + [quiet] * 40

    class FakeCapture:
        dropped_frames = overflows = 0

        def __init__(self, _cfg):
            pass

        def frames(self, _stop):
            for i, frame in enumerate(frames):
                yield frame, 1_700_000_000.0 + i * 0.03

    monkeypatch.setattr(capture, "Capture", FakeCapture)

    def run(enabled, name):
        cfg = C.Config()
        cfg.presence.enabled = enabled
        cfg.correct.enabled = False
        # Frames are delivered with no wall-clock gap, so the default 60 ms onset
        # would never elapse. Debounce itself is covered above; here onset and
        # hangover are zero so the recorder's wiring is what is under test.
        cfg.presence.onset_ms = 0
        cfg.presence.hangover_ms = 0
        store = Store(tmp_path / name, "none")
        Pipeline(cfg, [FakeT()], store).run()
        return cfg, store

    _off_cfg, off = run(False, "off")
    status = read_status(off.root)
    assert status["status"]["stopped"] is True
    assert not (off.root / "presence.json").exists()
    assert list(replay(off.root, types="presence")) == []

    on_cfg, on = run(True, "on")
    assert _states(replay(on.root, types="presence")) == ["speaking", "quiet"]
    snap = load_presence(on.root)
    assert snap["state"] == "quiet" and snap["since"] is not None
    # Just stopped, so the grace period still holds; a second later it does not.
    assert busy(on.root, on_cfg.presence, now=snap["since"] + 0.05) is True
    assert busy(on.root, on_cfg.presence, now=snap["updated"] + 1) is False


def test_flag_off_matches_the_heartbeat_and_ignores_the_presence_file(tmp_path):
    cfg = C.Config()
    assert cfg.presence.enabled is False
    now = time.time()
    assert user_busy(tmp_path, cfg) is False
    assert user_busy(tmp_path, cfg) == user_speaking(tmp_path)

    write_presence(tmp_path, file("speaking", now, now, True))
    assert user_busy(tmp_path, cfg) is False

    write_status(tmp_path, {"updated": now, "speaking": True})
    assert user_speaking(tmp_path) is True
    assert user_busy(tmp_path, cfg) is True

    write_status(tmp_path, {"updated": now, "speaking": False})
    assert user_busy(tmp_path, cfg) is False

    data = tmp_path / "spoken"
    append(data, brief("brief-1", "可以念。"))
    write_presence(data, file("speaking", time.time(), time.time(), True))
    calls = []

    def backend(text, voice, stop_check):
        calls.append((text, stop_check()))
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    assert speak_pending(data, cfg, backend=backend) == 1
    assert calls == [("可以念。", False)]
    spoken = list(replay(data, types="spoken"))
    assert spoken[0]["data"]["completed"] is True and spoken[0]["data"]["interrupted"] is False


def test_speaker_stays_quiet_while_speaking_and_through_the_grace_period(tmp_path):
    data = tmp_path / "data"
    cfg = cfg_on()
    append(data, brief("brief-1", "等一下。"))
    now = time.time()
    write_presence(data, file("speaking", now, now, True))
    calls = []

    def backend(text, _voice, _stop):
        calls.append(text)
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    stop = threading.Event()
    stop.set()
    assert speak_pending(data, cfg, backend=backend, stop=stop, poll_s=0.01) == 0
    assert calls == [] and list(replay(data, types="spoken")) == []

    write_presence(data, file("quiet", time.time(), time.time() - 0.05))
    assert speak_pending(data, cfg, backend=backend, stop=stop, poll_s=0.01) == 0
    assert calls == []

    write_presence(data, file("quiet", time.time(), time.time() - 10))
    assert speak_pending(data, cfg, backend=backend, poll_s=0.01) == 1
    assert calls == ["等一下。"]


def test_say_is_killed_within_a_third_of_a_second_and_not_left_running(tmp_path):
    cfg = cfg_on()
    now = time.time()
    write_presence(tmp_path, file("quiet", now, now - 10))
    proc = FakeProc()
    started = threading.Event()

    def popen(_cmd):
        started.set()
        return proc

    holder = {}

    def run():
        holder["result"] = say_backend(
            popen=popen, which=lambda _name: "/usr/bin/say", sleep=time.sleep, platform="darwin",
        )("你好", "Tingting", lambda: busy(tmp_path, cfg.presence))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(1)
    time.sleep(0.02)  # inside the poll, before the file changes
    t0 = time.monotonic()
    write_presence(tmp_path, file("speaking", time.time(), time.time(), True))
    thread.join(1)
    elapsed = time.monotonic() - t0
    assert not thread.is_alive()
    assert holder["result"]["interrupted"] is True and holder["result"]["completed"] is False
    assert proc.terminated and proc.poll() is not None
    assert elapsed < 0.3

    # `say` that ignores SIGTERM is killed, not left running. The first check is
    # quiet so the process actually starts; the next one is the barge-in.
    class IgnoresTerm(FakeProc):
        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("say", timeout or 1)
            return self.returncode

    stubborn = IgnoresTerm()
    checks = {"n": 0}

    def later(_=None):
        checks["n"] += 1
        return checks["n"] >= 2

    result = say_backend(
        popen=lambda _cmd: stubborn, which=lambda _name: "/usr/bin/say",
        sleep=lambda _s: None, platform="darwin",
    )("你好", "Tingting", later)
    assert result["interrupted"] is True
    assert stubborn.terminated and stubborn.killed and stubborn.poll() is not None


def test_say_does_not_start_while_the_presence_file_says_speaking(tmp_path):
    cfg = cfg_on()
    now = time.time()
    write_presence(tmp_path, file("speaking", now, now, True))

    def popen(_cmd):
        raise AssertionError("say should not start")

    result = say_backend(
        popen=popen, which=lambda _name: "/usr/bin/say", sleep=lambda _s: None, platform="darwin",
    )("你好", "Tingting", lambda: busy(tmp_path, cfg.presence))
    assert result["interrupted"] is True and result["completed"] is False and result["backend"] == "say"


def test_an_interruption_is_reoffered_once_and_the_finished_brief_is_not_reread(tmp_path):
    data = tmp_path / "data"
    cfg = cfg_on()
    append(data, reply("a", "甲。细节"))
    assert sync(data, cfg)[1]["wrote"] == 1
    now = time.time()
    write_presence(data, file("quiet", now, now - 10))
    calls = []

    def backend(text, _voice, stop_check):
        calls.append(text)
        if len(calls) == 1:
            write_presence(data, file("speaking", time.time(), time.time(), True))
            assert stop_check() is True
            return {"text": text, "completed": False, "interrupted": True, "backend": "mock"}
        assert stop_check() is False
        return {"text": text, "completed": True, "interrupted": False, "backend": "mock"}

    assert speak_pending(data, cfg, backend=backend) == 1
    assert calls == ["甲。"]
    assert unspoken(data) == []
    # The file is still "speaking". Put it back to quiet so a mistaken re-read would be heard.
    write_presence(data, file("quiet", time.time(), time.time() - 10))
    assert speak_pending(data, cfg, backend=backend) == 0
    assert calls == ["甲。"]

    _queue, written = sync(data, cfg)
    assert written["wrote"] == 1
    new_ids = [ev["id"] for ev in replay(data, types="brief")]
    assert new_ids == ["brief-a", "brief-a-2"]
    assert sync(data, cfg)[1]["wrote"] == 0  # the re-offer is already stored

    assert speak_pending(data, cfg, backend=backend) == 1
    assert calls == ["甲。", "甲。"]
    assert speak_pending(data, cfg, backend=backend) == 0
    assert sync(data, cfg)[1]["wrote"] == 0
    spoken = list(replay(data, types="spoken"))
    assert [(ev["ref"], ev["data"]["interrupted"], ev["data"]["completed"]) for ev in spoken] == [
        ("brief-a", True, False),
        ("brief-a-2", False, True),
    ]


def test_presence_config_defaults_off_and_supervisor_does_not_spawn_it(tmp_path):
    cfg = C.Config()
    assert cfg.presence.enabled is False
    assert cfg.presence.publish_s == 0.1 and cfg.presence.onset_ms == 60
    assert cfg.presence.hangover_ms == 400 and cfg.presence.grace_s == 0.3
    example = C.load(Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.presence.enabled is False and example.presence.publish_s == 0.1
    path = tmp_path / "c.toml"
    path.write_text(
        "[presence]\nenabled = true\npublish_s = 0.05\nstale_s = 2\n"
        "onset_ms = 40\nhangover_ms = 250\ngrace_s = 0.2\n")
    loaded = C.load(path)
    assert loaded.presence.enabled is True and loaded.presence.onset_ms == 40
    assert loaded.presence.hangover_ms == 250 and loaded.presence.grace_s == 0.2
    for bad in (
        "[presence]\nenabled = \"yes\"\n",
        "[presence]\npublish_s = 0\n",
        "[presence]\npublish_s = 0.5\n",
        "[presence]\npublish_s = 0.1\nstale_s = 0.1\n",
        "[presence]\nonset_ms = 500\n",
        "[presence]\nhangover_ms = -1\n",
        "[presence]\ngrace_s = -0.1\n",
        "[presence]\nnope = 1\n",
    ):
        path.write_text(bad)
        with pytest.raises(C.ConfigError):
            C.load(path)
    specs = child_specs(None, "127.0.0.1", 8727, turns=True, route=True,
                        dispatch=True, queue=True, speak=True)
    assert "presence" not in specs
    assert set(specs) == {"run", "web", "turns", "route", "dispatch", "queue", "speak"}


def test_presence_events_round_trip_through_replay(tmp_path):
    """A presence line is an ordinary event: stored, replayed, not rejected."""
    data = tmp_path / "data"
    append(data, make_event("presence", "capture", {"state": "speaking"}, id="p1", ts=TS))
    found = list(replay(data, types="presence"))
    assert found[0]["id"] == "p1" and found[0]["data"] == {"state": "speaking"}
