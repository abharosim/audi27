"""Streaming wake-word spotter. The player and the model are injected.

Nothing here opens a microphone or a sound device.
"""
import io
import json
import logging
import os
import tarfile
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from audi27 import config as C
from audi27.cli import main
from audi27.cue import SUPPRESS_S, playback_windows
from audi27.events import KNOWN_TYPES, make_event, replay
from audi27.kws import (
    MODEL_NAME,
    KwsError,
    Tap,
    detect,
    download_model,
    find_model,
    keyword_line,
    keyword_lines,
    load_engine,
    open_tap,
    read_wav,
    split_syllable,
)

TS = "2026-11-15T12:00:00+00:00"
SR = 16000


def _player(calls, *, ok=True, boom=False):
    def play():
        calls.append(time.monotonic())
        if boom:
            raise RuntimeError("no device")
        return {"ok": ok, "backend": "mock", "sound": "tone"}

    return play


def _tap(tmp_path, player, *, debounce_ms=1500, engine=None, start=False, queue_size=8):
    cfg = C.Config()
    cfg.kws.debounce_ms = debounce_ms
    return Tap(cfg, tmp_path, engine, player, start=start, queue_size=queue_size), cfg


def _wav(path, samples, sr=SR):
    pcm = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def _archive(members: dict) -> bytes:
    """bz2 tarball. `members` maps archive names to bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _model_bytes() -> bytes:
    root = MODEL_NAME
    return _archive({
        f"{root}/tokens.txt": b"d 1\n",
        f"{root}/encoder-epoch.onnx": b"enc",
        f"{root}/decoder-epoch.onnx": b"dec",
        f"{root}/joiner-epoch.onnx": b"join",
    })


# --- tokens ------------------------------------------------------------------


def test_dada_and_its_homophones_become_ppinyin_tokens():
    assert keyword_line("达达") == "d á d á @达达"
    assert keyword_line("哒哒") == "d á d á @哒哒"
    assert keyword_line("大大") == "d á d á @大大"
    assert keyword_line("DaDa") == "d á d á @DaDa"
    assert keyword_line("  dada  ") == "d á d á @dada"
    assert keyword_line("d á d á :2 #0.4 @达达") == "d á d á :2 #0.4 @达达"


def test_ppinyin_splits_initial_and_final():
    assert split_syllable("dá") == ["d", "á"]
    assert split_syllable("nǐ") == ["n", "ǐ"]
    assert split_syllable("hǎo") == ["h", "ǎo"]
    assert split_syllable("xiǎo") == ["x", "iǎo"]
    assert split_syllable("ài") == ["ài"]
    assert split_syllable("tóng") == ["t", "óng"]
    assert split_syllable("xué") == ["x", "ué"]
    assert split_syllable("zhōng") == ["zh", "ōng"]
    assert split_syllable("chī") == ["ch", "ī"]
    assert split_syllable("shì") == ["sh", "ì"]


def test_pypinyin_matches_the_sherpa_ppinyin_lines():
    pytest.importorskip("pypinyin")
    assert keyword_line("小爱同学") == "x iǎo ài t óng x ué @小爱同学"
    assert keyword_line("你好军哥") == "n ǐ h ǎo j ūn g ē @你好军哥"
    assert keyword_line("女儿") == "n ǚ ér @女儿"


def test_identical_pronunciations_keep_the_first_spelling():
    assert keyword_lines(["达达", "哒哒", "dada", "DaDa"]) == ["d á d á @达达"]


def test_unknown_keyword_is_skipped(caplog, monkeypatch):
    monkeypatch.setattr("audi27.kws._to_syllables", lambda _text: None)
    caplog.set_level(logging.WARNING)
    assert keyword_lines(["达达", "未知词"]) == ["d á d á @达达"]
    assert "未知词" in caplog.text


# --- hit, debounce, events ---------------------------------------------------


def test_a_hit_logs_latency_keyword_and_play_result(tmp_path, caplog):
    def play():
        return {"ok": True, "backend": "sounddevice", "sound": "tone", "device": "Speakers", "error": ""}

    tap, _cfg = _tap(tmp_path, play)
    caplog.set_level(logging.INFO)
    captured = time.time() - 0.25
    assert tap.handle("达达", captured) is True
    wake = list(replay(tmp_path, types="wake"))[0]["data"]
    assert wake["keyword"] == "达达"
    assert wake["latency_s"] == pytest.approx(0.25, abs=0.15)
    assert wake["play_ok"] is True
    assert wake["play_device"] == "Speakers"
    assert wake["play_error"] == ""
    assert "达达" in caplog.text and "Speakers" in caplog.text and "ok" in caplog.text


def test_a_failed_play_records_the_error_on_the_wake(tmp_path, caplog):
    def play():
        return {
            "ok": False, "backend": "log", "sound": "tone", "device": "",
            "error": "output stream did not start",
        }

    tap, _cfg = _tap(tmp_path, play)
    caplog.set_level(logging.INFO)
    assert tap.handle("达达", time.time()) is True
    wake = list(replay(tmp_path, types="wake"))[0]["data"]
    assert wake["play_ok"] is False
    assert wake["play_error"] == "output stream did not start"
    assert "output stream did not start" in caplog.text


def test_heartbeat_logs_counters_and_refreshes_the_live_file(tmp_path, caplog):
    class Player:
        def play(self):
            return {"ok": True, "backend": "mock", "sound": "tone", "device": "Speakers"}

        def status(self):
            return {"backend": "mock", "device": "Speakers", "stream": "open"}

        def close(self):
            return None

    tap, cfg = _tap(tmp_path, Player())
    cfg.kws.heartbeat_s = 60
    tap.frames = 40
    tap.dropped = 3
    tap.decode_s = 0.4
    tap.audio_s = 20.0
    tap.hits = 2
    tap.lag_s = 0.08
    tap._heartbeat_at = 1_000.0
    caplog.set_level(logging.INFO)
    tap._maybe_heartbeat(now=1_030.0)
    assert "heartbeat" not in caplog.text
    tap._maybe_heartbeat(now=1_060.0)
    assert "frames=40" in caplog.text
    assert "dropped=3" in caplog.text
    assert "hits=2" in caplog.text
    assert "open" in caplog.text
    live = json.loads((Path(tmp_path) / "kws.live").read_text(encoding="utf-8"))
    assert live["frames"] == 40 and live["dropped"] == 3 and live["hits"] == 2
    assert live["rtf"] == pytest.approx(0.02)
    assert live["lag_s"] == pytest.approx(0.08)
    assert live["stream"] == "open" and live["output_device"] == "Speakers"
    assert "pid" in live and "updated" in live and "started" in live


def test_the_worker_counts_frames_without_logging_each_one(tmp_path, caplog):
    class Engine:
        def accept(self, _samples, _rate):
            return None

    caplog.set_level(logging.INFO)
    tap, cfg = _tap(tmp_path, _player([]), engine=Engine(), start=True)
    cfg.kws.heartbeat_s = 3600
    for i in range(4):
        tap.offer(np.zeros(480, np.int16), 1_700_000_000.0 + i * 0.03)
    deadline = time.time() + 2
    while time.time() < deadline and tap.frames < 4:
        time.sleep(0.02)
    tap.close()
    assert tap.frames == 4
    assert tap.dropped == 0
    assert [rec for rec in caplog.records if rec.name == "audi27.kws" and "heartbeat" in rec.message] == []


def test_a_hit_plays_and_writes_wake_and_cue(tmp_path):
    calls = []
    tap, _cfg = _tap(tmp_path, _player(calls))
    audio_ts = 1_700_000_000.0
    assert tap.handle("达达", audio_ts) is True
    wakes = list(replay(tmp_path, types="wake"))
    cues = list(replay(tmp_path, types="cue_played"))
    assert len(calls) == 1
    assert len(wakes) == 1 and len(cues) == 1
    wake, cue = wakes[0], cues[0]
    assert wake["type"] == "wake" and wake["source"] == "kws"
    assert wake["data"]["keyword"] == "达达"
    assert wake["data"]["audio_ts"] == audio_ts
    assert wake["id"] == f"wake-{int(round(audio_ts * 1000))}"
    got = datetime.fromisoformat(wake["ts"])
    expect = datetime.fromtimestamp(audio_ts).astimezone()
    assert abs((got - expect).total_seconds()) < 0.001
    assert cue["ref"] == wake["id"]
    assert cue["id"] == "cue-" + wake["id"]
    assert cue["data"]["ok"] is True and cue["data"]["backend"] == "mock"
    assert cue["data"]["trigger"] == "达达"
    delta = datetime.fromisoformat(cue["data"]["until"]) - datetime.fromisoformat(cue["ts"])
    assert abs(delta.total_seconds() - SUPPRESS_S) < 0.05
    assert "wake" in KNOWN_TYPES
    windows = playback_windows(cues)
    assert len(windows) == 1
    assert abs((windows[0][1] - windows[0][0]).total_seconds() - SUPPRESS_S) < 0.05


def test_debounce_suppresses_a_second_hit_and_a_later_one_fires(tmp_path):
    calls = []
    tap, _cfg = _tap(tmp_path, _player(calls), debounce_ms=1500)
    assert tap.handle("达达", 1000.0) is True
    assert tap.handle("达达", 1000.5) is False
    assert tap.handle("达达", 1002.0) is True
    assert len(calls) == 2
    assert [ev["data"]["audio_ts"] for ev in replay(tmp_path, types="wake")] == [1000.0, 1002.0]


def test_a_player_failure_still_records_the_hit_and_does_not_retry_inside_the_window(tmp_path, caplog):
    calls = []
    caplog.set_level(logging.ERROR)
    tap, _cfg = _tap(tmp_path, _player(calls, boom=True))
    assert tap.handle("达达", 100.0) is True
    assert tap.handle("达达", 100.1) is False
    assert len(calls) == 1
    rows = list(replay(tmp_path, types="cue_played"))
    assert len(rows) == 1 and rows[0]["data"]["ok"] is False
    assert "no device" in caplog.text


def test_the_kws_cue_window_hides_a_bare_yes_and_a_later_yes_counts(tmp_path):
    from audi27.confirm import plan

    calls = []
    tap, _cfg = _tap(tmp_path, _player(calls))
    tap.handle("达达", 1_700_000_000.0)
    cue = list(replay(tmp_path, types="cue_played"))[0]
    begin = datetime.fromisoformat(cue["ts"])
    brief_ts = (begin - timedelta(seconds=1)).isoformat()
    task = make_event("task", "dispatch", {
        "route_id": "route-pay", "target": "grok", "adapter": None,
        "status": "awaiting_confirmation", "text": "帮我发邮件给张三",
    }, ts=(begin - timedelta(seconds=2)).isoformat(), ref="route-pay", id="task-pay")
    brief = make_event("brief", "queue", {
        "priority": "decision", "summary": "发给 Grok：帮我发邮件给张三。确认还是取消？",
        "text_ref": task["id"], "kind": "confirm", "item_id": task["id"], "task_id": task["id"],
    }, ts=brief_ts, ref=task["id"], id=f"brief-{task['id']}")

    def said(ident, text, when):
        return make_event("utterance", "transcribe", {
            "id": ident, "text": text, "status": "ok", "start": when, "end": when,
        }, ts=when, id=ident)

    during = said("u-echo", "确认", (begin + timedelta(milliseconds=200)).isoformat())
    after = said("u-yes", "好的", (begin + timedelta(seconds=3)).isoformat())
    assert plan([task, brief, cue, during]) == []
    decided = plan([task, brief, cue, after])
    assert [ev["data"]["decision"] for ev in decided] == ["approve"]
    missed = dict(cue)
    missed["data"] = dict(cue["data"])
    missed["data"]["ok"] = False
    heard = plan([task, brief, missed, during])
    assert [ev["data"]["decision"] for ev in heard] == ["approve"]


def test_offer_drops_the_oldest_frame_and_returns_immediately(tmp_path):
    calls = []
    tap, _cfg = _tap(tmp_path, _player(calls), start=False, queue_size=1)
    tap.offer(np.zeros(480, np.int16), 0.0)
    started = time.monotonic()
    tap.offer(np.full(480, 3, np.int16), 1.0)
    assert time.monotonic() - started < 0.05
    assert tap.dropped == 1
    kept_frame, kept_ts = tap.q.get_nowait()
    assert kept_ts == 1.0 and int(kept_frame[0]) == 3
    assert calls == []


def test_the_worker_plays_once_for_a_burst(tmp_path):
    calls = []

    class Always:
        def accept(self, samples, sample_rate):
            return "达达"

    tap, _cfg = _tap(tmp_path, _player(calls), engine=Always(), start=True, queue_size=8)
    for i in range(5):
        tap.offer(np.zeros(480, np.int16), 1_700_000_000.0 + i * 0.03)
    tap.close()
    assert len(calls) == 1
    assert len(list(replay(tmp_path, types="wake"))) == 1
    assert len(list(replay(tmp_path, types="cue_played"))) == 1


def test_a_running_spotter_publishes_kws_live_and_clears_it_on_close(tmp_path):
    from audi27.kws import spotter_live

    calls = []
    tap, _cfg = _tap(tmp_path, _player(calls), engine=None, start=True)
    deadline = time.time() + 2
    while time.time() < deadline and not spotter_live(tmp_path):
        time.sleep(0.02)
    assert spotter_live(tmp_path) is True
    tap.close()
    assert not (Path(tmp_path) / "kws.live").exists()
    assert spotter_live(tmp_path) is False


def test_the_hit_does_not_wait_for_the_event_write(tmp_path, monkeypatch):
    import audi27.kws as kws_mod

    real = kws_mod.append

    def slow(data_dir, event, if_absent=False):
        time.sleep(0.2)
        return real(data_dir, event, if_absent=if_absent)

    monkeypatch.setattr(kws_mod, "append", slow)
    calls = []
    tap, _cfg = _tap(tmp_path, _player(calls), engine=None, start=True)
    started = time.monotonic()
    assert tap.handle("达达", 1_700_000_000.0) is True
    assert time.monotonic() - started < 0.1
    tap.close()
    assert [ev["data"]["keyword"] for ev in replay(tmp_path, types="wake")] == ["达达"]
    assert len(calls) == 1


def test_player_reopens_when_the_stream_stops_or_the_device_changes(caplog):
    from audi27.kws import SoundDevicePlayer, tone

    caplog.set_level(logging.INFO)

    class Stream:
        def __init__(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

    class Backend:
        def __init__(self):
            self.device = "speakers"
            self.opens = 0
            self.fail = False
            self.streams = []

        def output_device(self):
            return self.device

        def open_stream(self, _callback, _rate):
            self.opens += 1
            if self.fail:
                raise RuntimeError("device gone")
            stream = Stream()
            self.streams.append(stream)
            return stream

    backend = Backend()
    player = SoundDevicePlayer(tone(880, 80), 16000, "tone", backend=backend)
    assert backend.opens == 1
    assert "output stream open" in caplog.text and "speakers" in caplog.text
    assert player.status()["stream"] == "open" and player.status()["device"] == "speakers"
    assert player.play()["ok"] is True and player.play()["backend"] == "sounddevice"
    assert player.play()["device"] == "speakers"
    assert backend.opens == 1
    backend.streams[0].active = False
    assert player.play()["ok"] is True
    assert backend.opens == 2
    assert "stream stopped" in caplog.text
    backend.device = "headphones"
    assert player.play()["ok"] is True
    assert backend.opens == 3
    assert "headphones" in caplog.text
    backend.fail = True
    backend.streams[-1].active = False
    missed = player.play()
    assert missed["ok"] is False and missed["backend"] == "log"
    assert missed["error"]
    assert any(rec.levelno >= logging.ERROR and "did not start" in rec.message for rec in caplog.records)
    backend.fail = False
    assert player.play()["ok"] is True
    assert backend.opens == 5
    player.close()


# --- model -------------------------------------------------------------------


def test_find_model_keeps_a_single_checkpoint(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    for kind in ("encoder", "decoder", "joiner"):
        (root / f"{kind}-epoch-99-avg-1-chunk-16-left-64.int8.onnx").write_bytes(b"a")
        (root / f"{kind}-epoch-12-avg-2-chunk-16-left-64.onnx").write_bytes(b"b")
        (root / f"{kind}-epoch-12-avg-2-chunk-16-left-64.int8.onnx").write_bytes(b"c")
    found = find_model(root)
    assert found is not None
    for kind in ("encoder", "decoder", "joiner"):
        assert found[kind].name == f"{kind}-epoch-12-avg-2-chunk-16-left-64.int8.onnx"


def test_find_model_prefers_int8_and_rejects_a_partial_directory(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    assert find_model(root) is None
    (root / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    (root / "encoder.onnx").write_bytes(b"fp32")
    (root / "encoder.int8.onnx").write_bytes(b"int8")
    (root / "decoder.onnx").write_bytes(b"dec")
    (root / "joiner.onnx").write_bytes(b"join")
    found = find_model(root)
    assert found is not None
    assert found["encoder"].name == "encoder.int8.onnx"
    assert found["decoder"].name == "decoder.onnx"
    assert found["tokens"].name == "tokens.txt"


def test_download_extracts_and_does_not_fetch_twice(tmp_path):
    blob = _model_bytes()
    calls = {"n": 0}

    def fetch(_url):
        calls["n"] += 1
        return blob

    model_dir = tmp_path / "models" / MODEL_NAME
    dest = download_model(model_dir, fetch=fetch)
    assert dest == model_dir
    assert find_model(model_dir)["joiner"].name.startswith("joiner")
    assert download_model(model_dir, fetch=fetch) == model_dir
    assert calls["n"] == 1


def test_download_rejects_an_archive_path_that_escapes(tmp_path):
    blob = _archive({"../evil.txt": b"x"})
    with pytest.raises(KwsError):
        download_model(tmp_path / "models" / MODEL_NAME, fetch=lambda _url: blob)
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "models" / "evil.txt").exists()


def test_missing_model_disables_the_spotter_and_logs_once(tmp_path, caplog):
    cfg = C.Config()
    cfg.kws.enabled = True
    cfg.kws.model_dir = str(tmp_path / "no-model")
    caplog.set_level(logging.DEBUG)
    assert open_tap(cfg, tmp_path) is None
    assert open_tap(cfg, tmp_path) is None
    errors = [rec for rec in caplog.records if rec.name == "audi27.kws" and rec.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "disabled" in errors[0].message and "no-model" in errors[0].message
    assert "audi27 kws download" in errors[0].message
    rows = list(replay(tmp_path, types="kws_status"))
    assert len(rows) == 1
    data = rows[0]["data"]
    assert data["enabled"] is True and data["live"] is False and data["state"] == "disabled"
    assert data["fix"] == "audi27 kws download"
    assert "no-model" in data["reason"]
    assert "kws_status" in KNOWN_TYPES


def test_missing_sherpa_logs_the_install_command(tmp_path, caplog, monkeypatch):
    cfg = C.Config()
    cfg.kws.enabled = True
    cfg.kws.model_dir = str(tmp_path / "model")
    monkeypatch.setattr(
        "audi27.kws.find_model",
        lambda _d: {"tokens": tmp_path, "encoder": tmp_path, "decoder": tmp_path, "joiner": tmp_path})

    def boom(_cfg, _files, module=None):
        raise KwsError("sherpa-onnx is not installed (uv sync --extra kws)")

    monkeypatch.setattr("audi27.kws.load_engine", boom)
    caplog.set_level(logging.ERROR)
    assert open_tap(cfg, tmp_path) is None
    assert any(
        rec.levelno >= logging.ERROR and "uv sync --extra kws" in rec.message
        for rec in caplog.records if rec.name == "audi27.kws")
    data = list(replay(tmp_path, types="kws_status"))[0]["data"]
    assert data["enabled"] is True and data["live"] is False
    assert data["fix"] == "uv sync --extra kws"
    assert "sherpa-onnx is not installed" in data["reason"]


def test_open_tap_records_a_live_status(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr("audi27.kws.input_device_name", lambda _cfg: "Test Mic")
    cfg = C.Config()
    cfg.kws.enabled = True

    class Engine:
        def accept(self, _samples, _rate):
            return None

    class Player:
        def play(self):
            return {"ok": True, "backend": "sounddevice", "sound": "tone", "device": "Speakers"}

        def status(self):
            return {"backend": "sounddevice", "device": "Speakers", "stream": "open"}

        def close(self):
            return None

    caplog.set_level(logging.INFO)
    tap = open_tap(cfg, tmp_path, engine=Engine(), player=Player())
    assert tap is not None
    tap.close()
    infos = [rec.message for rec in caplog.records if rec.name == "audi27.kws" and rec.levelno == logging.INFO]
    assert any("live" in msg and "Test Mic" in msg and "16000" in msg and "d á d á" in msg for msg in infos)
    data = list(replay(tmp_path, types="kws_status"))[0]["data"]
    assert data["live"] is True and data["state"] == "live" and data["enabled"] is True
    assert data["input_device"] == "Test Mic"
    assert data["sample_rate"] == 16000
    assert data["output_device"] == "Speakers"
    assert data["player"] == "sounddevice"
    assert data["threshold"] == 0.25 and data["score"] == 1.5
    assert data["model_dir"] == cfg.kws.model_dir
    assert any(line.startswith("d á d á") for line in data["keywords"])
    assert data["fix"] == "" and data["reason"] == ""


def test_disabled_flag_does_not_log(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    assert open_tap(C.Config(), tmp_path) is None
    assert [rec for rec in caplog.records if rec.name == "audi27.kws"] == []


def test_load_engine_passes_score_threshold_and_keyword_file(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "tokens.txt").write_text("d 1\n", encoding="utf-8")
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (root / name).write_bytes(b"x")
    files = find_model(root)
    seen = {}

    class Mod:
        def KeywordSpotter(self, **kwargs):
            seen.update(kwargs)

            class Spotter:
                def create_stream(self, keywords=None):
                    return object()

            return Spotter()

    cfg = C.Config()
    cfg.kws.score = 1.5
    cfg.kws.threshold = 0.25
    cfg.kws.keywords = ["达达"]
    engine = load_engine(cfg, files, module=Mod())
    assert engine is not None
    assert seen["keywords_score"] == 1.5
    assert seen["keywords_threshold"] == 0.25
    assert seen["num_threads"] == 1
    text = Path(seen["keywords_file"]).read_text(encoding="utf-8")
    assert text.strip() == "d á d á @达达"
    assert str(files["tokens"]) == seen["tokens"]


# --- offline latency ---------------------------------------------------------


def test_detect_reports_latency_from_the_keyword_end():
    sr = SR
    samples = np.zeros(sr, np.float32)
    samples[int(0.2 * sr):int(0.5 * sr)] = 0.1

    class Engine:
        def accept(self, chunk, sample_rate):
            self.seen = getattr(self, "seen", 0) + len(chunk)
            if self.seen >= int(0.8 * sr) and not getattr(self, "done", False):
                self.done = True
                return "达达"
            return None

    report = detect(samples, sr, Engine(), hop_ms=30, keyword_end=0.5)
    assert report["keyword"] == "达达"
    assert report["keyword_end_s"] == 0.5
    assert report["keyword_end_source"] == "argument"
    assert report["detected_s"] == pytest.approx(0.81, abs=0.001)
    assert report["latency_s"] == pytest.approx(report["detected_s"] - 0.5, abs=1e-6)


def test_detect_estimates_keyword_end_from_energy_and_counts_tail_padding():
    sr = SR
    samples = np.zeros(sr, np.float32)
    samples[int(0.2 * sr):int(0.5 * sr)] = 0.1

    class Engine:
        def accept(self, chunk, sample_rate):
            self.seen = getattr(self, "seen", 0) + len(chunk)
            if self.seen >= int(0.8 * sr) and not getattr(self, "done", False):
                self.done = True
                return "达达"
            return None

    report = detect(samples, sr, Engine(), hop_ms=30)
    assert report["keyword_end_source"] == "energy"
    assert report["keyword_end_s"] == pytest.approx(0.5, abs=0.02)

    class Late:
        def accept(self, chunk, sample_rate):
            self.seen = getattr(self, "seen", 0) + len(chunk)
            if self.seen >= sr + int(0.2 * sr) and not getattr(self, "done", False):
                self.done = True
                return "达达"
            return None

    late = detect(np.zeros(sr, np.float32), sr, Late(), hop_ms=30, keyword_end=0.0, tail_s=0.66)
    assert late["detected_s"] == pytest.approx(1.2, abs=0.03)
    assert late["latency_s"] == pytest.approx(late["detected_s"], abs=1e-6)


def test_read_wav_roundtrip(tmp_path):
    pcm = np.array([0, 1000, -1000, 32767], np.int16)
    path = tmp_path / "a.wav"
    _wav(path, pcm)
    got, sr = read_wav(path)
    assert sr == SR
    assert np.allclose(got * 32768.0, pcm.astype(np.float32), atol=1.0)


# --- text cue ----------------------------------------------------------------


def _utterance(ident, text, start, end=None):
    end = start if end is None else end
    return make_event("utterance", "transcribe", {
        "id": ident, "text": text, "status": "ok", "start": start, "end": end,
    }, ts=start, id=ident)


def _mark_live(data, *, updated, pid=None):
    root = Path(data)
    root.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid() if pid is None else pid, "started": updated, "updated": updated}
    (root / "kws.live").write_text(json.dumps(payload), encoding="utf-8")


def test_text_cue_plays_when_kws_is_enabled_but_the_spotter_is_not_live(tmp_path):
    from audi27.cue import ack

    data = tmp_path / "data"
    cfg = C.Config()
    cfg.kws.enabled = True
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    assert ack(data, cfg, _utterance("u-wake", "达达，今天天气", TS), player=player, pending={}) is True
    assert calls == [cfg.cue.sound]
    assert [ev["id"] for ev in replay(data, types="cue_played")] == ["cue-u-wake"]
    decision = list(replay(data, types="cue_decision"))
    assert [row["data"]["reason"] for row in decision] == ["played"]


def test_a_live_spotter_skips_the_text_cue_only_when_a_wake_covers_the_utterance(tmp_path):
    from audi27.cue import ack
    from audi27.kws import wake_covers
    from audi27.store import _dt

    data = tmp_path / "data"
    cfg = C.Config()
    cfg.kws.enabled = True
    cfg.turns.triggers = ["达达", "搭搭"]
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    start = datetime.fromisoformat(TS)
    _mark_live(data, updated=time.time())
    wake_at = start.timestamp()
    from audi27.events import append
    append(data, make_event("wake", "kws", {"keyword": "达达", "audio_ts": wake_at},
                            ts=_dt(wake_at), id="wake-covered"))
    covered = _utterance("u-dada", "达达，今天天气", TS, "2026-11-15T12:00:02+00:00")
    assert wake_covers(data, covered) is True
    assert ack(data, cfg, covered, player=player, pending={}) is True
    # 搭搭 is a turns trigger the default spotter list does not contain.
    missed = _utterance("u-da", "搭搭，记一下", "2026-11-15T12:00:20+00:00", "2026-11-15T12:00:22+00:00")
    assert wake_covers(data, missed) is False
    assert ack(data, cfg, missed, player=player, pending={}) is True
    assert calls == [cfg.cue.sound]
    assert [ev["ref"] for ev in replay(data, types="cue_played") if ev["source"] == "cue"] == ["u-da"]
    reasons = [ev["data"]["reason"] for ev in replay(data, types="cue_decision")]
    assert reasons == ["spotter live and a wake covered it", "played"]


def test_a_stale_or_dead_spotter_does_not_silence_the_text_cue(tmp_path):
    from audi27.cue import ack
    from audi27.kws import spotter_live
    from audi27.store import _dt

    data = tmp_path / "data"
    cfg = C.Config()
    calls = []

    def player(sound):
        calls.append(sound)
        return {"ok": True, "backend": "mock", "sound": sound}

    said = _utterance("u-wake", "达达，今天天气", TS)
    wake_at = datetime.fromisoformat(TS).timestamp()
    from audi27.events import append
    append(data, make_event("wake", "kws", {"keyword": "达达", "audio_ts": wake_at},
                            ts=_dt(wake_at), id="wake-old"))
    _mark_live(data, updated=time.time() - 30)
    assert spotter_live(data) is False
    assert ack(data, cfg, said, player=player, pending={}) is True
    _mark_live(data, updated=time.time(), pid=2**30)
    assert spotter_live(data) is False
    said2 = _utterance("u-wake-2", "达达，再来", "2026-11-15T12:00:05+00:00")
    assert ack(data, cfg, said2, player=player, pending={}) is True
    assert calls == [cfg.cue.sound, cfg.cue.sound]


# --- config / cli ------------------------------------------------------------


def test_config_defaults_and_example_document_kws():
    cfg = C.Config()
    from audi27.turns import DEFAULT_TRIGGERS
    assert cfg.kws.enabled is False
    assert cfg.kws.keywords == list(DEFAULT_TRIGGERS)
    assert cfg.kws.threshold == 0.25
    assert cfg.kws.score == 1.5
    assert cfg.kws.debounce_ms == 1500
    assert cfg.kws.heartbeat_s == 60
    assert cfg.kws.tone_hz == 880
    assert cfg.kws.tone_ms == 80
    assert cfg.kws.sound == ""
    assert cfg.kws.model_dir == "data/models/" + MODEL_NAME
    root = Path(__file__).resolve().parents[1]
    example = C.load(root / "config.example.toml")
    assert example.kws.enabled is False
    assert example.kws.keywords == list(DEFAULT_TRIGGERS)
    text = (root / "config.example.toml").read_text(encoding="utf-8")
    body = text.split("[kws]", 1)[1].split("\n[", 1)[0]
    assert "enabled = false" in body
    assert "keywords" in body and "threshold" in body and "debounce_ms" in body
    assert "heartbeat_s" in body
    assert example.kws.heartbeat_s == 60


def test_config_rejects_a_bad_kws_section(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[kws]\nenabled = true\nkeywords = ["达达"]\nthreshold = 0.2\nscore = 0\nheartbeat_s = 30\n', encoding="utf-8")
    loaded = C.load(path)
    assert loaded.kws.enabled is True and loaded.kws.threshold == 0.2 and loaded.kws.score == 0
    assert loaded.kws.heartbeat_s == 30
    for bad in (
        "[kws]\nenabled = \"yes\"\n",
        "[kws]\nkeywords = []\n",
        "[kws]\nthreshold = 0\n",
        "[kws]\nthreshold = 2\n",
        "[kws]\nscore = -1\n",
        "[kws]\ndebounce_ms = -5\n",
        "[kws]\nheartbeat_s = 0\n",
        "[kws]\nnope = 1\n",
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(C.ConfigError):
            C.load(path)


def test_kws_download_cli_prints_the_destination(tmp_path, capsys, monkeypatch):
    target = tmp_path / "model"

    def fake(dest, **_kwargs):
        return target

    monkeypatch.setattr("audi27.kws.download_model", fake)
    assert main(["kws", "download", "--dir", str(tmp_path / "in")]) == 0
    assert str(target) in capsys.readouterr().out


def test_kws_test_cli_prints_latency_and_json(tmp_path, capsys, monkeypatch):
    wav = tmp_path / "a.wav"
    _wav(wav, np.zeros(160, np.int16))

    def fake(_path, _cfg, keyword_end=None):
        return {
            "keyword": "达达", "detected_s": 0.84, "keyword_end_s": 0.70,
            "keyword_end_source": "argument" if keyword_end is not None else "energy",
            "latency_s": 0.14, "sample_rate": SR,
        }

    monkeypatch.setattr("audi27.kws.report_file", fake)
    assert main(["kws", "test", str(wav), "--keyword-end", "0.7"]) == 0
    out = capsys.readouterr().out
    assert "keyword: 达达" in out
    assert "latency_s: 0.140" in out
    assert main(["kws", "test", str(wav), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["keyword"] == "达达" and payload["latency_s"] == 0.14


def test_kws_test_cli_errors_without_a_file_or_a_model(tmp_path, capsys):
    assert main(["kws", "test", str(tmp_path / "missing.wav")]) == 2
    wav = tmp_path / "a.wav"
    _wav(wav, np.zeros(160, np.int16))
    assert main(["kws", "test", str(wav)]) == 2
    err = capsys.readouterr().err
    assert "model" in err


def test_kws_test_cli_returns_1_when_nothing_fires(tmp_path, capsys, monkeypatch):
    wav = tmp_path / "a.wav"
    _wav(wav, np.zeros(160, np.int16))

    def fake(_path, _cfg, keyword_end=None):
        return {
            "keyword": "", "detected_s": None, "keyword_end_s": 0.0,
            "keyword_end_source": "energy", "latency_s": None, "sample_rate": SR,
        }

    monkeypatch.setattr("audi27.kws.report_file", fake)
    assert main(["kws", "test", str(wav)]) == 1
    assert "no keyword detected" in capsys.readouterr().out


# --- recorder wiring ---------------------------------------------------------


def test_missing_model_leaves_transcription_running(tmp_path, monkeypatch, caplog):
    import audi27.capture as capture
    from audi27.pipeline import Pipeline
    from audi27.store import Store, search
    from audi27.transcribers import Piece, Transcriber, Transcript

    class FakeT(Transcriber):
        name = "fake"
        model = "m"

        def transcribe(self, wav, language=None, prompt=None):
            return Transcript([Piece(" hello from kws", 0, 2, -0.2, 0.01, 1.2)], "en", 0.99, "fake", "m")

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
    caplog.set_level(logging.WARNING)
    cfg = C.Config()
    cfg.correct.enabled = False
    cfg.kws.enabled = True
    cfg.kws.model_dir = str(tmp_path / "absent-model")
    store = Store(tmp_path / "rec", "none")
    Pipeline(cfg, [FakeT()], store).run()
    assert search(store.root, ["hello"])
    assert list(replay(store.root, types="wake")) == []
    assert any(rec.levelno >= logging.ERROR and "disabled" in rec.message and "absent-model" in rec.message
               and "audi27 kws download" in rec.message
               for rec in caplog.records if rec.name == "audi27.kws")
    status = list(replay(store.root, types="kws_status"))
    assert len(status) == 1 and status[0]["data"]["fix"] == "audi27 kws download"


def test_a_disabled_spotter_still_writes_a_startup_status(tmp_path, monkeypatch, caplog):
    import audi27.capture as capture
    from audi27.pipeline import Pipeline
    from audi27.store import Store
    from audi27.transcribers import Transcriber

    class FakeT(Transcriber):
        name = "fake"
        model = "m"

        def transcribe(self, wav, language=None, prompt=None):
            raise AssertionError("no speech in this clip")

    class FakeCapture:
        dropped_frames = overflows = 0

        def __init__(self, _cfg):
            pass

        def frames(self, _stop):
            yield np.zeros(480, np.int16), 1_700_000_000.0

    monkeypatch.setattr(capture, "Capture", FakeCapture)
    caplog.set_level(logging.INFO)
    cfg = C.Config()
    cfg.correct.enabled = False
    cfg.kws.enabled = False
    store = Store(tmp_path / "rec", "none")
    Pipeline(cfg, [FakeT()], store).run()
    rows = list(replay(store.root, types="kws_status"))
    assert len(rows) == 1
    assert rows[0]["data"]["enabled"] is False
    assert rows[0]["data"]["live"] is False
    assert rows[0]["data"]["state"] == "disabled"
    assert "enabled = false" in rows[0]["data"]["reason"]
    assert any(rec.levelno == logging.INFO and "enabled = false" in rec.message
               for rec in caplog.records if rec.name == "audi27.kws")
    assert not any(rec.levelno >= logging.ERROR and rec.name == "audi27.kws" for rec in caplog.records)


def test_spotter_does_not_block_the_capture_loop(tmp_path, monkeypatch):
    import audi27.capture as capture
    from audi27.kws import open_tap as real_open
    from audi27.pipeline import Pipeline
    from audi27.store import Store
    from audi27.transcribers import Transcriber

    class FakeT(Transcriber):
        name = "fake"
        model = "m"

        def transcribe(self, wav, language=None, prompt=None):
            raise AssertionError("these frames are too short to transcribe")

    holder = {}

    class FakeCapture:
        dropped_frames = overflows = 0

        def __init__(self, _cfg):
            holder["cap"] = self
            self.yield_s = None

        def frames(self, _stop):
            started = time.monotonic()
            for i in range(10):
                yield np.zeros(480, np.int16), 1_700_000_000.0 + i * 0.03
            self.yield_s = time.monotonic() - started

    class Slow:
        def __init__(self):
            self.n = 0

        def accept(self, samples, sample_rate):
            time.sleep(0.05)
            self.n += 1
            return "达达" if self.n == 1 else None

    calls = []

    def player():
        calls.append(1)
        return {"ok": True, "backend": "mock", "sound": "tone"}

    def fake_open(cfg, data_dir, **_kwargs):
        tap = Tap(cfg, data_dir, Slow(), player, start=True, queue_size=32)
        return tap

    monkeypatch.setattr(capture, "Capture", FakeCapture)
    monkeypatch.setattr("audi27.kws.open_tap", fake_open)
    assert real_open is not None
    cfg = C.Config()
    cfg.correct.enabled = False
    cfg.kws.enabled = True
    store = Store(tmp_path / "rec", "none")
    Pipeline(cfg, [FakeT()], store).run()
    assert holder["cap"].yield_s < 0.3
    assert calls == [1]
    wakes = list(replay(store.root, types="wake"))
    assert len(wakes) == 1 and wakes[0]["data"]["keyword"] == "达达"
    assert len(list(replay(store.root, types="cue_played"))) == 1
