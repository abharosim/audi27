"""`audi27 doctor`: one screen for "I said 达达 and heard no ding"."""
import json
import os
import time

from audi27 import config as C
from audi27.cli import main
from audi27.events import append, make_event


def test_doctor_summarizes_config_live_stats_events_and_warnings(tmp_path):
    from audi27.doctor import render

    cfg = C.Config()
    cfg.kws.enabled = True
    cfg.cue.enabled = False
    data = tmp_path / "data"
    data.mkdir()
    now = time.time()
    (data / "kws.live").write_text(json.dumps({
        "pid": os.getpid(), "started": now, "updated": now,
        "frames": 12, "dropped": 1, "rtf": 0.02, "lag_s": 0.05,
        "hits": 3, "stream": "open", "output_device": "Speakers", "player": "sounddevice",
    }), encoding="utf-8")
    append(data, make_event(
        "wake", "kws",
        {"keyword": "达达", "latency_s": 0.18, "play_ok": True, "play_device": "Speakers", "play_error": ""},
        id="wake-1"))
    append(data, make_event(
        "cue_decision", "cue",
        {"action": "skipped", "reason": "trigger not at start so no turn", "trigger": "达达", "status": "ok"},
        ref="u-mid", id="cuedec-u-mid"))
    append(data, make_event(
        "cue_played", "cue",
        {"ok": True, "backend": "afplay", "trigger": "达达", "sound": "/tmp/tink.aiff"},
        ref="u-ok", id="cue-u-ok"))
    (data / "audi27.log").write_text(
        "2026-11-17 10:00:00,000 INFO audi27.kws: kws live\n"
        "2026-11-17 10:00:01,000 ERROR audi27.kws: kws disabled: sherpa-onnx is not installed. Fix: uv sync --extra kws\n"
        "2026-11-17 10:00:02,000 WARNING audi27.cue: cue: sound file missing: /tmp/nope.aiff\n"
        "2026-11-17 10:00:03,000 INFO audi27.cue: cue decision u-ok played trigger=达达 reason=played\n",
        encoding="utf-8")
    text = render(
        cfg, data, now=now, sherpa=(False, "ModuleNotFoundError: sherpa_onnx"), input_device="Test Mic")
    assert "kws enabled: true" in text
    assert "cue enabled: false" in text
    assert "sherpa-onnx: not importable" in text
    assert "uv sync --extra kws" in text
    assert "audi27 kws download" in text
    assert "frames=12" in text and "dropped=1" in text and "hits=3" in text
    assert "fresh" in text and "Speakers" in text
    assert "latency_s=0.18" in text
    assert "trigger not at start so no turn" in text
    assert "sound file missing" in text
    assert "sherpa-onnx is not installed" in text
    assert "cue decision u-ok played" not in text


def test_doctor_cli_prints_a_summary(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("audi27.doctor.sherpa_status", lambda: (True, "1.10.0"))
    monkeypatch.setattr("audi27.doctor.input_device_name", lambda _cfg: "default")
    data = tmp_path / "data"
    data.mkdir()
    assert main(["doctor", "--data-dir", str(data)]) == 0
    out = capsys.readouterr().out
    assert "kws enabled: false" in out
    assert "cue enabled: false" in out
    assert "sherpa-onnx: importable" in out
    assert "kws.live: absent" in out
    assert "heartbeat_s: 60" in out
    assert "(none)" in out
