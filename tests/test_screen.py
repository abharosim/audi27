"""Screen context: front app and window title, and nothing else."""
import json
import logging
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.cli import main
from audi27.events import emit, emit_utterance, make_event, replay
from audi27.route import replay_routes
from audi27.screen import (
    _APP_SCRIPT,
    _TITLE_SCRIPT,
    Debouncer,
    FrontSampler,
    context_at,
    make_sampler,
    prepare,
    run,
)
from audi27.supervisor import child_specs
from audi27.turns import assemble_replay, event_for_turn, serve, turn_event, write_turns

BASE = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def _iso(seconds: float) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def _cfg(**kw):
    cfg = C.Config()
    cfg.screen.enabled = True
    for key, value in kw.items():
        setattr(cfg.screen, key, value)
    return cfg


def _sample(app, title, bundle="com.example.app"):
    return {"app": app, "bundle_id": bundle, "title": title}


class _Scripted:
    def __init__(self, rows):
        self.rows = list(rows)
        self.n = 0

    def sample(self):
        row = self.rows[min(self.n, len(self.rows) - 1)]
        self.n += 1
        return row


def _drive(data, rows, *, poll=2.0, debounce=1.0, limit=8.0, **cfg_kw):
    cfg = _cfg(poll_s=poll, debounce_s=debounce, **cfg_kw)
    clock = {"t": 0.0}
    stop = threading.Event()

    def wait(seconds):
        clock["t"] += seconds
        if clock["t"] > limit:
            stop.set()
            return True
        return False

    run(data, cfg, sampler=_Scripted(rows), stop=stop, clock=lambda: clock["t"], wait=wait)
    return list(replay(data, types="screen"))


def _screen(data, app, title, at, *, redacted=False, bundle="com.example.app"):
    emit(data, "screen", "screen", {
        "app": app, "bundle_id": bundle, "title": title, "title_redacted": redacted,
    }, ts=_iso(at), id=f"screen-{at}")


def _utt(data, text, start, end, ident="u1"):
    emit_utterance(data, {
        "id": ident, "text": text, "status": "ok",
        "start": _iso(start), "end": _iso(end),
    })


def test_sampler_scripts_ask_only_for_the_front_window_title():
    blob = (_TITLE_SCRIPT + "\n" + _APP_SCRIPT).lower()
    for banned in ("screenshot", "screencapture", "entire contents", "ui element", "value of", "ocr"):
        assert banned not in blob
    assert "name of front window" in _TITLE_SCRIPT
    assert "on error" not in _TITLE_SCRIPT
    assert "bundle identifier" in _APP_SCRIPT


def test_prepare_redacts_excludes_and_truncates():
    cfg = _cfg(max_title_chars=200)
    secret = "vault-password-9"
    excluded = prepare(_sample("1Password 7", secret, "com.1password.1password"), cfg.screen)
    assert excluded["app"] == "1Password 7"
    assert excluded["title"] is None and excluded["title_redacted"] is True
    assert excluded["title_reason"] == "excluded_app"
    assert secret not in json.dumps(excluded)

    bank = prepare(_sample("Chase", "checking"), cfg.screen)
    assert bank["title"] == "checking" and bank["title_redacted"] is False
    cfg.screen.exclude_apps = list(cfg.screen.exclude_apps) + ["Chase"]
    hidden = prepare(_sample("Chase", "checking"), cfg.screen)
    assert hidden["title"] is None and "checking" not in json.dumps(hidden)

    private = prepare(_sample("Google Chrome", "Secret Project - Google Chrome (Incognito)"), cfg.screen)
    assert private["title"] is None and private["title_reason"] == "private"
    assert "Secret Project" not in json.dumps(private)
    visible = prepare(_sample("Google Chrome", "Secret Project - Google Chrome"), cfg.screen)
    assert visible["title"] == "Secret Project - Google Chrome"

    cfg.screen.redact_title_patterns = [r"token-\d+"]
    matched = prepare(_sample("Notes", "notes token-99"), cfg.screen)
    assert matched["title"] is None and matched["title_reason"] == "pattern"
    assert "token-99" not in json.dumps(matched)

    long = "甲" * 250
    cut = prepare(_sample("Cursor", long), cfg.screen)
    assert cut["title"] == "甲" * 200 and cut["title_truncated"] is True
    assert long not in json.dumps(cut)
    messy = prepare(_sample("Cursor", "hello\x00world\nnext"), cfg.screen)
    assert messy["title"] == "hello world next"
    assert "\x00" not in json.dumps(messy)


def test_debouncer_keeps_only_the_settled_snapshot():
    gate = Debouncer(1.0)
    chrome = {"app": "Chrome", "bundle_id": "c", "title": "a", "title_redacted": False}
    terminal = {"app": "Terminal", "bundle_id": "t", "title": "b", "title_redacted": False}
    gate.note(chrome, 0.0)
    gate.note(terminal, 0.2)
    gate.note(chrome, 0.4)
    assert gate.take(1.3) is None
    assert gate.take(1.4) == chrome
    assert gate.take(1.5) is None
    gate.note(chrome, 2.0)
    assert gate.take(4.0) is None
    gate.note(terminal, 5.0)
    assert gate.take(6.0) == terminal


def test_run_emits_on_change_only_and_a_restart_does_not_repeat(tmp_path):
    data = tmp_path / "data"
    rows = [
        _sample("Cursor", "a.py", "com.cursor"),
        _sample("Cursor", "a.py", "com.cursor"),
        _sample("Safari", "inbox", "com.apple.Safari"),
    ]
    first = _drive(data, rows, limit=8)
    assert [ev["data"]["app"] for ev in first] == ["Cursor", "Safari"]
    assert first[0]["data"]["title"] == "a.py"
    assert first[0]["type"] == "screen" and first[0]["source"] == "screen"
    again = _drive(data, [_sample("Safari", "inbox", "com.apple.Safari")], limit=4)
    assert [ev["data"]["app"] for ev in again] == ["Cursor", "Safari"]


def test_debounce_drops_a_quick_alt_tab(tmp_path):
    data = tmp_path / "data"
    rows = [
        _sample("Google Chrome", "mail"),
        _sample("Terminal", "secret-session"),
        _sample("Google Chrome", "mail"),
    ]
    events = _drive(data, rows, poll=0.2, debounce=1.0, limit=3)
    assert [ev["data"]["app"] for ev in events] == ["Google Chrome"]
    blob = json.dumps(events)
    assert "Terminal" not in blob and "secret-session" not in blob


def test_excluded_title_changes_do_not_emit_again(tmp_path):
    data = tmp_path / "data"
    rows = [
        _sample("1Password", "SECRET-ONE", "com.1password.1password"),
        _sample("1Password", "SECRET-TWO", "com.1password.1password"),
    ]
    events = _drive(data, rows, limit=6)
    assert len(events) == 1
    assert events[0]["data"]["title"] is None
    assert events[0]["data"]["title_redacted"] is True
    blob = Path(list((data / "events").glob("*.jsonl"))[0]).read_text(encoding="utf-8")
    assert "SECRET-ONE" not in blob and "SECRET-TWO" not in blob


def test_disabled_sampler_is_not_called(tmp_path):
    data = tmp_path / "data"
    called = []

    class Sampler:
        def sample(self):
            called.append(1)
            return _sample("Cursor", "a.py")

    cfg = C.Config()
    assert cfg.screen.enabled is False
    run(data, cfg, sampler=Sampler(), stop=threading.Event())
    assert called == []
    assert list(replay(data, types="screen")) == []


def test_non_mac_sampler_is_a_noop(tmp_path, caplog):
    assert sys.platform != "darwin"
    assert make_sampler() is None
    data = tmp_path / "data"
    cfg = _cfg()
    stop = threading.Event()

    def wait(_seconds):
        stop.set()
        return True

    with caplog.at_level(logging.WARNING):
        run(data, cfg, sampler=None, stop=stop, wait=wait)
    assert "macOS" in caplog.text
    assert list(replay(data, types="screen")) == []


def test_permission_denial_records_the_app_and_logs_once(caplog):
    calls = {"title": 0}

    def run_cmd(argv):
        if argv[:2] == ["lsappinfo", "front"]:
            return 0, "ASN:0x0-0x10:", ""
        if argv[0] == "lsappinfo":
            return 0, '"LSDisplayName"="Safari" "CFBundleIdentifier"="com.apple.Safari"', ""
        assert argv[:2] == ["osascript", "-e"]
        assert "screenshot" not in argv[-1].lower()
        calls["title"] += 1
        return 1, "", "osascript is not allowed assistive access. (-1719)"

    sampler = FrontSampler(run=run_cmd)
    with caplog.at_level(logging.WARNING, logger="audi27.screen"):
        first = sampler.sample()
        second = sampler.sample()
    assert first == {"app": "Safari", "bundle_id": "com.apple.Safari", "title": None}
    assert second["title"] is None
    assert calls["title"] == 1
    denials = [rec for rec in caplog.records if "Accessibility permission" in rec.message]
    assert len(denials) == 1


def test_a_missing_window_is_an_empty_title_not_a_denial():
    def run_cmd(argv):
        if argv[:2] == ["lsappinfo", "front"]:
            return 0, "ASN:0x0-0x10:", ""
        if argv[0] == "lsappinfo":
            return 0, '"LSDisplayName"="Finder" "CFBundleIdentifier"="com.apple.finder"', ""
        return 1, "", "System Events got an error: Can't get window 1 of process Finder. (-1728)"

    sample = FrontSampler(run=run_cmd).sample()
    assert sample == {"app": "Finder", "bundle_id": "com.apple.finder", "title": ""}


def test_osascript_names_the_app_when_lsappinfo_is_missing():
    def run_cmd(argv):
        if argv[0] == "lsappinfo":
            return 1, "", "not found"
        script = argv[-1]
        if "front window" in script:
            return 0, "readme.md\n", ""
        return 0, "Terminal\ncom.apple.Terminal\n", ""

    sample = FrontSampler(run=run_cmd).sample()
    assert sample == {"app": "Terminal", "bundle_id": "com.apple.Terminal", "title": "readme.md"}


def test_context_at_uses_the_latest_snapshot_inside_the_window(tmp_path):
    data = tmp_path / "data"
    _screen(data, "Old", "stale", at=0)
    _screen(data, "Cursor", "main.py", at=10)
    _screen(data, "Later", "after", at=40)
    assert context_at(data, _iso(20), 30)["title"] == "main.py"
    assert context_at(data, _iso(20), 5) is None  # the Cursor sample is 10s old
    assert context_at(data, _iso(9), 30)["app"] == "Old"
    fresh = context_at(data, _iso(10), 30)
    assert fresh == {
        "app": "Cursor", "bundle_id": "com.example.app", "title": "main.py", "title_redacted": False,
    }


def test_turns_and_routes_attach_screen_inside_the_window(tmp_path):
    data = tmp_path / "data"
    _screen(data, "Cursor", "main.py", at=10)
    _screen(data, "Mail", "inbox", at=50)
    _utt(data, "达达，在吗", 0, 20)
    cfg = _cfg(stale_s=30)
    turns, _stats = assemble_replay(data, cfg)
    event = event_for_turn(data, cfg, turns[0])
    assert event["data"]["screen"]["app"] == "Cursor"
    assert event["data"]["screen"]["title"] == "main.py"
    assert "title_reason" not in event["data"]["screen"]
    assert write_turns(data, turns, cfg)["wrote"] == 1
    routes = replay_routes(data, cfg)
    assert routes[0]["data"]["screen"]["title"] == "main.py"
    assert routes[0]["data"]["target"] == "grok"

    stale = _cfg(stale_s=5)
    assert "screen" not in event_for_turn(data, stale, turns[0])["data"]


def test_flag_off_leaves_turns_and_routes_unchanged(tmp_path):
    data = tmp_path / "data"
    _screen(data, "Cursor", "secret-title", at=10)
    _utt(data, "达达，在吗", 0, 12)
    cfg = C.Config()
    assert cfg.screen.enabled is False
    turns, _stats = assemble_replay(data, cfg)
    plain = turn_event(turns[0])
    assert event_for_turn(data, cfg, turns[0])["data"] == plain["data"]
    assert "screen" not in plain["data"]
    assert write_turns(data, turns, cfg)["wrote"] == 1
    # A turn that already carries a snapshot must not leak it onto the route while the flag is off.
    stored = list(replay(data, types="turn"))[0]
    stored["data"]["screen"] = {
        "app": "Cursor", "bundle_id": "com.cursor", "title": "secret-title", "title_redacted": False,
    }
    lone = tmp_path / "off"
    emit(lone, "turn", "turns", stored["data"], ts=stored["ts"], ref=stored["ref"], id=stored["id"])
    routes = replay_routes(lone, cfg)
    assert "screen" not in routes[0]["data"]
    assert "secret-title" not in json.dumps(routes)


def test_route_reads_the_log_when_the_turn_has_no_snapshot(tmp_path):
    data = tmp_path / "data"
    _screen(data, "Cursor", "main.py", at=10)
    turn = make_event("turn", "turns", {
        "utterance_ids": ["u1"], "text": "在吗", "raw_text": "达达，在吗",
        "start": _iso(0), "end": _iso(12), "reason": "eof", "trigger": "达达", "has_suspect": False,
    }, ts=_iso(12), ref="u1", id="turn-u1")
    emit(data, "turn", "turns", turn["data"], ts=turn["ts"], ref=turn["ref"], id=turn["id"])
    routes = replay_routes(data, _cfg(stale_s=30))
    assert routes[0]["data"]["screen"]["title"] == "main.py"
    assert "screen" not in replay_routes(data, C.Config())[0]["data"]


def test_live_turn_carries_the_screen_snapshot(tmp_path):
    data = tmp_path / "data"
    cfg = _cfg(stale_s=3600)
    cfg.turns.wall_close_s = 0.3
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    end = _iso(12)
    emit(data, "screen", "screen", {
        "app": "Cursor", "bundle_id": "com.cursor", "title": "live.py", "title_redacted": False,
    }, ts=end)
    emit_utterance(data, {
        "id": "u1", "text": "达达，在吗", "status": "ok", "start": _iso(10), "end": end,
    })
    found = []
    deadline = time.time() + 3
    while time.time() < deadline:
        found = list(replay(data, types="turn"))
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert found and found[0]["data"]["text"] == "在吗"
    assert found[0]["data"]["screen"]["title"] == "live.py"


def test_config_defaults_and_validation(tmp_path):
    cfg = C.Config()
    assert cfg.screen.enabled is False
    assert cfg.screen.poll_s == 2.0 and cfg.screen.debounce_s == 1.0
    assert cfg.screen.stale_s == 3600 and cfg.screen.max_title_chars == 200
    assert cfg.screen.exclude_apps == ["1Password", "Bitwarden", "Keychain Access"]
    assert "无痕" in cfg.screen.private_title_markers
    assert cfg.grokbot.include_screen is False
    example = C.load(Path(__file__).resolve().parents[1] / "config.example.toml")
    assert example.screen.enabled is False
    assert example.screen.exclude_apps[0] == "1Password"
    assert example.grokbot.include_screen is False
    path = tmp_path / "c.toml"
    path.write_text("[screen]\nenabled = true\npoll_s = 0.5\ndebounce_s = 0\n", encoding="utf-8")
    loaded = C.load(path)
    assert loaded.screen.enabled is True and loaded.screen.poll_s == 0.5
    assert loaded.screen.debounce_s == 0.0
    for bad in (
        "[screen]\nenabled = \"yes\"\n",
        "[screen]\npoll_s = 0\n",
        "[screen]\ndebounce_s = -1\n",
        "[screen]\nmax_title_chars = 0\n",
        "[screen]\nexclude_apps = [\"\"]\n",
        "[screen]\nredact_title_patterns = [\"(\"]\n",
        "[screen]\nnope = 1\n",
        "[grokbot]\ninclude_screen = \"yes\"\n",
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(C.ConfigError):
            C.load(path)


def test_supervisor_starts_screen_only_when_asked():
    off = child_specs(None, "127.0.0.1", 8727)
    assert "screen" not in off
    on = child_specs("/tmp/audi27.toml", "127.0.0.1", 8727, screen=True)
    assert on["screen"][-2:] == ["screen", "run"]
    assert "screen" not in child_specs(None, "127.0.0.1", 8727, turns=True, speak=True)


def test_cli_screen_off_does_not_sample(capsys):
    assert main(["screen", "run"]) == 0
    err = capsys.readouterr().err
    assert "off" in err
