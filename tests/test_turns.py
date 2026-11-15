"""Turn assembler: trigger-word utterances become one turn. Pure logic, plus replay/live IO."""
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from audi27 import config as C
from audi27.cli import main
from audi27.events import emit_utterance, ref_chain, replay
from audi27.turns import TurnAssembler, format_turn, match_trigger, turn_event

BASE = datetime(2026, 11, 15, 14, 0, tzinfo=timezone(timedelta(hours=8)))
TRIGGERS = ["达达", "哒哒", "大大", "DaDa", "dada", "Dada"]


def utt(n, text, start_ms, end_ms, status="ok"):
    return {
        "id": f"u{n}",
        "text": text,
        "status": status,
        "start": (BASE + timedelta(milliseconds=start_ms)).isoformat(timespec="milliseconds"),
        "end": (BASE + timedelta(milliseconds=end_ms)).isoformat(timespec="milliseconds"),
    }


def _asm(**kw):
    kw.setdefault("triggers", TRIGGERS)
    return TurnAssembler(**kw)


# --- trigger matching -------------------------------------------------------
@pytest.mark.parametrize("text,rest,surface", [
    ("达达，帮我", "帮我", "达达"),
    ("哒哒 帮我", "帮我", "哒哒"),
    ("大大,帮我", "帮我", "大大"),
    ("大大的眼睛", "的眼睛", "大大"),
    ("DaDa, help", "help", "DaDa"),
    ("dada help", "help", "dada"),
    ("Dada：问", "问", "Dada"),
    ("DADA! go", "go", "DADA"),
    ("  达达！", "", "达达"),
    ("达达", "", "达达"),
    ("达达……关于面试", "关于面试", "达达"),
    ("dada...next", "next", "dada"),
    ("\u3000达达，帮我", "帮我", "达达"),
    ("达达\u3000在吗", "在吗", "达达"),
    ("\ufeff达达，帮我", "帮我", "达达"),
    ("\u200b达达", "", "达达"),
    ("ＤＡＤＡ help", "help", "ＤＡＤＡ"),
    ("ｄａｄａ，走", "走", "ｄａｄａ"),
])
def test_trigger_variants_and_punctuation(text, rest, surface):
    assert match_trigger(text, TRIGGERS) == surface
    turns = _asm().feed([utt(1, text, 0, 400)])
    assert len(turns) == 1
    assert turns[0].text == rest
    assert turns[0].trigger == surface
    assert turns[0].raw_text == " ".join(text.split())
    assert turns[0].reason == "eof"


@pytest.mark.parametrize("text", [
    "我跟大大说",
    "然后大大，你听一下",
    "眼睛大大的",
    "这个dada不行",
    "，达达你好",
    "。达达",
    "dadapple pie",
    "hello 达达",
    "好的大大",
    "dada's house",
    "dada\u2019s house",
    "dada\u2018s house",
    "dada\uff07s house",
    "dada２",
    "dada２go",
    "ＤＡＤＡｐｉｅ",
    "大大小小",
    "大大小小的事情",
    "大大方方",
    "大大咧咧地走",
    "有大大小小的问题",
])
def test_trigger_only_at_the_very_start(text):
    assert match_trigger(text, TRIGGERS) is None
    assert _asm().feed([utt(1, text, 0, 400)]) == []


def test_configured_triggers_replace_the_defaults():
    assert TurnAssembler(triggers=["嘿"]).feed([utt(1, "达达，在吗", 0, 400)]) == []
    turns = TurnAssembler(triggers=["嘿"]).feed([utt(1, "嘿，在吗", 0, 400)])
    assert turns[0].text == "在吗" and turns[0].trigger == "嘿"


def test_duplicate_utterance_id_is_kept_once():
    first = utt(1, "达达，甲", 0, 1000)
    again = utt(1, "乙", 1200, 1800)
    turns = _asm().feed([first, again, utt(2, "丙", 2000, 2500)])
    assert turns[0].utterance_ids == ["u1", "u2"]
    assert turns[0].text == "甲 丙"
    asm = _asm()
    assert asm.push(first) == []
    assert asm.push(first) == []
    assert asm.finish()[0].text == "甲"


def test_trigger_alone_then_the_command():
    turns = _asm().feed([
        utt(1, "达达", 0, 400),
        utt(2, "帮我问一下 Muse", 1000, 2000),
    ])
    assert len(turns) == 1
    assert turns[0].text == "帮我问一下 Muse"
    assert turns[0].raw_text == "达达 帮我问一下 Muse"
    assert turns[0].utterance_ids == ["u1", "u2"]


# --- merging ----------------------------------------------------------------
def test_merge_across_a_short_pause_and_leave_ambient_out():
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，帮我问一下 Muse", 0, 1500),
        utt(2, "关于下个月的面试", 2300, 4000),
        utt(3, "今天天气不错", 10000, 11000),
    ])
    assert len(turns) == 1
    assert turns[0].reason == "gap"
    assert turns[0].utterance_ids == ["u1", "u2"]
    assert turns[0].text == "帮我问一下 Muse 关于下个月的面试"
    assert turns[0].raw_text == "达达，帮我问一下 Muse 关于下个月的面试"
    assert turns[0].start == utt(1, "", 0, 0)["start"]
    assert turns[0].end == utt(2, "", 0, 4000)["end"]


def test_gap_equal_to_the_timeout_still_merges():
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，甲", 0, 1000),
        utt(2, "乙", 3000, 4000),       # gap == 2.000 s
        utt(3, "丙", 6001, 7000),       # gap == 2.001 s
    ])
    assert turns[0].utterance_ids == ["u1", "u2"]
    assert turns[0].reason == "gap"
    assert turns[0].text == "甲 乙"


def test_max_duration_includes_the_utterance_that_crosses_it():
    turns = _asm(max_duration_s=10).feed([
        utt(1, "达达，甲", 0, 4000),
        utt(2, "乙", 5000, 9000),
        utt(3, "丙", 10000, 14000),
        utt(4, "丁", 15000, 16000),
    ])
    assert len(turns) == 1
    assert turns[0].reason == "max"
    assert turns[0].utterance_ids == ["u1", "u2", "u3"]
    assert turns[0].text == "甲 乙 丙"


def test_opening_utterance_at_the_cap_closes_immediately():
    closed = _asm(max_duration_s=10).push(utt(1, "达达，很长的一段", 0, 10000))
    assert len(closed) == 1 and closed[0].reason == "max"
    assert closed[0].utterance_ids == ["u1"]


def test_new_trigger_splits_an_open_turn():
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，第一件", 0, 1000),
        utt(2, "还没说完", 1500, 2000),
        utt(3, "哒哒，第二件", 2500, 3500),
    ])
    assert [t.reason for t in turns] == ["new_trigger", "eof"]
    assert turns[0].text == "第一件 还没说完"
    assert turns[0].utterance_ids == ["u1", "u2"]
    assert turns[1].text == "第二件" and turns[1].trigger == "哒哒"


def test_trigger_after_a_long_pause_closes_the_previous_as_gap():
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，第一件", 0, 1000),
        utt(2, "达达，第二件", 5000, 6000),
    ])
    assert [t.reason for t in turns] == ["gap", "eof"]
    assert turns[0].text == "第一件" and turns[0].utterance_ids == ["u1"]
    assert turns[1].text == "第二件"


def test_audio_order_wins_over_arrival_order():
    """A retried or backfilled segment can land later in the file than it happened."""
    turns = _asm(gap_s=2).feed([
        utt(3, "今天天气不错", 8000, 9000),
        utt(2, "查日程", 1500, 2500),
        utt(1, "达达，帮我", 0, 1000),
    ])
    assert len(turns) == 1
    assert turns[0].reason == "gap"
    assert turns[0].utterance_ids == ["u1", "u2"]
    assert turns[0].text == "帮我 查日程"

    # Earlier ambient speech must not be glued onto the command just because it arrived late.
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，帮我", 5000, 6000),
        utt(2, "今天天气不错", 0, 1000),
    ])
    assert turns[0].utterance_ids == ["u1"]
    assert turns[0].text == "帮我"
    assert turns[0].reason == "eof"


def test_new_trigger_that_is_itself_too_long():
    turns = _asm(max_duration_s=5, gap_s=2).feed([
        utt(1, "达达，短", 0, 1000),
        utt(2, "达达，很长很长", 1500, 8000),
    ])
    assert [t.reason for t in turns] == ["new_trigger", "max"]
    assert turns[1].utterance_ids == ["u2"]


# --- status -----------------------------------------------------------------
def test_rejected_is_skipped_and_does_not_move_the_gap():
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，帮我", 0, 1000, "ok"),
        utt(2, "谢谢观看", 1200, 1500, "rejected"),
        utt(3, "订明天的票", 2000, 3000, "ok"),
        utt(4, "达达，别执行", 3200, 4000, "rejected"),
    ])
    assert len(turns) == 1
    assert turns[0].utterance_ids == ["u1", "u3"]
    assert turns[0].text == "帮我 订明天的票"
    assert turns[0].has_suspect is False
    assert turns[0].reason == "eof"


def test_rejected_gap_is_measured_between_kept_utterances():
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，帮我", 0, 1000),
        utt(2, "谢谢观看", 1200, 8000, "rejected"),
        utt(3, "订票", 8500, 9000),
    ])
    assert len(turns) == 1
    assert turns[0].reason == "gap"
    assert turns[0].utterance_ids == ["u1"]


def test_suspect_is_kept_and_flagged():
    turns = _asm().feed([
        utt(1, "哒哒，开会", 0, 1000, "suspect"),
        utt(2, "下午三点", 1500, 2500, "ok"),
    ])
    assert turns[0].has_suspect is True
    assert turns[0].text == "开会 下午三点"
    turns = _asm().feed([
        utt(1, "达达，开会", 0, 1000, "ok"),
        utt(2, "下午三点", 1500, 2500, "suspect"),
    ])
    assert turns[0].has_suspect is True
    assert turns[0].utterance_ids == ["u1", "u2"]


def test_unknown_status_and_blank_text_are_skipped():
    assert _asm().feed([utt(1, "达达，在吗", 0, 400, "nope")]) == []
    assert _asm().feed([utt(1, "   ", 0, 400)]) == []
    turns = _asm(gap_s=2).feed([
        utt(1, "达达，甲", 0, 1000),
        utt(2, "   ", 1200, 1400),
        utt(3, "乙", 2000, 2500),
    ])
    assert turns[0].utterance_ids == ["u1", "u3"]


def test_non_utterance_events_are_ignored_and_envelopes_are_accepted():
    screen = {"type": "screen", "id": "s", "data": {"text": "达达，从屏幕来的", "start": utt(1, "", 0, 0)["start"]}}
    assert _asm().feed([screen]) == []
    rec = utt(1, "达达，在吗", 0, 400)
    ev = {"id": "u1", "type": "utterance", "source": "transcribe", "ts": rec["start"],
          "ref": None, "data": rec}
    turns = _asm().feed([ev])
    assert turns[0].text == "在吗" and turns[0].utterance_ids == ["u1"]


def test_unparseable_utterance_does_not_abort_the_sequence():
    bad = {"id": "bad", "text": "达达，坏时间", "status": "ok", "start": "not-a-time", "end": "also-bad"}
    turns = _asm().feed([bad, utt(1, "今天天气不错", 0, 400), utt(2, "达达，在吗", 5000, 5500)])
    assert len(turns) == 1 and turns[0].text == "在吗"


# --- wall clock (live) ------------------------------------------------------
def test_live_wall_clock_closes_after_receipt_not_audio_end():
    clock = {"t": 100.0}
    asm = _asm(wall_close_s=2.0, clock=lambda: clock["t"])
    # Audio ended long before we saw the event. Do not close just because of that.
    assert asm.push(utt(1, "达达，在吗", -30_000, -29_000)) == []
    assert asm.poll() == []
    clock["t"] = 101.9
    assert asm.poll() == []
    clock["t"] = 102.0
    closed = asm.poll()
    assert len(closed) == 1 and closed[0].reason == "timeout"
    assert closed[0].text == "在吗"
    assert asm.poll() == []


def test_continuation_resets_the_wall_clock():
    clock = {"t": 0.0}
    asm = _asm(gap_s=2.0, wall_close_s=2.0, clock=lambda: clock["t"])
    asm.push(utt(1, "达达，帮我", 0, 1000))
    clock["t"] = 1.5
    assert asm.push(utt(2, "查日程", 1500, 2500)) == []
    clock["t"] = 3.4  # 1.9 s after the continuation, 3.4 s after the trigger
    assert asm.poll() == []
    clock["t"] = 3.5
    closed = asm.poll()
    assert closed[0].reason == "timeout"
    assert closed[0].utterance_ids == ["u1", "u2"]


def test_wall_timeout_drops_a_later_continuation():
    clock = {"t": 0.0}
    asm = _asm(gap_s=2.0, wall_close_s=2.0, clock=lambda: clock["t"])
    asm.push(utt(1, "达达，帮我", 0, 1000))
    clock["t"] = 2.0
    closed = asm.poll()
    assert closed[0].reason == "timeout" and closed[0].text == "帮我"
    # Audio gap would have been 0.5 s, but the turn is already closed.
    assert asm.push(utt(2, "查日程", 1500, 2500)) == []
    assert asm.finish() == []


def test_feed_does_not_apply_the_wall_clock():
    """Closing a replay uses utterance timestamps only. A zero idle timeout must not win."""
    clock = {"t": 0.0}
    turns = _asm(wall_close_s=0.0, clock=lambda: clock["t"]).feed([
        utt(1, "达达，甲", 0, 1000),
        utt(2, "乙", 1500, 2000),
    ])
    assert turns[0].reason == "eof" and turns[0].utterance_ids == ["u1", "u2"]


# --- events -----------------------------------------------------------------
def test_turn_event_points_at_the_first_utterance():
    turn = _asm().feed([utt(1, "达达，在吗", 0, 500), utt(2, "收到", 800, 1200)])[0]
    ev = turn_event(turn)
    again = turn_event(turn)
    assert ev == again
    assert ev["type"] == "turn" and ev["source"] == "turns"
    assert ev["ref"] == "u1" and ev["id"] == "turn-u1"
    assert ev["data"]["utterance_ids"] == ["u1", "u2"]
    assert ev["data"]["text"] == "在吗 收到"
    assert ev["data"]["raw_text"] == "达达，在吗 收到"
    assert ev["data"]["reason"] == "eof"
    assert ev["data"]["has_suspect"] is False
    assert datetime.fromisoformat(ev["ts"]) == datetime.fromisoformat(turn.end)
    assert "timeout" not in format_turn(turn)
    assert "在吗 收到" in format_turn(turn)


def _write_utterances(data, rows):
    for rec in rows:
        emit_utterance(data, rec)


def test_replay_is_deterministic_and_does_not_touch_the_live_stream(tmp_path):
    from audi27.turns import assemble_replay

    data = tmp_path / "data"
    rows = [
        utt(1, "今天天气不错", 0, 800),
        utt(2, "达达，帮我问一下 Muse", 5000, 6500),
        utt(3, "关于下个月的面试", 7300, 9000),
        utt(4, "谢谢观看", 9100, 9400, "rejected"),
    ]
    _write_utterances(data, rows)
    cfg = C.Config()
    first, stats = assemble_replay(data, cfg)
    second, stats2 = assemble_replay(data, cfg)
    assert stats == stats2 == {"turns": 1, "utterances": 2, "seen": 4}
    assert [turn_event(t) for t in first] == [turn_event(t) for t in second]
    assert first[0].text == "帮我问一下 Muse 关于下个月的面试"
    assert list(replay(data, types="turn")) == []


def _wait_turns(data, n, timeout=3.0):
    deadline = time.time() + timeout
    found = []
    while time.time() < deadline:
        found = list(replay(data, types="turn"))
        if len(found) >= n:
            return found
        time.sleep(0.05)
    return found


def _serve(data, cfg, wall=0.4):
    from audi27.turns import serve

    cfg.turns.wall_close_s = wall
    ready, stop = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    return thread, stop


def test_live_serve_closes_on_the_wall_clock(tmp_path):
    from audi27.turns import serve

    data = tmp_path / "data"
    cfg = C.Config()
    cfg.turns.wall_close_s = 0.4
    ready = threading.Event()
    stop = threading.Event()
    thread = threading.Thread(
        target=serve, args=(data, cfg), kwargs={"poll_s": 0.05, "stop": stop, "ready": ready}, daemon=True)
    thread.start()
    assert ready.wait(2)
    rec = utt(1, "达达，在吗", 0, 400)
    emit_utterance(data, rec)
    found = []
    deadline = time.time() + 3
    while time.time() < deadline:
        found = list(replay(data, types="turn"))
        if found:
            break
        time.sleep(0.05)
    stop.set()
    thread.join(2)
    assert found, "live assembler did not close the pending turn"
    assert found[0]["data"]["reason"] == "timeout"
    assert found[0]["data"]["text"] == "在吗"
    assert found[0]["ref"] == "u1"
    assert not thread.is_alive()


def test_live_orders_a_late_segment_before_it_closes(tmp_path):
    from audi27.turns import serve

    data = tmp_path / "data"
    thread, stop = _serve(data, C.Config(), wall=1.0)
    # Continuation is written first; the trigger's transcription finishes later.
    # wall_close_s is long enough that a scheduling gap between these writes
    # cannot close the batch before the trigger lands.
    emit_utterance(data, utt(2, "查日程", 1500, 2500))
    emit_utterance(data, utt(3, "今天天气不错", 8000, 9000))
    emit_utterance(data, utt(1, "达达，帮我", 0, 1000))
    found = _wait_turns(data, 1, timeout=4)
    stop.set()
    thread.join(2)
    assert len(found) == 1
    assert found[0]["data"]["reason"] == "gap"
    assert found[0]["data"]["utterance_ids"] == ["u1", "u2"]
    assert found[0]["data"]["text"] == "帮我 查日程"
    assert not thread.is_alive()


def test_restart_does_not_backfill_or_duplicate_and_does_not_drop(tmp_path):
    from audi27.turns import serve

    data = tmp_path / "data"
    emit_utterance(data, utt(0, "达达，旧的", 0, 400))
    thread, stop = _serve(data, C.Config(), wall=0.3)
    time.sleep(0.8)
    assert list(replay(data, types="turn")) == []
    stop.set()
    thread.join(2)
    assert not thread.is_alive()

    # Written while the assembler was down: the next process resumes past the snapshot.
    emit_utterance(data, utt(2, "查日程", 6500, 7500))
    emit_utterance(data, utt(1, "达达，在吗", 5000, 5400))
    thread, stop = _serve(data, C.Config(), wall=0.3)
    found = _wait_turns(data, 1)
    stop.set()
    thread.join(2)
    assert len(found) == 1
    assert found[0]["id"] == "turn-u1"
    assert found[0]["data"]["utterance_ids"] == ["u1", "u2"]
    assert found[0]["data"]["text"] == "在吗 查日程"
    assert not thread.is_alive()

    thread, stop = _serve(data, C.Config(), wall=0.3)
    time.sleep(0.8)
    stop.set()
    thread.join(2)
    assert [ev["id"] for ev in replay(data, types="turn")] == ["turn-u1"]
    assert not thread.is_alive()


# --- config, supervisor, CLI ------------------------------------------------
def test_turns_config_defaults_and_rejects_typos(tmp_path):
    cfg = C.Config()
    assert cfg.turns.enabled is False
    assert cfg.turns.gap_s == 2.0
    assert cfg.turns.max_duration_s == 120.0
    assert cfg.turns.wall_close_s == 8.0
    assert cfg.turns.triggers == TRIGGERS
    from audi27.turns import DEFAULT_TRIGGERS
    assert list(DEFAULT_TRIGGERS) == list(cfg.turns.triggers)
    example = C.load(Path(__file__).resolve().parent.parent / "config.example.toml")
    assert example.turns.enabled is False and example.turns.triggers == TRIGGERS
    f = tmp_path / "c.toml"
    f.write_text('[turns]\nenabled = true\ngap_s = 3\ntriggers = ["嘿", " dada "]\n')
    loaded = C.load(f)
    assert loaded.turns.enabled is True and loaded.turns.gap_s == 3.0
    assert loaded.turns.triggers == ["嘿", "dada"]
    for bad in ("[turns]\nwall = 1\n", '[turns]\ntriggers = "达达"\n', "[turns]\ngap_s = -1\n",
                "[turns]\nmax_duration_s = 0\n", '[turns]\nenabled = "yes"\n', "[turns]\ntriggers = []\n"):
        f.write_text(bad)
        with pytest.raises(C.ConfigError):
            C.load(f)


def test_supervisor_starts_turns_only_when_enabled():
    from audi27.supervisor import child_specs

    off = child_specs(None, "127.0.0.1", 8727, turns=False)
    on = child_specs("/tmp/audi27.toml", "127.0.0.1", 8727, turns=True)
    assert set(off) == {"run", "web"}
    assert on["turns"][-2:] == ["turns", "run"]
    assert on["run"][:4] == on["turns"][:4]
    assert "-c" in on["run"] and "/tmp/audi27.toml" in on["run"]


def test_cli_replay_json_stats_output_and_idempotent_write(tmp_path, capsys):
    data = tmp_path / "data"
    _write_utterances(data, [
        utt(1, "路过的话", 0, 500),
        utt(2, "达达，帮我问一下 Muse", 2000, 3000),
        utt(3, "关于下个月的面试", 3500, 5000),
    ])
    # Explicit offsets: a bare calendar day is the machine's local midnight, which can exclude +08 timestamps.
    args = ["turns", "replay", "--data-dir", str(data),
            "--since", "2026-11-15T00:00:00+00:00", "--until", "2026-11-16T00:00:00+00:00"]
    assert main(args + ["--json"]) == 0
    out1 = capsys.readouterr()
    assert main(args + ["--json"]) == 0
    out2 = capsys.readouterr()
    assert out1.out == out2.out
    ev = json.loads(out1.out.strip())
    assert ev["type"] == "turn" and ev["id"] == "turn-u2" and ev["ref"] == "u2"
    assert ev["data"]["text"] == "帮我问一下 Muse 关于下个月的面试"
    assert ev["data"]["reason"] == "eof"
    assert "turns found: 1, utterances merged: 2" in out1.err
    assert list(replay(data, types="turn")) == []

    out_path = tmp_path / "out" / "turns.jsonl"
    assert main(args + ["--output", str(out_path)]) == 0
    capsys.readouterr()
    first_bytes = out_path.read_bytes()
    assert main(args + ["--output", str(out_path)]) == 0
    capsys.readouterr()
    assert out_path.read_bytes() == first_bytes
    assert json.loads(first_bytes.decode().strip())["id"] == "turn-u2"

    assert main(args + ["--write"]) == 0
    err = capsys.readouterr().err
    assert "wrote 1" in err
    assert main(args + ["--write"]) == 0
    err = capsys.readouterr().err
    assert "already present 1" in err
    stored = list(replay(data, types="turn"))
    assert len(stored) == 1 and stored[0]["id"] == "turn-u2"
    assert [e["type"] for e in ref_chain(data, "turn-u2")] == ["utterance", "turn"]

    assert main(["turns", "replay", "--since", "not-a-time"]) == 2


def test_cli_replay_honours_configured_triggers(tmp_path, capsys):
    data = tmp_path / "data"
    _write_utterances(data, [utt(1, "达达，在吗", 0, 400), utt(2, "嘿，看日程", 5000, 6000)])
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text('[turns]\ntriggers = ["嘿"]\n')
    assert main(["-c", str(cfg_path), "turns", "replay", "--data-dir", str(data), "--json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 1 and lines[0]["data"]["text"] == "看日程" and lines[0]["ref"] == "u2"
