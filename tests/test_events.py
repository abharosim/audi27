"""Event stream: append-only JSON lines under data/events/, independent of transcripts."""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from audi27.cli import main
from audi27.events import (
    KNOWN_TYPES,
    EventError,
    append,
    backfill_utterances,
    emit,
    follow,
    make_event,
    read_new,
    ref_chain,
    replay,
    snapshot_offsets,
    utterance_event,
)
from audi27.store import Store

SR = 16000


def _record(rec_id, start, text="hello", status="ok"):
    return {
        "id": rec_id,
        "start": start,
        "end": start,
        "duration": 1.0,
        "text": text,
        "status": status,
        "reasons": [],
        "language": "zh",
        "language_prob": 0.9,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.1,
        "backend": "fake",
        "model": "m",
        "audio": None,
    }


def _ids(events):
    return [e["id"] for e in events]


# --- format -----------------------------------------------------------------
def test_event_envelope_and_unknown_type(tmp_path):
    data = tmp_path / "data"
    ev = emit(data, "ping", "future-module", {"text": "达达"}, ref=None)
    assert ev["type"] == "ping" and "ping" not in KNOWN_TYPES
    assert set(KNOWN_TYPES) >= {"utterance", "screen", "presence", "turn", "route", "task", "reply", "brief", "spoken"}
    assert ev["source"] == "future-module" and ev["ref"] is None and ev["data"] == {"text": "达达"}
    assert ev["id"] and " " not in ev["id"]
    ts = datetime.fromisoformat(ev["ts"])
    assert ts.tzinfo is not None and ts.utcoffset() is not None
    path = next((data / "events").glob("*.jsonl"))
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert "达达".encode() in raw
    stored = json.loads(raw)
    assert set(stored) == {"id", "ts", "type", "source", "ref", "data"}
    assert list(replay(data)) == [ev]


def test_make_event_rejects_bad_envelope():
    with pytest.raises(EventError):
        make_event("ping", "mod", {}, ts=datetime(2026, 11, 15, 12, 0, 0))  # no timezone
    with pytest.raises(EventError):
        make_event("", "mod", {})
    with pytest.raises(EventError):
        make_event("ping", "mod", ["not", "a", "dict"])
    with pytest.raises(EventError):
        make_event("ping", "mod", {}, ref="")
    with pytest.raises(EventError):
        append("/tmp", {"id": "x", "type": "ping"})  # incomplete


def test_explicit_id_ref_and_nan(tmp_path):
    data = tmp_path / "data"
    ts = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    ev = emit(data, "screen", "screen", {"title": "x", "n": float("nan")}, id="s1", ref="u1", ts=ts)
    assert ev["id"] == "s1" and ev["ref"] == "u1" and ev["data"]["n"] is None
    assert datetime.fromisoformat(ev["ts"]) == ts.astimezone()


# --- append, rollover, concurrency ------------------------------------------
def test_append_closes_a_torn_tail(tmp_path):
    data = tmp_path / "data"
    ev = make_event("ping", "mod", {"n": 1}, id="whole")
    day = datetime.fromisoformat(ev["ts"]).astimezone().strftime("%Y-%m-%d")
    path = data / "events" / f"{day}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"id": "torn", "ts": "2026-11-')
    append(data, ev)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])
    assert json.loads(lines[1])["id"] == "whole"
    assert _ids(replay(data)) == ["whole"]


def test_day_rollover_uses_local_time(tmp_path):
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Shanghai"
    time.tzset()
    try:
        data = tmp_path / "data"
        late = datetime.fromisoformat("2026-11-15T23:30:00+08:00")
        # 16:30Z is 00:30 the next day in Shanghai, even though the original date says the 15th.
        early = datetime.fromisoformat("2026-11-15T16:30:00+00:00")
        edge = datetime.fromisoformat("2026-11-15T23:59:59.999+08:00")
        midnight = datetime.fromisoformat("2026-11-16T00:00:00+08:00")
        emit(data, "presence", "capture", {"state": "speaking"}, id="late", ts=late)
        emit(data, "presence", "capture", {"state": "speaking"}, id="edge", ts=edge)
        emit(data, "presence", "capture", {"state": "quiet"}, id="midnight", ts=midnight)
        emit(data, "presence", "capture", {"state": "quiet"}, id="early", ts=early)
        files = sorted(p.name for p in (data / "events").glob("*.jsonl"))
        assert files == ["2026-11-15.jsonl", "2026-11-16.jsonl"]
        # since is inclusive, until is exclusive, and the file prefilter must not drop the boundary day.
        from audi27.store import parse_when
        assert _ids(replay(data, since=parse_when("2026-11-15"),
                           until=parse_when("2026-11-15", end_of_day=True))) == ["late", "edge"]
        assert _ids(replay(data, since=midnight)) == ["midnight", "early"]
        for line in (data / "events" / "2026-11-16.jsonl").read_text().splitlines():
            assert json.loads(line)["ts"].endswith("+08:00")
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_concurrent_appends_do_not_interleave(tmp_path):
    data = str(tmp_path / "data")
    n, writers = 30, 6
    # The test process is chdir'd to a temp dir, so point the children at the repo that holds audi27.
    root = str(Path(__file__).resolve().parents[1])
    code = (
        "import sys\n"
        "from audi27.events import emit\n"
        "data, n, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
        "for i in range(n):\n"
        "    emit(data, 'marker', prefix, {'i': i, 'blob': prefix * 400}, id=f'{prefix}-{i}')\n"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code, data, str(n), f"p{i}"],
                         cwd=root, stderr=subprocess.PIPE, text=True)
        for i in range(writers)
    ]
    codes = []
    for p in procs:
        _, err = p.communicate(timeout=60)
        codes.append((p.returncode, err))
    assert codes == [(0, "")] * writers, codes
    lines = []
    for path in sorted((tmp_path / "data" / "events").glob("*.jsonl")):
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    assert len(lines) == n * writers
    ids = []
    for line in lines:
        ev = json.loads(line)
        assert ev["data"]["blob"] == ev["source"] * 400
        ids.append(ev["id"])
    assert len(set(ids)) == len(ids)


# --- replay, follow, partial lines ------------------------------------------
def test_replay_filters_type_source_and_time(tmp_path):
    data = tmp_path / "data"
    t0 = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    emit(data, "utterance", "transcribe", {"text": "a"}, id="a", ts=t0)
    emit(data, "screen", "screen", {"app": "Terminal"}, id="b", ts=t0 + timedelta(hours=1))
    emit(data, "utterance", "other", {"text": "c"}, id="c", ts=t0 + timedelta(hours=2))
    assert _ids(replay(data, types="utterance")) == ["a", "c"]
    assert _ids(replay(data, types=["utterance", "screen"], sources="screen")) == ["b"]
    assert _ids(replay(data, since=t0 + timedelta(hours=1), until=t0 + timedelta(hours=2))) == ["b"]
    assert _ids(replay(data, types="nope")) == []


def test_replay_skips_corrupt_lines_and_a_partial_tail(tmp_path):
    data = tmp_path / "data"
    ts = datetime.now().astimezone().isoformat(timespec="milliseconds")
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    path = data / "events" / f"{day}.jsonl"
    path.parent.mkdir(parents=True)
    good = {"id": "ok1", "ts": ts, "type": "ping", "source": "mod", "ref": None, "data": {"n": 1}}
    good2 = {**good, "id": "ok2", "data": {"n": 2}}
    path.write_text(
        json.dumps(good) + "\n"
        + "{not json\n"
        + "\n"
        + json.dumps({"id": 5, "ts": ts, "type": "ping", "source": "mod", "data": {}}) + "\n"
        + json.dumps(good2) + "\n"
        + '{"id": "partial", "ts": "'
        , encoding="utf-8")
    assert _ids(replay(data)) == ["ok1", "ok2"]


def test_follow_starts_at_end_unless_since_is_set(tmp_path):
    data = tmp_path / "data"
    emit(data, "ping", "mod", {"n": 0}, id="old")
    ready, stop = threading.Event(), threading.Event()
    seen = []

    def consume():
        for ev in follow(data, poll=0.02, stop=stop, ready=ready):
            seen.append(ev["id"])

    thread = threading.Thread(target=consume)
    thread.start()
    assert ready.wait(2)
    emit(data, "ping", "mod", {"n": 1}, id="new")
    deadline = time.time() + 2
    while "new" not in seen and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(2)
    assert seen == ["new"] and not thread.is_alive()


def test_follow_replays_since_then_new_events(tmp_path):
    data = tmp_path / "data"
    emit(data, "task", "agent", {"n": 1}, id="old")
    stop = threading.Event()
    seen = []

    def consume():
        for ev in follow(data, since=datetime.now().astimezone() - timedelta(hours=1), poll=0.02, stop=stop):
            seen.append(ev["id"])
            if len(seen) >= 2:
                stop.set()

    thread = threading.Thread(target=consume)
    thread.start()
    deadline = time.time() + 2
    while "old" not in seen and time.time() < deadline:
        time.sleep(0.02)
    emit(data, "reply", "agent", {"n": 2}, id="new", ref="old")
    thread.join(2)
    assert seen == ["old", "new"]


def test_follow_waits_for_a_partial_line_to_finish(tmp_path):
    data = tmp_path / "data"
    emit(data, "ping", "mod", {"n": 1}, id="first")
    path = next((data / "events").glob("*.jsonl"))
    done = make_event("ping", "mod", {"n": 2}, id="second")
    raw = (json.dumps(done) + "\n").encode()
    with open(path, "ab") as f:
        f.write(raw[:20])
    stop = threading.Event()
    seen = []

    def consume():
        for ev in follow(data, since=datetime.now().astimezone() - timedelta(hours=1), poll=0.02, stop=stop):
            seen.append(ev["id"])

    thread = threading.Thread(target=consume)
    thread.start()
    deadline = time.time() + 2
    while seen != ["first"] and time.time() < deadline:
        time.sleep(0.02)
    assert seen == ["first"]
    time.sleep(0.1)  # a few polls must leave the unfinished line unread
    assert seen == ["first"]
    with open(path, "ab") as f:
        f.write(raw[20:])
    deadline = time.time() + 2
    while "second" not in seen and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(2)
    assert seen == ["first", "second"]


def test_follow_sees_a_file_created_after_it_starts(tmp_path):
    """Tail must notice a day file that does not exist yet (midnight rollover, first event of a run)."""
    data = tmp_path / "data"
    ready, stop = threading.Event(), threading.Event()
    seen = []

    def consume():
        for ev in follow(data, poll=0.02, stop=stop, ready=ready):
            seen.append(ev["id"])

    thread = threading.Thread(target=consume)
    thread.start()
    assert ready.wait(2)
    emit(data, "presence", "capture", {"state": "speaking"}, id="born")
    deadline = time.time() + 2
    while "born" not in seen and time.time() < deadline:
        time.sleep(0.02)
    emit(data, "presence", "capture", {"state": "quiet"}, id="next-day",
         ts=datetime.now().astimezone() + timedelta(days=1))
    deadline = time.time() + 2
    while "next-day" not in seen and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(2)
    assert len(list((data / "events").glob("*.jsonl"))) == 2
    assert seen == ["born", "next-day"] and not thread.is_alive()


def test_read_new_resumes_after_the_cursor_and_waits_for_a_partial_line(tmp_path):
    data = tmp_path / "data"
    emit(data, "utterance", "transcribe", {"text": "old"}, id="old")
    cursor = snapshot_offsets(data)
    found, cursor = read_new(data, cursor, types="utterance")
    assert found == []
    path = next((data / "events").glob("*.jsonl"))
    done = make_event("utterance", "transcribe", {"text": "new"}, id="new")
    raw = (json.dumps(done, ensure_ascii=False) + "\n").encode()
    with open(path, "ab") as f:
        f.write(raw[:20])
    found, mid = read_new(data, cursor, types="utterance")
    assert found == [] and mid == cursor
    with open(path, "ab") as f:
        f.write(raw[20:])
    # A turn written into the same file must not come back as an utterance, and must not hide `new`.
    emit(data, "turn", "turns", {"text": "x"}, id="turn-old")
    found, cursor = read_new(data, mid, types="utterance")
    assert [ev["id"] for ev in found] == ["new"]
    found, cursor = read_new(data, cursor, types="utterance")
    assert found == []


def test_replay_since_inclusive_until_exclusive(tmp_path):
    data = tmp_path / "data"
    before = datetime(2026, 11, 15, 23, 59, 59, 999000, tzinfo=timezone.utc)
    at = datetime(2026, 11, 16, 0, 0, tzinfo=timezone.utc)
    after = datetime(2026, 11, 16, 0, 0, 0, 1000, tzinfo=timezone.utc)
    emit(data, "ping", "mod", {}, id="before", ts=before)
    emit(data, "ping", "mod", {}, id="at", ts=at)
    emit(data, "ping", "mod", {}, id="after", ts=after)
    assert _ids(replay(data, since=before, until=at)) == ["before"]
    assert _ids(replay(data, since=at, until=after)) == ["at"]
    assert _ids(replay(data, since=at)) == ["at", "after"]


# --- ref chain --------------------------------------------------------------
def test_ref_chain_walks_back_to_the_origin(tmp_path):
    data = tmp_path / "data"
    yesterday = datetime.now().astimezone() - timedelta(days=1)
    emit(data, "utterance", "transcribe", {"text": "达达 查一下"}, id="u", ts=yesterday)
    emit(data, "route", "router", {"agent": "claude"}, id="r", ref="u")
    emit(data, "spoken", "speaker", {"text": "查完了"}, id="s", ref="r")
    assert _ids(ref_chain(data, "s")) == ["u", "r", "s"]
    assert _ids(ref_chain(data, "u")) == ["u"]
    emit(data, "spoken", "speaker", {}, id="dangling", ref="missing")
    assert _ids(ref_chain(data, "dangling")) == ["dangling"]
    emit(data, "task", "agent", {}, id="a", ref="b")
    emit(data, "reply", "agent", {}, id="b", ref="a")
    assert _ids(ref_chain(data, "a")) == ["b", "a"]
    with pytest.raises(EventError):
        ref_chain(data, "no-such-id")


# --- utterance backfill and live pipeline -----------------------------------
def test_backfill_utterances_is_idempotent(tmp_path):
    store = Store(tmp_path / "data")
    morning = "2026-11-14T12:00:00+00:00"
    evening = "2026-11-15T12:00:00+00:00"
    store.append(_record("m", morning, "早上"))
    store.append(_record("e", evening, "谢谢观看", status="rejected"))
    assert list(replay(store.root)) == []
    assert backfill_utterances(store.root, since=datetime.fromisoformat(evening)) == {"wrote": 1, "skipped": 0}
    assert _ids(replay(store.root)) == ["e"]
    assert backfill_utterances(store.root) == {"wrote": 1, "skipped": 1}
    events = list(replay(store.root, types="utterance"))
    assert _ids(events) == ["m", "e"]
    by_id = {e["id"]: e for e in events}
    assert by_id["m"]["source"] == "transcribe" and by_id["m"]["ref"] is None
    assert by_id["m"]["data"]["text"] == "早上" and by_id["m"]["data"]["status"] == "ok"
    assert by_id["e"]["data"]["status"] == "rejected"
    assert by_id["m"]["ts"] == datetime.fromisoformat(morning).astimezone().isoformat(timespec="milliseconds")
    assert backfill_utterances(store.root) == {"wrote": 0, "skipped": 2}
    # same window as a datetime, passed the way replay accepts an ISO string
    assert backfill_utterances(store.root, since="2026-11-15T12:00:00+00:00") == {"wrote": 0, "skipped": 1}
    assert len(list(replay(store.root))) == 2
    built = utterance_event(_record("m", morning, "早上"))
    assert built["id"] == "m" and built["type"] == "utterance" and built["data"]["text"] == "早上"


def test_if_absent_repairs_a_torn_complete_line(tmp_path):
    """JSON written, newline not: the id is already there, but readers must not skip the line forever."""
    data = tmp_path / "data"
    ev = make_event("utterance", "transcribe", {"text": "hello"}, id="u1",
                    ts=datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc))
    day = datetime.fromisoformat(ev["ts"]).astimezone().strftime("%Y-%m-%d")
    path = data / "events" / f"{day}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(json.dumps(ev, ensure_ascii=False).encode())
    assert list(replay(data)) == []
    assert append(data, ev, if_absent=True) is None
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert _ids(replay(data)) == ["u1"]
    assert append(data, ev, if_absent=True) is None
    assert raw == path.read_bytes()


def test_backfill_repairs_a_torn_utterance_without_duplicating(tmp_path):
    store = Store(tmp_path / "data")
    start = "2026-11-14T12:00:00+00:00"
    rec = _record("m", start, "早上")
    store.append(rec)
    ev = utterance_event(rec)
    day = datetime.fromisoformat(ev["ts"]).astimezone().strftime("%Y-%m-%d")
    path = store.root / "events" / f"{day}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(json.dumps(ev, ensure_ascii=False).encode())
    assert list(replay(store.root)) == []
    assert backfill_utterances(store.root) == {"wrote": 0, "skipped": 1}
    assert _ids(replay(store.root)) == ["m"]
    assert path.read_bytes().count(b"\n") == 1
    assert backfill_utterances(store.root) == {"wrote": 0, "skipped": 1}


def test_unicode_line_separators_stay_inside_one_record(tmp_path):
    data = tmp_path / "data"
    text = "a\u2028b\u2029c\u0085d"
    ts = "2026-11-15T12:00:00.000+00:00"
    hand = {"id": "hand", "ts": ts, "type": "screen", "source": "screen", "ref": None, "data": {"text": text}}
    path = data / "events" / "x.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes((json.dumps(hand, ensure_ascii=False) + "\n").encode())
    assert list(replay(data))[0]["data"]["text"] == text

    emit(data, "screen", "screen", {"text": text}, id="esc", ts=datetime.fromisoformat(ts))
    esc_line = next(line for p in (data / "events").glob("*.jsonl") for line in p.read_bytes().split(b"\n")
                    if b'"id": "esc"' in line)
    assert b"\xe2\x80\xa8" not in esc_line and b"\xe2\x80\xa9" not in esc_line and b"\xc2\x85" not in esc_line
    by_id = {e["id"]: e for e in replay(data)}
    assert by_id["hand"]["data"]["text"] == text and by_id["esc"]["data"]["text"] == text


def test_pipeline_emits_utterance_without_changing_transcripts(tmp_path):
    from audi27.pipeline import Pipeline
    from audi27.segmenter import Segment
    from audi27.store import parse_when
    from audi27.transcribers import Piece, Transcriber, Transcript
    import numpy as np

    class Fake(Transcriber):
        name, model = "fake", "m"

        def __init__(self):
            self.done = False

        def transcribe(self, wav, language=None, prompt=None):
            self.done = True
            return Transcript([Piece(" deploy the fetcher ", 0, 2, -0.2, 0.01, 1.2)], "en", 0.99, "fake", "m")

    from audi27 import config as C
    cfg = C.Config()
    store = Store(tmp_path / "data", "suspect")
    p = Pipeline(cfg, [Fake()], store)
    ts = parse_when("2026-11-14T10:00:00").timestamp()
    store.spool_wav(Segment(np.zeros(SR * 2, np.int16), ts, SR, 2.0))
    p.drain()
    rec = json.loads((store.transcripts / "2026-11-14.jsonl").read_text().splitlines()[0])
    assert rec["text"] == "deploy the fetcher" and "type" not in rec
    assert (store.transcripts / "2026-11-14.md").read_text() == "- 10:00:00 deploy the fetcher\n"
    events = list(replay(store.root))
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "utterance" and ev["source"] == "transcribe" and ev["ref"] is None
    assert ev["id"] == rec["id"] and ev["data"]["text"] == rec["text"] and ev["data"]["status"] == "ok"
    assert store.pending() == []


def test_spool_left_after_the_transcript_emits_one_utterance(tmp_path):
    """Crash after the transcript is durable and before the spool file is removed: one utterance, no second transcript."""
    from audi27 import config as C
    from audi27.pipeline import Pipeline
    from audi27.segmenter import Segment
    from audi27.store import _dt, search
    from audi27.transcribers import Transcriber
    import numpy as np

    class Boom(Transcriber):
        name, model = "fake", "m"

        def transcribe(self, wav, language=None, prompt=None):
            raise AssertionError("transcript already stored")

    ts = 1_800_000_000.0
    store = Store(tmp_path / "data", "none")
    wav = store.spool_wav(Segment(np.zeros(SR, np.int16), ts, SR, 1.0))
    original = wav.read_bytes()
    store.append(_record(wav.stem, _dt(ts).isoformat(timespec="milliseconds"), "kept"))
    p = Pipeline(C.Config(), [Boom()], store)
    p.drain()
    assert [r["text"] for r in search(store.root, ["kept"])] == ["kept"]
    assert _ids(replay(store.root)) == [wav.stem]
    assert store.pending() == []
    wav.write_bytes(original)  # same crash window again, after the utterance exists
    p.drain()
    assert [r["text"] for r in search(store.root, ["kept"])] == ["kept"]
    assert _ids(replay(store.root)) == [wav.stem]
    assert store.pending() == []


def test_event_failure_does_not_drop_the_transcript(tmp_path):
    from audi27.pipeline import Pipeline
    from audi27.segmenter import Segment
    from audi27.store import search
    from audi27.transcribers import Piece, Transcriber, Transcript
    import numpy as np
    from audi27 import config as C

    class Fake(Transcriber):
        name, model = "fake", "m"

        def transcribe(self, wav, language=None, prompt=None):
            return Transcript([Piece(" kept ", 0, 2, -0.2, 0.01, 1.2)], "en", 0.99, "fake", "m")

    store = Store(tmp_path / "data", "none")
    (store.root / "events").write_text("not a directory\n", encoding="utf-8")
    p = Pipeline(C.Config(), [Fake()], store)
    store.spool_wav(Segment(np.zeros(SR, np.int16), 1_800_000_000.0, SR, 1.0))
    p.drain()
    assert [r["text"] for r in search(store.root, ["kept"])] == ["kept"]
    assert store.pending() == [] and list(store.failed.glob("*.wav")) == []


# --- CLI --------------------------------------------------------------------
def test_cli_replay_backfill_and_chain(tmp_path, capsys):
    data = tmp_path / "data"
    store = Store(data)
    store.append(_record("u", "2026-11-14T12:00:00+00:00", "hello connector"))
    assert main(["events", "backfill", "--data-dir", str(data), "--since", "2026-11-14", "--until", "2026-11-14"]) == 0
    assert "wrote 1" in capsys.readouterr().out
    assert main(["events", "replay", "--data-dir", str(data), "--since", "2h", "--type", "utterance"]) == 0
    # the record is in 2026, not the last 2 hours
    assert capsys.readouterr().out == ""
    rc = main(["events", "replay", "--data-dir", str(data), "--since", "2026-11-14", "--until", "2026-11-14",
               "--type", "utterance", "--json"])
    assert rc == 0
    ev = json.loads(capsys.readouterr().out.strip())
    assert ev["id"] == "u" and ev["data"]["text"] == "hello connector"
    emit(data, "route", "router", {"agent": "claude"}, id="r", ref="u",
         ts=datetime.fromisoformat("2026-11-14T12:01:00+00:00"))
    assert main(["events", "chain", "r", "--data-dir", str(data), "--json"]) == 0
    chain = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert _ids(chain) == ["u", "r"]
    assert main(["events", "replay", "--since", "not-a-time"]) == 2
    assert main(["events", "chain", "missing", "--data-dir", str(data)]) == 2
