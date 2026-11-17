import json
import wave
from pathlib import Path

import numpy as np
import pytest

from audi27 import config as C
from audi27.pipeline import Pipeline
from audi27.segmenter import Segment, Segmenter
from audi27.store import Store, parse_when, search
from audi27.transcribers import Piece, TranscribeError, Transcriber, Transcript, build
from audi27.validate import validate

SR, FRAME = 16000, 480
rng = np.random.default_rng(0)


def noise(n_frames, amp):
    return [(rng.standard_normal(FRAME) * amp).astype(np.int16) for _ in range(n_frames)]


def run_seg(frames, cfg=None):
    seg = Segmenter(cfg or C.SegmentCfg(), SR, 30)
    out = [s for i, f in enumerate(frames) if (s := seg.feed(f, 1000 + i * 0.03))]
    return out, seg


def T(*pieces, lang_prob=0.99):
    return Transcript(list(pieces), "en", lang_prob, "fake", "m")


def good(text):
    return Piece(text, 0, 2, -0.2, 0.01, 1.2)


# --- segmenter ---------------------------------------------------------------
def test_segmenter_finds_speech_and_timestamps():
    frames = noise(50, 20) + noise(60, 3000) + noise(50, 20)
    segs, _ = run_seg(frames)
    assert len(segs) == 1
    s = segs[0]
    assert 1.75 <= s.voiced_s <= 2.0
    assert abs(s.start_ts - (1000 + 50 * 0.03)) < 0.35  # speech onset minus pre-roll
    assert 2.0 < s.duration < 3.2


def test_segmenter_ignores_silence_and_short_blips():
    segs, seg = run_seg(noise(100, 20) + noise(7, 3000) + noise(100, 20))
    assert segs == [] and seg.dropped_short == 1
    assert run_seg(noise(300, 20))[0] == []
    assert run_seg([np.zeros(FRAME, np.int16)] * 300)[0] == []


def test_segmenter_hard_cut():
    segs, _ = run_seg(noise(10, 20) + noise(1200, 3000))  # 36 s of continuous speech
    assert len(segs) == 1 and segs[0].duration <= 30.01


# --- validation --------------------------------------------------------------
def test_validate_ok():
    v = validate(T(good(" Let's ship the connector fix today.")), 2.5, C.ValidateCfg())
    assert v.status == "ok" and v.text == "Let's ship the connector fix today."


@pytest.mark.parametrize("pieces,reason", [
    ([], "empty"),
    ([Piece(" Thanks for watching!", 0, 2, -0.3, 0.1, 1.0)], "known_hallucination"),
    ([Piece("谢谢观看", 0, 2, -0.3, 0.1, 1.0)], "known_hallucination"),
    ([Piece("字幕由Amara.org社区提供", 0, 2, -0.3, 0.1, 1.0)], "known_hallucination"),
    ([Piece(" something", 0, 2, -1.4, 0.9, 1.0)], "piece_dropped:no_speech"),
    ([Piece(" la la la la la la la la", 0, 2, -0.3, 0.1, 3.5)], "piece_dropped:high_compression_ratio"),
    ([Piece("好的好的好的好的好的好的", 0, 2, -0.3, 0.1, 1.5)], "repetition_loop"),
])
def test_validate_rejects(pieces, reason):
    v = validate(T(*pieces), 2.0, C.ValidateCfg())
    assert v.status == "rejected" and any(r.startswith(reason) for r in v.reasons)


def test_validate_suspects():
    cfg = C.ValidateCfg()
    assert validate(T(Piece(" maybe this", 0, 2, -1.3, 0.2, 1.0)), 2, cfg).status == "suspect"
    assert validate(T(good(" hello there my friend")), 2, cfg, prev_text="Hello there my friend").status == "suspect"
    assert validate(T(good(" the quick brown fox jumps over the lazy dog near a river bank")), 1.0, cfg).status == "suspect"
    assert validate(T(good(" hello there"), lang_prob=0.3), 2, cfg).status == "suspect"
    # a hallucination phrase inside a real sentence is not rejected
    assert validate(T(good(" He said thanks for watching the kids yesterday afternoon.")), 4, cfg).status == "ok"


def test_validate_prompt_echo_and_partial_drop():
    cfg = C.ValidateCfg()
    assert validate(T(good(" Vanta, Connectors")), 2, cfg, prompt="Vanta, Connectors, Fetcher").status == "rejected"
    v = validate(T(good(" real words here."), Piece(" ghost", 2, 4, -1.5, 0.95, 1.0)), 4, cfg)
    assert v.status == "ok" and v.text == "real words here." and "piece_dropped:no_speech" in v.reasons


def test_validate_without_metrics():
    v = validate(T(Piece("plain text from an api")), 2, C.ValidateCfg())
    assert v.status == "ok" and "no_confidence_metrics" in v.reasons


# --- pipeline + store + search ---------------------------------------------
class Fake(Transcriber):
    name = "fake"
    model = "m"

    def __init__(self, replies):
        self.replies = list(replies)

    def transcribe(self, wav, language=None, prompt=None):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return T(*r)


def spool(store, ts, seconds=2.0):
    return store.spool_wav(Segment(np.zeros(int(SR * seconds), np.int16), ts, SR, seconds))


def make(tmp_path, *backends, keep="suspect"):
    cfg = C.Config()
    store = Store(tmp_path / "data", keep)
    return Pipeline(cfg, list(backends), store), store


def test_pipeline_records_all_statuses(tmp_path):
    p, store = make(tmp_path, Fake([[good(" deploy the fetcher at noon")], [Piece(" Thanks for watching!", 0, 2, -0.3, 0.1, 1)],
                                    [Piece(" unsure words", 0, 2, -1.2, 0.1, 1)]]))
    t0 = parse_when("2026-11-14T10:00:00").timestamp()
    for i in range(3):
        spool(store, t0 + i * 10)
    p.drain()
    lines = [json.loads(x) for x in (store.transcripts / "2026-11-14.jsonl").read_text().splitlines()]
    assert [r["status"] for r in lines] == ["ok", "rejected", "suspect"]
    assert lines[0]["audio"] is None and lines[1]["audio"] and (store.root / lines[1]["audio"]).exists()
    assert store.pending() == []
    md = (store.transcripts / "2026-11-14.md").read_text()
    assert md == "- 10:00:00 deploy the fetcher at noon\n- 10:00:20 [?] unsure words\n"
    assert [r["text"] for r in search(store.root, ["FETCHER", "noon"])] == ["deploy the fetcher at noon"]
    assert len(search(store.root, [])) == 2 and len(search(store.root, [], statuses=("ok", "suspect", "rejected"))) == 3
    assert search(store.root, ["fetcher"], since=parse_when("2026-11-15")) == []
    assert len(search(store.root, ["fetcher"], until=parse_when("2026-11-14", end_of_day=True))) == 1
    ctx = search(store.root, ["fetcher"], context=1)
    assert [r["match"] for r in ctx] == [True, False]


def test_backend_failure_never_writes_text(tmp_path):
    p, store = make(tmp_path, Fake([TranscribeError("boom")]))
    wav = spool(store, 1_800_000_000)
    p.drain()
    assert store.pending() == [wav] and not list(store.transcripts.glob("*.jsonl"))
    p.transcribers = [Fake([[good(" recovered text")]])]
    p.drain()
    assert store.pending() == [] and search(store.root, ["recovered"])


def test_fallback_backend(tmp_path):
    p, store = make(tmp_path, Fake([TranscribeError("local down")]), Fake([[good(" from fallback")]]))
    spool(store, 1_800_000_000)
    p.drain()
    assert search(store.root, ["fallback"])[0]["backend"] == "fake"


def test_worker_retries_then_fails(tmp_path):
    p, store = make(tmp_path, Fake([TranscribeError("x")]))
    p.cfg.transcribe.max_attempts = 1
    wav = spool(store, 1_800_000_000)
    p.q.put((wav, 0)); p.q.put(None)
    p._worker()
    assert store.pending() == [] and (store.failed / wav.name).exists()
    assert "x" in (store.failed / wav.name).with_suffix(".err").read_text()
    assert store.requeue_failed() == 1 and store.pending() == [wav]


def test_crash_between_record_and_cleanup_is_not_duplicated(tmp_path):
    p, store = make(tmp_path, Fake([[good(" only once")], [good(" only once")]]))
    wav = spool(store, 1_800_000_000)
    data = wav.read_bytes()
    p.drain()
    wav.write_bytes(data)  # simulate: record appended, then crash before the spool file was removed
    p.drain()
    assert len(search(store.root, ["once"])) == 1 and store.pending() == []


def test_corrupt_spool_file_goes_to_failed(tmp_path):
    p, store = make(tmp_path, Fake([]))
    (store.spool / "1800000000000.wav").write_bytes(b"not a wav")
    p.drain()
    assert store.pending() == [] and (store.failed / "1800000000000.wav").exists()


def test_torn_jsonl_line_is_skipped(tmp_path):
    p, store = make(tmp_path, Fake([[good(" intact line")]]))
    spool(store, parse_when("2026-11-14T10:00:00").timestamp())
    p.drain()
    with open(store.transcripts / "2026-11-14.jsonl", "a") as f:
        f.write('{"id": "torn", "start": "2026-11-')
    assert len(search(store.root, [])) == 1


def test_keep_audio_modes(tmp_path):
    p, store = make(tmp_path, Fake([[good(" keep me")]]), keep="all")
    spool(store, 1_800_000_000)
    p.drain()
    rec = search(store.root, ["keep"])[0]
    with wave.open(str(store.root / rec["audio"])) as w:
        assert w.getframerate() == SR and w.getnframes() == SR * 2


# --- config ------------------------------------------------------------------
def test_config(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[segment]\nend_silence_ms = 500\n[backend.openai]\nmodel = "x"\n')
    cfg = C.load(f)
    assert cfg.segment.end_silence_ms == 500 and cfg.backend["openai"] == {"model": "x"}
    f.write_text("[segment]\nend_silense_ms = 500\n")
    with pytest.raises(C.ConfigError):
        C.load(f)
    assert C.load(Path(__file__).parent.parent / "config.example.toml").transcribe.backends == ["mlx_whisper"]
    with pytest.raises(TranscribeError):
        build("nope")
    with pytest.raises(TranscribeError):
        build("openai", {"api_key_env": "AUDI27_NO_SUCH_KEY"})


# --- status + dashboard -------------------------------------------------------
def _get(port, path, headers=None):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_status_and_dashboard(tmp_path):
    import threading
    import time

    from audi27.status import read_status, write_status
    from audi27.web import make_server

    p, store = make(tmp_path, Fake([[good(" 部署 fetcher")], [Piece(" Thanks for watching!", 0, 2, -0.3, 0.1, 1)]]))
    t0 = parse_when("2026-11-14T10:00:00").timestamp()
    spool(store, t0); spool(store, t0 + 10)
    p.drain()
    (tmp_path / "secret.wav").write_bytes(b"RIFFsecret")

    assert read_status(store.root) == {"running": False, "age": None, "status": None}
    write_status(store.root, {"updated": time.time(), "started": time.time() - 5, "levels": [0.1, 0.9], "speaking": True})
    assert read_status(store.root)["running"] is True
    write_status(store.root, {"updated": time.time() - 60})
    assert read_status(store.root)["running"] is False
    write_status(store.root, {"updated": time.time(), "stopped": True})
    assert read_status(store.root)["running"] is False

    srv = make_server(store.root, port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, body, _ = _get(port, "/")
        assert code == 200 and b"<canvas" in body
        write_status(store.root, {"updated": time.time(), "started": time.time(), "levels": [0.5], "speaking": False})
        s = json.loads(_get(port, "/api/status")[1])
        assert s["running"] is True and s["spool"] == 0 and s["failed"] == 0 and s["status"]["levels"] == [0.5]
        assert json.loads(_get(port, "/api/days")[1]) == {"days": ["2026-11-14"]}
        recs = json.loads(_get(port, "/api/records?date=2026-11-14")[1])["records"]
        assert [r["status"] for r in recs] == ["ok", "rejected"]
        assert len(json.loads(_get(port, "/api/records?q=FETCHER")[1])["records"]) == 1
        assert json.loads(_get(port, "/api/records?date=2026-11-13")[1])["records"] == []
        assert _get(port, "/api/records?date=bad")[0] == 400
        audio = "/" + recs[1]["audio"]
        code, body, hdr = _get(port, audio)
        assert code == 200 and body[:4] == b"RIFF" and hdr["Content-Type"] == "audio/wav"
        code, part, hdr = _get(port, audio, {"Range": "bytes=0-3"})
        assert code == 206 and part == b"RIFF" and hdr["Content-Range"].startswith("bytes 0-3/")
        for bad in ("/audio/../../secret.wav", "/audio/%2e%2e/%2e%2e/secret.wav", "/audio/../status.json", "/nope"):
            assert _get(port, bad)[0] == 404
    finally:
        srv.shutdown(); srv.server_close()


def test_run_loop_writes_heartbeat_and_transcribes(tmp_path, monkeypatch):
    import audi27.capture as capture
    from audi27.status import read_status

    frames = noise(40, 20) + noise(60, 3000) + noise(40, 20)

    class FakeCapture:
        dropped_frames = overflows = 0

        def __init__(self, cfg):
            pass

        def frames(self, stop):
            for i, f in enumerate(frames):
                yield f, 1_800_000_000 + i * 0.03

    monkeypatch.setattr(capture, "Capture", FakeCapture)
    p, store = make(tmp_path, Fake([[good(" heard through the run loop")]]))
    p.run()
    assert search(store.root, ["run", "loop"]) and store.pending() == []
    s = read_status(store.root)
    assert s["running"] is False and s["status"]["stopped"] is True and s["status"]["counts"]["ok"] == 1
    assert s["status"]["backends"] == ["fake"]


def test_mlx_backend_maps_segments_and_wraps_errors(tmp_path, monkeypatch):
    import sys
    import types

    calls = {}

    def fake_transcribe(audio, **kw):
        calls["audio"], calls["kw"] = audio, kw
        if kw["path_or_hf_repo"] == "broken":
            raise RuntimeError("metal exploded")
        return {"text": " hi", "language": "zh", "segments": [
            {"start": 0.0, "end": 1.5, "text": " 你好 world", "avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.1}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    store = Store(tmp_path / "data")
    wav = store.spool_wav(Segment(np.full(SR, 16384, np.int16), 1_800_000_000, SR, 1.0))
    t = build("mlx_whisper").transcribe(wav, None, "Vanta")
    assert t.text == "你好 world" and t.language == "zh" and t.backend == "mlx_whisper"
    assert t.model == "mlx-community/whisper-turbo" and t.pieces[0].avg_logprob == -0.2
    assert calls["audio"].dtype == np.float32 and len(calls["audio"]) == SR and abs(calls["audio"][0] - 0.5) < 1e-6
    assert calls["kw"]["condition_on_previous_text"] is False and calls["kw"]["temperature"] == 0.0
    assert calls["kw"]["language"] is None and calls["kw"]["initial_prompt"] == "Vanta"
    with pytest.raises(TranscribeError, match="metal exploded"):
        build("mlx_whisper", {"model": "broken"}).transcribe(wav)
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with pytest.raises(TranscribeError, match="not installed"):
        build("mlx_whisper")


class LangFake(Transcriber):
    """Auto-detect reports `detected`; forcing a language returns that language's canned result."""
    name = "fake"
    model = "m"

    def __init__(self, detected, forced):
        self.detected, self.forced, self.calls = detected, forced, []

    def transcribe(self, wav, language=None, prompt=None):
        self.calls.append(language)
        lang = language or self.detected
        text, lp = self.forced.get(lang, (" 안녕하세요", -0.9))
        return Transcript([Piece(text, 0, 2, lp, 0.05, 1.1)], None if language else lang, 0.9, "fake", "m")


def test_languages_default_to_zh_and_en(tmp_path):
    assert C.Config().transcribe.languages == ["zh", "en"]
    forced = {"zh": (" 帮我看一下 connector", -0.3), "en": (" bang wo kan yi xia connector", -0.8)}

    # detected language is allowed: a single call, auto-detect, no flag
    f = LangFake("zh", forced)
    p, store = make(tmp_path / "a", f)
    spool(store, 1_800_000_000); p.drain()
    r = search(store.root, [])[0]
    assert f.calls == [None] and r["status"] == "ok" and r["language"] == "zh"

    # "chinese"/"english" names from API backends count as allowed
    f = LangFake("english", {"english": (" plain english here", -0.2)})
    p, store = make(tmp_path / "b", f)
    spool(store, 1_800_000_000); p.drain()
    assert f.calls == [None] and search(store.root, [])[0]["status"] == "ok"

    # detected Korean: force zh and en, keep the more confident, flag as suspect
    f = LangFake("ko", forced)
    p, store = make(tmp_path / "c", f)
    spool(store, 1_800_000_000); p.drain()
    r = search(store.root, [])[0]
    assert f.calls == [None, "zh", "en"]
    assert r["text"] == "帮我看一下 connector" and r["language"] == "zh"
    assert r["status"] == "suspect" and "language_forced:ko->zh" in r["reasons"]

    # one configured language is always forced; an empty list never restricts
    f = LangFake("ko", forced)
    p, store = make(tmp_path / "d", f)
    p.cfg.transcribe.languages = ["en"]
    spool(store, 1_800_000_000); p.drain()
    assert f.calls == ["en"] and search(store.root, [])[0]["language"] == "en"
    f = LangFake("ko", forced)
    p, store = make(tmp_path / "e", f)
    p.cfg.transcribe.languages = []
    spool(store, 1_800_000_000); p.drain()
    assert f.calls == [None] and search(store.root, [])[0]["language"] == "ko"


def test_languages_config_validation(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[transcribe]\nlanguages = ["ZH", "en"]\n')
    assert C.load(f).transcribe.languages == ["zh", "en"]
    f.write_text('[transcribe]\nlanguages = "zh"\n')
    with pytest.raises(C.ConfigError):
        C.load(f)
    f.write_text('[transcribe]\nlanguage = "zh"\n')
    with pytest.raises(C.ConfigError):
        C.load(f)


# --- replay / eval -----------------------------------------------------------
class Canned(Transcriber):
    """Returns a fixed text per (variant, call order); registered in the tests as backend type 'canned'."""
    name = "canned"
    texts = {
        "gold": [" 先把 connector 的重试逻辑改掉", " Remind me to review the pull request.", ""],
        "small": [" 先把 connector 的重试逻辑改调", " Remind me to review the poll request.", " 谢谢观看"],
    }

    def __init__(self, variant="gold", fail_on=-1):
        self.variant, self.model, self.n, self.fail_on = variant, f"canned-{variant}", 0, fail_on

    def transcribe(self, wav, language=None, prompt=None):
        i, self.n = self.n, self.n + 1
        if i == self.fail_on:
            raise TranscribeError("flaky")
        text = self.texts[self.variant][i % 3]
        return Transcript([Piece(text, 0, 2, -0.2, 0.01, 1.1)] if text else [], "zh", 0.9, self.name, self.model)


def test_tokens_and_edit_distance():
    from audi27.evals import edit_distance, tokens

    assert tokens("先把 Connector 的重试，改掉！don't") == ["先", "把", "connector", "的", "重", "试", "改", "掉", "don't"]
    assert tokens("ＡＢＣ　１２３") == ["abc", "123"] and tokens("") == [] and tokens("。，！") == []
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], ["a", "b"]) == 2 and edit_distance(["a"], ["a"]) == 0


def test_eval_replay_and_compare(tmp_path, monkeypatch):
    from audi27 import evals as E
    from audi27.transcribers import REGISTRY
    from audi27.web import make_server

    monkeypatch.setitem(REGISTRY, "canned", f"{__name__}:Canned")
    cfg = C.Config()
    cfg.store.data_dir = str(tmp_path / "data")
    cfg.backend = {"gold": {"type": "canned", "variant": "gold"}, "small": {"type": "canned", "variant": "small"},
                   "flaky": {"type": "canned", "variant": "gold", "fail_on": 1}}
    # live recording: what the "production" model heard (third segment is noise it turned into text)
    live = Fake([[good(" 先把 connector 的重试逻辑改掉")], [good(" Remind me to review the full request.")], [good(" 再见")]])
    store = Store(cfg.store.data_dir, "all")
    p = Pipeline(cfg, [live], store)
    t0 = parse_when("2026-11-14T10:00:00").timestamp()
    for i in range(3):
        spool(store, t0 + i * 10)
    p.drain()
    assert all(r["audio"] for r in search(store.root, []))  # keep_audio=all keeps every clip for replay

    logs = []
    s = E.run_eval(cfg, "gold", log=logs.append)
    assert (s["ran"], s["failed"], s["no_audio"], s["already_done"]) == (3, 0, 0, 0) and len(logs) == 3
    assert E.run_eval(cfg, "gold", log=logs.append)["ran"] == 0          # resumable: nothing left to do
    assert E.run_eval(cfg, "small", label="small-v1", log=logs.append)["ran"] == 3
    s = E.run_eval(cfg, "flaky", log=logs.append)                        # a failed segment is not recorded...
    assert (s["ran"], s["failed"]) == (2, 1) and len(E.load(store.root, "flaky")) == 2
    cfg.backend["flaky"]["fail_on"] = -1
    s = E.run_eval(cfg, "flaky", log=logs.append)                        # ...so the next run picks it up
    assert (s["ran"], s["already_done"]) == (1, 2)
    assert E.labels(store.root) == ["live", "flaky", "gold", "small-v1"]

    r = E.compare(store.root, "gold", ["live", "small-v1"])
    by = {x["label"]: x for x in r["summary"]}
    assert r["ref_segments"] == 3 and r["ref_model"] == "canned-gold"
    # reference tokens: 10 (zh) + 7 (en) + 0 (silence) = 17
    assert by["live"]["ref_tokens"] == 17 and by["live"]["errors"] == 1 + 2   # full/pull + "再见" on silence
    assert by["small-v1"]["errors"] == 1 + 1 + 0   # 调/掉, poll/pull; "谢谢观看" was rejected -> empty -> correct
    assert by["small-v1"]["exact_rate"] == round(1 / 3, 4) and by["small-v1"]["model"] == "canned-small"
    assert by["small-v1"]["rtf"] is not None and by["live"]["rtf"] is None
    assert [x["label"] for x in r["summary"]] == ["small-v1", "live"]          # best first
    assert r["rows"][0]["ref"] == "" and r["rows"][0]["outputs"]["live"]["text"] == "再见"   # worst first
    assert r["rows"][0]["audio"] and r["rows"][0]["outputs"]["small-v1"]["status"] == "rejected"
    assert E.compare(store.root, "gold", since=parse_when("2026-11-15"))["rows"] == []
    # any label can be the reference, including the live output
    assert {x["label"] for x in E.compare(store.root, "live")["summary"]} == {"gold", "small-v1", "flaky"}

    for bad in ("live", "../x", "a b"):
        with pytest.raises(E.EvalError):
            E.run_eval(cfg, "gold", label=bad)
    with pytest.raises(E.EvalError):
        E.compare(store.root, "nope")
    with pytest.raises(TranscribeError):
        E.run_eval(cfg, "undefined-backend", force=True)

    from audi27.cli import main
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(f'[store]\ndata_dir = "{cfg.store.data_dir}"\n')
    assert main(["-c", str(cfgfile), "eval", "compare", "--ref", "gold"]) == 0
    assert main(["-c", str(cfgfile), "eval", "list"]) == 0
    assert main(["-c", str(cfgfile), "eval", "compare", "--ref", "nope"]) == 2

    srv = make_server(store.root, port=0)
    port = srv.server_address[1]
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert json.loads(_get(port, "/api/eval/labels")[1])["labels"] == ["live", "flaky", "gold", "small-v1"]
        d = json.loads(_get(port, "/api/eval/compare?ref=gold")[1])
        assert d["total_rows"] == 3 and len(d["summary"]) == 3
        assert _get(port, "/api/eval/compare?ref=..%2Fx")[0] == 400 and _get(port, "/api/eval/compare")[0] == 400
    finally:
        srv.shutdown(); srv.server_close()


def test_nan_metrics_never_reach_disk_or_the_browser(tmp_path):
    """Regression: mlx-whisper returned avg_logprob=NaN for a "!" segment; the NaN in the JSONL
    made the dashboard's JSON unparseable, so the transcript list stayed empty."""
    import threading

    from audi27.web import make_server

    nan = float("nan")
    v = validate(T(Piece("!", 0, 2, nan, 0.0, 0.111)), 1.9, C.ValidateCfg())
    assert v.status == "rejected" and "no_words" in v.reasons and v.avg_logprob is None
    v = validate(T(Piece(" real words", 0, 2, nan, 0.0, float("inf"))), 2, C.ValidateCfg())
    assert v.status == "suspect" and "invalid_confidence_metrics" in v.reasons and v.compression_ratio is None

    p, store = make(tmp_path, Fake([[Piece(" real words", 0, 2, nan, 0.0, 1.0)]]))
    t0 = parse_when("2026-11-14T10:00:00").timestamp()
    spool(store, t0)
    p.drain()
    path = store.transcripts / "2026-11-14.jsonl"
    assert "NaN" not in path.read_text()
    with open(path, "a") as f:  # a line written by the old code
        f.write('{"id": "old", "start": "2026-11-14T11:00:00.000+00:00", "text": "!", "status": "ok", "reasons": [], "avg_logprob": NaN}\n')
    assert [r["avg_logprob"] for r in search(store.root, [])] == [None, None]

    srv = make_server(store.root, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = _get(srv.server_address[1], "/api/records")[1].decode()
        assert "NaN" not in body and len(json.loads(body, parse_constant=lambda c: 1 / 0)["records"]) == 2
    finally:
        srv.shutdown(); srv.server_close()


def test_supervisor_restarts_crashed_child_and_reexecs_on_request(tmp_path):
    import sys
    import threading
    import time

    from audi27.supervisor import supervise

    marks = tmp_path / "marks.txt"
    crasher = [sys.executable, "-c", f"open({str(marks)!r}, 'a').write('x')"]          # exits at once
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    data = tmp_path / "data"
    data.mkdir()
    (data / "restart.request").write_text("stale")   # left over from before this start: must be ignored

    class Reexec(Exception):
        pass

    def reexec():
        raise Reexec

    def request_restart():
        deadline = time.time() + 10
        while time.time() < deadline and (not marks.exists() or len(marks.read_text()) < 2):
            time.sleep(0.05)
        (data / "restart.request").write_text("code changed")

    threading.Thread(target=request_restart, daemon=True).start()
    t0 = time.time()
    with pytest.raises(Reexec):
        supervise(data, {"crasher": crasher, "sleeper": sleeper}, reexec, poll_s=0.05, grace_s=5)
    assert time.time() - t0 < 20
    assert len(marks.read_text()) >= 2                       # the crashed child was started again
    assert not (data / "restart.request").exists()
    state = json.loads((data / "supervisor.json").read_text())
    assert state["crashes"]["crasher"] >= 1 and "sleeper" not in state["crashes"]
    import os
    assert os.environ.pop("AUDI27_RESTARTS") == "1" and os.environ.pop("AUDI27_RESTART_REASON") == "code changed"


def test_a_restart_sync_keeps_the_kws_extra_only_when_enabled(tmp_path, monkeypatch):
    """A pyproject change must not `uv sync --extra dev` alone while the spotter is on."""
    import os
    import subprocess
    import threading
    import time

    from audi27.supervisor import supervise

    proj = tmp_path / "proj"
    proj.mkdir()
    pyproject = proj / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setattr("audi27.supervisor.PROJECT", proj)
    calls = []

    def fake_run(argv, cwd=None, **_kwargs):
        calls.append((list(argv), Path(cwd) if cwd is not None else None))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("audi27.supervisor.subprocess.run", fake_run)
    monkeypatch.setattr("audi27.supervisor.shutil.which", lambda _name: "/usr/bin/uv")

    class Reexec(Exception):
        pass

    def once(text: str):
        calls.clear()
        data = tmp_path / f"data-{abs(hash(text))}"
        data.mkdir()
        cfg = data / "config.toml"
        cfg.write_text(text, encoding="utf-8")

        def request():
            time.sleep(0.15)
            current = pyproject.stat().st_mtime
            os.utime(pyproject, (current + 10, current + 10))
            (data / "restart.request").write_text("deps")

        threading.Thread(target=request, daemon=True).start()
        with pytest.raises(Reexec):
            supervise(
                data, {}, lambda: (_ for _ in ()).throw(Reexec()),
                poll_s=0.05, grace_s=1, config_path=str(cfg))
        os.environ.pop("AUDI27_RESTARTS", None)
        os.environ.pop("AUDI27_RESTART_REASON", None)
        return list(calls)

    assert once("[kws]\nenabled = true\n") == [
        (["uv", "sync", "--extra", "dev", "--extra", "kws"], proj)]
    assert once("[kws]\nenabled = false\n") == [
        (["uv", "sync", "--extra", "dev"], proj)]


def _post(port, path, body, headers=None):
    import urllib.error
    import urllib.request

    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_review_matrix_verdicts_and_run_from_dashboard(tmp_path, monkeypatch):
    import threading
    import time

    from audi27 import evals as E
    from audi27.transcribers import REGISTRY
    from audi27.web import make_server

    monkeypatch.setitem(REGISTRY, "canned", f"{__name__}:Canned")
    data = tmp_path / "data"
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text(f'[store]\ndata_dir = "{data}"\n[backend.gold]\ntype = "canned"\nvariant = "gold"\n'
                       '[backend.small]\ntype = "canned"\nvariant = "small"\n')
    cfg = C.load(cfgfile)
    live = Fake([[good(" 先把 connector 的重试逻辑改掉")], [good(" Remind me to review the full request.")], [good(" 再见")]])
    store = Store(data, "all")
    p = Pipeline(cfg, [live], store)
    t0 = parse_when("2026-11-14T10:00:00").timestamp()
    ids = [spool(store, t0 + i * 10).stem for i in range(3)]
    p.drain()

    srv = make_server(data, port=0, config_path=str(cfgfile))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        names = {b["name"]: b for b in json.loads(_get(port, "/api/eval/backends")[1])["backends"]}
        assert set(names) == {"mlx_whisper", "gold", "small", "@correct"} and names["gold"]["type"] == "canned"
        plan = json.loads(_get(port, "/api/eval/plan?label=gold")[1])
        assert plan["todo"] == 3 and plan["todo_seconds"] == 6.0 and plan["already_done"] == 0

        # refused: other origin, non-JSON, unknown backend, bad label
        assert _post(port, "/api/eval/run", {"backend": "gold"}, {"Origin": "http://evil.example"})[0] == 403
        assert _post(port, "/api/eval/run", {"backend": "gold"}, {"Content-Type": "text/plain"})[0] == 415
        assert _post(port, "/api/eval/run", {"backend": "nope"})[0] == 400
        assert _post(port, "/api/eval/run", {"backend": "gold", "label": "../x"})[0] == 400
        assert not (data / "evals").exists()

        for backend in ("gold", "small"):
            code, job = _post(port, "/api/eval/run", {"backend": backend}, {"Origin": f"http://127.0.0.1:{port}"})
            assert code == 200 and job["running"] is True
            for _ in range(200):
                job = json.loads(_get(port, "/api/eval/job")[1])
                if not job["running"]:
                    break
                time.sleep(0.05)
            assert job.get("done") and job["ran"] == 3 and not job.get("error")
        assert json.loads(_get(port, "/api/eval/plan?label=gold")[1])["todo"] == 0

        m = json.loads(_get(port, "/api/eval/matrix")[1])
        assert m["base"] == "gold" and [x["label"] for x in m["labels"]] == ["live", "gold", "small"]
        rows = {r["id"]: r for r in m["rows"]}
        assert [r["id"] for r in m["rows"]] == ids[::-1]                      # newest first
        assert rows[ids[0]]["cells"]["live"]["text"] == "先把 connector 的重试逻辑改掉" and not rows[ids[0]]["agree"]
        assert rows[ids[2]]["cells"]["small"] == {"text": "", "status": "rejected", "raw": "谢谢观看",
                                                  "reasons": ["known_hallucination:谢谢观看"]}
        by = {x["label"]: x for x in m["labels"]}
        assert by["live"]["same_rate"] == round(1 / 3, 4) and by["small"]["error_rate"] == round(2 / 17, 4)
        assert by["gold"]["judged"] == 0 and by["gold"]["correct_rate"] is None

        # judge: segment 0 -> gold's text is right; segment 1 -> nobody; segment 2 -> silence ("") is right
        assert _post(port, "/api/eval/verdict", {"id": ids[0], "labels": ["gold"], "texts": ["先把 connector 的重试逻辑改掉"]})[0] == 200
        assert _post(port, "/api/eval/verdict", {"id": ids[1], "labels": [], "texts": []})[0] == 200
        assert _post(port, "/api/eval/verdict", {"id": ids[2], "labels": ["gold"], "texts": [""]})[0] == 200
        assert _post(port, "/api/eval/verdict", {"id": "../x", "texts": []})[0] == 400
        assert _post(port, "/api/eval/verdict", {"id": ids[0], "texts": []}, {"Origin": "http://evil.example"})[0] == 403
        m = json.loads(_get(port, "/api/eval/matrix?base=live")[1])
        assert m["base"] == "live"
        by = {x["label"]: x for x in m["labels"]}
        rows = {r["id"]: r for r in m["rows"]}
        # live matched gold on segment 0 (same text => also correct); hallucinated on the silent one
        assert (by["live"]["correct"], by["live"]["judged"]) == (1, 3)
        assert (by["gold"]["correct"], by["small"]["correct"]) == (2, 1)
        assert rows[ids[0]]["cells"]["live"]["correct"] is True and rows[ids[0]]["cells"]["small"]["correct"] is False
        assert rows[ids[2]]["cells"]["small"]["correct"] is True and rows[ids[2]]["cells"]["live"]["correct"] is False
        assert all(r["judged"] for r in m["rows"])

        # a label run later is scored against the texts already judged correct, with no new clicks
        (data / "evals" / "tuned.jsonl").write_text(json.dumps(
            {"id": ids[0], "start": rows[ids[0]]["start"], "text": "先把 Connector 的重试逻辑改掉！", "status": "ok"}, ensure_ascii=False) + "\n")
        by = {x["label"]: x for x in json.loads(_get(port, "/api/eval/matrix")[1])["labels"]}
        assert (by["tuned"]["correct"], by["tuned"]["judged"]) == (1, 1)

        assert _post(port, "/api/eval/verdict", {"id": ids[1], "cleared": True})[0] == 200
        m = json.loads(_get(port, "/api/eval/matrix")[1])
        assert {r["id"]: r["judged"] for r in m["rows"]}[ids[1]] is False
        assert len(E.load_verdicts(data)) == 2
    finally:
        srv.shutdown(); srv.server_close()


def test_dotenv_and_openai_minimal_retry(tmp_path, monkeypatch):
    import sys
    import types

    from audi27.cli import _load_dotenv

    env = tmp_path / ".env"
    env.write_text('# comment\nAUDI27_T1=abc\nexport AUDI27_T2="quoted value"\nAUDI27_T3=keep\nbroken line\n')
    monkeypatch.delenv("AUDI27_T1", raising=False); monkeypatch.delenv("AUDI27_T2", raising=False)
    monkeypatch.setenv("AUDI27_T3", "already set")
    _load_dotenv(env)
    import os
    assert (os.environ["AUDI27_T1"], os.environ["AUDI27_T2"], os.environ["AUDI27_T3"]) == ("abc", "quoted value", "already set")
    monkeypatch.delenv("AUDI27_T1"); monkeypatch.delenv("AUDI27_T2")
    _load_dotenv(tmp_path / "missing.env")

    calls = []

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._body, self.text = code, body, json.dumps(body)

        def json(self):
            return self._body

    def post(url, headers, data, files, timeout):
        calls.append(dict(data))
        if "temperature" in data:
            return Resp(400, {"error": {"message": "Unsupported parameter: temperature"}})
        return Resp(200, {"text": "你好 world"})

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=post, HTTPError=OSError))
    monkeypatch.setenv("AUDI27_KEY", "k")
    store = Store(tmp_path / "data")
    wav = spool(store, 1_800_000_000)
    t = build("gold", {"type": "openai", "model": "gpt-transcribe", "response_format": "json", "api_key_env": "AUDI27_KEY"})
    out = t.transcribe(wav, "zh", "hint")
    assert out.text == "你好 world" and out.model == "gpt-transcribe" and out.pieces[0].avg_logprob is None
    assert calls[0]["prompt"] == "hint" and calls[1] == {"model": "gpt-transcribe", "response_format": "json"}


def test_xai_backend(tmp_path, monkeypatch):
    import sys
    import types

    seen = {}

    class Resp:
        def __init__(self, code, body, text=None):
            self.status_code, self._body, self.text = code, body, text or json.dumps(body)

        def json(self):
            if self._body is None:
                raise ValueError("not json")
            return self._body

    replies = [Resp(200, {"text": " 你好 Grok ", "language": "zh", "duration": 2.0}), Resp(200, {"text": "  "}),
               Resp(401, {"error": "bad key"}), Resp(200, {"words": []}), Resp(200, None, "<html>")]

    def post(url, headers, data, files, timeout):
        seen.update(url=url, auth=headers["Authorization"], data=dict(data), name=files["file"][0], ctype=files["file"][2])
        return replies.pop(0)

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=post, HTTPError=OSError))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(TranscribeError, match="XAI_API_KEY"):
        build("gold", {"type": "xai"})
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    t = build("gold", {"type": "xai"})
    store = Store(tmp_path / "data")
    wav = spool(store, 1_800_000_000)
    out = t.transcribe(wav, "zh", "ignored hint")
    assert out.text == "你好 Grok" and out.language == "zh" and out.backend == "xai" and out.model == "grok-voice-transcribe-2.0"
    assert seen == {"url": "https://api.x.ai/v1/stt", "auth": "Bearer xai-test", "data": {"model": "grok-voice-transcribe-2.0"},
                    "name": wav.name, "ctype": "audio/wav"}
    assert t.transcribe(wav).pieces == []
    for pattern in ("HTTP 401", "unexpected body", "unexpected body"):
        with pytest.raises(TranscribeError, match=pattern):
            t.transcribe(wav)


def test_dotenv_edits_are_picked_up(tmp_path, monkeypatch):
    import os

    env = tmp_path / ".env"
    monkeypatch.delenv("AUDI27_T9", raising=False)
    env.write_text("AUDI27_T9=typo\n")
    C.load_dotenv(env)
    env.write_text("AUDI27_T9=fixed\n")
    C.load_dotenv(env)
    assert os.environ.pop("AUDI27_T9") == "fixed"
    C._FROM_DOTENV.discard("AUDI27_T9")


# --- fillers, context, correction ---------------------------------------------
CONTEXT = """# 当前上下文
正在做 audi27：常驻录音转写，用 Claude Code 和 Grokbot 调度 agent，对比页用 gold 做基准。

## 术语
- Grokbot = Grog Bot, Grobbbord, grog 机器人
- Claude Code = Cloud Code, Crowed Code
- Grok = grog
- notification
- connector = 连接器

## 备注
- 这一行不是术语
"""


class FakeLLM:
    model = "fake-llm"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, max_tokens):
        from audi27.correct import CorrectorError

        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise CorrectorError(str(r))
        return r


def test_textnorm_and_filler_validation():
    from audi27.textnorm import cmp_tokens, is_filler_only, raw_tokens

    assert raw_tokens("嗯，他他慢慢地说 um OK") == ["嗯", "他", "他", "慢", "慢", "地", "说", "um", "ok"]
    # fillers and 儿 dropped, 的/地/得 merged, immediate repeats collapsed: both sides compare equal
    assert cmp_tokens("呃，他他有很多件事儿，会会慢慢地念") == cmp_tokens("他有很多件事,会慢慢的念")
    assert cmp_tokens("我Mac上") == cmp_tokens("我 Mac 上") and cmp_tokens("改掉") != cmp_tokens("改调")
    assert is_filler_only("嗯") and is_filler_only("Mm-hmm.") and is_filler_only("Um, uh…") and is_filler_only("嗯嗯 啊")
    assert not is_filler_only("嗯，好的") and not is_filler_only("") and not is_filler_only("。")
    cfg = C.ValidateCfg()
    for text in ("嗯", " um", "Mmm.", "Mm-hmm."):
        v = validate(T(Piece(text, 0, 1, -0.3, 0.1, 1.0)), 1.0, cfg)
        assert v.status == "rejected" and v.reasons == ["filler_only"], text
    assert validate(T(good(" 嗯，我觉得可以")), 2.0, cfg).status == "ok"
    v = validate(T(Piece("词曲 李宗盛", 0, 2, -0.3, 0.1, 1.0)), 2.0, cfg)
    assert v.status == "rejected" and v.reasons[0].startswith("known_hallucination")


def test_context_parsing_and_aliases(tmp_path):
    from audi27.correct import apply_aliases, load_context, parse_context

    ctx = parse_context(CONTEXT)
    assert ctx.terms == ["Grokbot", "Claude Code", "Grok", "notification", "connector"]   # only bullets under 术语
    assert "audi27" in ctx.vocab and "gold" in ctx.vocab and "grobbbord" in ctx.vocab
    fix = lambda t: apply_aliases(t, ctx)[0]
    assert fix("我在手机和 Mac 上都装了 Grog Bot。") == "我在手机和 Mac 上都装了 Grokbot。"
    assert fix("同时有Grobbbord的时候") == "同时有Grokbot的时候"
    assert fix("我有一个 grog key,换成 grog 行不行?") == "我有一个 Grok key,换成 Grok 行不行?"
    assert fix("那个Crowed Code的令牌，还有 cloud  code") == "那个Claude Code的令牌，还有 Claude Code"
    assert fix("这个连接器的重试") == "这个connector的重试"
    assert fix("groggy 和 cloudcoder 不该被改") == "groggy 和 cloudcoder 不该被改"       # word boundaries
    text, changes = apply_aliases("Grog Bot 用 grog", ctx)
    assert text == "Grokbot 用 Grok" and [c["to"] for c in changes] == ["Grokbot", "Grok"] and changes[0]["by"] == "alias"

    f = tmp_path / "context.md"
    assert load_context(f).terms == []                       # missing file: no context, no error
    f.write_text(CONTEXT)
    assert len(load_context(f).aliases) == 7
    import os
    f.write_text("## terms\n- Foo = fu\n"); os.utime(f, (1, 1))   # edits are picked up via mtime
    assert load_context(f).terms == ["Foo"]


def test_correction_guards():
    from audi27.correct import check, parse_context

    ctx = parse_context(CONTEXT)
    ok = lambda o, p: check(o, p, ctx, 0.3)
    assert ok("感觉这个 notify cation 被发送了两次。", "感觉这个 notification 被发送了两次。") == ("感觉这个 notification 被发送了两次。", "corrected")
    assert ok("把它推到我的听头上的那个通知", "“把它推到我的镜头上的那个通知”")[1] == "corrected"          # quotes stripped
    assert ok("今天下午三点开会讨论进度", "今天下午三点开会讨论进度。") == (None, "unchanged")                   # punctuation only
    assert ok("今天下午三点开会讨论进度", "好的，我会在今天下午三点提醒你开会并整理讨论要点和进度。")[1] == "too_many_changes"
    assert ok("我们今天下午用那个来部署所有的服务", "我们今天下午用 Kubernetes 来部署所有的服务")[1].startswith("invented_term:kubernetes")
    assert ok("我们今天下午用那个来部署所有的服务", "我们今天下午用 Grokbot 来部署所有的服务")[1] == "corrected"                         # term from the context
    assert ok("今天下午三点开会", "修正后：今天下午三点开会\n说明：没有错误")[1] == "not_a_single_line"
    assert ok("今天下午三点开会", "  ")[1] == "empty"


def test_correction_stage_and_pipeline(tmp_path, monkeypatch):
    from audi27.correct import CorrectionStage

    ctxfile = tmp_path / "context.md"
    ctxfile.write_text(CONTEXT)
    cfg = C.Config()
    cfg.correct.context_file = str(ctxfile)
    cfg.store.data_dir = str(tmp_path / "data")

    llm = FakeLLM(["感觉这个 notification 被发送了两次。", "我会帮你把所有通知都整理好然后发一份完整的报告给你。", RuntimeError("down")])
    stage = CorrectionStage(cfg, llm)
    assert stage.prompt_terms() == "Grokbot, Claude Code, Grok, notification, connector"
    text, info = stage.run("感觉这个 notify cation 被发送了两次。", ["前一句"])
    assert text == "感觉这个 notification 被发送了两次。" and info["llm"] == "corrected" and info["changes"][0]["by"] == "llm"
    user = llm.calls[0][1]["content"]
    assert "Grokbot" in user and "前一句" in user and "notify cation" in user
    assert stage.run("感觉这个通知被发送了两次", [])[0] == "感觉这个通知被发送了两次"       # rewrite refused: ASR text stands
    assert stage.run("Grog Bot 的通知被发送了两次", [])[0] == "Grokbot 的通知被发送了两次"  # LLM down: alias still applied
    assert stage.run("嗯好", [])[0] == "嗯好" and len(llm.calls) == 3                       # too short: no LLM call

    # three consecutive failures disable the LLM for the session; aliases keep working
    stage = CorrectionStage(cfg, FakeLLM([RuntimeError("x")] * 3))
    for _ in range(4):
        assert stage.run("Grog Bot 今天下午三点开会", [])[0] == "Grokbot 今天下午三点开会"
    assert stage.disabled_reason == "x" and len(stage.llm.calls) == 3

    # disabled, or no context file: text passes through untouched and the LLM is never built
    cfg2 = C.Config(); cfg2.correct.enabled = False; cfg2.correct.context_file = str(ctxfile)
    assert CorrectionStage(cfg2).run("Grog Bot", []) == ("Grog Bot", {"changes": []})
    assert CorrectionStage(C.Config()).run("Grog Bot 今天下午三点开会", [])[0] == "Grog Bot 今天下午三点开会"
    cfg3 = C.Config(); cfg3.correct.context_file = str(ctxfile); cfg3.correct.llm = False
    assert CorrectionStage(cfg3).run("Grog Bot 今天下午三点开会", [])[0] == "Grokbot 今天下午三点开会"

    # pipeline: corrected text is stored, the recogniser's text is kept, glossary terms reach the prompt
    seen = {}

    class Rec(Fake):
        def transcribe(self, wav, language=None, prompt=None):
            seen["prompt"] = prompt
            return super().transcribe(wav, language, prompt)

    store = Store(cfg.store.data_dir, "all")
    p = Pipeline(cfg, [Rec([[good(" 我现在在手机和 Mac 上都装了 Grog Bot。")], [good(" 感觉这个 notify cation 被发送了两次。")],
                           [Piece(" Thanks for watching!", 0, 2, -0.3, 0.1, 1)], [good(" 嗯")]])], store)
    p.corrector.llm = FakeLLM(["我现在在手机和 Mac 上都装了 Grokbot。", "感觉这个 notification 被发送了两次。"])
    t0 = parse_when("2026-11-14T10:00:00").timestamp()
    for i in range(4):
        spool(store, t0 + i * 10)
    p.drain()
    assert seen["prompt"].endswith("Grokbot, Claude Code, Grok, notification, connector")
    recs = search(store.root, [], statuses=("ok", "suspect", "rejected"))
    assert recs[0]["text"] == "我现在在手机和 Mac 上都装了 Grokbot。" and recs[0]["asr_text"].endswith("Grog Bot。")
    assert recs[0]["corrections"][0] == {"by": "alias", "from": "Grog Bot", "to": "Grokbot"} and "corrected" in recs[0]["reasons"]
    assert recs[1]["text"] == "感觉这个 notification 被发送了两次。" and recs[1]["corrections"][0]["by"] == "llm"
    assert "Grokbot" in p.corrector.llm.calls[1][1]["content"].split("前文")[1]            # previous line passed as context
    assert recs[2]["status"] == "rejected" and "asr_text" not in recs[2]                    # rejected text is never "corrected"
    assert recs[3]["status"] == "rejected" and recs[3]["reasons"] == ["filler_only"]
    assert len(p.corrector.llm.calls) == 2
    assert [r["text"] for r in search(store.root, ["grokbot"])] == [recs[0]["text"]]

    # comparison: the pre-correction text appears as its own column; a corrector-only replay adds another
    from audi27 import evals as E
    m = E.matrix(store.root)
    assert [x["label"] for x in m["labels"]] == ["live", "live-asr"]
    row = [r for r in m["rows"] if r["id"] == recs[0]["id"]][0]
    assert row["cells"]["live-asr"]["text"].endswith("Grog Bot。") and row["cells"]["live"]["text"].endswith("Grokbot。")
    with pytest.raises(E.EvalError):
        E.run_correct(cfg, "live-asr")
    ctxfile.write_text(CONTEXT.replace("- notification", "- notification = notify cation"))
    import os
    os.utime(ctxfile, (2, 2))
    cfg.correct.llm = False
    s = E.run_correct(cfg, "ctx-v2", log=lambda *_: None)
    assert (s["ran"], s["changed"]) == (4, 2) and E.run_correct(cfg, "ctx-v2")["ran"] == 0
    out = E.load(store.root, "ctx-v2")
    assert out[recs[1]["id"]]["text"] == "感觉这个 notification 被发送了两次。" and out[recs[1]["id"]]["corrections"][0]["by"] == "alias"
    assert out[recs[2]["id"]]["status"] == "rejected" and E.plan_correct(store.root, "new")[1]["todo"] == 4


def test_correct_config_and_backends(tmp_path):
    from audi27.correct import CorrectorError, build_corrector

    f = tmp_path / "c.toml"
    f.write_text('[correct]\nllm = false\ncontext_file = "x.md"\n[corrector.local]\ntype = "openai_chat"\nmodel = "qwen"\n')
    cfg = C.load(f)
    assert cfg.correct.llm is False and cfg.correct.context_file == "x.md" and cfg.corrector["local"]["model"] == "qwen"
    assert build_corrector("local", cfg.corrector["local"]).url == "http://127.0.0.1:1234/v1/chat/completions"
    assert build_corrector("mlx_lm", None).model == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    for name, opts in (("nope", None), ("mlx_lm", {"bogus": 1})):
        with pytest.raises(CorrectorError):
            build_corrector(name, opts)
    f.write_text("[correct]\nenable = true\n")
    with pytest.raises(C.ConfigError):
        C.load(f)
