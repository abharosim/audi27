from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

from . import config as C
from .store import Store, parse_when, search

_load_dotenv = C.load_dotenv


def _setup_logging(data_dir: Path, verbose: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(data_dir / "audi27.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def _pipeline(cfg):
    from .pipeline import Pipeline, build_transcribers

    store = Store(cfg.store.data_dir, cfg.store.keep_audio)
    return Pipeline(cfg, build_transcribers(cfg), store)


def cmd_run(args, cfg) -> int:
    _setup_logging(Path(cfg.store.data_dir), args.verbose)

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    _pipeline(cfg).run()
    return 0


def cmd_devices(args, cfg) -> int:
    from .capture import list_devices

    print(list_devices())
    return 0


def cmd_file(args, cfg) -> int:
    """Transcribe existing 16 kHz mono WAV files through the full pipeline (backend + validation test)."""
    _setup_logging(Path(cfg.store.data_dir), args.verbose)
    p = _pipeline(cfg)
    for i, src in enumerate(args.wav):
        ts_ms = int(Path(src).stat().st_mtime * 1000) + i
        shutil.copyfile(src, p.store.spool / f"{ts_ms}.wav")
    p.drain()
    return 0


def cmd_retry(args, cfg) -> int:
    _setup_logging(Path(cfg.store.data_dir), args.verbose)
    p = _pipeline(cfg)
    print(f"requeued {p.store.requeue_failed()} failed segment(s)")
    p.drain()
    return 0


def cmd_eval(args, cfg) -> int:
    from . import evals as E

    root = args.data_dir or cfg.store.data_dir
    cfg.store.data_dir = root
    try:
        since = parse_when(args.since) if getattr(args, "since", None) else None
        until = parse_when(args.until, end_of_day=True) if getattr(args, "until", None) else None
        if args.eval_cmd == "list":
            for name in E.labels(root):
                print(f"{name:20} {len(E.load(root, name)):6d} segments")
        elif args.eval_cmd == "correct":
            s = E.run_correct(cfg, args.label, args.force)
            print(f"label={s['label']}: {s['ran']} segments re-corrected, {s.get('changed', 0)} changed, "
                  f"{s['already_done']} already done" + (f"; LLM unavailable: {s['last_error']}" if s.get("last_error") else ""))
        elif args.eval_cmd == "run":
            s = E.run_eval(cfg, args.backend, args.label, since, until, args.limit, args.force)
            print(f"label={s['label']}: ran {s['ran']}, failed {s['failed']}, already done {s['already_done']}, "
                  f"skipped {s['no_audio']} without audio (of {s['segments']} segments)")
            return 1 if s["failed"] else 0
        else:
            r = E.compare(root, args.ref, args.labels or None, since, until)
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
                return 0
            print(f"reference: {r['ref']} ({r['ref_model'] or '-'}), {r['ref_segments']} segments\n")
            print(f"{'label':20} {'segments':>8} {'error':>8} {'exact':>8} {'rtf':>6}  model")
            for s in r["summary"]:
                if not s["segments"]:
                    print(f"{s['label']:20} {0:8d}  (no segments in common with the reference)")
                    continue
                rtf = "-" if s["rtf"] is None else f"{s['rtf']:.2f}"
                print(f"{s['label']:20} {s['segments']:8d} {s['error_rate']:8.1%} {s['exact_rate']:8.1%} {rtf:>6}  {s['model']}")
            for row in r["rows"][:args.worst]:
                if row["worst"] == 0:
                    break
                print(f"\n{row['start'][:19]}  id={row['id']}  audio={row['audio']}")
                print(f"  {r['ref']:>14} | {row['ref']}")
                for name, o in row["outputs"].items():
                    print(f"  {name:>14} | {o['text']}   [{o['error_rate']:.0%}]")
    except E.EvalError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"bad time: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_supervise(args, cfg) -> int:
    from . import supervisor

    _setup_logging(Path(cfg.store.data_dir), args.verbose)
    supervisor.main(args.config, cfg.store.data_dir, args.host, args.port)
    return 0


def cmd_web(args, cfg) -> int:
    from .web import serve

    serve(args.data_dir or cfg.store.data_dir, args.host, args.port, args.config)
    return 0


def _print_event(ev: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(ev, ensure_ascii=False), flush=True)
        return
    try:
        stamp = datetime.fromisoformat(ev["ts"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        stamp = str(ev.get("ts", ""))
    data = ev.get("data") or {}
    if ev.get("type") == "utterance":
        summary = " ".join(str(data.get("text", "")).split())
    else:
        summary = json.dumps(data, ensure_ascii=False)
        if len(summary) > 160:
            summary = summary[:157] + "..."
    ref = ev.get("ref") or "-"
    print(f"{stamp}  {ev.get('type', ''):<10} {ev.get('source', ''):<12} {ev.get('id', '')}  ref={ref}  {summary}",
          flush=True)


def cmd_events(args, cfg) -> int:
    from .events import EventError, backfill_utterances, follow, ref_chain, replay

    data_dir = args.data_dir or cfg.store.data_dir
    try:
        since = parse_when(args.since) if getattr(args, "since", None) else None
        until = parse_when(args.until, end_of_day=True) if getattr(args, "until", None) else None
    except ValueError as e:
        print(f"bad time: {e}", file=sys.stderr)
        return 2
    types = getattr(args, "type", None) or None
    sources = getattr(args, "source", None) or None
    try:
        if args.events_cmd == "backfill":
            stats = backfill_utterances(data_dir, since=since, until=until)
            print(f"backfill: wrote {stats['wrote']}, already present {stats['skipped']}")
            return 0
        if args.events_cmd == "chain":
            rows = ref_chain(data_dir, args.id)
        elif args.events_cmd == "tail":
            rows = follow(data_dir, since=since, until=until, types=types, sources=sources, poll=args.poll)
        else:
            rows = replay(data_dir, since=since, until=until, types=types, sources=sources)
        for ev in rows:
            _print_event(ev, getattr(args, "json", False))
    except EventError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


def cmd_search(args, cfg) -> int:
    data_dir = args.data_dir or cfg.store.data_dir
    try:
        since = parse_when(args.since) if args.since else None
        until = parse_when(args.until, end_of_day=True) if args.until else None
    except ValueError as e:
        print(f"bad time: {e}", file=sys.stderr)
        return 2
    statuses = ("ok", "suspect", "rejected") if args.all else ("ok",) if args.ok_only else ("ok", "suspect")
    for r in search(data_dir, args.terms, since, until, statuses, args.context, args.limit):
        if args.json:
            print(json.dumps(r, ensure_ascii=False))
        else:
            start = datetime.fromisoformat(r["start"])
            mark = {"ok": " ", "suspect": "?", "rejected": "x"}[r["status"]]
            print(f"{start:%Y-%m-%d %H:%M:%S} {mark} {' '.join(r['text'].split())}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="audi27")
    ap.add_argument("-c", "--config", help="path to config.toml (default: ./config.toml if present)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="record continuously and transcribe").set_defaults(fn=cmd_run)
    sub.add_parser("devices", help="list audio input devices").set_defaults(fn=cmd_devices)
    f = sub.add_parser("file", help="transcribe WAV file(s) through the pipeline")
    f.add_argument("wav", nargs="+")
    f.set_defaults(fn=cmd_file)
    sub.add_parser("retry", help="retry segments in data/failed and data/spool").set_defaults(fn=cmd_retry)
    s = sub.add_parser("search", help="search transcripts")
    s.add_argument("terms", nargs="*", help="all terms must appear (case-insensitive); none = list everything")
    s.add_argument("--since", help="YYYY-MM-DD | ISO datetime | today | yesterday | 30m | 2h | 3d")
    s.add_argument("--until", help="same formats; a bare date includes that whole day")
    s.add_argument("--context", type=int, default=0, help="neighbouring segments to include")
    s.add_argument("--limit", type=int, default=0, help="keep only the N most recent matches")
    s.add_argument("--json", action="store_true", help="one JSON record per line")
    s.add_argument("--all", action="store_true", help="include rejected segments")
    s.add_argument("--ok-only", action="store_true", help="exclude suspect segments")
    s.add_argument("--data-dir")
    s.set_defaults(fn=cmd_search)
    evp = sub.add_parser("events", help="append-only event stream shared by modules")
    evsub = evp.add_subparsers(dest="events_cmd", required=True)

    def _window(p, *, kinds=False):
        p.add_argument("--data-dir")
        p.add_argument("--since", help="YYYY-MM-DD | ISO datetime | today | yesterday | 30m | 2h | 3d")
        p.add_argument("--until", help="same formats; a bare date includes that whole day")
        if kinds:
            p.add_argument("--type", action="append", help="event type (repeatable; unknown types are allowed)")
            p.add_argument("--source", action="append", help="writing module (repeatable)")
            p.add_argument("--json", action="store_true", help="one JSON event per line")

    et = evsub.add_parser("tail", help="print events as they are appended")
    _window(et, kinds=True)
    et.add_argument("--poll", type=float, default=0.1, help="seconds between polls")
    er = evsub.add_parser("replay", help="print events in a time range")
    _window(er, kinds=True)
    eb = evsub.add_parser("backfill", help="copy existing transcripts into utterance events")
    _window(eb)
    ec = evsub.add_parser("chain", help="follow ref from an event back to its origin")
    ec.add_argument("id")
    ec.add_argument("--data-dir")
    ec.add_argument("--json", action="store_true")
    evp.set_defaults(fn=cmd_events)
    e = sub.add_parser("eval", help="replay stored audio through other backends and compare")
    e.add_argument("--data-dir")
    esub = e.add_subparsers(dest="eval_cmd", required=True)
    esub.add_parser("list", help="list eval labels")
    er = esub.add_parser("run", help="replay stored audio through a backend")
    er.add_argument("backend", help="a [backend.<name>] entry from config.toml")
    er.add_argument("--label", help="name for this run (default: the backend name)")
    er.add_argument("--since"); er.add_argument("--until")
    er.add_argument("--limit", type=int, default=0, help="only the N most recent segments")
    er.add_argument("--force", action="store_true", help="redo segments this label already has")
    ek = esub.add_parser("correct", help="re-run only the context correction over stored ASR text (no audio)")
    ek.add_argument("--label", default="corrected")
    ek.add_argument("--force", action="store_true")
    ec = esub.add_parser("compare", help="score labels against a reference label")
    ec.add_argument("--ref", required=True, help="reference label, e.g. gold")
    ec.add_argument("labels", nargs="*", help="labels to score (default: all others, including live)")
    ec.add_argument("--since"); ec.add_argument("--until")
    ec.add_argument("--worst", type=int, default=10, help="show the N segments with the largest disagreement")
    ec.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_eval)
    w = sub.add_parser("web", help="local dashboard: live status, waveform, transcripts")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8727)
    w.add_argument("--data-dir")
    w.set_defaults(fn=cmd_web)
    sv = sub.add_parser("supervise", help="run recorder + dashboard, restart on crash or on data/restart.request")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8727)
    sv.set_defaults(fn=cmd_supervise)
    args = ap.parse_args(argv)
    C.load_dotenv()
    try:
        cfg = C.load(args.config)
        return args.fn(args, cfg)
    except C.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        from .transcribers import TranscribeError

        if isinstance(e, TranscribeError):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
