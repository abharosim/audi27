"""Replay stored audio through other backends and compare the outputs.

  data/evals/<label>.jsonl   one record per replayed segment, keyed by the live record id

"live" is a reserved label: what the recorder actually wrote to data/transcripts. Any label can
serve as the reference ("gold") in a comparison. A rejected output counts as empty text, so the
comparison scores the whole pipeline (model + validation), not just the raw model.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from . import validate as V
from .config import Config
from .store import Store, clean_json, iter_records
from .textnorm import cmp_tokens as tokens
from .textnorm import edit_distance  # noqa: F401  (re-exported)
from .transcribers import TranscribeError, build

LIVE = "live"
LIVE_ASR = "live-asr"   # the live output before context correction (only when some record was corrected)
_LABEL = re.compile(r"[A-Za-z0-9_.-]+")


class EvalError(ValueError):
    pass


def effective_text(rec: dict) -> str:
    return "" if rec.get("status") == "rejected" else rec.get("text", "")


def _check_label(label: str) -> str:
    if not _LABEL.fullmatch(label or "") or label in (LIVE, LIVE_ASR):
        raise EvalError(f"bad label '{label}': use letters, digits, _ . - (and not '{LIVE}')")
    return label


def labels(root: str | Path) -> list[str]:
    return [LIVE] + sorted(p.stem for p in (Path(root) / "evals").glob("*.jsonl"))


def load(root: str | Path, label: str, since: datetime | None = None, until: datetime | None = None) -> dict:
    """id -> record for one label. For replays, a later line for the same id replaces the earlier one."""
    if label == LIVE:
        return {r["id"]: r for r in iter_records(root, since, until)}
    if label == LIVE_ASR:
        return {r["id"]: {**r, "text": r.get("asr_text", r.get("text", ""))} for r in iter_records(root, since, until)}
    path = Path(root) / "evals" / f"{_check_label(label)}.jsonl"
    out: dict = {}
    if not path.exists():
        raise EvalError(f"no eval run labelled '{label}'; known: {labels(root)}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = clean_json(json.loads(line))
                out[rec["id"]] = rec
            except (ValueError, KeyError):
                continue
    return out


def plan(root: str | Path, label: str, since: datetime | None = None, until: datetime | None = None,
         limit: int = 0, force: bool = False) -> tuple[list[dict], dict]:
    """Which stored segments a run under `label` would process, and the totals (for cost estimates)."""
    root = Path(root)
    label = _check_label(label)
    out_path = root / "evals" / f"{label}.jsonl"
    done = set() if force or not out_path.exists() else set(load(root, label))
    recs = list(iter_records(root, since, until))
    with_audio = [r for r in recs if r.get("audio") and (root / r["audio"]).is_file()]
    todo = [r for r in with_audio if r["id"] not in done]
    if limit:
        todo = todo[-limit:]
    stats = {"label": label, "segments": len(recs), "no_audio": len(recs) - len(with_audio),
             "already_done": len(with_audio) - len([r for r in with_audio if r["id"] not in done]),
             "todo": len(todo), "todo_seconds": round(sum(r.get("duration") or 0.0 for r in todo), 1),
             "ran": 0, "failed": 0}
    return todo, stats


def run_eval(cfg: Config, backend: str, label: str | None = None, since: datetime | None = None,
             until: datetime | None = None, limit: int = 0, force: bool = False, log=print,
             progress=None) -> dict:
    """Replay stored audio through `backend` (a [backend.<name>] entry) and save under `label`."""
    from .pipeline import Pipeline

    label = _check_label(label or backend)
    root = Path(cfg.store.data_dir)
    out_dir = root / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.jsonl"
    todo, stats = plan(root, label, since, until, limit, force)
    if not todo:
        return stats
    transcriber = build(backend, cfg.backend.get(backend))
    pipe = Pipeline(cfg, [transcriber], Store(root, cfg.store.keep_audio))
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                t = pipe._transcribe(root / r["audio"])
            except TranscribeError as e:  # not written, so the next run retries it
                stats["failed"] += 1
                stats["last_error"] = str(e)
                log(f"[{i}/{len(todo)}] {r['id']} FAILED: {e}")
                if progress:
                    progress(dict(stats))
                continue
            seconds = time.perf_counter() - t0
            v = V.validate(t, r.get("duration", 0.0), cfg.validate, "", cfg.transcribe.initial_prompt)
            if t.notes:
                v.reasons.extend(t.notes)
                if v.status == V.OK:
                    v.status = V.SUSPECT
            rec = {"id": r["id"], "start": r["start"], "label": label, "text": v.text, "status": v.status,
                   "reasons": v.reasons, "language": t.language, "avg_logprob": v.avg_logprob,
                   "no_speech_prob": v.no_speech_prob, "compression_ratio": v.compression_ratio,
                   "backend": t.backend, "model": t.model, "seconds": round(seconds, 3),
                   "created": datetime.now().astimezone().isoformat(timespec="seconds")}
            if t.text != v.text:
                rec["raw_text"] = t.text
            f.write(json.dumps(clean_json(rec), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            stats["ran"] += 1
            log(f"[{i}/{len(todo)}] {v.status:8} {seconds:5.1f}s  {v.text[:70]}")
            if progress:
                progress(dict(stats))
    return stats


def compare(root: str | Path, ref: str, others: list[str] | None = None,
            since: datetime | None = None, until: datetime | None = None) -> dict:
    """Score each label against `ref`. error_rate = token edit distance / reference tokens,
    summed over the segments both sides have (so labels with partial coverage stay comparable)."""
    root = Path(root)
    live = load(root, LIVE, since, until)
    ref_recs = load(root, ref)
    names = others or [x for x in labels(root) if x != ref]
    data = {name: load(root, name, since, until) if name == LIVE else load(root, name) for name in names}
    summary = {name: {"label": name, "segments": 0, "ref_tokens": 0, "errors": 0, "exact": 0,
                      "seconds": 0.0, "audio_seconds": 0.0, "models": set()} for name in names}
    rows = []
    for rec_id, base in live.items():
        if rec_id not in ref_recs:
            continue
        ref_text = effective_text(ref_recs[rec_id])
        ref_tok = tokens(ref_text)
        row = {"id": rec_id, "start": base["start"], "duration": base.get("duration"), "audio": base.get("audio"),
               "ref": ref_text, "outputs": {}, "worst": 0.0}
        for name in names:
            rec = data[name].get(rec_id)
            if rec is None:
                continue
            text = effective_text(rec)
            dist = edit_distance(ref_tok, tokens(text))
            rate = dist / max(1, len(ref_tok))
            row["outputs"][name] = {"text": text, "status": rec.get("status"), "errors": dist, "error_rate": round(rate, 4)}
            row["worst"] = max(row["worst"], rate)
            s = summary[name]
            s["segments"] += 1
            s["ref_tokens"] += len(ref_tok)
            s["errors"] += dist
            s["exact"] += dist == 0
            s["models"].add(rec.get("model") or "")
            if rec.get("seconds") is not None:
                s["seconds"] += rec["seconds"]
                s["audio_seconds"] += base.get("duration") or 0.0
        if row["outputs"]:
            rows.append(row)
    out = []
    for s in summary.values():
        n = s["segments"]
        out.append({
            "label": s["label"], "model": ", ".join(sorted(m for m in s["models"] if m)), "segments": n,
            "ref_tokens": s["ref_tokens"], "errors": s["errors"],
            "error_rate": round(s["errors"] / max(1, s["ref_tokens"]), 4) if n else None,
            "exact_rate": round(s["exact"] / n, 4) if n else None,
            # seconds of compute per second of audio; < 1 is faster than real time
            "rtf": round(s["seconds"] / s["audio_seconds"], 3) if s["audio_seconds"] else None,
        })
    out.sort(key=lambda s: (s["error_rate"] is None, s["error_rate"]))
    rows.sort(key=lambda r: -r["worst"])
    ref_models = sorted({r.get("model") for r in ref_recs.values() if r.get("model")})
    return {"ref": ref, "ref_model": ", ".join(ref_models), "ref_segments": len([i for i in live if i in ref_recs]),
            "summary": out, "rows": rows}


# --- human verdicts -----------------------------------------------------------
# data/verdicts.jsonl, append-only, last line per id wins. A verdict stores the *texts* judged
# correct, not just the labels, so it keeps scoring labels that are (re)run later: a judged
# segment becomes a gold answer without anyone typing a transcript.
def save_verdict(root: str | Path, rec_id: str, texts: list[str], labels_marked: list[str],
                 cleared: bool = False) -> dict:
    root = Path(root)
    v = {"id": str(rec_id), "texts": [t for t in texts if isinstance(t, str)],
         "labels": [x for x in labels_marked if isinstance(x, str)], "cleared": bool(cleared),
         "ts": datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(root / "verdicts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(v, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return v


def load_verdicts(root: str | Path) -> dict:
    out: dict = {}
    path = Path(root) / "verdicts.jsonl"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                v = json.loads(line)
                if v.get("cleared"):
                    out.pop(v["id"], None)
                else:
                    out[v["id"]] = v
            except (ValueError, KeyError):
                continue
    return out


def matrix(root: str | Path, base: str | None = None, since: datetime | None = None,
           until: datetime | None = None) -> dict:
    """One row per stored segment, one cell per label. `base` is the column the others are diffed
    and scored against (default: gold if it exists, else live)."""
    root = Path(root)
    names = labels(root)
    live = load(root, LIVE, since, until)
    data = {name: live if name == LIVE else load(root, name) for name in names}
    if any("asr_text" in r for r in live.values()):
        names.insert(1, LIVE_ASR)
        data[LIVE_ASR] = load(root, LIVE_ASR, since, until)
    if base not in names:
        base = "gold" if "gold" in names else LIVE
    verdicts = load_verdicts(root)
    summary = {name: {"label": name, "models": set(), "segments": 0, "judged": 0, "correct": 0,
                      "base_tokens": 0, "errors": 0, "same_as_base": 0, "compared": 0} for name in names}
    rows = []
    for rec_id, rec in live.items():
        cells = {}
        for name in names:
            r = data[name].get(rec_id)
            if r is None:
                continue
            cells[name] = {"text": effective_text(r), "status": r.get("status"), "raw": r.get("text", ""),
                           "reasons": r.get("reasons") or []}
            summary[name]["segments"] += 1
            summary[name]["models"].add(r.get("model") or "")
        toks = {name: tokens(c["text"]) for name, c in cells.items()}
        verdict = verdicts.get(rec_id)
        good = [tokens(t) for t in verdict["texts"]] if verdict else []
        for name, c in cells.items():
            if verdict:
                c["correct"] = toks[name] in good
                summary[name]["judged"] += 1
                summary[name]["correct"] += c["correct"]
            if base in cells and name != base:
                dist = edit_distance(toks[base], toks[name])
                s = summary[name]
                s["compared"] += 1
                s["errors"] += dist
                s["base_tokens"] += len(toks[base])
                s["same_as_base"] += dist == 0
        rows.append({"id": rec_id, "start": rec["start"], "duration": rec.get("duration"), "audio": rec.get("audio"),
                     "cells": cells, "judged": bool(verdict),
                     "agree": len(cells) > 1 and all(t == toks[next(iter(toks))] for t in toks.values())})
    out = []
    for name in names:
        s = summary[name]
        out.append({"label": name, "model": ", ".join(sorted(m for m in s["models"] if m)), "segments": s["segments"],
                    "judged": s["judged"], "correct": s["correct"],
                    "correct_rate": round(s["correct"] / s["judged"], 4) if s["judged"] else None,
                    "compared": s["compared"],
                    "error_rate": round(s["errors"] / max(1, s["base_tokens"]), 4) if s["compared"] else None,
                    "same_rate": round(s["same_as_base"] / s["compared"], 4) if s["compared"] else None})
    rows.sort(key=lambda r: r["start"], reverse=True)
    return {"base": base, "labels": out, "rows": rows}


def plan_correct(root: str | Path, label: str, force: bool = False) -> tuple[list[dict], dict]:
    root = Path(root)
    label = _check_label(label)
    out_path = root / "evals" / f"{label}.jsonl"
    done = set() if force or not out_path.exists() else set(load(root, label))
    recs = list(iter_records(root))
    todo = [r for r in recs if r["id"] not in done]
    return todo, {"label": label, "segments": len(recs), "no_audio": 0, "already_done": len(recs) - len(todo),
                  "todo": len(todo), "todo_seconds": 0.0, "ran": 0, "failed": 0}


def run_correct(cfg: Config, label: str = "corrected", force: bool = False, log=print, progress=None) -> dict:
    """Re-run only the context correction over the stored ASR text (no audio, no recognition).
    Use it after editing context.md or the corrector settings, then compare the new column."""
    from .correct import CorrectionStage

    root = Path(cfg.store.data_dir)
    todo, stats = plan_correct(root, label, force)
    if not todo:
        return stats
    (root / "evals").mkdir(parents=True, exist_ok=True)
    stage = CorrectionStage(cfg)
    recent: list[str] = []
    with open(root / "evals" / f"{_check_label(label)}.jsonl", "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            original = r.get("asr_text", r.get("text", ""))
            t0 = time.perf_counter()
            text, info = (original, {"changes": []}) if r.get("status") == "rejected" else stage.run(original, recent)
            rec = {"id": r["id"], "start": r["start"], "label": label, "text": text, "status": r.get("status"),
                   "reasons": list(r.get("reasons") or []), "model": stage.model, "backend": "correct",
                   "seconds": round(time.perf_counter() - t0, 3),
                   "created": datetime.now().astimezone().isoformat(timespec="seconds")}
            if text != original:
                rec["asr_text"] = original
                rec["corrections"] = info["changes"]
            f.write(json.dumps(clean_json(rec), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            if r.get("status") == "ok":
                recent.append(text)
                del recent[:-10]
            stats["ran"] += 1
            if text != original:
                stats["changed"] = stats.get("changed", 0) + 1
                log(f"[{i}/{len(todo)}] {original[:50]!r} -> {text[:50]!r}")
            if progress:
                progress(dict(stats))
    if stage.disabled_reason:
        stats["last_error"] = stage.disabled_reason
    return stats
