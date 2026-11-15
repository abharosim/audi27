"""On-disk layout under data_dir:

  spool/<epoch_ms>.wav           segments waiting for transcription (crash-safe queue)
  failed/<epoch_ms>.wav + .err   segments whose transcription kept failing
  audio/YYYY-MM-DD/<id>.wav      audio kept per store.keep_audio
  transcripts/YYYY-MM-DD.jsonl   one record per segment, every status (source of truth)
  transcripts/YYYY-MM-DD.md      readable log of ok + suspect lines, for grep
"""
from __future__ import annotations

import json
import math
import os
import re
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from .segmenter import Segment


def clean_json(obj):
    """Replace NaN/Infinity with None. Python's json accepts them, browsers and other agents do not."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts).astimezone()


class Store:
    def __init__(self, data_dir: str | Path, keep_audio: str = "suspect"):
        self.root = Path(data_dir)
        self.keep_audio = keep_audio
        self.spool = self.root / "spool"
        self.failed = self.root / "failed"
        self.transcripts = self.root / "transcripts"
        for d in (self.spool, self.failed, self.transcripts):
            d.mkdir(parents=True, exist_ok=True)

    # --- spool -----------------------------------------------------------
    def spool_wav(self, seg: Segment) -> Path:
        path = self.spool / f"{int(seg.start_ts * 1000)}.wav"
        tmp = path.with_suffix(".tmp")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(seg.sample_rate)
            w.writeframes(seg.samples.astype("<i2").tobytes())
        os.replace(tmp, path)  # atomic: the spool never holds a partial file
        return path

    def pending(self) -> list[Path]:
        return sorted(self.spool.glob("*.wav"))

    def mark_failed(self, wav: Path, error: str) -> None:
        dest = self.failed / wav.name
        os.replace(wav, dest)
        dest.with_suffix(".err").write_text(error + "\n", encoding="utf-8")

    def requeue_failed(self) -> int:
        n = 0
        for wav in sorted(self.failed.glob("*.wav")):
            os.replace(wav, self.spool / wav.name)
            wav.with_suffix(".err").unlink(missing_ok=True)
            n += 1
        return n

    # --- records ---------------------------------------------------------
    def audio_dest(self, wav: Path, start_ts: float, status: str) -> Path | None:
        keep = self.keep_audio == "all" or (self.keep_audio == "suspect" and status != "ok")
        return self.root / "audio" / _dt(start_ts).strftime("%Y-%m-%d") / wav.name if keep else None

    def has_id(self, rec_id: str, start_ts: float) -> bool:
        path = self.transcripts / f"{_dt(start_ts):%Y-%m-%d}.jsonl"
        if not path.exists():
            return False
        needle = f'"id": "{rec_id}"'
        with open(path, encoding="utf-8") as f:
            return any(needle in line for line in f)

    def append(self, record: dict) -> None:
        start = datetime.fromisoformat(record["start"])
        day = start.strftime("%Y-%m-%d")
        with open(self.transcripts / f"{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_json(record), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if record["status"] != "rejected":
            mark = "" if record["status"] == "ok" else "[?] "
            text = " ".join(record["text"].split())
            with open(self.transcripts / f"{day}.md", "a", encoding="utf-8") as f:
                f.write(f"- {start:%H:%M:%S} {mark}{text}\n")

    def finish_audio(self, wav: Path, dest: Path | None) -> None:
        if dest is None:
            wav.unlink(missing_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(wav, dest)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --- search (read-only; safe while the recorder is running) ----------------
def parse_when(s: str, end_of_day: bool = False) -> datetime:
    now = datetime.now().astimezone()
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([mhd])", s)
    if m:
        return now - timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[m.group(2)]: int(m.group(1))})
    if s in ("today", "yesterday"):
        d = now if s == "today" else now - timedelta(days=1)
        s = d.strftime("%Y-%m-%d")
    dt = datetime.fromisoformat(s)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) and end_of_day:
        dt += timedelta(days=1)
    return dt.astimezone()


def iter_records(data_dir: str | Path, since: datetime | None = None,
                 until: datetime | None = None) -> Iterator[dict]:
    for path in sorted((Path(data_dir) / "transcripts").glob("*.jsonl")):
        day = path.stem
        if since and day < (since - timedelta(days=1)).strftime("%Y-%m-%d"):
            continue
        if until and day > (until + timedelta(days=1)).strftime("%Y-%m-%d"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = clean_json(json.loads(line))
                    start = datetime.fromisoformat(rec["start"])
                except (ValueError, KeyError):
                    continue  # a torn final line from a crash must not break readers
                if (since and start < since) or (until and start >= until):
                    continue
                yield rec


def search(data_dir: str | Path, terms: list[str], since: datetime | None = None,
           until: datetime | None = None, statuses: tuple = ("ok", "suspect"),
           context: int = 0, limit: int = 0) -> list[dict]:
    recs = [r for r in iter_records(data_dir, since, until) if r.get("status") in statuses]
    wanted = [t.lower() for t in terms]
    hits = [i for i, r in enumerate(recs) if all(t in r.get("text", "").lower() for t in wanted)]
    if limit:
        hits = hits[-limit:]
    out, seen, hit_set = [], set(), set(hits)
    for i in hits:
        for j in range(max(0, i - context), min(len(recs), i + context + 1)):
            if j not in seen:
                seen.add(j)
                out.append({**recs[j], "match": j in hit_set})
    return out
