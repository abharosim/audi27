"""capture -> segment -> spool -> transcribe (with fallback + retry) -> validate -> store."""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
import os
import wave
from collections import deque
from pathlib import Path

import numpy as np

from . import validate as V
from .config import Config
from .correct import CorrectionStage
from .segmenter import Segmenter, frame_db
from .status import write_status
from .store import Store, _dt, wav_duration
from .transcribers import TranscribeError, Transcriber, Transcript, build

log = logging.getLogger("audi27")


_LANG_NAMES = {"chinese": "zh", "mandarin": "zh", "cantonese": "yue", "english": "en", "japanese": "ja", "korean": "ko"}


def _lang_code(lang: str | None) -> str | None:
    """Backends report either codes ("zh") or names ("chinese"); normalise to codes."""
    if not lang:
        return None
    lang = lang.strip().lower()
    return _LANG_NAMES.get(lang, lang)


def _confidence(t: Transcript) -> float:
    lps = [p.avg_logprob for p in t.pieces if p.text.strip() and p.avg_logprob is not None and math.isfinite(p.avg_logprob)]
    return sum(lps) / len(lps) if lps and t.text else float("-inf")


def build_transcribers(cfg: Config) -> list[Transcriber]:
    out = []
    for name in cfg.transcribe.backends:
        try:
            out.append(build(name, cfg.backend.get(name)))
            log.info("backend ready: %s", name)
        except TranscribeError as e:
            log.error("backend '%s' unavailable: %s", name, e)
    if not out:
        raise TranscribeError("no transcription backend could be started")
    return out


class Pipeline:
    def __init__(self, cfg: Config, transcribers: list[Transcriber], store: Store):
        self.cfg = cfg
        self.transcribers = transcribers
        self.store = store
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.prev_text = ""
        self.corrector = CorrectionStage(cfg)
        self.recent: deque = deque(maxlen=10)   # accepted lines, context for the corrector
        self.counts = {"ok": 0, "suspect": 0, "rejected": 0, "failed": 0}

    # --- one segment -----------------------------------------------------
    def _transcribe(self, wav: Path) -> Transcript:
        """Transcribe within the configured languages. With several allowed languages the backend
        auto-detects; if it detects something else (typical for noise and very short clips), each
        allowed language is forced in turn and the most confident result wins."""
        langs = self.cfg.transcribe.languages
        t = self._transcribe_once(wav, langs[0] if len(langs) == 1 else None)
        detected = _lang_code(t.language)
        if len(langs) < 2 or not t.text or detected is None or detected in langs:
            return t
        best = max((self._transcribe_once(wav, lang) for lang in langs), key=_confidence)
        best.notes.append(f"language_forced:{detected}->{_lang_code(best.language) or '?'}")
        return best

    def _prompt(self) -> str:
        """The configured prompt plus the glossary terms from the context file."""
        base, terms = self.cfg.transcribe.initial_prompt, self.corrector.prompt_terms()
        return f"{base} {terms}".strip() if terms else base

    def _transcribe_once(self, wav: Path, language: str | None) -> Transcript:
        last: Exception | None = None
        for t in self.transcribers:
            try:
                result = t.transcribe(wav, language, self._prompt() or None)
                if language and not result.language:
                    result.language = language
                return result
            except TranscribeError as e:
                log.warning("%s failed on %s: %s", t.name, wav.name, e)
                last = e
        raise TranscribeError(f"all backends failed; last error: {last}")

    def process_file(self, wav: Path) -> dict | None:
        """Transcribe + validate + record one spooled WAV. Raises TranscribeError to request a retry."""
        try:
            start_ts = int(wav.stem) / 1000.0
            duration = wav_duration(wav)
        except (ValueError, wave.Error, EOFError, OSError) as e:
            self.store.mark_failed(wav, f"unreadable audio: {e}")
            self.counts["failed"] += 1
            return None
        if self.store.has_id(wav.stem, start_ts):  # recorded before a crash; only cleanup is left
            wav.unlink(missing_ok=True)
            return None
        t = self._transcribe(wav)
        v = V.validate(t, duration, self.cfg.validate, self.prev_text, self.cfg.transcribe.initial_prompt)
        if t.notes:  # the detector disagreed with the allowed languages: keep, but flag it
            v.reasons.extend(t.notes)
            if v.status == V.OK:
                v.status = V.SUSPECT
        asr_text, corrections = v.text, []
        if v.status != V.REJECTED:
            v.text, info = self.corrector.run(v.text, list(self.recent))
            corrections = info["changes"]
            if v.text != asr_text:
                v.reasons.append("corrected")
        dest = self.store.audio_dest(wav, start_ts, v.status)
        record = {
            "id": wav.stem,
            "start": _dt(start_ts).isoformat(timespec="milliseconds"),
            "end": _dt(start_ts + duration).isoformat(timespec="milliseconds"),
            "duration": round(duration, 2),
            "text": v.text,
            "status": v.status,
            "reasons": v.reasons,
            "language": t.language,
            "language_prob": None if t.language_prob is None else round(t.language_prob, 3),
            "avg_logprob": v.avg_logprob,
            "no_speech_prob": v.no_speech_prob,
            "compression_ratio": v.compression_ratio,
            "backend": t.backend,
            "model": t.model,
            "audio": str(dest.relative_to(self.store.root)) if dest else None,
        }
        if v.text != asr_text:
            record["asr_text"] = asr_text      # what the recogniser said, before correction
            record["corrections"] = corrections
        if t.text != asr_text:
            record["raw_text"] = t.text
        self.store.append(record)       # record first, then move audio: a crash in between is
        self.store.finish_audio(wav, dest)  # resolved by has_id() on the next start
        self.counts[v.status] += 1
        if v.status == V.OK:
            self.prev_text = v.text
            self.recent.append(v.text)
        log.info("[%s] %s %s", v.status, v.text[:100] or "-", ",".join(v.reasons))
        return record

    # --- worker ----------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            wav, attempt = item
            try:
                if wav.exists():
                    self.process_file(wav)
            except TranscribeError as e:
                if attempt + 1 >= self.cfg.transcribe.max_attempts:
                    log.error("giving up on %s after %d attempts: %s", wav.name, attempt + 1, e)
                    self.store.mark_failed(wav, str(e))
                    self.counts["failed"] += 1
                else:
                    delay = min(60.0, 2.0 * 2 ** attempt)
                    log.warning("retry %s in %.0f s (attempt %d)", wav.name, delay, attempt + 1)
                    timer = threading.Timer(delay, self.q.put, args=((wav, attempt + 1),))
                    timer.daemon = True
                    timer.start()
            except Exception as e:  # never let one segment kill the worker
                log.exception("unexpected error on %s", wav.name)
                try:
                    self.store.mark_failed(wav, repr(e))
                except OSError:
                    pass
                self.counts["failed"] += 1

    def drain(self) -> None:
        """Process everything in the spool synchronously (startup recovery, `retry`, tests)."""
        for wav in self.store.pending():
            try:
                self.process_file(wav)
            except TranscribeError as e:
                log.error("still failing: %s: %s", wav.name, e)

    # --- live run --------------------------------------------------------
    def run(self) -> None:
        from .capture import Capture

        for wav in self.store.pending():
            self.q.put((wav, 0))
        if self.q.qsize():
            log.info("recovering %d spooled segment(s) from a previous run", self.q.qsize())
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        cap = Capture(self.cfg.audio)
        seg = Segmenter(self.cfg.segment, self.cfg.audio.sample_rate, self.cfg.audio.frame_ms)
        zero_since: float | None = None
        warned_zero = False
        last_stats = started = time.time()
        last_beat = 0.0
        levels: deque = deque(maxlen=64)  # ~2 s of per-frame loudness for the dashboard waveform

        def beat(stopped: bool = False) -> None:
            try:
                write_status(self.store.root, {
                    "pid": os.getpid(), "started": started, "updated": time.time(), "stopped": stopped,
                    "speaking": seg.active and not stopped, "levels": [] if stopped else list(levels),
                    "counts": self.counts, "backlog": self.q.qsize(), "mic_silent": warned_zero,
                    "dropped_frames": cap.dropped_frames, "overflows": cap.overflows,
                    "backends": [t.name for t in self.transcribers],
                    "device": str(self.cfg.audio.device or ""),
                })
            except OSError as e:
                log.debug("cannot write status: %s", e)

        try:
            for frame, ts in cap.frames(self.stop):
                # Exact digital silence for 10 s = macOS mic permission denied or a muted device.
                if not np.any(frame):
                    zero_since = zero_since or ts
                    if not warned_zero and ts - zero_since > 10:
                        log.error("microphone returns pure silence. Check System Settings > Privacy & "
                                  "Security > Microphone for your terminal app, and the input device.")
                        warned_zero = True
                else:
                    zero_since, warned_zero = None, False
                s = seg.feed(frame, ts)
                if s is not None:
                    self.q.put((self.store.spool_wav(s), 0))
                # map -60..-10 dBFS to 0..1
                levels.append(round(min(1.0, max(0.0, (frame_db(frame) + 60.0) / 50.0)), 3))
                if time.time() - last_beat >= 0.4:
                    last_beat = time.time()
                    beat()
                if time.time() - last_stats > 300:
                    last_stats = time.time()
                    log.info("stats %s backlog=%d dropped_frames=%d overflows=%d short_dropped=%d",
                             self.counts, self.q.qsize(), cap.dropped_frames, cap.overflows, seg.dropped_short)
                    if self.q.qsize() > 20:
                        log.warning("transcription is falling behind; use a smaller model or a cloud backend")
        except KeyboardInterrupt:
            log.info("stopping")
        finally:
            self.stop.set()
            s = seg.flush()
            if s is not None:
                self.q.put((self.store.spool_wav(s), 0))
            self.q.put(None)
            worker.join(timeout=120)
            beat(stopped=True)
            log.info("stopped. %s; %d segment(s) left in spool", self.counts, len(self.store.pending()))
