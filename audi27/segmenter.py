"""Energy-based speech segmenter with an adaptive noise floor. Pure numpy, no I/O."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SegmentCfg


@dataclass
class Segment:
    samples: np.ndarray  # int16 mono
    start_ts: float      # epoch seconds of the first sample
    sample_rate: int
    voiced_s: float

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def frame_db(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms / 32768.0 + 1e-10)


class Segmenter:
    def __init__(self, cfg: SegmentCfg, sample_rate: int, frame_ms: int):
        self.cfg = cfg
        self.sr = sample_rate
        self.frame_s = frame_ms / 1000.0
        self.start_frames = max(1, round(cfg.start_ms / frame_ms))
        self.end_frames = max(1, round(cfg.end_silence_ms / frame_ms))
        self.max_frames = max(1, round(cfg.max_segment_s * 1000 / frame_ms))
        self.pre = deque(maxlen=max(self.start_frames, round(cfg.pre_roll_ms / frame_ms)))
        self.floor: float | None = None
        self.dropped_short = 0
        self._reset()

    def _reset(self) -> None:
        self.buf: list[np.ndarray] = []
        self.active = False
        self.run = 0        # consecutive speech frames while idle
        self.silence = 0    # consecutive silent frames while active
        self.voiced = 0
        self.start_ts = 0.0

    def _is_speech(self, db: float) -> bool:
        if self.floor is None:
            self.floor = db
        speech = db > max(self.floor + self.cfg.start_margin_db, self.cfg.min_db)
        if not self.active:
            # floor follows drops quickly and rises slowly (~6 s), so speech does not lift it
            rate = 0.2 if db < self.floor else 0.005
            self.floor += rate * (db - self.floor)
        return speech

    def feed(self, frame: np.ndarray, ts: float) -> Segment | None:
        """Feed one frame (int16) whose first sample was captured at `ts`."""
        speech = self._is_speech(frame_db(frame))
        if not self.active:
            self.pre.append(frame)
            self.run = self.run + 1 if speech else 0
            if self.run >= self.start_frames:
                self.active = True
                self.buf = list(self.pre)
                self.start_ts = ts - (len(self.buf) - 1) * self.frame_s
                self.voiced = self.run
                self.silence = 0
                self.pre.clear()
            return None
        self.buf.append(frame)
        if speech:
            self.voiced += 1
            self.silence = 0
        else:
            self.silence += 1
        if self.silence >= self.end_frames or len(self.buf) >= self.max_frames:
            return self._emit()
        return None

    def flush(self) -> Segment | None:
        return self._emit() if self.active else None

    def _emit(self) -> Segment | None:
        voiced_s = self.voiced * self.frame_s
        seg = Segment(np.concatenate(self.buf), self.start_ts, self.sr, voiced_s)
        self._reset()
        if voiced_s * 1000 < self.cfg.min_speech_ms:
            self.dropped_short += 1
            return None
        return seg
