"""Microphone capture. Yields (int16 frame, epoch ts) and reopens the stream on device errors."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Iterator

import numpy as np

from .config import AudioCfg

log = logging.getLogger("audi27.capture")


class Capture:
    def __init__(self, cfg: AudioCfg):
        self.cfg = cfg
        self.frame_samples = cfg.sample_rate * cfg.frame_ms // 1000
        self.q: queue.Queue = queue.Queue(maxsize=4000)  # ~2 min of frames
        self.dropped_frames = 0
        self.overflows = 0

    def _device(self):
        d = self.cfg.device
        return None if d in ("", None) else d

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.overflows += 1
        ts = time.time() - frames / self.cfg.sample_rate
        try:
            self.q.put_nowait((indata[:, 0].copy(), ts))
        except queue.Full:
            self.dropped_frames += 1

    def frames(self, stop: threading.Event) -> Iterator[tuple[np.ndarray, float]]:
        import sounddevice as sd

        backoff = 1.0
        while not stop.is_set():
            try:
                with sd.InputStream(
                    samplerate=self.cfg.sample_rate, channels=1, dtype="int16",
                    blocksize=self.frame_samples, device=self._device(), callback=self._callback,
                ):
                    log.info("capture started (device=%s, %d Hz)", self._device() or "default", self.cfg.sample_rate)
                    backoff = 1.0
                    idle = 0
                    while not stop.is_set():
                        try:
                            yield self.q.get(timeout=1.0)
                            idle = 0
                        except queue.Empty:
                            idle += 1
                            if idle >= 5:
                                raise sd.PortAudioError("no audio frames for 5 s")
            except sd.PortAudioError as e:
                log.error("audio device error: %s; reopening in %.0f s", e, backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                try:  # pick up a changed default device
                    sd._terminate(); sd._initialize()
                except Exception:
                    pass


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())
