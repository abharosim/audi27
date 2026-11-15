from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class TranscribeError(Exception):
    """Backend failed (network, model, bad response). The segment is retried, never recorded as text."""


@dataclass
class Piece:
    text: str
    start: float = 0.0
    end: float = 0.0
    # Confidence metrics; None when the backend does not report them.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


@dataclass
class Transcript:
    pieces: list[Piece] = field(default_factory=list)
    language: str | None = None
    language_prob: float | None = None
    backend: str = ""
    model: str = ""
    notes: list[str] = field(default_factory=list)  # set by the pipeline, e.g. language_forced

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pieces).strip()


class Transcriber(ABC):
    name: str = ""
    model: str = ""

    @abstractmethod
    def transcribe(self, wav: Path, language: str | None = None, prompt: str | None = None) -> Transcript:
        """Transcribe a 16 kHz mono WAV. Raise TranscribeError on any failure."""
