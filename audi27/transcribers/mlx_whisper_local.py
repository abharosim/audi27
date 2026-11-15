from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .base import Piece, TranscribeError, Transcriber, Transcript


def _read_wav(path: Path) -> np.ndarray:
    """16 kHz mono int16 WAV -> float32 in [-1, 1]. Passing an array avoids mlx-whisper's ffmpeg dependency."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise TranscribeError(f"{path.name}: expected 16 kHz mono 16-bit WAV")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


class MLXWhisperTranscriber(Transcriber):
    """Whisper on Apple Silicon via MLX. `model` is a Hugging Face repo (served from the local
    ~/.cache/huggingface cache when already downloaded) or a local directory."""

    name = "mlx_whisper"

    def __init__(self, model: str = "mlx-community/whisper-turbo"):
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise TranscribeError("mlx-whisper is not installed (Apple Silicon only)") from e
        self.model = model
        # The model itself loads on the first transcribe() call, in the thread that transcribes:
        # MLX GPU streams are per-thread, so loading here (main thread) can break the worker thread.

    def transcribe(self, wav: Path, language: str | None = None, prompt: str | None = None) -> Transcript:
        import mlx_whisper

        try:
            result = mlx_whisper.transcribe(
                _read_wav(wav),
                path_or_hf_repo=self.model,
                language=language or None,
                initial_prompt=prompt or None,
                # Both reduce hallucination: no carry-over between windows, no sampling fallback.
                condition_on_previous_text=False,
                temperature=0.0,
                verbose=None,
            )
            pieces = [
                Piece(s["text"], s.get("start", 0.0), s.get("end", 0.0), s.get("avg_logprob"),
                      s.get("no_speech_prob"), s.get("compression_ratio"))
                for s in result["segments"]
            ]
        except TranscribeError:
            raise
        except Exception as e:
            raise TranscribeError(f"mlx_whisper failed: {e}") from e
        return Transcript(pieces, result.get("language"), None, self.name, self.model)
