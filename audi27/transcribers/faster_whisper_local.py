from __future__ import annotations

from pathlib import Path

from .base import Piece, TranscribeError, Transcriber, Transcript


class FasterWhisperTranscriber(Transcriber):
    name = "faster_whisper"

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto",
                 compute_type: str = "int8", beam_size: int = 5, vad_filter: bool = True):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscribeError("faster-whisper is not installed") from e
        self.model = model
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        try:
            self._model = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as e:
            raise TranscribeError(f"cannot load whisper model '{model}': {e}") from e

    def transcribe(self, wav: Path, language: str | None = None, prompt: str | None = None) -> Transcript:
        try:
            segments, info = self._model.transcribe(
                str(wav),
                language=language or None,
                initial_prompt=prompt or None,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                # Both reduce hallucination: no carry-over between windows, no sampling fallback.
                condition_on_previous_text=False,
                temperature=0.0,
            )
            pieces = [
                Piece(s.text, s.start, s.end, s.avg_logprob, s.no_speech_prob, s.compression_ratio)
                for s in segments
            ]
        except Exception as e:
            raise TranscribeError(f"faster_whisper failed: {e}") from e
        return Transcript(pieces, info.language, info.language_probability, self.name, self.model)
