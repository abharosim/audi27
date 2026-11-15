"""Transcription backends. To add one: subclass Transcriber in a new module and add it to REGISTRY."""
from __future__ import annotations

import importlib

from .base import Piece, TranscribeError, Transcriber, Transcript

# name -> "module:Class"; imported lazily so unused backends need no dependencies
REGISTRY = {
    "mlx_whisper": "audi27.transcribers.mlx_whisper_local:MLXWhisperTranscriber",
    "faster_whisper": "audi27.transcribers.faster_whisper_local:FasterWhisperTranscriber",
    "openai": "audi27.transcribers.openai_api:OpenAITranscriber",
    "xai": "audi27.transcribers.xai_api:XAITranscriber",
}


def build(name: str, options: dict | None = None) -> Transcriber:
    # [backend.<name>] may set type = "<registry key>", so one implementation can be configured
    # several times under different names (e.g. a local model and a "gold" cloud model).
    options = dict(options or {})
    kind = options.pop("type", name)
    if kind not in REGISTRY:
        raise TranscribeError(f"unknown backend '{kind}'; known: {sorted(REGISTRY)}. "
                              f'For a custom name add type = "..." under [backend.{name}]')
    module, cls = REGISTRY[kind].split(":")
    try:
        return getattr(importlib.import_module(module), cls)(**options)
    except TypeError as e:
        raise TranscribeError(f"bad options for backend '{name}': {e}") from e


__all__ = ["Piece", "TranscribeError", "Transcriber", "Transcript", "REGISTRY", "build"]
