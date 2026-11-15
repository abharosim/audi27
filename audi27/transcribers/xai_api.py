from __future__ import annotations

import os
from pathlib import Path

from .base import Piece, TranscribeError, Transcriber, Transcript


class XAITranscriber(Transcriber):
    """xAI Grok Speech to Text, batch REST: POST {base_url}/stt with multipart `model` + `file`.
    The response carries the transcript in `text`; no confidence metrics, so only the text-level
    validation rules apply to its output."""

    name = "xai"

    def __init__(self, base_url: str = "https://api.x.ai/v1", model: str = "grok-voice-transcribe-2.0",
                 api_key_env: str = "XAI_API_KEY", timeout_s: float = 60.0):
        self.url = base_url.rstrip("/") + "/stt"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        if not os.environ.get(api_key_env):
            raise TranscribeError(f"environment variable {api_key_env} is not set (put it in .env)")

    def transcribe(self, wav: Path, language: str | None = None, prompt: str | None = None) -> Transcript:
        import httpx

        try:
            with open(wav, "rb") as f:
                r = httpx.post(self.url, headers={"Authorization": f"Bearer {os.environ[self.api_key_env]}"},
                               data={"model": self.model}, files={"file": (wav.name, f, "audio/wav")},
                               timeout=self.timeout_s)
        except (httpx.HTTPError, OSError) as e:
            raise TranscribeError(f"xai request failed: {e}") from e
        if r.status_code != 200:
            raise TranscribeError(f"xai HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
            text = body["text"]
            if not isinstance(text, str):
                raise TypeError("text is not a string")
        except (ValueError, KeyError, TypeError) as e:
            raise TranscribeError(f"xai returned an unexpected body: {r.text[:300]}") from e
        lang = body.get("language") if isinstance(body.get("language"), str) else None
        return Transcript([Piece(text)] if text.strip() else [], lang, None, self.name, self.model)
