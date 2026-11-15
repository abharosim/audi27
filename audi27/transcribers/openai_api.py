from __future__ import annotations

import os
from pathlib import Path

from .base import Piece, TranscribeError, Transcriber, Transcript


class OpenAITranscriber(Transcriber):
    """Any OpenAI-compatible POST {base_url}/audio/transcriptions endpoint."""

    name = "openai"

    def __init__(self, base_url: str = "https://api.openai.com/v1", model: str = "whisper-1",
                 api_key_env: str = "OPENAI_API_KEY", response_format: str = "verbose_json",
                 timeout_s: float = 60.0):
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model = model
        self.response_format = response_format
        self.timeout_s = timeout_s
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise TranscribeError(f"environment variable {api_key_env} is not set")

    def transcribe(self, wav: Path, language: str | None = None, prompt: str | None = None) -> Transcript:
        import httpx

        data = {"model": self.model, "response_format": self.response_format, "temperature": "0"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        def post(fields: dict):
            try:
                with open(wav, "rb") as f:
                    return httpx.post(self.url, headers={"Authorization": f"Bearer {self.api_key}"},
                                      data=fields, files={"file": (wav.name, f, "audio/wav")},
                                      timeout=self.timeout_s)
            except (httpx.HTTPError, OSError) as e:
                raise TranscribeError(f"openai request failed: {e}") from e

        r = post(data)
        if r.status_code == 400 and len(data) > 2:
            # Models differ in which optional parameters they accept; retry with the required ones only.
            r = post({"model": self.model, "response_format": self.response_format})
        if r.status_code != 200:
            raise TranscribeError(f"openai HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
            if "segments" in body:
                pieces = [
                    Piece(s["text"], s.get("start", 0.0), s.get("end", 0.0), s.get("avg_logprob"),
                          s.get("no_speech_prob"), s.get("compression_ratio"))
                    for s in body["segments"]
                ]
            else:
                pieces = [Piece(body["text"])] if body["text"].strip() else []
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise TranscribeError(f"openai returned an unexpected body: {r.text[:300]}") from e
        return Transcript(pieces, body.get("language"), None, self.name, self.model)
