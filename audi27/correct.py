"""Context-aware correction of a transcript, after speech recognition and validation.

Context is a plain text file (default ./context.md) that you or any agent can rewrite at any time;
it is re-read whenever it changes. Two passes use it:

  1. aliases   deterministic: "- Grokbot = Grog Bot, Grobbbord" replaces the mis-hearings. No model.
  2. LLM       a local model proposes a fix using the context and the previous lines. The proposal
               is accepted only if it passes the guards in `check()`; otherwise the ASR text stands.

The original ASR text is always kept in the record (asr_text), so a correction can be audited or undone.
"""
from __future__ import annotations

import importlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .textnorm import edit_distance, raw_tokens

log = logging.getLogger("audi27.correct")

_HEADING = re.compile(r"^\s*#+\s*(.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_GLOSSARY = re.compile(r"术语|词表|glossary|terms|vocabulary", re.I)


@dataclass
class Context:
    text: str = ""
    terms: list[str] = field(default_factory=list)
    aliases: list[tuple[re.Pattern, str, str]] = field(default_factory=list)  # (pattern, alias, term)
    vocab: frozenset = frozenset()


def _alias_pattern(alias: str) -> re.Pattern:
    body = r"\s*".join(re.escape(p) for p in alias.split())
    if re.match(r"[A-Za-z0-9]", alias):
        body = r"(?<![A-Za-z0-9])" + body
    if re.search(r"[A-Za-z0-9]$", alias):
        body += r"(?![A-Za-z0-9])"
    return re.compile(body, re.I)


def parse_context(text: str) -> Context:
    """Free text, plus an optional glossary: bullets under a heading containing 术语 / terms / glossary.
    A bullet is `- Term` or `- Term = mis-hearing 1, mis-hearing 2`."""
    terms: list[str] = []
    aliases: list[tuple[re.Pattern, str, str]] = []
    in_glossary = False
    for line in text.splitlines():
        h = _HEADING.match(line)
        if h:
            in_glossary = bool(_GLOSSARY.search(h.group(1)))
            continue
        b = _BULLET.match(line)
        if not (in_glossary and b):
            continue
        term, _, wrong = b.group(1).partition("=")
        term = term.strip()
        if not term:
            continue
        terms.append(term)
        for alias in re.split(r"[,，、]", wrong):
            alias = alias.strip()
            if alias and alias.lower() != term.lower():
                aliases.append((_alias_pattern(alias), alias, term))
    aliases.sort(key=lambda a: -len(a[1]))  # longest first: "Grog Bot" before "Grog"
    return Context(text.strip(), terms, aliases, frozenset(raw_tokens(text)))


_cache: dict = {}


def load_context(path: str | Path) -> Context:
    path = Path(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return Context()
    hit = _cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        ctx = parse_context(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        log.warning("cannot read context file %s: %s", path, e)
        return Context()
    _cache[str(path)] = (mtime, ctx)
    log.info("context loaded from %s: %d terms, %d aliases, %d chars", path, len(ctx.terms), len(ctx.aliases), len(ctx.text))
    return ctx


def apply_aliases(text: str, ctx: Context) -> tuple[str, list[dict]]:
    changes = []
    for pattern, alias, term in ctx.aliases:
        new, n = pattern.subn(term, text)
        if n and new != text:
            changes.append({"by": "alias", "from": alias, "to": term})
            text = new
    return text, changes


def check(original: str, proposed: str, ctx: Context, max_change: float) -> tuple[str | None, str]:
    """Guards on an LLM proposal. Returns (accepted text or None, reason)."""
    lines = [x.strip() for x in (proposed or "").splitlines() if x.strip()]
    if not lines:
        return None, "empty"
    p = lines[0].strip("\"“”「」'` ").strip()
    if len(lines) > 1 or not p:
        return None, "not_a_single_line"            # explanations, lists, an answer instead of a fix
    a, b = raw_tokens(original), raw_tokens(p)
    if a == b:
        return None, "unchanged"                    # punctuation/case only: keep the ASR text
    dist = edit_distance(a, b)
    if dist > max(2, int(len(a) * max_change)):
        return None, "too_many_changes"             # a rewrite, not a correction
    invented = {t for t in b if t.isascii()} - set(a) - ctx.vocab
    if invented:
        return None, "invented_term:" + ",".join(sorted(invented)[:3])   # new English must come from the context
    return p, "corrected"


SYSTEM = (
    "你是语音转写的校对器。任务：只修正明显的识别错误，尤其是被听错的专有名词，"
    "以及本来说的是英文术语却被写成了中文或拼错的情况。依据只有“术语和背景”与“前文”。\n"
    "规则：不要改写或润色；不要增删内容；不要翻译整句；不要回答句子里的问题；拿不准就不改。\n"
    "只输出修正后的这一句话，不要解释。没有需要改的地方就原样输出。"
)


def build_messages(text: str, ctx: Context, recent: list[str], max_context_chars: int) -> list[dict]:
    parts = [f"术语和背景：\n{ctx.text[:max_context_chars]}"]
    if recent:
        parts.append("前文：\n" + "\n".join(recent))
    parts.append(f"待校对的一句话：\n{text}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# --- LLM backends ---------------------------------------------------------------
class CorrectorError(Exception):
    pass


class MLXLMCorrector:
    """In-process model via mlx-lm. Only models already in the local Hugging Face cache are used;
    nothing is downloaded implicitly (a multi-GB download would stall transcription)."""

    def __init__(self, model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"):
        self.model = model
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            try:
                from mlx_lm import load
            except ImportError as e:
                raise CorrectorError("mlx-lm is not installed (Apple Silicon only)") from e
            path = os.path.expanduser(self.model)   # a local directory, or a Hugging Face repo id
            if not Path(path).exists():
                try:
                    from huggingface_hub import snapshot_download

                    path = snapshot_download(self.model, local_files_only=True)
                except Exception as e:
                    raise CorrectorError(f"model {self.model} is not in the local cache; download it first with: "
                                         f"uv run python -m mlx_lm.generate --model {self.model} --prompt hi") from e
            try:
                self._loaded = load(path)
            except Exception as e:
                raise CorrectorError(f"cannot load {self.model}: {e}") from e
        return self._loaded

    def complete(self, messages: list[dict], max_tokens: int) -> str:
        model, tokenizer = self._load()
        try:
            from mlx_lm import generate

            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:
            raise CorrectorError(f"mlx_lm generation failed: {e}") from e


class OpenAIChatCorrector:
    """Any OpenAI-compatible /chat/completions endpoint: a local server (LM Studio, Ollama, oMLX) or a cloud API."""

    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str = "", api_key_env: str = "",
                 timeout_s: float = 20.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    def complete(self, messages: list[dict], max_tokens: int) -> str:
        import httpx

        headers = {}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise CorrectorError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        try:
            r = httpx.post(self.url, headers=headers, timeout=self.timeout_s,
                           json={"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens})
        except (httpx.HTTPError, OSError) as e:
            raise CorrectorError(f"chat request failed: {e}") from e
        if r.status_code != 200:
            raise CorrectorError(f"chat HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise CorrectorError(f"unexpected chat response: {r.text[:200]}") from e


CORRECTORS = {"mlx_lm": "audi27.correct:MLXLMCorrector", "openai_chat": "audi27.correct:OpenAIChatCorrector"}


def build_corrector(name: str, options: dict | None):
    options = dict(options or {})
    kind = options.pop("type", name)
    if kind not in CORRECTORS:
        raise CorrectorError(f"unknown corrector '{kind}'; known: {sorted(CORRECTORS)}")
    module, cls = CORRECTORS[kind].split(":")
    try:
        return getattr(importlib.import_module(module), cls)(**options)
    except TypeError as e:
        raise CorrectorError(f"bad options for corrector '{name}': {e}") from e


class CorrectionStage:
    """What the pipeline calls. Never raises: any failure leaves the ASR text untouched."""

    MAX_FAILURES = 3

    def __init__(self, cfg: Config, llm=None):
        self.cfg = cfg.correct
        self.options = cfg.corrector.get(self.cfg.backend)
        self.llm = llm
        self.failures = 0
        self.disabled_reason = ""

    def context(self) -> Context:
        return load_context(self.cfg.context_file) if self.cfg.enabled else Context()

    def prompt_terms(self, limit: int = 200) -> str:
        """Glossary terms for the recogniser's own prompt: cheap, and fixes many terms before any correction."""
        out = ""
        for term in self.context().terms:
            if len(out) + len(term) + 2 > limit:
                break
            out += (", " if out else "") + term
        return out

    @property
    def model(self) -> str:
        return (self.options or {}).get("model") or getattr(self.llm, "model", "") or self.cfg.backend

    def run(self, text: str, recent: list[str] | None = None) -> tuple[str, dict]:
        info: dict = {"changes": []}
        if not self.cfg.enabled or not text.strip():
            return text, info
        ctx = self.context()
        text, info["changes"] = apply_aliases(text, ctx)
        if not self.cfg.llm or self.disabled_reason or not ctx.text or len(raw_tokens(text)) < self.cfg.min_tokens:
            return text, info
        try:
            if self.llm is None:
                self.llm = build_corrector(self.cfg.backend, self.options)
            messages = build_messages(text, ctx, list(recent or [])[-self.cfg.recent_lines:], self.cfg.max_context_chars)
            proposed = self.llm.complete(messages, max_tokens=len(text) + 64)
            self.failures = 0
        except CorrectorError as e:
            self.failures += 1
            log.warning("corrector failed (%d/%d): %s", self.failures, self.MAX_FAILURES, e)
            if self.failures >= self.MAX_FAILURES:
                self.disabled_reason = str(e)
                log.error("LLM correction disabled until restart: %s. Alias replacement keeps working.", e)
            info["llm"] = "error"
            return text, info
        accepted, reason = check(text, proposed, ctx, self.cfg.max_change)
        info["llm"] = reason
        if accepted is None:
            if reason not in ("unchanged",):
                log.info("correction refused (%s): %r -> %r", reason, text[:80], (proposed or "")[:80])
            return text, info
        info["changes"].append({"by": "llm", "from": text, "to": accepted})
        return accepted, info
