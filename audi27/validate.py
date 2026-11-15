"""Decide whether a transcript can be trusted. Nothing is deleted here: every result is
labelled ok / suspect / rejected with reasons, and the store records all three."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .config import ValidateCfg
from .textnorm import is_filler_only
from .transcribers.base import Transcript

OK, SUSPECT, REJECTED = "ok", "suspect", "rejected"

# Phrases Whisper emits on silence/noise (learned from subtitle training data).
# Rejected when they make up essentially the whole text.
HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "like and subscribe", "subtitles by", "see you next time",
    "谢谢观看", "感谢观看", "谢谢大家观看", "感谢收看", "请订阅", "订阅我的频道",
    "请不吝点赞", "字幕由", "字幕提供", "字幕志愿者", "下期再见",
    "謝謝觀看", "感謝觀看", "請訂閱", "字幕志願者",
    "词曲", "作词", "作曲", "编曲", "詞曲", "作詞",
]
# Rejected wherever they appear.
ALWAYS_HALLUCINATION = ["李宗盛", "amara.org", "明镜与点点", "明鏡與點點", "优优独播剧场", "優優獨播劇場"]

_REPEAT = re.compile(r"(.{2,30}?)\1{3,}", re.S)  # same 2-30 char unit 4+ times in a row
_WS = re.compile(r"\s+")


@dataclass
class Verdict:
    status: str
    text: str                       # text of the pieces that passed
    reasons: list[str] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def validate(t: Transcript, duration: float, cfg: ValidateCfg,
             prev_text: str = "", prompt: str = "") -> Verdict:
    reasons: list[str] = []
    suspect = False
    kept = []
    for p in t.pieces:
        if not p.text.strip():
            continue
        lp, nsp, cr = (_finite(x) for x in (p.avg_logprob, p.no_speech_prob, p.compression_ratio))
        if (lp, nsp, cr) != (p.avg_logprob, p.no_speech_prob, p.compression_ratio):
            reasons.append("invalid_confidence_metrics")  # NaN/inf from the backend: cannot be trusted
            suspect = True
        if nsp is not None and lp is not None and nsp > cfg.max_no_speech_prob and lp < cfg.min_avg_logprob:
            reasons.append("piece_dropped:no_speech")
            continue
        if cr is not None and cr > cfg.max_compression_ratio:
            reasons.append("piece_dropped:high_compression_ratio")
            continue
        if lp is not None and lp < cfg.min_avg_logprob:
            reasons.append("low_confidence")
            suspect = True
        elif nsp is not None and nsp > cfg.max_no_speech_prob:
            reasons.append("high_no_speech_prob")
            suspect = True
        kept.append(p)

    text = "".join(p.text for p in kept).strip()
    lps = [x for x in (_finite(p.avg_logprob) for p in kept) if x is not None]
    nsps = [x for x in (_finite(p.no_speech_prob) for p in kept) if x is not None]
    crs = [x for x in (_finite(p.compression_ratio) for p in kept) if x is not None]
    v = Verdict(OK, text, reasons,
                round(sum(lps) / len(lps), 3) if lps else None,
                round(max(nsps), 3) if nsps else None,
                round(max(crs), 3) if crs else None)

    def reject(reason: str) -> Verdict:
        v.status = REJECTED
        v.reasons.append(reason)
        return v

    if not text:
        return reject("empty")
    n = _norm(text)
    core = re.sub(r"[\W_]+", "", n)
    if not core:
        return reject("no_words")  # punctuation only, e.g. "!" or "..."
    for phrase in ALWAYS_HALLUCINATION:
        if phrase in n:
            return reject(f"known_hallucination:{phrase}")
    for phrase in HALLUCINATION_PHRASES + [_norm(x) for x in cfg.extra_hallucination_phrases]:
        if phrase and phrase in n and len(core) <= len(re.sub(r"[\W_]+", "", phrase)) + 6:
            return reject(f"known_hallucination:{phrase}")
    if is_filler_only(text):
        return reject("filler_only")  # "嗯", "um", "Mm-hmm": real sound, no content
    if _REPEAT.search(_WS.sub("", text)):
        return reject("repetition_loop")
    np_ = _norm(prompt)
    if np_ and len(core) >= 4 and n in np_:
        return reject("prompt_echo")

    if not lps and not nsps:
        v.reasons.append("no_confidence_metrics")  # informational: text heuristics only
    if duration > 0 and len(core) / duration > cfg.max_chars_per_sec:
        v.reasons.append("too_much_text_for_duration")
        suspect = True
    if t.language_prob is not None and t.language_prob < cfg.min_language_prob:
        v.reasons.append("language_uncertain")
        suspect = True
    if prev_text and len(core) >= 6 and n == _norm(prev_text):
        v.reasons.append("duplicate_of_previous")
        suspect = True
    if suspect:
        v.status = SUSPECT
    return v
