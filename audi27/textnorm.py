"""Tokenisation shared by validation, correction guards and model comparison."""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[㐀-鿿豈-﫿]|[a-z0-9]+(?:'[a-z]+)?")

# Hesitation sounds. A segment made only of these carries no content; inside a sentence they are
# ignored when two transcripts are compared (one model keeps "呃", another drops it: not an error).
FILLERS = frozenset("嗯 呃 唔 啊 哦 噢 哎 um umm uh uhh uhm er erm hmm hm mm mmm mhm huh".split())
# Spelling variants that are not recognition errors: 的/地/得, and the optional 儿 suffix.
_VARIANTS = {"地": "的", "得": "的"}
_DROP = frozenset("儿")


def raw_tokens(text: str) -> list[str]:
    """CJK characters one by one, Latin words lower-cased, punctuation dropped."""
    return _TOKEN.findall(unicodedata.normalize("NFKC", text or "").lower())


def cmp_tokens(text: str) -> list[str]:
    """Tokens for comparing two transcripts of the same audio: fillers and 儿 removed, 的/地/得 merged,
    immediate repeats ("他他", "会会") collapsed. Applied to both sides, so it only hides differences
    that are not worth a human's attention."""
    out: list[str] = []
    for t in raw_tokens(text):
        if t in FILLERS or t in _DROP:
            continue
        t = _VARIANTS.get(t, t)
        if out and out[-1] == t:
            continue
        out.append(t)
    return out


def is_filler_only(text: str) -> bool:
    toks = raw_tokens(text)
    return bool(toks) and all(t in FILLERS for t in toks)


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
