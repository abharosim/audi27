"""Route one `turn` to an agent. Display only: this module never sends anything.

It reads `turn` events and writes one `route` event per turn. The id is
`route-<turn id>` and `ref` is the turn id, so writing the same turn again is
a no-op. Rules come from a TOML file (routes.toml if it exists, otherwise
routes.example.toml) that is re-read whenever it changes, so a save applies on
the next turn without a restart.

Decision order, first match wins:

1. an agent name in the text (发给 Muse / Cloud Code / 本地模型 / …),
   unless that name is explicitly negated (别用 / 不要用 / don't use)
2. a queue-control phrase that is the whole turn (跳过, 详细说说, …),
   including a repeat of that same phrase
3. a whole-turn "not a command" phrase (算了, 不用了, …) -> target `none`,
   including several of them stacked ("算了算了", "算了，不用了")
4. keyword rules, in file order
5. the optional local classifier, only if nothing above matched
6. the file's default target (grok unless the file says otherwise)

Claude Code and the local model are never chosen by a keyword or by the
classifier. Naming them is the only way. The classifier is the same backend
the corrector uses (`[correct] backend`), behind `[route] llm` (default off),
with a hard timeout. A timeout, a bad answer, or a missing model falls through
to the default. Tests pass a `classify` callable and never load a model.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .events import append, make_event, read_new, replay, snapshot_offsets

log = logging.getLogger("audi27.route")

ROUTE_SOURCE = "route"
AGENT_TARGETS = ("grok", "muse", "claude_code", "local")
KEYWORD_TARGETS = ("grok", "muse", "none")
LLM_TARGETS = ("grok", "muse", "none")
_CURSOR_NAME = "route.cursor.json"
_VERDICT_NAME = "route_verdicts.jsonl"
_EXAMPLE_NAME = "routes.example.toml"
_ROUTE_ID = re.compile(r"route-[^\s/]{1,240}\Z")
_NAME_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,40}\Z")
_TOP_KEYS = {"default", "agents", "address", "keyword", "whole", "queue", "confirm"}

# May sit around a queue or "not a command" phrase. Anything else
# ("跳过这个文件", "算了下个月的账") means the turn is a real request.
# Longest match wins, so list order is not the matcher. "谢谢" is the trailing
# thanks on "不用了谢谢"; "please" is the same after "skip".
_FILLERS = (
    "谢谢你", "谢谢了", "谢谢", "please",
    "这个", "这条", "一下", "当前", "汇报",
    "吧", "啊", "呀", "了", "的", "哦", "噢", "哈", "呢", "吗", "嘛", "哟",
)
_COMPACT_DROP = re.compile(r"[\s\u3000，。！？、,.!?;；:：\"'“”‘’「」『』（）()\[\]【】…·~～]+")
# Longest first. A trailing one counts only as its own word (the 请 in 申请 does not).
_POLITE = ("请帮我", "帮我", "麻烦你", "麻烦", "请你", "请", "please")
# Spoken Chinese puts one of these between a verb and its object: 推一下代码, 发封邮件.
_CONFIRM_GAP = "(?:一下子|一下|一个|一封|个|封)?"
_LABEL_PUNCT = "。.`\"'“”「」『』,，！!？?；;:：()（）[]【】"
# Longer alternative first. The name itself is not part of the match: the
# prefix must end on the negation, so "用 Claude Code" still addresses it
# and "不要用" is not swallowed by the shorter "别用".
_NEGATED_TAIL = re.compile(
    r"(?:不要用|别用|do\s+not\s+use|don't\s+use|dont\s+use)\s*$",
    re.IGNORECASE,
)

_LABEL = {
    "grok": "Grok",
    "muse": "Muse",
    "claude_code": "Claude Code",
    "local": "本地模型",
    "none": "不分派",
    "queue": "队列",
}

CLASSIFY_SYSTEM = (
    "你是语音秘书的分派器。这句话已经确认是对秘书说的。"
    "只判断交给谁：muse（职业规划、面试、简历、跳槽、薪资、晋升），"
    "none（不是要做事的指令，或听不清），grok（其余所有事）。"
    "不要选择 claude 或本地模型。只输出一个词：muse、grok 或 none。"
)

# Used only when neither routes.toml nor routes.example.toml can be read.
# The example file is the real rule list; keep this able to route the same
# kinds of sentence if that file is missing.
_FALLBACK_TOML = """
default = "grok"
address = ["send to", "问一下", "发给", "使用", "tell", "ask", "让", "用", "问", "给", "请"]
confirm = ["发邮件", "发送邮件", "付款", "支付", "付钱", "付一下", "转账", "买", "删除", "删掉", "删了", "删一下", "删个", "delete", "git push", "推代码", "推一下", "push", "发帖", "tweet", "post", "send the email", "send email"]
[agents]
muse = ["muse", "缪斯"]
grok = ["grok", "grog", "grokbot", "grok bot"]
claude_code = ["claude code", "cloud code", "crowed code"]
local = ["本地模型", "本地大模型", "local model", "local llm"]
[[keyword]]
name = "career"
target = "muse"
words = ["职业规划", "面试", "简历", "跳槽", "求职", "offer", "career", "resume", "interview"]
[[whole]]
name = "not_a_command"
target = "none"
phrases = ["算了", "不用了", "没事了", "不是对你说"]
[[queue]]
command = "details"
phrases = ["详细说说", "详细说"]
[[queue]]
command = "skip"
phrases = ["跳过"]
[[queue]]
command = "later"
phrases = ["稍后", "等一下"]
[[queue]]
command = "pending"
phrases = ["还有什么没处理", "还有什么"]
[[queue]]
command = "close"
phrases = ["关掉", "关闭", "别说了", "别念了"]
"""


class RulesError(ValueError):
    pass


@dataclass
class Rules:
    default: str = "grok"
    agents: list = field(default_factory=list)  # (target, compiled pattern)
    address: list = field(default_factory=list)  # verbs, longest first
    keywords: list = field(default_factory=list)  # (name, target, words)
    whole: list = field(default_factory=list)  # (name, compact phrase), longest first
    queue: list = field(default_factory=list)  # (command, compact phrase), longest first
    confirm: list = field(default_factory=list)
    path: str = ""


@dataclass
class Decision:
    target: str
    text: str
    turn_text: str
    rule: str
    reason: str
    method: str
    needs_confirmation: bool
    command: str | None = None

    def data(self) -> dict:
        return {
            "target": self.target,
            "text": self.text,
            "turn_text": self.turn_text,
            "rule": self.rule,
            "reason": self.reason,
            "method": self.method,
            "needs_confirmation": self.needs_confirmation,
            "command": self.command,
        }


def _example_path() -> Path:
    return Path(__file__).resolve().parent.parent / _EXAMPLE_NAME


def _strings(value, where: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise RulesError(f"{where} must be a non-empty list of strings")
    return [x.strip() for x in value]


def _target(value, allowed, where: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise RulesError(f"{where} must be one of: {', '.join(allowed)}")
    return value


def _token(value, where: str) -> str:
    if not isinstance(value, str) or not _NAME_TOKEN.fullmatch(value):
        raise RulesError(f"{where} must be a short lowercase name such as career or skip")
    return value


def _name_pattern(name: str) -> re.Pattern:
    """Match `name` with flexible spaces. Latin names stop at a word boundary, so museum is not muse."""
    parts = unicodedata.normalize("NFKC", name).split()
    body = r"\s*".join(re.escape(p) for p in parts)
    if re.match(r"[A-Za-z0-9]", parts[0]):
        body = r"(?<![A-Za-z0-9])" + body
    if re.search(r"[A-Za-z0-9]$", parts[-1]):
        body += r"(?![A-Za-z0-9])"
    return re.compile(body, re.IGNORECASE)


def load_rules_text(text: str, *, path: str = "") -> Rules:
    """Parse rules from TOML text. Unknown keys are an error, same as config.toml."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RulesError(f"cannot parse routes: {e}") from e
    if not isinstance(raw, dict):
        raise RulesError("routes file must be a TOML table")
    extra = set(raw) - _TOP_KEYS
    if extra:
        raise RulesError("unknown routes keys: " + ", ".join(sorted(extra)))
    default = _target(raw.get("default", "grok"), KEYWORD_TARGETS, "default")
    agents = []
    agent_table = raw.get("agents", {})
    if not isinstance(agent_table, dict):
        raise RulesError("[agents] must be a table of names")
    unknown_agents = set(agent_table) - set(AGENT_TARGETS)
    if unknown_agents:
        raise RulesError("unknown agent: " + ", ".join(sorted(unknown_agents)))
    for target in AGENT_TARGETS:
        names = agent_table.get(target)
        if names is None:
            continue
        for name in _strings(names, f"agents.{target}"):
            agents.append((target, _name_pattern(name)))
    address = []
    if "address" in raw:
        address = _strings(raw["address"], "address")
        address.sort(key=len, reverse=True)
    keywords = []
    for i, item in enumerate(raw.get("keyword", []), 1):
        if not isinstance(item, dict):
            raise RulesError(f"keyword #{i} must be a table")
        extra = set(item) - {"name", "target", "words"}
        if extra:
            raise RulesError(f"keyword #{i} has unknown keys: {', '.join(sorted(extra))}")
        name = _token(item.get("name"), f"keyword #{i} name")
        target = _target(item.get("target"), KEYWORD_TARGETS, f"keyword #{i} target")
        keywords.append((name, target, _strings(item.get("words"), f"keyword #{i} words")))
    whole = []
    if "whole" in raw:
        if not isinstance(raw["whole"], list):
            raise RulesError("[[whole]] must be an array of tables")
        for i, item in enumerate(raw["whole"], 1):
            if not isinstance(item, dict):
                raise RulesError(f"whole #{i} must be a table")
            extra = set(item) - {"name", "target", "phrases"}
            if extra:
                raise RulesError(f"whole #{i} has unknown keys: {', '.join(sorted(extra))}")
            _token(item.get("name"), f"whole #{i} name")
            _target(item.get("target"), ("none",), f"whole #{i} target")
            for phrase in _strings(item.get("phrases"), f"whole #{i} phrases"):
                compact = _compact(phrase)
                if not compact:
                    raise RulesError(f"whole #{i} phrase {phrase!r} is only punctuation")
                # target is none; the rule id keeps the table name (none:not_a_command).
                whole.append((item["name"], compact))
        whole.sort(key=lambda row: len(row[1]), reverse=True)
    queue = []
    if "queue" in raw:
        if not isinstance(raw["queue"], list):
            raise RulesError("[[queue]] must be an array of tables")
        for i, item in enumerate(raw["queue"], 1):
            if not isinstance(item, dict):
                raise RulesError(f"queue #{i} must be a table")
            extra = set(item) - {"command", "phrases"}
            if extra:
                raise RulesError(f"queue #{i} has unknown keys: {', '.join(sorted(extra))}")
            command = _token(item.get("command"), f"queue #{i} command")
            for phrase in _strings(item.get("phrases"), f"queue #{i} phrases"):
                compact = _compact(phrase)
                if not compact:
                    raise RulesError(f"queue #{i} phrase {phrase!r} is only punctuation")
                queue.append((command, compact))
        queue.sort(key=lambda row: len(row[1]), reverse=True)
    confirm = _strings(raw["confirm"], "confirm") if "confirm" in raw else []
    return Rules(default, agents, address, keywords, whole, queue, confirm, path)


def load_rules(path: str | Path) -> Rules:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise RulesError(f"cannot read {path}: {e}") from e
    rules = load_rules_text(text, path=str(path))
    rules.path = str(path)
    return rules


_cache: dict = {}
_warned: set = set()
_fallback: Rules | None = None


def _fallback_rules() -> Rules:
    global _fallback
    if _fallback is None:
        _fallback = load_rules_text(_FALLBACK_TOML, path="fallback")
    return _fallback


def resolve_rules_path(configured: str = "routes.toml") -> Path | None:
    """routes.toml if it is there, otherwise the committed example next to the package."""
    raw = Path(configured)
    path = raw if raw.is_absolute() else Path.cwd() / raw
    if path.is_file():
        return path
    example = _example_path()
    if example.is_file():
        key = str(path)
        if key not in _warned and path.resolve() != example.resolve():
            log.info("routes file %s is missing; using %s", path, example.name)
            _warned.add(key)
        return example
    return None


def get_rules(configured: str = "routes.toml", *, strict: bool = False) -> Rules:
    """Rules from `configured`, re-read when the file's mtime changes.

    A broken save keeps the previous good copy (and is logged). With no good
    copy yet, the example file is used so a typo does not drop the confirm
    list. `strict` raises instead, which is what `route test` and `route replay`
    want so a typo is visible. The live process uses the lenient path and keeps routing.
    """
    path = resolve_rules_path(configured)
    if path is None:
        return _fallback_rules()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError as e:
        if strict:
            raise RulesError(f"cannot read {path}: {e}") from e
        log.warning("routes file %s: %s", path, e)
        hit = _cache.get(str(path.resolve()))
        return hit[1] if hit else _backup_rules(path)
    key = str(path.resolve())
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        rules = load_rules(path)
    except (RulesError, OSError, UnicodeError) as e:
        log.warning("routes file %s: %s", path, e)
        if strict:
            raise RulesError(str(e)) from e
        return hit[1] if hit else _backup_rules(path)
    _cache[key] = (mtime, rules)
    return rules


def _backup_rules(failed: Path) -> Rules:
    """Example file when `failed` will not parse, else the embedded copy.

    A typo in routes.toml must not silently drop to the shorter embedded list
    while routes.example.toml is sitting next to the package.
    """
    example = _example_path()
    try:
        same = example.resolve() == Path(failed).resolve()
    except OSError:
        same = False
    if example.is_file() and not same:
        try:
            rules = load_rules(example)
        except (RulesError, OSError, UnicodeError) as e:
            log.warning("routes example %s: %s", example, e)
        else:
            key = f"backup:{failed}"
            if key not in _warned:
                log.warning("routes file %s is unusable; using %s", failed, example.name)
                _warned.add(key)
            return rules
    return _fallback_rules()


def _compact(text: str) -> str:
    return _COMPACT_DROP.sub("", unicodedata.normalize("NFKC", text).casefold())


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _rstrip_sep(text: str) -> str:
    i = len(text)
    while i > 0:
        ch = text[i - 1]
        if ch.isspace() or unicodedata.category(ch)[:1] == "P":
            i -= 1
            continue
        break
    return text[:i]


def _lstrip_sep(text: str) -> str:
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or unicodedata.category(ch)[:1] == "P":
            i += 1
            continue
        break
    return text[i:]


def _cut_suffix(text: str, suffix: str) -> str | None:
    if len(text) < len(suffix) or not suffix:
        return None
    if text[-len(suffix):].casefold() != suffix.casefold():
        return None
    return text[:-len(suffix)]


def _peel_polite(text: str) -> str:
    """Drop one trailing 帮我/请/please that stands alone, not the 请 inside 申请 or 邀请."""
    for polite in _POLITE:
        cut = _cut_suffix(text, polite)
        if cut is None:
            continue
        if cut == "" or _rstrip_sep(cut) != cut:
            return cut
    return text


def _verb_before(left: str, verbs) -> str | None:
    """The address verb ending `left`, if it is the whole left side, after punctuation, or after 帮我/请/please.

    使用 ends with 用, but 用 is not an address there unless 使用 itself is the verb.
    """
    bare = _rstrip_sep(left)
    for verb in verbs:
        before = _cut_suffix(bare, verb)
        if before is None:
            continue
        if before == "":
            return verb
        if _rstrip_sep(before) != before:
            return verb
        if _peel_polite(before) != before:
            return verb
    return None


def _message(text: str, start: int, end: int, verbs) -> str:
    """Drop the address verb and the agent name. Leave the text alone when the name is just mentioned."""
    if _verb_before(text[:start], verbs) is None:
        return _collapse(text)
    before = _rstrip_sep(text[:start])
    for verb in verbs:
        cut = _cut_suffix(before, verb)
        if cut is None:
            continue
        if cut == "" or _rstrip_sep(cut) != cut or _peel_polite(cut) != cut:
            before = _rstrip_sep(_peel_polite(_rstrip_sep(cut)))
            break
    right = _lstrip_sep(text[end:])
    return _collapse(f"{before} {right}".strip())


@dataclass
class _Hit:
    target: str
    start: int
    end: int
    addressed: bool


def _is_negated(prefix: str) -> bool:
    """True when `prefix` ends with 别用 / 不要用 / don't use (optional space before the name)."""
    text = unicodedata.normalize("NFKC", prefix).replace("\u2019", "'").rstrip()
    return _NEGATED_TAIL.search(text) is not None


def _split_agents(text: str, rules: Rules) -> tuple[_Hit | None, set[str]]:
    """Positive name hits, plus targets whose every mention is negated.

    A negated mention is not a candidate. "别用 Claude Code，问 Muse" keeps Muse.
    A target that is only ever negated (别用 Grok, and Grok is also the default)
    is returned in the blocked set so a later rule cannot select it either.
    """
    hits = []
    flags: dict[str, list[bool]] = {}
    for target, pattern in rules.agents:
        for match in pattern.finditer(text):
            negated = _is_negated(text[:match.start()])
            flags.setdefault(target, []).append(negated)
            if negated:
                continue
            hits.append(_Hit(target, match.start(), match.end(),
                             _verb_before(text[:match.start()], rules.address) is not None))
    blocked = {target for target, seen in flags.items() if seen and all(seen)}
    if not hits:
        return None, blocked
    # An addressed name beats a bare mention. Otherwise the earliest, then the longest
    # (grok bot rather than the grok inside it).
    return max(hits, key=lambda h: (h.addressed, -h.start, h.end - h.start)), blocked


def _starts_with_polite(body: str) -> bool:
    for polite in _POLITE:
        token = _compact(polite)
        if token and body.startswith(token):
            return True
    return False


def _strip_leading_polite(body: str, rows) -> str:
    """Drop 请/帮我/please only when what follows is still a command, not the 请 in 请求."""
    while body:
        stripped = False
        for polite in _POLITE:
            token = _compact(polite)
            if not token or not body.startswith(token):
                continue
            rest = body[len(token):]
            if any(rest.startswith(phrase) for _, phrase in rows) or _starts_with_polite(rest):
                body = rest
                stripped = True
                break
        if not stripped:
            break
    return body


def _phrase_cover(body: str, phrases: list[str]) -> bool:
    """True when `body` is only these phrases and fillers, and at least one phrase."""
    phrases = sorted(phrases, key=len, reverse=True)
    saw = False
    while body:
        best = 0
        is_phrase = False
        for phrase in phrases:
            if len(phrase) > best and body.startswith(phrase):
                best = len(phrase)
                is_phrase = True
        for filler in _FILLERS:
            if len(filler) > best and body.startswith(filler):
                best = len(filler)
                is_phrase = False
        if best == 0:
            return False
        saw = saw or is_phrase
        body = body[best:]
    return saw


def _match_phrase(compact: str, rows, *, mix: bool) -> str | None:
    """Whole-turn match. `mix` allows different phrases from `rows` in one turn.

    Queue commands stay unmixed, so "跳过稍后" is not a command. "算了，不用了"
    and "跳过跳过" still match. A phrase has to start the turn (after 请/帮我).
    """
    body = _strip_leading_polite(compact, rows)
    grouped: dict[str, list[str]] = {}
    for label, phrase in rows:
        grouped.setdefault(label, []).append(phrase)
    for label, phrase in rows:
        if not body.startswith(phrase):
            continue
        allowed = [item for group in grouped.values() for item in group] if mix else grouped[label]
        if _phrase_cover(body, allowed):
            return label
    return None


def _has(text_folded: str, word: str, *, gap: bool = False) -> bool:
    word_f = unicodedata.normalize("NFKC", word).casefold().strip()
    if not word_f:
        return False
    if re.search(r"[a-z0-9]", word_f):
        parts = [p for p in word_f.split() if p]
        if not parts:
            return False
        body = r"\s+".join(re.escape(p) for p in parts)
        if re.match(r"[a-z0-9]", parts[0]):
            body = r"(?<![a-z0-9])" + body
        if re.search(r"[a-z0-9]$", parts[-1]):
            body += r"(?![a-z0-9])"
        return re.search(body, text_folded) is not None
    if gap and len(word_f) > 1:
        body = _CONFIRM_GAP.join(re.escape(ch) for ch in word_f)
        # 对付一下 is "deal with it", not a payment. 帮我付一下 is.
        if word_f.startswith("付"):
            body = r"(?<!对)" + body
        return re.search(body, text_folded) is not None
    return word_f in text_folded


def needs_confirmation(text: str, rules: Rules) -> bool:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return any(_has(folded, word, gap=True) for word in rules.confirm)


def _reason(rule: str) -> str:
    if rule.startswith("negated:"):
        return "点名排除了 " + _LABEL.get(rule.split(":", 1)[1], rule.split(":", 1)[1])
    if rule.startswith("explicit:"):
        return "点名了 " + _LABEL.get(rule.split(":", 1)[1], rule.split(":", 1)[1])
    if rule.startswith("keyword:"):
        return "关键词 " + rule.split(":", 1)[1]
    if rule.startswith("queue:"):
        return "队列指令 " + rule.split(":", 1)[1]
    if rule.startswith("none:"):
        return "不像指令"
    return {"empty": "没有内容", "llm": "本地模型判断", "default": "默认"}.get(rule, rule)


def _parse_label(raw) -> str | None:
    """muse / grok / none, after fullwidth and surrounding punctuation. Anything else is refused."""
    if not isinstance(raw, str):
        return None
    token = unicodedata.normalize("NFKC", raw).strip().split()[:1]
    if not token:
        return None
    label = token[0].strip(_LABEL_PUNCT).casefold()
    return label if label in LLM_TARGETS else None


def _within(fn, timeout_s: float):
    """Run `fn` on a daemon thread. None on timeout or any exception."""
    if timeout_s <= 0:
        return None
    box: dict = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as e:
            box["e"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive() or "e" in box:
        return None
    return box.get("v")


def decide(text: str, rules: Rules, *, classify=None, timeout_s: float = 8.0) -> Decision:
    """Route one turn's text. `classify(text)` may return muse, grok, or none. It is not called when a rule matches."""
    raw = text if isinstance(text, str) else ""
    turn_text = raw.strip()
    norm = unicodedata.normalize("NFKC", turn_text)
    confirm = needs_confirmation(norm, rules) if norm else False

    def done(target, message, rule, method, command=None) -> Decision:
        return Decision(target, message, turn_text, rule, _reason(rule), method, confirm, command)

    if _compact(norm) == "":
        return done("none", "", "empty", "rule")

    hit, blocked = _split_agents(norm, rules)

    def finish(decision: Decision) -> Decision:
        # "别用 Grok" must not fall through to the default and select Grok anyway.
        if decision.target not in blocked:
            return decision
        rule = f"negated:{decision.target}"
        return Decision("none", decision.text, decision.turn_text, rule, _reason(rule), "rule",
                        decision.needs_confirmation, None)

    if hit is not None:
        return finish(done(hit.target, _message(norm, hit.start, hit.end, rules.address),
                           f"explicit:{hit.target}", "rule"))

    command = _match_phrase(_compact(norm), rules.queue, mix=False)
    if command is not None:
        return finish(done("queue", _collapse(norm), f"queue:{command}", "rule", command))

    whole = _match_phrase(_compact(norm), rules.whole, mix=True)
    if whole is not None:
        return finish(done("none", _collapse(norm), f"none:{whole}", "rule"))

    folded = norm.casefold()
    for name, target, words in rules.keywords:
        if any(_has(folded, word) for word in words):
            return finish(done(target, _collapse(norm), f"keyword:{name}", "rule"))

    if classify is not None:
        label = _parse_label(_within(lambda: classify(norm), timeout_s))
        if label is not None:
            return finish(done(label, _collapse(norm), "llm", "llm"))

    return finish(done(rules.default, _collapse(norm), "default", "default"))


def route_id(turn_id: str) -> str:
    return "route-" + turn_id


def route_event(turn: dict, decision: Decision, *, screen: dict | None = None) -> dict:
    """One `route` event. The id is fixed from the turn id, so a replay can be repeated.

    `screen` is the public app/title snapshot already chosen for this turn.
    Omit it and the event has no screen field.
    """
    data = decision.data()
    if screen:
        from .screen import public_context

        view = public_context(screen)
        if view:
            data["screen"] = view
    return make_event("route", ROUTE_SOURCE, data, ts=turn["ts"],
                      ref=turn["id"], id=route_id(turn["id"]))


def _screen_for_route(data_dir, cfg, turn: dict) -> dict | None:
    """Screen context for this turn when `[screen] enabled`, else None.

    A snapshot already stored on the turn wins. Otherwise the latest screen
    event at the turn's timestamp is used, when the sampler's heartbeat says
    that snapshot was still current. An excluded app's title is dropped again
    here, including one that was stored before the app was excluded.
    """
    from .screen import _strip_excluded, context_for, enabled, public_context

    if not enabled(cfg):
        return None
    data = turn.get("data") if isinstance(turn.get("data"), dict) else {}
    attached = data.get("screen")
    if isinstance(attached, dict):
        view = public_context(attached)
        if view is not None:
            return _strip_excluded(view, cfg)
    return context_for(data_dir, cfg, turn.get("ts"))


def format_decision(decision: Decision) -> str:
    command = decision.command or "-"
    flag = "yes" if decision.needs_confirmation else "no"
    lines = [
        f"target: {decision.target}",
        f"command: {command}",
        f"method: {decision.method}",
        f"rule: {decision.rule}",
        f"reason: {decision.reason}",
        f"needs_confirmation: {flag}",
        f"text: {decision.text}",
    ]
    return "\n".join(lines)


def format_route(event: dict) -> str:
    data = event.get("data") or {}
    try:
        stamp = datetime.fromisoformat(event["ts"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, KeyError):
        stamp = str(event.get("ts", ""))
    flag = "!" if data.get("needs_confirmation") else " "
    command = data.get("command") or ""
    command = f" {command}" if command else ""
    rule = data.get("rule") or ""
    text = data.get("text") if data.get("text") else data.get("turn_text") or ""
    return f"{stamp} {flag} {data.get('target', ''):<12}{command:<10} {rule:<22} {text}"


def make_classifier(cfg):
    """The corrector's local model, or None when `[route] llm` is off or the backend cannot be built.

    The model is not loaded here. The first call loads it, and `decide` gives up at `timeout_s`.
    """
    if not getattr(getattr(cfg, "route", None), "llm", False):
        return None
    try:
        from .correct import build_corrector

        corrector = build_corrector(cfg.correct.backend, cfg.corrector.get(cfg.correct.backend))
    except Exception as e:
        log.warning("route classifier is off: %s", e)
        return None

    lock = threading.Lock()

    def classify(text: str) -> str:
        # One call at a time. A timed-out call may still be running; the next
        # turn waits only inside `decide`'s timeout and then uses the default.
        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": text},
        ]
        with lock:
            return corrector.complete(messages, 16)

    return classify


def _turn_text(event: dict) -> str:
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    text = data.get("text")
    return text if isinstance(text, str) else ""


def replay_routes(data_dir, cfg, *, since=None, until=None, classify=None) -> list[dict]:
    """Route stored turns in `[since, until)`. Does not write. Re-reads the rules file per turn."""
    timeout = cfg.route.timeout_s
    if classify is None and cfg.route.llm:
        classify = make_classifier(cfg)
    out = []
    for event in replay(data_dir, since=since, until=until, types="turn"):
        rules = get_rules(cfg.route.rules, strict=True)
        decision = decide(_turn_text(event), rules, classify=classify, timeout_s=timeout)
        out.append(route_event(event, decision, screen=_screen_for_route(data_dir, cfg, event)))
    return out


def write_routes(data_dir, events) -> dict:
    """Append route events. An id already in that day's file is skipped."""
    wrote = skipped = 0
    for event in events:
        stored = append(data_dir, event, if_absent=True)
        if stored is None:
            skipped += 1
        else:
            wrote += 1
    return {"wrote": wrote, "skipped": skipped}


def _cursor_path(data_dir) -> Path:
    return Path(data_dir) / _CURSOR_NAME


def _save_cursor(data_dir, offsets: dict) -> None:
    path = _cursor_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"offsets": offsets}, ensure_ascii=False, sort_keys=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _load_cursor(data_dir) -> dict:
    """Resume offset, or the current end when this stream has no cursor yet.

    The first start snapshots the end, so history is not backfilled.
    """
    path = _cursor_path(data_dir)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            offs = raw.get("offsets") if isinstance(raw, dict) else None
            if isinstance(offs, dict) and all(
                    isinstance(k, str) and type(v) is int and v >= 0 for k, v in offs.items()):
                return dict(offs)
        except (OSError, json.JSONDecodeError, UnicodeError):
            log.warning("ignoring unreadable route cursor %s", path)
    offsets = snapshot_offsets(data_dir)
    try:
        _save_cursor(data_dir, offsets)
    except OSError:
        log.exception("route cursor was not written")
    return offsets


def _emit(data_dir, turn: dict, cfg, classify) -> bool:
    """Append the route. False only when the write failed; a bad turn is skipped so it cannot block the cursor."""
    try:
        rules = get_rules(cfg.route.rules, strict=False)
        decision = decide(_turn_text(turn), rules, classify=classify, timeout_s=cfg.route.timeout_s)
        event = route_event(turn, decision, screen=_screen_for_route(data_dir, cfg, turn))
    except Exception:
        log.exception("route skipped turn %s", turn.get("id"))
        return True
    try:
        stored = append(data_dir, event, if_absent=True)
    except Exception:
        log.exception("route event was not written for %s", turn.get("id"))
        return False
    if stored is not None:
        log.info("%s %s", decision.target, (decision.text or decision.turn_text)[:100] or "-")
    return True


def serve(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None,
          ready: threading.Event | None = None, classify=None) -> None:
    """Follow new turn events and emit routes. Blocks until `stop` or Ctrl-C.

    The first start snapshots the end of the stream and does not backfill.
    Later starts resume from `route.cursor.json`. Rules are re-read per turn.
    Nothing is sent to an agent.
    """
    if classify is None and cfg.route.llm:
        classify = make_classifier(cfg)
    stop = stop or threading.Event()
    offsets = _load_cursor(data_dir)
    durable = dict(offsets)
    cursor_ok = True

    def save(cursor: dict) -> None:
        nonlocal durable
        if cursor == durable:
            return
        try:
            _save_cursor(data_dir, cursor)
        except OSError:
            log.exception("route cursor was not written")
            return
        durable = dict(cursor)

    log.info("router watching %s (rules %s, llm %s)", data_dir, cfg.route.rules, cfg.route.llm)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            found, offsets = read_new(data_dir, offsets, types="turn")
            for event in found:
                cursor_ok = _emit(data_dir, event, cfg, classify) and cursor_ok
            if cursor_ok:
                save(offsets)
            if stop.wait(poll_s):
                break
    except KeyboardInterrupt:
        log.info("stopping")
        stop.set()
    finally:
        try:
            found, offsets = read_new(data_dir, offsets, types="turn")
            for event in found:
                cursor_ok = _emit(data_dir, event, cfg, classify) and cursor_ok
            if cursor_ok:
                save(offsets)
        except Exception:
            log.exception("router did not flush on the way out")


# --- dashboard verdicts -------------------------------------------------------
# Eval verdicts (data/verdicts.jsonl) store the transcript text judged correct and only
# accept a numeric segment id. A route mark is correct/incorrect on a decision, so it
# lives in its own append-only file with the same last-line-wins rule.

def save_route_verdict(root, route_id: str, *, correct: bool | None = None, cleared: bool = False) -> dict:
    if not isinstance(route_id, str) or not _ROUTE_ID.fullmatch(route_id):
        raise ValueError("bad id")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if cleared:
        rec = {"id": route_id, "cleared": True}
    else:
        rec = {"id": route_id, "correct": bool(correct)}
    rec["ts"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path = root / _VERDICT_NAME
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def load_route_verdicts(root) -> dict:
    out: dict = {}
    path = Path(root) / _VERDICT_NAME
    if not path.is_file():
        return out
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return out
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str):
            continue
        if rec.get("cleared"):
            out.pop(rec["id"], None)
        elif isinstance(rec.get("correct"), bool):
            out[rec["id"]] = rec
    return out


def list_routes(data_dir, *, limit: int = 300) -> dict:
    """Recent route events, newest first, plus a running accuracy over every saved mark."""
    verdicts = load_route_verdicts(data_dir)
    judged = [v for v in verdicts.values() if isinstance(v.get("correct"), bool)]
    correct = sum(1 for v in judged if v["correct"])
    events = list(replay(data_dir, types="route"))
    if limit > 0:
        events = events[-limit:]
    rows = []
    for event in reversed(events):
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        mark = verdicts.get(event["id"])
        rows.append({
            "id": event["id"],
            "ts": event.get("ts"),
            "turn_text": data.get("turn_text") or "",
            "text": data.get("text") or "",
            "target": data.get("target") or "",
            "rule": data.get("rule") or "",
            "reason": data.get("reason") or data.get("rule") or "",
            "method": data.get("method") or "",
            "needs_confirmation": bool(data.get("needs_confirmation")),
            "command": data.get("command"),
            "verdict": None if mark is None else mark.get("correct"),
        })
    return {
        "routes": rows,
        "accuracy": {
            "correct": correct,
            "judged": len(judged),
            "rate": None if not judged else round(correct / len(judged), 4),
        },
    }
