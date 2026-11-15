"""Typed config loaded from TOML. Unknown keys raise, so typos fail loudly."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    pass


@dataclass
class AudioCfg:
    device: Any = ""
    sample_rate: int = 16000
    frame_ms: int = 30


@dataclass
class SegmentCfg:
    start_margin_db: float = 10.0
    min_db: float = -50.0
    start_ms: int = 150
    end_silence_ms: int = 800
    pre_roll_ms: int = 300
    min_speech_ms: int = 400
    max_segment_s: float = 30.0


@dataclass
class TranscribeCfg:
    backends: list = field(default_factory=lambda: ["mlx_whisper"])
    # [] = any language (auto-detect); one = forced; several = auto-detect restricted to these
    languages: list = field(default_factory=lambda: ["zh", "en"])
    initial_prompt: str = "以下是普通话和 English 混合的对话。"
    max_attempts: int = 5


@dataclass
class ValidateCfg:
    max_no_speech_prob: float = 0.6
    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    min_language_prob: float = 0.5
    max_chars_per_sec: float = 25.0
    extra_hallucination_phrases: list = field(default_factory=list)


@dataclass
class CorrectCfg:
    enabled: bool = True
    context_file: str = "context.md"   # plain text; re-read whenever it changes
    llm: bool = True                   # false = alias replacement only
    backend: str = "mlx_lm"            # a [corrector.<name>] entry
    min_tokens: int = 4                # shorter texts skip the LLM
    recent_lines: int = 5              # previous accepted lines shown to the LLM
    max_change: float = 0.3            # an LLM proposal changing more than this share of tokens is refused
    max_context_chars: int = 1500


@dataclass
class StoreCfg:
    data_dir: str = "data"
    keep_audio: str = "all"


@dataclass
class RouteCfg:
    # Live router under `audi27 supervise` only when this is true. `route run` ignores it.
    enabled: bool = False
    rules: str = "routes.toml"   # missing file falls back to routes.example.toml; re-read every turn
    llm: bool = False             # classify leftovers with the [correct] backend; timeout falls back to default
    timeout_s: float = 8.0


@dataclass
class DispatchCfg:
    # `audi27 supervise` starts `dispatch run` only when this is true. Default off:
    # routes stay display-only until execution is turned on. `dispatch run` itself
    # always dispatches.
    enabled: bool = False
    # target -> adapter name. Missing targets use "fake".
    adapters: dict = field(default_factory=dict)


@dataclass
class FakeCfg:
    delay_s: float = 0.0
    fail_marker: str = "模拟失败"       # route text containing this makes the fake agent fail
    decision_marker: str = "需要你决定"  # route text containing this asks for a decision


@dataclass
class GrokbotCfg:
    # Auth header for the webhook. `{key}` is replaced with AUDI27_GROKBOT_WEBHOOK_KEY.
    # The URL and the key are environment variables, never keys in this file.
    header_name: str = "Authorization"
    header_value: str = "Bearer {key}"
    timeout_s: float = 10.0
    context_utterances: int = 0  # last N utterances in the POST; 0 sends none
    # Front app and window title on the POST, and only then. Default off:
    # screen context stays on this machine. Requires [screen] enabled as well.
    include_screen: bool = False
    reply_timeout_s: float = 1800.0
    parse_retries: int = 3  # JSON parse errors before a reply file is quarantined


@dataclass
class QueueCfg:
    # `audi27 supervise` starts `queue run` only when this is true. `queue run` ignores it.
    enabled: bool = False
    summary_chars: int = 80


@dataclass
class SpeakCfg:
    # `audi27 supervise` starts `speak run` only when this is true. `speak run` ignores it.
    enabled: bool = False
    voice: str = "Tingting"  # macOS zh_CN. Older systems call this voice Ting-Ting.


@dataclass
class PresenceCfg:
    # The recorder publishes data/presence.json only when this is true. Default off:
    # the speaker keeps using the status.json heartbeat. Not a separate process.
    enabled: bool = False
    publish_s: float = 0.1     # refresh the file at least this often; also on every change
    stale_s: float = 1.0       # older than this, or timestamped in the future, is unknown
    onset_ms: int = 60         # continuous speech before "speaking"; capped at 200
    hangover_ms: int = 400     # continuous silence before "quiet"
    grace_s: float = 0.3       # after quiet, the speaker waits this long before starting


@dataclass
class ScreenCfg:
    # `audi27 supervise` starts `screen run` only when this is true. Default off:
    # no sampling and no screen events, and turns/routes are unchanged.
    enabled: bool = False
    poll_s: float = 2.0          # how often the front app is read
    debounce_s: float = 1.0      # a snapshot must hold this long before it is written
    stale_s: float = 3600.0      # older than this at turn time, it is not attached
    max_title_chars: int = 200
    # Substring of the app name or bundle id. A match keeps the app name and drops the title.
    # Banking apps are added here; they are not in the default list.
    exclude_apps: list = field(default_factory=lambda: ["1Password", "Bitwarden", "Keychain Access"])
    # A title matching any of these (case-insensitive) is dropped whole.
    redact_title_patterns: list = field(default_factory=list)
    # Substrings that mark a private/incognito browser window. The title is then dropped.
    private_title_markers: list = field(default_factory=lambda: [
        "incognito", "inprivate", "private browsing", "private window", "无痕", "隐私浏览",
    ])


@dataclass
class TurnsCfg:
    # Live assembler under `audi27 supervise` only when this is true. `turns run` ignores it.
    enabled: bool = False
    # Keep this list in sync with audi27.turns.DEFAULT_TRIGGERS.
    triggers: list = field(default_factory=lambda: ["达达", "哒哒", "大大", "DaDa", "dada", "Dada"])
    gap_s: float = 2.0
    max_duration_s: float = 120.0
    wall_close_s: float = 8.0


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    segment: SegmentCfg = field(default_factory=SegmentCfg)
    transcribe: TranscribeCfg = field(default_factory=TranscribeCfg)
    validate: ValidateCfg = field(default_factory=ValidateCfg)
    store: StoreCfg = field(default_factory=StoreCfg)
    correct: CorrectCfg = field(default_factory=CorrectCfg)
    turns: TurnsCfg = field(default_factory=TurnsCfg)
    route: RouteCfg = field(default_factory=RouteCfg)
    dispatch: DispatchCfg = field(default_factory=DispatchCfg)
    fake: FakeCfg = field(default_factory=FakeCfg)
    grokbot: GrokbotCfg = field(default_factory=GrokbotCfg)
    queue: QueueCfg = field(default_factory=QueueCfg)
    speak: SpeakCfg = field(default_factory=SpeakCfg)
    presence: PresenceCfg = field(default_factory=PresenceCfg)
    screen: ScreenCfg = field(default_factory=ScreenCfg)
    backend: dict = field(default_factory=dict)  # name -> constructor kwargs
    corrector: dict = field(default_factory=dict)  # name -> constructor kwargs


def _fill(obj, data: dict, where: str) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown config key [{where}] {key}")
        setattr(obj, key, value)


def load(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        path = Path("config.toml")
        if not path.exists():
            _check_all(cfg)
            return cfg
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}")
    for section, data in raw.items():
        if section in ("backend", "corrector"):
            setattr(cfg, section, {k: dict(v) for k, v in data.items()})
        elif section in {f.name for f in fields(cfg)}:
            _fill(getattr(cfg, section), data, section)
        else:
            raise ConfigError(f"unknown config section [{section}]")
    if cfg.store.keep_audio not in ("all", "suspect", "none"):
        raise ConfigError("store.keep_audio must be all | suspect | none")
    langs = cfg.transcribe.languages
    if not isinstance(langs, list) or not all(isinstance(x, str) and x for x in langs):
        raise ConfigError('transcribe.languages must be a list of codes, e.g. ["zh", "en"]')
    cfg.transcribe.languages = [x.lower() for x in langs]
    if not cfg.transcribe.backends:
        raise ConfigError("transcribe.backends is empty")
    _check_all(cfg)
    return cfg


def _check_turns(cfg: Config) -> None:
    t = cfg.turns
    if not isinstance(t.enabled, bool):
        raise ConfigError("turns.enabled must be true or false")
    if (not isinstance(t.triggers, list) or not t.triggers
            or not all(isinstance(x, str) and x.strip() for x in t.triggers)):
        raise ConfigError('turns.triggers must be a non-empty list of strings, e.g. ["达达", "dada"]')
    t.triggers = [x.strip() for x in t.triggers]
    for name in ("gap_s", "max_duration_s", "wall_close_s"):
        value = getattr(t, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"turns.{name} must be a number")
        setattr(t, name, float(value))
    if t.gap_s < 0 or t.wall_close_s < 0:
        raise ConfigError("turns.gap_s and turns.wall_close_s must be >= 0")
    if t.max_duration_s <= 0:
        raise ConfigError("turns.max_duration_s must be > 0")


def _check_all(cfg: Config) -> None:
    _check_turns(cfg)
    _check_route(cfg)
    _check_dispatch(cfg)
    _check_fake(cfg)
    _check_grokbot(cfg)
    _check_queue(cfg)
    _check_speak(cfg)
    _check_presence(cfg)
    _check_screen(cfg)


def _check_route(cfg: Config) -> None:
    r = cfg.route
    if not isinstance(r.enabled, bool):
        raise ConfigError("route.enabled must be true or false")
    if not isinstance(r.llm, bool):
        raise ConfigError("route.llm must be true or false")
    if not isinstance(r.rules, str) or not r.rules.strip():
        raise ConfigError("route.rules must be a file path, e.g. routes.toml")
    r.rules = r.rules.strip()
    if isinstance(r.timeout_s, bool) or not isinstance(r.timeout_s, (int, float)):
        raise ConfigError("route.timeout_s must be a number")
    r.timeout_s = float(r.timeout_s)
    if r.timeout_s <= 0:
        raise ConfigError("route.timeout_s must be > 0")


_DISPATCH_TARGETS = ("grok", "muse", "claude_code", "local")


def _check_dispatch(cfg: Config) -> None:
    d = cfg.dispatch
    if not isinstance(d.enabled, bool):
        raise ConfigError("dispatch.enabled must be true or false")
    if not isinstance(d.adapters, dict):
        raise ConfigError("dispatch.adapters must be a table of target = \"fake\"")
    cleaned = {}
    for key, value in d.adapters.items():
        if key not in _DISPATCH_TARGETS:
            raise ConfigError(f"dispatch.adapters.{key} is not a target ({', '.join(_DISPATCH_TARGETS)})")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"dispatch.adapters.{key} must be an adapter name")
        cleaned[key] = value.strip()
    d.adapters = cleaned


def _check_fake(cfg: Config) -> None:
    f = cfg.fake
    if isinstance(f.delay_s, bool) or not isinstance(f.delay_s, (int, float)):
        raise ConfigError("fake.delay_s must be a number")
    f.delay_s = float(f.delay_s)
    if f.delay_s < 0:
        raise ConfigError("fake.delay_s must be >= 0")
    for name in ("fail_marker", "decision_marker"):
        value = getattr(f, name)
        if not isinstance(value, str):
            raise ConfigError(f"fake.{name} must be a string")
        setattr(f, name, value.strip())


_HEADER_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _check_grokbot(cfg: Config) -> None:
    g = cfg.grokbot
    if not isinstance(g.header_name, str) or not _HEADER_TOKEN.match(g.header_name):
        raise ConfigError("grokbot.header_name must be an HTTP header name, e.g. Authorization")
    if (not isinstance(g.header_value, str) or not g.header_value.strip()
            or any(c in g.header_value for c in "\r\n")):
        raise ConfigError("grokbot.header_value must be a header value; {key} is the webhook key")
    g.header_value = g.header_value.strip()
    if isinstance(g.timeout_s, bool) or not isinstance(g.timeout_s, (int, float)):
        raise ConfigError("grokbot.timeout_s must be a number")
    g.timeout_s = float(g.timeout_s)
    if g.timeout_s <= 0:
        raise ConfigError("grokbot.timeout_s must be > 0")
    if isinstance(g.reply_timeout_s, bool) or not isinstance(g.reply_timeout_s, (int, float)):
        raise ConfigError("grokbot.reply_timeout_s must be a number")
    g.reply_timeout_s = float(g.reply_timeout_s)
    if g.reply_timeout_s <= 0:
        raise ConfigError("grokbot.reply_timeout_s must be > 0")
    if isinstance(g.context_utterances, bool) or not isinstance(g.context_utterances, int):
        raise ConfigError("grokbot.context_utterances must be an integer")
    if not 0 <= g.context_utterances <= 20:
        raise ConfigError("grokbot.context_utterances must be from 0 to 20")
    if isinstance(g.parse_retries, bool) or not isinstance(g.parse_retries, int):
        raise ConfigError("grokbot.parse_retries must be an integer")
    if g.parse_retries < 1:
        raise ConfigError("grokbot.parse_retries must be >= 1")
    if not isinstance(g.include_screen, bool):
        raise ConfigError("grokbot.include_screen must be true or false")


def _check_queue(cfg: Config) -> None:
    q = cfg.queue
    if not isinstance(q.enabled, bool):
        raise ConfigError("queue.enabled must be true or false")
    if isinstance(q.summary_chars, bool) or not isinstance(q.summary_chars, int):
        raise ConfigError("queue.summary_chars must be an integer")
    if q.summary_chars < 1:
        raise ConfigError("queue.summary_chars must be >= 1")


def _check_speak(cfg: Config) -> None:
    s = cfg.speak
    if not isinstance(s.enabled, bool):
        raise ConfigError("speak.enabled must be true or false")
    if not isinstance(s.voice, str) or not s.voice.strip():
        raise ConfigError("speak.voice must be a voice name, e.g. Tingting")
    s.voice = s.voice.strip()


def _as_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def _string_list(value, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{name} must be a list of strings")
    cleaned = []
    for item in value:
        text = item.strip()
        if not text:
            raise ConfigError(f"{name} must not contain a blank entry")
        cleaned.append(text)
    return cleaned


def _check_screen(cfg: Config) -> None:
    s = cfg.screen
    if not isinstance(s.enabled, bool):
        raise ConfigError("screen.enabled must be true or false")
    s.poll_s = _as_float(s.poll_s, "screen.poll_s")
    s.debounce_s = _as_float(s.debounce_s, "screen.debounce_s")
    s.stale_s = _as_float(s.stale_s, "screen.stale_s")
    if s.poll_s <= 0 or s.poll_s > 3600:
        raise ConfigError("screen.poll_s must be > 0 and <= 3600")
    if s.debounce_s < 0 or s.debounce_s > 3600:
        raise ConfigError("screen.debounce_s must be between 0 and 3600")
    if s.stale_s < 0:
        raise ConfigError("screen.stale_s must be >= 0")
    if isinstance(s.max_title_chars, bool) or not isinstance(s.max_title_chars, int):
        raise ConfigError("screen.max_title_chars must be an integer")
    if not 1 <= s.max_title_chars <= 2000:
        raise ConfigError("screen.max_title_chars must be from 1 to 2000")
    s.exclude_apps = _string_list(s.exclude_apps, "screen.exclude_apps")
    s.private_title_markers = _string_list(s.private_title_markers, "screen.private_title_markers")
    s.redact_title_patterns = _string_list(s.redact_title_patterns, "screen.redact_title_patterns")
    for pattern in s.redact_title_patterns:
        try:
            re.compile(pattern)
        except re.error as e:
            raise ConfigError(f"screen.redact_title_patterns {pattern!r} is not a valid regex: {e}") from e


def _check_presence(cfg: Config) -> None:
    p = cfg.presence
    if not isinstance(p.enabled, bool):
        raise ConfigError("presence.enabled must be true or false")
    p.publish_s = _as_float(p.publish_s, "presence.publish_s")
    p.stale_s = _as_float(p.stale_s, "presence.stale_s")
    p.grace_s = _as_float(p.grace_s, "presence.grace_s")
    if not 0 < p.publish_s <= 0.2:
        raise ConfigError("presence.publish_s must be > 0 and <= 0.2")
    if p.stale_s <= p.publish_s:
        raise ConfigError("presence.stale_s must be greater than presence.publish_s")
    if p.grace_s < 0 or p.grace_s > 5:
        raise ConfigError("presence.grace_s must be between 0 and 5")
    for name, limit in (("onset_ms", 200), ("hangover_ms", 5000)):
        value = getattr(p, name)
        number = _as_float(value, f"presence.{name}")
        if number < 0 or number > limit:
            raise ConfigError(f"presence.{name} must be between 0 and {limit}")
        setattr(p, name, int(round(number)))


_FROM_DOTENV: set = set()


def _dotenv_value(raw: str) -> str:
    """Value of one .env assignment. Quotes and a trailing ` # comment` are not part of it."""
    value = raw.strip()
    if value[:1] in "\"'":
        quote = value[0]
        i = 1
        while i < len(value):
            if value[i] == "\\" and i + 1 < len(value):
                i += 2
                continue
            if value[i] == quote:
                rest = value[i + 1:].strip()
                if rest == "" or rest.startswith("#"):
                    return value[1:i]
                break
            i += 1
        return value
    cut = value.find(" #")
    if cut != -1:
        value = value[:cut].rstrip()
    return value


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines from ./.env into the environment (existing variables win). Keeps API keys out of config.toml.

    A missing file is ignored. `#` comments and optional quotes are stripped; the real environment wins.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _dotenv_value(value)
        if key not in os.environ or key in _FROM_DOTENV:  # real environment wins; .env edits are picked up
            os.environ[key] = value
            _FROM_DOTENV.add(key)
