"""Typed config loaded from TOML. Unknown keys raise, so typos fail loudly."""
from __future__ import annotations

import os
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
            _check_turns(cfg)
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
    _check_turns(cfg)
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


_FROM_DOTENV: set = set()


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines from ./.env into the environment (existing variables win). Keeps API keys out of config.toml."""
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
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        key = key.strip()
        if key not in os.environ or key in _FROM_DOTENV:  # real environment wins; .env edits are picked up
            os.environ[key] = value
            _FROM_DOTENV.add(key)
