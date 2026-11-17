"""Streaming wake-word spotter on the frames the recorder already captured.

The capture callback stays a copy onto the recorder queue. `audi27 run` hands
each frame to a second queue; this thread runs sherpa-onnx there, so a slow
spotter cannot stall segmentation. No second input stream is opened.

A hit plays a preloaded tone (or wav) on an output stream that is already
running, then appends `wake` (audio time of the frame) and `cue_played` (the
same echo window confirm already reads). Hits closer than `debounce_ms` on
the audio clock are ignored. Turns still come from Whisper text.

The weights are not in the repo. `audi27 kws download` fetches them. A missing
package or a missing model logs once and the spotter stays off.

Latency, after the last sample of the wake word is on the mic:
the capture frame is `frame_ms` (default 30), the zipformer is chunk-16
(about 160 ms of features) plus one trailing blank, and the tone is already
sitting in an open output stream (one 128-sample block, about 8 ms). That is
the path aimed at under 400 ms. It does not wait for segment silence or Whisper.
"""
from __future__ import annotations

import io
import logging
import math
import queue
import sys
import tarfile
import threading
import urllib.request
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .cue import SUPPRESS_S
from .events import append, make_event
from .store import _dt

log = logging.getLogger("audi27.kws")

KWS_SOURCE = "kws"
MODEL_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    + MODEL_NAME + ".tar.bz2"
)
MODEL_RATE = 16000
# ppinyin: the longest initial wins, so zh/ch/sh come before z/c/s.
_INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)
# Phrase table for the wake sound. 大大 is the same spoken wake as 达达, not the
# lexical fourth-tone reading of 大. Other phrases go through pypinyin when it
# is installed, or a sherpa line (`d á d á @达达`).
_PHRASES = {
    "达达": "d á d á",
    "哒哒": "d á d á",
    "大大": "d á d á",
    "dada": "d á d á",
}
_CHAR_PINYIN = {"达": "dá", "哒": "dá", "大": "dà"}
_VOWELS = set("aeiouüvāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ")
_warned: set[str] = set()


class KwsError(RuntimeError):
    pass


def _warn_once(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    log.warning("%s", message)


def split_syllable(syllable: str) -> list[str]:
    """Split one tone-marked syllable into sherpa ppinyin tokens (`dá` -> `d`, `á`)."""
    for initial in _INITIALS:
        if syllable.startswith(initial) and len(syllable) > len(initial):
            return [initial, syllable[len(initial):]]
    return [syllable]


def _has_vowel(token: str) -> bool:
    return any(ch.lower() in _VOWELS for ch in token)


def _is_syllable(token: str) -> bool:
    return bool(token) and token[0] not in ":#@" and _has_vowel(token)


def _to_syllables(text: str) -> list[str] | None:
    parts = text.split()
    if len(parts) > 1 and all(_is_syllable(part) for part in parts):
        return parts
    if _is_syllable(text) and " " not in text and not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return [text]
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return None
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        syllables = []
        for ch in text:
            if not ("\u4e00" <= ch <= "\u9fff"):
                continue
            syl = _CHAR_PINYIN.get(ch)
            if syl is None:
                return None
            syllables.append(syl)
        return syllables or None
    syllables = [syl for syl in lazy_pinyin(text, style=Style.TONE, errors="ignore") if syl]
    return syllables or None


def _normalize_raw(text: str) -> str | None:
    parts = text.split()
    try:
        at = next(i for i, part in enumerate(parts) if part.startswith("@"))
    except StopIteration:
        return None
    if at == 0:
        return None
    display = "".join(parts[at:])[1:].replace(" ", "")
    if not display:
        return None
    return " ".join(parts[:at] + ["@" + display])


def keyword_line(text: str) -> str | None:
    """One sherpa keywords.txt line, or None when the phrase cannot be encoded."""
    raw = "" if text is None else str(text).strip()
    if not raw:
        return None
    if "@" in raw:
        return _normalize_raw(raw)
    key = raw.casefold() if raw.isascii() else raw
    tokens = _PHRASES.get(key)
    if tokens is None:
        syllables = _to_syllables(raw)
        if not syllables:
            return None
        pieces: list[str] = []
        for syllable in syllables:
            pieces.extend(split_syllable(syllable))
        tokens = " ".join(pieces)
    display = raw.replace(" ", "_")
    return f"{tokens} @{display}"


def keyword_lines(words) -> list[str]:
    lines = []
    for word in words:
        line = keyword_line(word)
        if line is None:
            log.warning(
                "kws: skipping keyword %r; use a sherpa line such as 'd á d á @达达'", word)
            continue
        lines.append(line)
    return lines


def _prefer_int8(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    int8 = [path for path in paths if "int8" in path.name]
    return sorted(int8 or paths)[0]


def _variant(path: Path) -> str:
    """Checkpoint id shared by an encoder/decoder/joiner trio (`epoch-12-avg-2-...`)."""
    name = path.name
    for kind in ("encoder", "decoder", "joiner"):
        if name.startswith(kind):
            name = name[len(kind):]
            break
    for suffix in (".int8.onnx", ".onnx"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lstrip("-")


def find_model(model_dir) -> dict | None:
    """`tokens` / `encoder` / `decoder` / `joiner` paths, or None if any are missing.

    The archive ships more than one checkpoint. Encoder, decoder, and joiner are
    taken from the same one (the epoch-12 streaming graph when it is present).
    An int8 file wins over the float file of that checkpoint.
    """
    root = Path(model_dir)
    tokens = root / "tokens.txt"
    if not tokens.is_file():
        return None
    groups: dict[str, dict[str, list[Path]]] = {}
    for kind in ("encoder", "decoder", "joiner"):
        for path in root.glob(f"{kind}*.onnx"):
            if path.is_file():
                groups.setdefault(_variant(path), {}).setdefault(kind, []).append(path)
    complete = [key for key, kinds in groups.items() if {"encoder", "decoder", "joiner"} <= set(kinds)]
    if not complete:
        return None
    preferred = [key for key in complete if "epoch-12-avg-2" in key]
    key = sorted(preferred or complete)[0]
    return {
        "tokens": tokens,
        "encoder": _prefer_int8(groups[key]["encoder"]),
        "decoder": _prefer_int8(groups[key]["decoder"]),
        "joiner": _prefer_int8(groups[key]["joiner"]),
    }


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in tar.getmembers():
        name = member.name
        if member.issym() or member.islnk() or name.startswith("/") or ".." in Path(name).parts:
            raise KwsError(f"unsafe path in model archive: {name}")
        target = (root / name).resolve()
        if target != root and root not in target.parents:
            raise KwsError(f"unsafe path in model archive: {name}")
    kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
    tar.extractall(root, **kwargs)


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "audi27"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def download_model(model_dir, *, url: str | None = None, fetch=None) -> Path:
    """Download and extract the keyword model into `model_dir`. A present model is left as-is."""
    model_dir = Path(model_dir)
    if find_model(model_dir):
        return model_dir
    blob = (fetch or _fetch)(url or MODEL_URL)
    parent = model_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
        _safe_extract(tar, parent)
    if find_model(model_dir):
        return model_dir
    raise KwsError(f"archive did not contain a keyword model at {model_dir}")


def tone(hz: float, ms: int, sample_rate: int = MODEL_RATE) -> np.ndarray:
    """A short sine with a few milliseconds of fade so the speaker does not click."""
    n = max(1, int(sample_rate * ms / 1000))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    wave_f = (0.2 * np.sin(2.0 * math.pi * float(hz) * t)).astype(np.float32)
    fade = min(n // 2, int(sample_rate * 0.005))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wave_f[:fade] *= ramp
        wave_f[-fade:] *= ramp[::-1]
    return wave_f


def _resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(samples) == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)
    n = int(round(len(samples) * dst / src))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(new, old, np.asarray(samples, dtype=np.float64)).astype(np.float32)


def load_wav(path, sample_rate: int = MODEL_RATE) -> np.ndarray | None:
    try:
        with wave.open(str(path)) as handle:
            if handle.getsampwidth() != 2 or handle.getnchannels() not in (1, 2):
                return None
            rate = handle.getframerate()
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, OSError, EOFError):
        return None
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        data = data.reshape(-1, 2)[:, 0]
    if rate != sample_rate:
        data = _resample(data, rate, sample_rate)
    return np.ascontiguousarray(data, dtype=np.float32)


def _preload(cfg) -> tuple[np.ndarray, str]:
    if cfg.kws.sound:
        pcm = load_wav(cfg.kws.sound, MODEL_RATE)
        if pcm is not None and len(pcm):
            return pcm, cfg.kws.sound
        log.warning("kws: cannot read %s; using the tone", cfg.kws.sound)
    return tone(cfg.kws.tone_hz, cfg.kws.tone_ms, MODEL_RATE), f"tone:{cfg.kws.tone_hz:g}Hz/{cfg.kws.tone_ms}ms"


class _LogPlayer:
    def __init__(self, label: str):
        self.label = label

    def play(self) -> dict:
        log.info("kws: not playing %s", self.label)
        return {"ok": False, "backend": "log", "sound": self.label}

    def close(self) -> None:
        return None


class SoundDevicePlayer:
    """An output stream that is already running. `play` only rewinds a buffer."""

    def __init__(self, pcm: np.ndarray, sample_rate: int, label: str):
        self.pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        self.sample_rate = sample_rate
        self.label = label
        self._lock = threading.Lock()
        self._pos = None
        self.stream = None
        self._ok = False
        try:
            import sounddevice as sd
            self.stream = self._open(sd)
            self._ok = True
        except Exception:
            log.warning("kws: output stream did not start; hits will be logged")
            self.stream = None

    def _open(self, sd):
        kwargs = dict(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            callback=self._on_audio, blocksize=128,
        )
        try:
            stream = sd.OutputStream(latency="low", **kwargs)
        except Exception:
            stream = sd.OutputStream(**kwargs)
        stream.start()
        return stream

    def _on_audio(self, outdata, frames, _time_info, _status) -> None:
        outdata.fill(0)
        with self._lock:
            if self._pos is None:
                return
            n = min(frames, len(self.pcm) - self._pos)
            if n > 0:
                outdata[:n, 0] = self.pcm[self._pos:self._pos + n]
            self._pos += n
            if self._pos >= len(self.pcm):
                self._pos = None

    def play(self) -> dict:
        if not self._ok:
            return {"ok": False, "backend": "log", "sound": self.label}
        with self._lock:
            self._pos = 0
        return {"ok": True, "backend": "sounddevice", "sound": self.label}

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.exception("kws: output stream did not close")


def build_player(cfg):
    pcm, label = _preload(cfg)
    try:
        return SoundDevicePlayer(pcm, MODEL_RATE, label)
    except Exception:
        log.exception("kws: player was not created")
        return _LogPlayer(label)


def _as_keyword(result) -> str:
    if result is None:
        return ""
    if isinstance(result, bytes):
        result = result.decode("utf-8", "replace")
    if isinstance(result, str):
        return result.strip()
    keyword = getattr(result, "keyword", None)
    if isinstance(keyword, str) and keyword.strip():
        return keyword.strip()
    text = str(result).strip()
    if not text or text == "None" or text.startswith("<"):
        return ""
    return text


def _display(result) -> str:
    text = _as_keyword(result)
    if "@" in text:
        text = text.split("@", 1)[1]
    text = " ".join(text.split())
    return text[:80]


class SherpaEngine:
    """One streaming spotter. `accept` returns the display keyword, or None."""

    def __init__(self, spotter):
        self.spotter = spotter
        self.stream = spotter.create_stream()

    def _drain(self):
        hit = None
        for _ in range(10000):
            if not self.spotter.is_ready(self.stream):
                return hit
            self.spotter.decode_stream(self.stream)
            keyword = _as_keyword(self.spotter.get_result(self.stream))
            if keyword:
                hit = keyword
                self.spotter.reset_stream(self.stream)
        log.warning("kws: decode did not drain")
        return hit

    def accept(self, samples, sample_rate: int):
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if len(audio) == 0:
            return None
        self.stream.accept_waveform(int(sample_rate), audio)
        return self._drain()

    def finish(self):
        try:
            self.stream.input_finished()
        except Exception:
            log.exception("kws: stream did not finish")
            return None
        return self._drain()


def load_engine(cfg, files: dict, *, module=None):
    """Build a sherpa KeywordSpotter. `module` is the sherpa_onnx module (tests inject one)."""
    sherpa = module
    if sherpa is None:
        try:
            import sherpa_onnx as sherpa
        except ImportError as e:
            raise KwsError("sherpa-onnx is not installed (uv sync --extra kws)") from e
    lines = keyword_lines(cfg.kws.keywords)
    if not lines:
        raise KwsError("no keyword could be encoded")
    path = Path(files["tokens"]).parent / "keywords.audi27.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spotter = sherpa.KeywordSpotter(
        tokens=str(files["tokens"]),
        encoder=str(files["encoder"]),
        decoder=str(files["decoder"]),
        joiner=str(files["joiner"]),
        keywords_file=str(path),
        keywords_score=float(cfg.kws.score),
        keywords_threshold=float(cfg.kws.threshold),
        num_threads=1,
        provider="cpu",
        max_active_paths=4,
        num_trailing_blanks=1,
    )
    return SherpaEngine(spotter)


def _frames_to_model(frame, sample_rate: int) -> np.ndarray:
    raw = np.asarray(frame)
    if np.issubdtype(raw.dtype, np.integer):
        samples = raw.astype(np.float32) / 32768.0
    else:
        samples = raw.astype(np.float32)
    if sample_rate != MODEL_RATE:
        samples = _resample(samples, sample_rate, MODEL_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _write_event(data_dir, event) -> bool:
    last = None
    for _ in range(2):
        try:
            append(data_dir, event)
            return True
        except OSError as e:
            last = e
    log.error("kws: event %s was not written: %s", event.get("id"), last)
    return False


class Tap:
    """Queue plus worker. `offer` never blocks. `handle` is the hit itself."""

    def __init__(self, cfg, data_dir, engine, player, *, start=True, queue_size=200):
        self.cfg = cfg
        self.data_dir = data_dir
        self.engine = engine
        self.player = player
        self.sample_rate = int(cfg.audio.sample_rate)
        self.debounce_s = float(cfg.kws.debounce_ms) / 1000.0
        self.q = queue.Queue(maxsize=max(1, int(queue_size)))
        self.dropped = 0
        self.stop = threading.Event()
        self._closed = False
        self._started = False
        self._failed = False
        self._last_ts = None
        self._drop_logged = 0
        play = getattr(player, "play", None)
        self._play = play if callable(play) else player
        self.thread = threading.Thread(target=self._loop, name="audi27-kws", daemon=True)
        if start:
            self.start()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.thread.start()

    def offer(self, frame, ts: float) -> None:
        """Copy one captured frame onto the spotter queue. A full queue drops the frame."""
        if self._closed:
            return
        try:
            copied = np.asarray(frame, dtype=np.int16).copy()
            self.q.put_nowait((copied, float(ts)))
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 200 == 0:
                if self.dropped != self._drop_logged:
                    self._drop_logged = self.dropped
                    log.warning("kws dropped %d frame(s); the spotter is behind", self.dropped)

    def handle(self, keyword: str, audio_ts: float) -> bool:
        """Play and append `wake` + `cue_played`. False when debounce swallows the hit."""
        keyword = " ".join(str(keyword).split())[:80]
        if not keyword:
            return False
        if self._last_ts is not None:
            delta = float(audio_ts) - self._last_ts
            if 0 <= delta < self.debounce_s:
                return False
        self._last_ts = float(audio_ts)
        started = datetime.now().astimezone()
        try:
            result = self._play()
        except Exception:
            log.exception("kws: cue failed")
            result = None
        if not isinstance(result, dict):
            result = {"ok": False, "backend": "log", "sound": ""}
        ok = bool(result.get("ok"))
        backend = result.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            backend = "sounddevice" if ok else "log"
        sound = result.get("sound")
        if not isinstance(sound, str):
            sound = ""
        ms = int(round(float(audio_ts) * 1000))
        wake_id = f"wake-{ms}"
        data = {
            "keyword": keyword,
            "audio_ts": round(float(audio_ts), 3),
        }
        try:
            wake = make_event(
                "wake", KWS_SOURCE, data, ts=_dt(float(audio_ts)), id=wake_id)
        except Exception:
            log.exception("kws: wake event was not built")
            return True
        cue_data = {
            "sound": sound,
            "backend": backend.strip()[:40],
            "ok": ok,
            "trigger": keyword,
        }
        if ok:
            cue_data["until"] = (started + timedelta(seconds=SUPPRESS_S)).isoformat(timespec="milliseconds")
        try:
            cue = make_event(
                "cue_played", KWS_SOURCE, cue_data, ts=started, ref=wake_id, id="cue-" + wake_id)
        except Exception:
            log.exception("kws: cue event was not built")
            cue = None
        _write_event(self.data_dir, wake)
        if cue is not None:
            _write_event(self.data_dir, cue)
        if ok:
            log.info("kws hit %s", keyword)
        else:
            log.info("kws hit %s; cue was not played", keyword)
        return True

    def _loop(self) -> None:
        while True:
            if self.stop.is_set() and self.q.empty():
                break
            try:
                frame, ts = self.q.get(timeout=0.2)
            except queue.Empty:
                if self.stop.is_set():
                    break
                continue
            if frame is None:
                break
            if len(frame) == 0 or self.engine is None:
                continue
            try:
                samples = _frames_to_model(frame, self.sample_rate)
                keyword = self.engine.accept(samples, MODEL_RATE)
            except Exception:
                if not self._failed:
                    log.exception("kws: spotter failed; this hit is skipped")
                    self._failed = True
                continue
            if not keyword:
                continue
            audio_ts = float(ts) + len(frame) / float(self.sample_rate)
            try:
                self.handle(_display(keyword), audio_ts)
            except Exception:
                log.exception("kws: cue failed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        if self._started:
            try:
                self.q.put((None, 0.0), timeout=2)
            except queue.Full:
                pass
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                log.warning("kws worker did not stop")
        closer = getattr(self.player, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                log.exception("kws: player did not close")


def open_tap(cfg, data_dir, *, engine=None, player=None):
    """Start the spotter thread, or return None when it should stay off.

    A missing model or a missing sherpa-onnx install logs once and returns None.
    The recorder keeps running.
    """
    if not cfg.kws.enabled:
        return None
    if engine is None:
        files = find_model(cfg.kws.model_dir)
        if files is None:
            _warn_once(
                f"kws disabled: model not found in {cfg.kws.model_dir}. Run: audi27 kws download")
            return None
        try:
            engine = load_engine(cfg, files)
        except KwsError as e:
            _warn_once(f"kws disabled: {e}")
            return None
        except Exception:
            _warn_once(f"kws disabled: spotter failed to load from {cfg.kws.model_dir}")
            log.exception("kws: spotter failed to load")
            return None
    if player is None:
        player = build_player(cfg)
    log.info("kws watching %s", ", ".join(cfg.kws.keywords))
    return Tap(cfg, data_dir, engine, player, start=True)


def read_wav(path) -> tuple[np.ndarray, int]:
    """16-bit PCM wav as float32 in [-1, 1], mono. Stereo keeps the first channel."""
    try:
        with wave.open(str(path)) as handle:
            if handle.getnchannels() not in (1, 2):
                raise KwsError(f"{path}: need mono or stereo")
            if handle.getsampwidth() != 2:
                raise KwsError(f"{path}: need 16-bit PCM")
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
    except KwsError:
        raise
    except (wave.Error, OSError, EOFError) as e:
        raise KwsError(f"{path}: {e}") from e
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        data = data.reshape(-1, 2)[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)


def energy_end(samples, sample_rate: int, thresh_db: float = -35.0) -> float:
    """Seconds until the last 10 ms frame at or above `thresh_db`. Silence is 0."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    hop = max(1, int(sample_rate * 0.01))
    last = 0
    for i in range(0, len(audio), hop):
        chunk = audio[i:i + hop]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        db = 20.0 * math.log10(rms + 1e-10)
        if db >= thresh_db:
            last = i + len(chunk)
    return last / float(sample_rate)


def detect(samples, sample_rate: int, engine, *, hop_ms: int = 30,
           keyword_end: float | None = None, tail_s: float = 0.66) -> dict:
    """Run `engine` over `samples` and report detection time minus the keyword end.

    `keyword_end` is seconds from the start of the file. When it is omitted, the
    end of energy in the file is used (right for a clip that goes quiet after
    the wake word). `tail_s` of silence is fed after the file so the last
    chunk can flush, the same way a live mic keeps delivering frames.
    """
    audio = np.asarray(samples)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if keyword_end is None:
        end_s = energy_end(audio, sample_rate)
        source = "energy"
    else:
        end_s = float(keyword_end)
        source = "argument"
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    pieces = [audio]
    if tail_s > 0:
        pieces.append(np.zeros(int(sample_rate * tail_s), dtype=np.float32))
    stream = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    detected = None
    keyword = ""
    fed = 0
    for i in range(0, len(stream), hop):
        chunk = stream[i:i + hop]
        fed += len(chunk)
        hit = engine.accept(np.ascontiguousarray(chunk, dtype=np.float32), sample_rate)
        if hit:
            keyword = _display(hit)
            detected = fed / float(sample_rate)
            break
    if detected is None:
        finish = getattr(engine, "finish", None)
        if finish is not None:
            hit = finish()
            if hit:
                keyword = _display(hit)
                detected = fed / float(sample_rate)
    latency = None if detected is None else detected - end_s
    return {
        "keyword": keyword,
        "detected_s": detected,
        "keyword_end_s": end_s,
        "keyword_end_source": source,
        "latency_s": latency,
        "sample_rate": int(sample_rate),
    }


def report_file(path, cfg, *, keyword_end: float | None = None) -> dict:
    """Spot keywords in one wav. Raises KwsError when the model or the package is missing."""
    files = find_model(cfg.kws.model_dir)
    if files is None:
        raise KwsError(f"model not found in {cfg.kws.model_dir}. Run: audi27 kws download")
    samples, sample_rate = read_wav(path)
    if sample_rate != MODEL_RATE:
        samples = _resample(samples, sample_rate, MODEL_RATE)
        sample_rate = MODEL_RATE
    engine = load_engine(cfg, files)
    return detect(
        samples, sample_rate, engine,
        hop_ms=cfg.audio.frame_ms, keyword_end=keyword_end)


def format_report(report: dict) -> str:
    def num(key):
        value = report.get(key)
        return "" if value is None else f"{float(value):.3f}"

    end = num("keyword_end_s")
    source = report.get("keyword_end_source") or ""
    end_bit = f"{end} ({source})" if end else ""
    lines = [
        f"keyword: {report.get('keyword') or ''}",
        f"detected_s: {num('detected_s')}",
        f"keyword_end_s: {end_bit}",
        f"latency_s: {num('latency_s')}",
    ]
    if report.get("detected_s") is None:
        lines.append("no keyword detected")
    return "\n".join(lines)
