"""One-screen summary for "I said 达达 and heard no ding".

Reads config, `kws.live`, and the event log. Does not start the recorder, the
spotter, or the text cue.
"""
from __future__ import annotations

from pathlib import Path

from .events import replay
from .kws import FIX_PACKAGE, find_model, keyword_lines, read_live, spotter_live


def sherpa_status() -> tuple[bool, str]:
    try:
        import sherpa_onnx
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    version = getattr(sherpa_onnx, "__version__", "")
    text = version if isinstance(version, str) and version.strip() else "yes"
    return True, text


def input_device_name(cfg) -> str:
    from .kws import input_device_name as lookup
    return lookup(cfg)


def _shown(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _warnings(text: str, limit: int) -> list[str]:
    found = []
    for line in text.splitlines():
        if "audi27.kws" not in line and "audi27.cue" not in line:
            continue
        if " WARNING " not in line and " ERROR " not in line:
            continue
        found.append(line.rstrip())
    return found[-limit:]


def _brief(ev: dict) -> str:
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    kind = ev.get("type")
    stamp = ev.get("ts") or ""
    if kind == "wake":
        return (
            f"{stamp} keyword={data.get('keyword')} latency_s={data.get('latency_s')} "
            f"play_ok={data.get('play_ok')} device={data.get('play_device')} "
            f"error={data.get('play_error')}"
        )
    if kind == "cue_decision":
        return (
            f"{stamp} {data.get('action')} {data.get('reason')} "
            f"trigger={data.get('trigger')} ref={ev.get('ref')}"
        )
    if kind == "cue_played":
        return (
            f"{stamp} ok={data.get('ok')} backend={data.get('backend')} "
            f"trigger={data.get('trigger')} ref={ev.get('ref')}"
        )
    if kind == "kws_status":
        return (
            f"{stamp} state={data.get('state')} live={data.get('live')} "
            f"reason={data.get('reason')} fix={data.get('fix')}"
        )
    return stamp


def _section(title: str, rows: list[str]) -> list[str]:
    if not rows:
        return [f"{title}:", "  (none)"]
    return [f"{title}:", *[f"  {row}" for row in rows]]


def _live_line(data_dir, now: float) -> str:
    data = read_live(data_dir)
    if not data:
        return "kws.live: absent"
    updated = data.get("updated")
    age = ""
    if isinstance(updated, (int, float)) and not isinstance(updated, bool):
        age = f" age={float(now) - float(updated):.1f}s"
    state = "fresh" if spotter_live(data_dir, now=now) else "stale"

    def num(key):
        value = data.get(key, "-")
        return "-" if value is None else value

    return (
        f"kws.live: {state}{age} pid={data.get('pid')} "
        f"frames={num('frames')} dropped={num('dropped')} rtf={num('rtf')} "
        f"lag_s={num('lag_s')} hits={num('hits')} stream={num('stream')} "
        f"device={num('output_device')} player={num('player')}"
    )


def render(cfg, data_dir, *, limit: int = 5, now: float | None = None,
           log_text: str | None = None, sherpa=None, input_device: str | None = None) -> str:
    """Plain-text summary. `sherpa` is `(ok, detail)` when the caller already checked."""
    import time

    root = Path(data_dir)
    if now is None:
        now = time.time()
    if sherpa is None:
        sherpa = sherpa_status()
    if input_device is None:
        input_device = input_device_name(cfg)
    ok, detail = sherpa
    model = find_model(cfg.kws.model_dir)
    lines = [
        f"kws enabled: {str(bool(cfg.kws.enabled)).lower()}",
        f"cue enabled: {str(bool(cfg.cue.enabled)).lower()}",
        f"sherpa-onnx: {'importable' if ok else 'not importable'} ({detail})",
        f"model: {'present' if model else 'missing'} ({cfg.kws.model_dir})",
        "keywords: " + (", ".join(keyword_lines(cfg.kws.keywords)) or "-"),
        f"threshold: {_shown(cfg.kws.threshold)}",
        f"score: {_shown(cfg.kws.score)}",
        f"heartbeat_s: {_shown(cfg.kws.heartbeat_s)}",
        f"sample_rate: {cfg.audio.sample_rate}",
        f"input_device: {input_device}",
    ]
    if cfg.kws.enabled and not ok:
        lines.append(f"fix: {FIX_PACKAGE}")
    if cfg.kws.enabled and model is None:
        lines.append("fix: audi27 kws download")
    lines.append(_live_line(root, float(now)))
    n = max(1, int(limit))
    for kind in ("wake", "cue_played", "cue_decision", "kws_status"):
        rows = list(replay(root, types=kind))
        lines.extend(_section(kind, [_brief(ev) for ev in rows[-n:]]))
    if log_text is None:
        path = root / "audi27.log"
        try:
            log_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
    lines.extend(_section("log", _warnings(log_text, n)))
    return "\n".join(lines)
