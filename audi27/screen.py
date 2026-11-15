"""Frontmost app and window title, as `screen` events.

Privacy boundary: a sample is the frontmost app name, its bundle id, and the
front window's title. Nothing else. No screenshots, no OCR, no window contents,
and no accessibility tree beyond that one title. Titles of excluded apps,
private browser windows, and configured patterns are dropped before the event
is built, so the dropped text is never written.

Sampling is off unless `[screen] enabled` is true. On macOS the app name and
bundle id come from `lsappinfo` (no Accessibility permission). The window title
comes from `osascript` and needs Accessibility; if that is denied, events keep
the app and leave the title empty, and the denial is logged once. Other
platforms stay idle and write nothing.

Events are written only when the recorded snapshot changes, and only after it
has held still for `debounce_s`, so a burst of alt-tab does not flood the log.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime, timedelta

from .events import emit, replay

log = logging.getLogger("audi27.screen")

SCREEN_SOURCE = "screen"
_APP_LIMIT = 200
# osascript errors that mean Accessibility was not granted. The title is then
# omitted; the app name from lsappinfo still stands. -1728 is "no such window",
# which is an empty title, not a denial.
_DENIED_MARKERS = (
    "not allowed assistive",
    "assistive access",
    "not authorized",
    "-25211",
    "-1743",
    "1002",
)
# "Can't get window" is an app with no window, not a permission failure.
_NO_WINDOW_MARKERS = ("-1728", "can't get", "cannot get")

# Front window title only. Do not ask for UI elements, values, or the document.
# No `try` around it: a permission error must reach us, or it looks like "no window".
_TITLE_SCRIPT = """
tell application "System Events"
    return name of front window of first application process whose frontmost is true
end tell
"""

# Used only when lsappinfo cannot name the front app.
_APP_SCRIPT = """
tell application "System Events"
    set p to first application process whose frontmost is true
    set appName to name of p
    set bid to ""
    try
        set bid to bundle identifier of p
    end try
    return appName & linefeed & bid
end tell
"""

_ASN = re.compile(r"ASN:0x[0-9a-fA-F]+-0x[0-9a-fA-F]+:?")
_INFO_FIELD = re.compile(
    r'"(LSDisplayName|CFBundleIdentifier|name|bundleid)"\s*=\s*"((?:\\.|[^"\\])*)"'
)


def enabled(cfg) -> bool:
    screen = getattr(cfg, "screen", None)
    return screen is not None and bool(getattr(screen, "enabled", False))


def _run_command(argv: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except OSError as e:
        return 1, "", str(e)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _denied(text: str) -> bool:
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in _DENIED_MARKERS)


def _unescape(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def parse_lsappinfo(front_text: str, info_text: str) -> tuple[str, str] | None:
    """App name and bundle id from `lsappinfo front` plus `lsappinfo info`."""
    if not _ASN.search(front_text or ""):
        return None
    found = {}
    for key, value in _INFO_FIELD.findall(info_text or ""):
        found[key] = _unescape(value)
    name = (found.get("LSDisplayName") or found.get("name") or "").strip()
    bundle = (found.get("CFBundleIdentifier") or found.get("bundleid") or "").strip()
    if not name:
        return None
    return name, bundle


def _asn(text: str) -> str:
    match = _ASN.search(text or "")
    if match is None:
        return ""
    asn = match.group(0)
    return asn if asn.endswith(":") else asn + ":"


def _one_line(text: str, limit: int | None = None) -> str:
    """Drop control characters and collapse whitespace. `limit` cuts the result."""
    chars = []
    for ch in text:
        if ch.isspace() or unicodedata.category(ch) == "Cc":
            chars.append(" ")
        else:
            chars.append(ch)
    cleaned = " ".join("".join(chars).split())
    if limit is not None and limit >= 0:
        return cleaned[:limit]
    return cleaned


class FrontSampler:
    """macOS frontmost app, bundle id, and front window title.

    `run(argv) -> (code, stdout, stderr)` is injectable. Tests never launch
    `osascript` or `lsappinfo`.
    """

    def __init__(self, run=None):
        self._run = _run_command if run is None else run
        self._logged_denied = False
        self._logged_title = False
        self._logged_app = False

    def sample(self) -> dict | None:
        """`{app, bundle_id, title}` or None when the front app cannot be named.

        `title` is None when the window title could not be read (Accessibility
        denied, or the title command failed). It is "" when the front app has
        no window. The dict has no other keys.
        """
        named = self._app()
        if named is None:
            return None
        app, bundle = named
        return {"app": app, "bundle_id": bundle, "title": self._title()}

    def _app(self) -> tuple[str, str] | None:
        named = self._app_from_lsappinfo()
        if named is not None:
            return named
        return self._app_from_osascript()

    def _app_from_lsappinfo(self) -> tuple[str, str] | None:
        code, out, _err = self._run(["lsappinfo", "front"])
        asn = _asn(out) if code == 0 else ""
        if not asn:
            return None
        code, info, _err = self._run(["lsappinfo", "info", "-only", "name,bundleid", asn])
        if code != 0:
            return None
        return parse_lsappinfo(out, info)

    def _app_from_osascript(self) -> tuple[str, str] | None:
        code, out, err = self._run(["osascript", "-e", _APP_SCRIPT])
        if code != 0 or _denied(err) or _denied(out):
            if not self._logged_app:
                self._logged_app = True
                log.warning("frontmost app is unavailable; no screen events until it can be named")
            return None
        lines = [line.strip() for line in out.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            return None
        bundle = lines[1] if len(lines) > 1 else ""
        return lines[0], bundle

    def _title(self) -> str | None:
        if self._logged_denied:
            return None
        code, out, err = self._run(["osascript", "-e", _TITLE_SCRIPT])
        blob = f"{err}\n{out}"
        if _denied(blob):
            if not self._logged_denied:
                self._logged_denied = True
                log.warning(
                    "window title needs Accessibility permission for the app running audi27 "
                    "(System Settings → Privacy & Security → Accessibility); recording the app name only"
                )
            return None
        if code != 0:
            if any(marker in blob.casefold() for marker in _NO_WINDOW_MARKERS):
                return ""
            if not self._logged_title:
                self._logged_title = True
                log.warning("window title is unavailable; recording the app name only")
            return None
        text = out.replace("\r\n", "\n").replace("\r", "\n").strip()
        if text.casefold() == "missing value":
            return ""
        return text


def make_sampler():
    """The macOS sampler, or None on any other platform."""
    if sys.platform != "darwin":
        return None
    return FrontSampler()


def _excluded(app: str, bundle: str, names) -> bool:
    hay_app = app.casefold()
    hay_bundle = bundle.casefold()
    for name in names or []:
        if not isinstance(name, str):
            continue
        needle = name.strip().casefold()
        if needle and (needle in hay_app or needle in hay_bundle):
            return True
    return False


def _markers_hit(title: str, markers) -> bool:
    folded = title.casefold()
    for marker in markers or []:
        if not isinstance(marker, str):
            continue
        needle = marker.strip().casefold()
        if needle and needle in folded:
            return True
    return False


def _patterns(patterns):
    compiled = []
    for raw in patterns or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error:
            log.warning("ignoring an invalid screen title pattern")
    return compiled


def prepare(raw: dict, cfg) -> dict | None:
    """The event `data` for one sample. Redacted titles are not copied in.

    Returns None when the sample has no app name. `title` is None when it was
    redacted or could not be read. `title_redacted` distinguishes those cases
    together with `title_reason` (`excluded_app`, `private`, `pattern`,
    `unavailable`).
    """
    if not isinstance(raw, dict):
        return None
    app = raw.get("app")
    if not isinstance(app, str):
        return None
    app = _one_line(app, _APP_LIMIT)
    if not app:
        return None
    bundle = raw.get("bundle_id") if isinstance(raw.get("bundle_id"), str) else ""
    bundle = _one_line(bundle, _APP_LIMIT)
    raw_title = raw.get("title")
    unavailable = raw_title is None or not isinstance(raw_title, str)
    title = "" if unavailable else _one_line(raw_title)
    limit = int(getattr(cfg, "max_title_chars", 200) or 200)

    redacted = False
    reason = None
    truncated = False
    if _excluded(app, bundle, getattr(cfg, "exclude_apps", ())):
        title = None
        redacted = True
        reason = "excluded_app"
    elif unavailable:
        title = None
        reason = "unavailable"
    elif _markers_hit(title, getattr(cfg, "private_title_markers", ())):
        title = None
        redacted = True
        reason = "private"
    elif any(pattern.search(title) for pattern in _patterns(getattr(cfg, "redact_title_patterns", ()))):
        title = None
        redacted = True
        reason = "pattern"
    elif len(title) > limit:
        title = title[:limit]
        truncated = True

    data = {
        "app": app,
        "bundle_id": bundle,
        "title": title,
        "title_redacted": redacted,
    }
    if reason:
        data["title_reason"] = reason
    if truncated:
        data["title_truncated"] = True
    return data


def public_context(data) -> dict | None:
    """App, bundle id, and title only. A redacted title is forced to None."""
    if not isinstance(data, dict):
        return None
    app = data.get("app")
    if not isinstance(app, str) or not _one_line(app):
        return None
    bundle = data.get("bundle_id") if isinstance(data.get("bundle_id"), str) else ""
    redacted = bool(data.get("title_redacted"))
    title = data.get("title")
    if redacted or not isinstance(title, str):
        title = None
    else:
        title = _one_line(title)
    return {
        "app": _one_line(app, _APP_LIMIT),
        "bundle_id": _one_line(bundle, _APP_LIMIT),
        "title": title,
        "title_redacted": redacted,
    }


def _parse_when(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt


def context_at(data_dir, when, stale_s: float) -> dict | None:
    """Latest screen snapshot at `when`, if it is not older than `stale_s`.

    Events after `when` are ignored, so a replay does not use a window the
    turn had not reached yet. The returned dict is the public view.
    """
    end = _parse_when(when)
    if end is None:
        return None
    try:
        stale = float(stale_s)
    except (TypeError, ValueError):
        return None
    if stale < 0:
        return None
    start = end - timedelta(seconds=stale)
    chosen = None
    for ev in replay(data_dir, since=start, until=end + timedelta(milliseconds=1), types="screen"):
        ts = _parse_when(ev.get("ts"))
        if ts is None or ts < start or ts > end:
            continue
        view = public_context(ev.get("data"))
        if view is not None:
            chosen = view
    return chosen


def context_for(data_dir, cfg, when) -> dict | None:
    """`context_at` when screen is enabled, else None. A read error attaches nothing."""
    if not enabled(cfg):
        return None
    try:
        return context_at(data_dir, when, cfg.screen.stale_s)
    except Exception:
        log.exception("screen context was not read")
        return None


def snapshot_key(payload: dict) -> tuple:
    return (
        payload.get("app"),
        payload.get("bundle_id"),
        payload.get("title"),
        bool(payload.get("title_redacted")),
    )


class Debouncer:
    """Emit a snapshot only after it has been stable for `debounce_s`.

    A new snapshot replaces one that has not been emitted yet, which is what
    drops the apps flashed past in an alt-tab. The same snapshot is not emitted
    twice. `debounce_s` of 0 emits on the sample that changed it.
    """

    def __init__(self, debounce_s: float):
        self.debounce_s = max(0.0, float(debounce_s))
        self._key = None
        self._since = None
        self._payload = None
        self._emitted = None
        self.due_at: float | None = None

    def seed(self, payload: dict) -> None:
        """Treat `payload` as already emitted, so a restart does not repeat it."""
        key = snapshot_key(payload)
        self._key = key
        self._payload = payload
        self._emitted = key
        self._since = None
        self.due_at = None

    def note(self, payload: dict | None, now: float) -> None:
        if not payload:
            return
        key = snapshot_key(payload)
        if key != self._key:
            self._key = key
            self._since = now
            self._payload = payload
        if key == self._emitted:
            self.due_at = None
            return
        since = now if self._since is None else self._since
        self.due_at = since + self.debounce_s

    def take(self, now: float) -> dict | None:
        if self.due_at is None or now < self.due_at:
            return None
        if self._key == self._emitted or self._payload is None:
            self.due_at = None
            return None
        self._emitted = self._key
        self.due_at = None
        return self._payload


def _seed(debouncer: Debouncer, data_dir) -> None:
    last = None
    try:
        for ev in replay(data_dir, types="screen", sources=SCREEN_SOURCE):
            last = ev
    except Exception:
        log.exception("screen history was not read")
        return
    if not last or not isinstance(last.get("data"), dict):
        return
    if last["data"].get("app"):
        debouncer.seed(last["data"])


def _write(data_dir, payload: dict) -> None:
    stored = emit(data_dir, "screen", SCREEN_SOURCE, payload)
    # The title stays out of the log. The app name is not the sensitive part.
    log.info("screen %s", stored["data"].get("app"))


def run(data_dir, cfg, *, sampler=None, stop: threading.Event | None = None,
        poll_s: float | None = None, clock=None, wait=None) -> None:
    """Sample until `stop`. `sampler` None on a non-Mac host idles without writing.

    `clock` returns monotonic seconds. `wait(seconds)` sleeps and returns true
    when the loop should stop. Tests pass both so they do not sleep.
    """
    if not enabled(cfg):
        log.info("screen is disabled; not sampling")
        return
    stop = stop or threading.Event()
    clock = time.monotonic if clock is None else clock

    def _wait(seconds: float) -> bool:
        if wait is not None:
            return bool(wait(seconds))
        if seconds <= 0:
            return stop.is_set()
        return stop.wait(seconds)

    if sampler is None:
        log.warning(
            "screen sampler reads the frontmost app only on macOS (this host is %s). "
            "It will stay idle and write no screen events.",
            sys.platform,
        )
        while not stop.is_set():
            if _wait(1.0):
                return
        return

    interval = cfg.screen.poll_s if not poll_s else float(poll_s)
    if interval <= 0:
        interval = cfg.screen.poll_s
    debouncer = Debouncer(cfg.screen.debounce_s)
    _seed(debouncer, data_dir)
    next_sample = clock()
    while not stop.is_set():
        now = clock()
        if now >= next_sample:
            try:
                raw = sampler.sample()
            except Exception:
                log.exception("screen sample failed")
                raw = None
            next_sample = now + interval
            if raw:
                debouncer.note(prepare(raw, cfg.screen), now)
        emitted = debouncer.take(clock())
        if emitted is not None:
            try:
                _write(data_dir, emitted)
            except Exception:
                log.exception("screen event was not written")
        now = clock()
        wake = next_sample
        if debouncer.due_at is not None:
            wake = min(wake, debouncer.due_at)
        delay = wake - now
        if delay < 0:
            delay = 0.0
        if _wait(delay):
            return


def serve(data_dir, cfg, *, poll_s: float | None = None, stop: threading.Event | None = None,
          sampler="auto") -> None:
    """`audi27 screen run`. Picks the platform sampler unless one is injected."""
    if sampler == "auto":
        sampler = make_sampler()
    run(data_dir, cfg, sampler=sampler, stop=stop, poll_s=poll_s)
