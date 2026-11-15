"""Frontmost app and window title, as `screen` events.

Privacy boundary: a sample is the frontmost app name, its bundle id, and the
front window's title. Nothing else. No screenshots, no OCR, no window contents,
and no accessibility tree beyond that one title. Titles of excluded apps,
private browser windows, and configured patterns are dropped before the event
is built, so the dropped text is never written and is not logged.

Sampling is off unless `[screen] enabled` is true. On macOS one `osascript`
reads the front app, its bundle id, and that same process's window title. The
script is a fixed string: the app name and the title are never interpolated.
`lsappinfo` is only the fallback when that call is denied or the app has no
window. A denial is logged once, and later polls do not call `osascript`.
Other platforms stay idle and write nothing.

Each command has a timeout. On timeout the process group is killed and any
partial output is discarded, so a title that was already printed is not kept.
Events are written only when the recorded snapshot changes, and only after it
has held still for `debounce_s`, so a burst of alt-tab does not flood the log.

`data/screen.json` is a liveness heartbeat: `started` and `updated` only, no
app and no title. A turn attaches the latest change-only snapshot when that
heartbeat shows the sampler was running at the turn. A stopped sampler does
not: an old event is not the current window just because it is younger than
`stale_s`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

from .events import emit, replay

log = logging.getLogger("audi27.screen")

SCREEN_SOURCE = "screen"
HEARTBEAT_NAME = "screen.json"
_APP_LIMIT = 200
# How far back a live sampler may look. Change-only events mean a window that
# has not changed is still front, so the last event can be days old.
_LIVE_LOOKBACK_S = 7 * 24 * 3600.0
_HEARTBEAT_SKEW_S = 2.0
_FUTURE_S = 60.0
# osascript errors that mean Accessibility was not granted. Matched only
# against a failed command, never against a window title that happened to
# contain a number. -1728 is "no such window", an empty title, not a denial.
_DENIED_MARKERS = (
    "not allowed assistive",
    "assistive access",
    "not authorized to send apple events",
    "(-25211)",
    "(-1743)",
    "(-1719)",
    "(1002)",
)
# "Can't get window" is an app with no window, not a permission failure.
_NO_WINDOW_MARKERS = ("-1728", "can't get", "cannot get")

# One process, so the title cannot belong to a different app than the name.
# Do not interpolate the app name or the title into this string, and do not
# wrap the window read in `try`: a permission error must reach us, or it looks
# like "no window". `try` is only around the bundle id, which some processes
# do not have.
_FRONT_SCRIPT = """
tell application "System Events"
    set p to first application process whose frontmost is true
    set appName to name of p
    set bid to ""
    try
        set bid to bundle identifier of p
    end try
    set winName to name of front window of p
    return appName & linefeed & bid & linefeed & winName
end tell
"""

_ASN = re.compile(r"ASN:0x[0-9a-fA-F]+-0x[0-9a-fA-F]+:?")
_INFO_FIELD = re.compile(
    r'"(LSDisplayName|CFBundleIdentifier|name|bundleid)"\s*=\s*"((?:\\.|[^"\\])*)"'
)


def enabled(cfg) -> bool:
    screen = getattr(cfg, "screen", None)
    return screen is not None and bool(getattr(screen, "enabled", False))


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the command and anything it started. Partial output is not read."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if pid and pid > 0:
            os.killpg(pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _run_command(argv: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run `argv` without a shell. On timeout, kill the process group and return no output.

    A title may already have been printed when the timer fires. That text is
    discarded, and it is not included in the returned error.
    """
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as e:
        return 1, "", str(e)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return 1, "", "timeout"
        code = proc.returncode if proc.returncode is not None else 1
        return code, out or "", err or ""
    finally:
        if proc.poll() is None:
            _kill_group(proc)


def _denied(text: str) -> bool:
    """True when `text` is an osascript failure caused by a missing Accessibility grant.

    Callers pass stderr (and stdout only when the command failed). A window
    title that contains "1002" or "not authorized" is not a denial.
    """
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


def _split_front(text: str) -> tuple[str, str, str] | None:
    """App, bundle id, and title from one osascript result. The title may contain newlines."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not text.strip():
        return None
    parts = text.split("\n", 2)
    app = parts[0].strip()
    if not app or app.casefold() == "missing value":
        return None
    bundle = parts[1].strip() if len(parts) > 1 else ""
    if bundle.casefold() == "missing value":
        bundle = ""
    title = parts[2] if len(parts) > 2 else ""
    if title.strip().casefold() == "missing value":
        title = ""
    return app, bundle, title


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
        no window. The dict has no other keys. App name, bundle id, and title
        come from one osascript call so they belong to the same process.
        """
        if not self._logged_denied:
            kind = self._from_osascript()
            if kind[0] == "ok":
                return {"app": kind[1], "bundle_id": kind[2], "title": kind[3]}
            if kind[0] == "denied":
                self._note_denied()
            elif kind[0] == "empty":
                named = self._app_from_lsappinfo()
                if named is None:
                    return None
                return {"app": named[0], "bundle_id": named[1], "title": ""}
        named = self._app_from_lsappinfo()
        if named is None:
            if not self._logged_app and not self._logged_denied:
                self._logged_app = True
                log.warning("frontmost app is unavailable; no screen events until it can be named")
            return None
        return {"app": named[0], "bundle_id": named[1], "title": None}

    def _note_denied(self) -> None:
        if self._logged_denied:
            return
        self._logged_denied = True
        log.warning(
            "window title needs Accessibility permission for the app running audi27 "
            "(System Settings → Privacy & Security → Accessibility); recording the app name only"
        )

    def _from_osascript(self) -> tuple:
        """`("ok", app, bundle, title)`, `("denied",)`, `("empty",)`, or `("fail",)`.

        A zero exit is a title, even when the text contains "1002" or
        "not authorized". Denial is decided from a failed command only.
        """
        code, out, err = self._run(["osascript", "-e", _FRONT_SCRIPT])
        if code == 0:
            parsed = _split_front(out)
            if parsed is None:
                return ("fail",)
            return ("ok", parsed[0], parsed[1], parsed[2])
        blob = f"{err}\n{out}"
        if _denied(blob):
            return ("denied",)
        if any(marker in blob.casefold() for marker in _NO_WINDOW_MARKERS):
            return ("empty",)
        if not self._logged_title:
            self._logged_title = True
            log.warning("window title is unavailable; recording the app name only")
        return ("fail",)

    def _app_from_lsappinfo(self) -> tuple[str, str] | None:
        code, out, _err = self._run(["lsappinfo", "front"])
        asn = _asn(out) if code == 0 else ""
        if not asn:
            return None
        code, info, _err = self._run(["lsappinfo", "info", "-only", "name,bundleid", asn])
        if code != 0:
            return None
        return parse_lsappinfo(out, info)


def make_sampler():
    """The macOS sampler, or None on any other platform."""
    if sys.platform != "darwin":
        return None
    return FrontSampler()


def _compact(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _alias(compact: str) -> str:
    """1Password's bundle id says onepassword, not 1password."""
    return compact.replace("1password", "onepassword")


def _excluded(app: str, bundle: str, names) -> bool:
    """Case-insensitive match on the app name or the bundle id.

    A needle with spaces also matches those letters run together, so
    "Keychain Access" matches `com.apple.keychainaccess` and a localized
    display name still excludes the app. Short needles stay substring matches
    on the raw text only, so "app" does not exclude every Apple bundle.
    """
    folded_app = (app or "").casefold()
    folded_bundle = (bundle or "").casefold()
    compact_app = _alias(_compact(app or ""))
    compact_bundle = _alias(_compact(bundle or ""))
    for name in names or []:
        if not isinstance(name, str):
            continue
        raw = name.strip()
        if not raw:
            continue
        folded = raw.casefold()
        if folded in folded_app or folded in folded_bundle:
            return True
        compact = _alias(_compact(raw))
        if len(compact) >= 6 and (compact in compact_app or compact in compact_bundle):
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


_DROP_REASONS = frozenset({"excluded_app", "private", "pattern"})


def public_context(data) -> dict | None:
    """App, bundle id, and title only. A redacted title is forced to None.

    `title_reason` of excluded_app, private, or pattern drops the title even
    when `title_redacted` was left false.
    """
    if not isinstance(data, dict):
        return None
    app = data.get("app")
    if not isinstance(app, str) or not _one_line(app):
        return None
    bundle = data.get("bundle_id") if isinstance(data.get("bundle_id"), str) else ""
    redacted = bool(data.get("title_redacted")) or data.get("title_reason") in _DROP_REASONS
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


def liveness_window(cfg) -> float:
    """How long a sampler may go without a heartbeat and still count as running.

    A hung `osascript` blocks one poll, so this is several poll intervals and
    at least 15 s. It is not `stale_s`.
    """
    screen = getattr(cfg, "screen", None)
    poll = float(getattr(screen, "poll_s", 2.0) or 2.0)
    debounce = float(getattr(screen, "debounce_s", 0.0) or 0.0)
    return max(15.0, poll * 3 + debounce + 5.0)


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def write_heartbeat(data_dir, started: float, updated: float) -> None:
    """Atomically replace `screen.json` with two timestamps. No app and no title."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {"started": float(started), "updated": float(updated)}
    tmp = root / (HEARTBEAT_NAME + ".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, root / HEARTBEAT_NAME)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_heartbeat(data_dir) -> dict | None:
    """The heartbeat file, or None when it is missing or not a JSON object."""
    path = Path(data_dir) / HEARTBEAT_NAME
    for _ in range(2):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _known_heartbeat(data) -> bool:
    return isinstance(data, dict) and _number(data.get("started")) is not None and _number(data.get("updated")) is not None


def heartbeat_covers(data, when: datetime, liveness_s: float, now: float | None = None) -> bool:
    """True when this sampler process was running at `when` and had checked in recently.

    `started` is when the process began. `updated` is the last successful sample.
    A process that started after the turn cannot describe it. A process whose
    last sample is older than `liveness_s` before the turn had already stopped.
    """
    if not _known_heartbeat(data):
        return False
    started = _number(data.get("started"))
    updated = _number(data.get("updated"))
    if started is None or updated is None:
        return False
    now = time.time() if now is None else now
    when_e = when.timestamp()
    # Ahead of both the machine clock and the turn: a clock jump, not a replay
    # of an older log on a machine whose clock is behind the timestamps.
    horizon = max(now, when_e) + _FUTURE_S
    if updated > horizon or started > horizon:
        return False
    if started > when_e + _HEARTBEAT_SKEW_S:
        return False
    if updated < when_e - liveness_s:
        return False
    return True


def _strip_excluded(view, cfg):
    """Drop a title the current exclude list rejects, including one stored earlier."""
    if not view:
        return None
    names = getattr(getattr(cfg, "screen", None), "exclude_apps", ())
    if _excluded(view.get("app") or "", view.get("bundle_id") or "", names):
        return {**view, "title": None, "title_redacted": True}
    return view


def context_for(data_dir, cfg, when) -> dict | None:
    """Screen snapshot for a turn, or None when screen is off or the sampler was not running.

    A heartbeat that covers `when` means the change-only log is still current,
    so the latest event may be older than `stale_s`. A heartbeat that does not
    cover `when` means the sampler had stopped (or this process started later):
    nothing is attached. With no heartbeat file, `stale_s` is the age limit.
    """
    if not enabled(cfg):
        return None
    try:
        return _context_for(data_dir, cfg, when)
    except Exception:
        log.exception("screen context was not read")
        return None


def _context_for(data_dir, cfg, when) -> dict | None:
    end = _parse_when(when)
    if end is None:
        return None
    try:
        stale = float(cfg.screen.stale_s)
    except (TypeError, ValueError):
        return None
    if stale < 0:
        return None
    hb = load_heartbeat(data_dir)
    if _known_heartbeat(hb):
        if heartbeat_covers(hb, end, liveness_window(cfg)):
            view = context_at(data_dir, end, max(stale, _LIVE_LOOKBACK_S))
        else:
            view = None
    else:
        view = context_at(data_dir, end, stale)
    return _strip_excluded(view, cfg)


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
    started = time.time()
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
            prepared = prepare(raw, cfg.screen) if raw else None
            if prepared:
                debouncer.note(prepared, now)
                try:
                    write_heartbeat(data_dir, started, time.time())
                except Exception:
                    log.exception("screen heartbeat was not written")
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
