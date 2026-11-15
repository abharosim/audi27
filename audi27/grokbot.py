"""Grok Bot adapter. There is no SDK: a webhook wakes the bot, and the reply is a file.

`handle` emits `sent` before the POST. The bot writes
`<data_dir>/inbox/grokbot/<task_id>.json` later (seconds or many minutes;
sometimes twice; sometimes never). `dispatch run` polls that directory, and
`audi27 grokbot inbox` is the same loop on its own. A file is moved to
`processed/` after one reply event, or to `quarantine/` when it is not a
reply we sent. The same `task_id` is not emitted twice; a restart re-reads
the event log. A JSON parse error is retried for a few ticks (a partial
write) and only then quarantined.

URL and key come from `AUDI27_GROKBOT_WEBHOOK_URL` and
`AUDI27_GROKBOT_WEBHOOK_KEY` (environment or `.env`), never from config.toml,
and are never written to the log or the event stream. The auth header is
`[grokbot] header_name` / `header_value`, default `Authorization: Bearer {key}`.
Reply text is plain and capped before it is spoken. Symlinks under the inbox
are not followed.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import stat
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from .agent import Agent, reply_event, route_text, task_event
from .events import append, replay

log = logging.getLogger("audi27.grokbot")

ADAPTER = "grokbot"
TIMEOUT_REASON = "Grok Bot 没有回复"
_URL_ENV = "AUDI27_GROKBOT_WEBHOOK_URL"
_KEY_ENV = "AUDI27_GROKBOT_WEBHOOK_KEY"
_MISSING = f"没有配置 Grok Bot webhook（{_URL_ENV} / {_KEY_ENV}）"
_STATUSES = frozenset({"done", "failed", "needs_decision"})
_FIELDS = ("task_id", "route_id", "status", "summary", "text")
_STATE_NAME = "grokbot.inbox.json"
_UTTERANCE_CHARS = 500
_MAX_BODY = 65536
_MAX_REPLY_BYTES = 1024 * 1024
_TEXT_LIMIT = 4000
_SUMMARY_LIMIT = 240


class WebhookError(Exception):
    def __init__(self, message: str, *, network: bool = False):
        super().__init__(message)
        self.network = network


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def inbox_dir(data_dir) -> Path:
    return Path(data_dir) / "inbox" / "grokbot"


def _now() -> datetime:
    return datetime.now().astimezone()


def _header(name: str, template: str, key: str) -> dict:
    return {name: template.replace("{key}", key)}


def _public_failure(exc: BaseException) -> str:
    """Class name only. `str(exc)` from urllib can contain the URL or the key."""
    seen: set[int] = set()

    def walk(err, depth: int) -> str:
        if depth > 4 or id(err) in seen or isinstance(err, str):
            return "error"
        seen.add(id(err))
        name = err.__class__.__name__
        number = getattr(err, "errno", None)
        if isinstance(number, int):
            return f"{name} ({number})"
        reason = getattr(err, "reason", None)
        if reason is not None and reason is not err and not isinstance(reason, str):
            inner = walk(reason, depth + 1)
            if inner != "error":
                return f"{name}: {inner}"
        return name

    return walk(exc, 0)


def _is_timeout(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    while exc is not None and not isinstance(exc, str) and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, TimeoutError) or exc.__class__.__name__ in ("TimeoutError", "timeout"):
            return True
        nxt = getattr(exc, "reason", None)
        exc = None if nxt is exc or isinstance(nxt, str) else nxt
    return False


def _redact(text: str, *secrets: str) -> str:
    out = text
    for secret in secrets:
        if isinstance(secret, str) and secret:
            out = out.replace(secret, "***")
    return out


def _plain(text: str, limit: int) -> str:
    """Drop control characters and cap length so `say` and the dashboard get plain text."""
    if not isinstance(text, str) or limit <= 0:
        return ""
    chars = []
    for ch in text:
        if ch in "\n\r\t":
            chars.append(" ")
        elif unicodedata.category(ch)[0] == "C":
            continue
        else:
            chars.append(ch)
    out = " ".join("".join(chars).split()).strip()
    if len(out) > limit:
        out = out[:limit - 1].rstrip() + "…"
    return out


def _decode(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="replace")
    return ""


def _relax_timeout(resp) -> None:
    """So a quiet body cannot sit past the caller's deadline between reads."""
    fp = getattr(resp, "fp", None)
    raw = getattr(fp, "raw", None) if fp is not None else None
    sock = getattr(raw, "_sock", None) if raw is not None else None
    if sock is None:
        return
    try:
        sock.settimeout(0.2)
    except Exception:
        pass


def _read_some(resp, seconds: float) -> bytes:
    """Read a response body, but stop after `seconds` even if the server keeps writing.

    `read(n)` waits until n bytes arrive. One byte at a time returns to the clock,
    so a trickle cannot hold the dispatcher for the whole body.
    """
    if resp is None:
        return b""
    _relax_timeout(resp)
    deadline = time.monotonic() + max(0.0, seconds)
    chunks = []
    total = 0
    while total < _MAX_BODY and time.monotonic() < deadline:
        try:
            part = resp.read(1)
        except Exception:
            break
        if not part:
            break
        chunks.append(part)
        total += len(part)
    return b"".join(chunks)


def _once(url: str, body: bytes, headers: dict, timeout_s: float):
    """One POST. Returns ('status', code, text, None), ('pending', ...), or ('error', ..., exc).

    Headers are waited on for `timeout_s`. The body is abandoned after a short budget, from
    the same thread that owns the socket. A still-running connect is not a finished attempt.
    """
    box: dict = {}

    def run() -> None:
        resp = None
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                resp = urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout_s)
            except urllib.error.HTTPError as e:
                box["status"] = e.code
                try:
                    box["body"] = _read_some(e, 0.25) if e.fp is not None else b""
                except Exception:
                    box["body"] = b""
                finally:
                    try:
                        e.close()
                    except Exception:
                        pass
                return
            box["status"] = resp.status
            try:
                box["body"] = _read_some(resp, 0.25)
            except Exception:
                box["body"] = b""
        except Exception as e:
            box["error"] = e
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if "status" in box or "error" in box or not thread.is_alive():
            break
        thread.join(min(0.05, max(0.0, deadline - time.monotonic())))
    if thread.is_alive() and "status" not in box and "error" not in box:
        # The socket timeout and this deadline land together. Collect a finished error.
        thread.join(0.25)
    if "status" in box:
        thread.join(0.5)  # body budget is 0.25s and runs on this same thread
        return "status", box["status"], _decode(box.get("body")), None
    if thread.is_alive():
        return "pending", None, "", None
    err = box.get("error")
    return "error", None, "", err if isinstance(err, BaseException) else WebhookError("error", network=True)


def post_webhook(url: str, payload: dict, *, header_name: str, header_value: str,
                 key: str, timeout_s: float) -> tuple[int, str]:
    """POST `payload`. One retry, and only when the attempt finished with a network error.

    A non-2xx response is returned, not retried. Redirects are not followed.
    The wait is capped at `timeout_s` per attempt. An attempt that is still running
    is not retried: it may yet send. Returned text has the key and URL removed.
    """
    if (not isinstance(url, str) or not isinstance(key, str)
            or any(c in url or c in key for c in "\r\n\x00")):
        raise WebhookError("invalid webhook", network=False)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **_header(header_name, header_value, key)}
    try:
        for value in headers.values():
            value.encode("latin-1")
    except UnicodeEncodeError:
        raise WebhookError("invalid webhook", network=False) from None
    last = WebhookError("error", network=True)
    for attempt in (1, 2):
        kind, status, text, err = _once(url, body, headers, timeout_s)
        if kind == "status":
            return status, _redact(text, key, url)
        if kind == "pending":
            # The thread may still send. A second POST would deliver the task twice.
            raise WebhookError("timeout", network=True)
        last = WebhookError(_public_failure(err), network=True)
        if attempt == 2:
            raise last
    raise last


def recent_utterances(data_dir, n: int) -> list[str]:
    """The last `n` non-rejected utterance texts, oldest first. Empty when `n` is 0."""
    if n <= 0:
        return []
    texts: list[str] = []
    for ev in replay(data_dir, types="utterance"):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if data.get("status") == "rejected":
            continue
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip()[:_UTTERANCE_CHARS])
    return texts[-n:]


class GrokbotAgent(Agent):
    """POST the route to Grok Bot. The reply is emitted later, from the inbox."""

    name = ADAPTER

    def __init__(self, *, data_dir, url: str, key: str, header_name: str, header_value: str,
                 timeout_s: float, context_utterances: int):
        self.data_dir = str(data_dir)
        self.url = url
        self.key = key
        self.header_name = header_name
        self.header_value = header_value
        self.timeout_s = float(timeout_s)
        self.context_utterances = int(context_utterances)

    @classmethod
    def from_cfg(cls, cfg) -> "GrokbotAgent":
        g = cfg.grokbot
        return cls(
            data_dir=cfg.store.data_dir,
            url=os.environ.get(_URL_ENV, ""),
            key=os.environ.get(_KEY_ENV, ""),
            header_name=g.header_name,
            header_value=g.header_value,
            timeout_s=g.timeout_s,
            context_utterances=g.context_utterances,
        )

    def handle(self, route: dict, emit) -> None:
        url = self.url.strip() if isinstance(self.url, str) else ""
        key = self.key.strip() if isinstance(self.key, str) else ""
        if not url or not key or any(c in url or c in key for c in "\r\n"):
            emit(task_event(route, status="failed", ref=route["id"], adapter=ADAPTER, error=_MISSING))
            return
        now = _now()
        sent = task_event(route, status="sent", ref=route["id"], adapter=ADAPTER, ts=now)
        emit(sent)  # before any network call; a crash after this must not send twice
        payload = {
            "task_id": sent["id"],
            "route_id": route["id"],
            "text": route_text(route),
            "ts": now.isoformat(timespec="milliseconds"),
        }
        if self.context_utterances > 0:
            payload["context"] = recent_utterances(self.data_dir, self.context_utterances)
        try:
            status, body = post_webhook(
                url, payload, header_name=self.header_name, header_value=self.header_value,
                key=key, timeout_s=self.timeout_s)
        except WebhookError as e:
            log.error("grokbot post failed for %s: %s", route.get("id"), e)
            emit(task_event(route, status="failed", ref=sent["id"], adapter=ADAPTER,
                            error=f"Grok Bot 连不上：{e}", ts=_now()))
            return
        except Exception:
            log.error("grokbot post failed for %s", route.get("id"))
            emit(task_event(route, status="failed", ref=sent["id"], adapter=ADAPTER,
                            error="Grok Bot 连不上", ts=_now()))
            return
        if status < 200 or status >= 300:
            snippet = _plain(_redact(body, key, url), 80)
            reason = f"Grok Bot 返回 {status}" + (f"：{snippet}" if snippet else "")
            log.error("grokbot post for %s returned %s", route.get("id"), status)
            emit(task_event(route, status="failed", ref=sent["id"], adapter=ADAPTER,
                            error=reason, ts=_now()))
            return
        log.info("grokbot posted %s", route.get("id"))
        emit(task_event(route, status="running", ref=sent["id"], adapter=ADAPTER, ts=_now()))


def _state_path(data_dir) -> Path:
    return Path(data_dir) / _STATE_NAME


def _load_attempts(data_dir) -> dict:
    try:
        raw = json.loads(_state_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    attempts = raw.get("attempts") if isinstance(raw, dict) else None
    if not isinstance(attempts, dict):
        return {}
    out = {}
    for name, count in attempts.items():
        if isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool) and count > 0:
            out[name] = count
    return out


def _save_attempts(data_dir, attempts: dict) -> None:
    path = _state_path(data_dir)
    if not attempts:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"attempts": attempts}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _safe_id(value: str) -> bool:
    """An id used only as a lookup key, never as a path."""
    if not value or len(value) > 200 or value in (".", ".."):
        return False
    if any(c in value for c in "/\\\x00") or Path(value).name != value:
        return False
    return True


def _subdir(root: Path, name: str) -> Path | None:
    """A real directory inside `root`. A symlink is not followed."""
    dest = root / name
    if dest.is_symlink():
        log.error("grokbot inbox %s is a symlink; not writing through it", name)
        return None
    if dest.exists() and not dest.is_dir():
        log.error("grokbot inbox %s is not a directory", name)
        return None
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.exception("grokbot inbox could not create %s", name)
        return None
    if dest.is_symlink() or not dest.is_dir():
        return None
    try:
        dest.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        log.error("grokbot inbox %s is outside the inbox", name)
        return None
    return dest


def _move(src: Path, dest_dir: Path) -> None:
    target = dest_dir / src.name
    if target.is_symlink() or target.exists():
        target = dest_dir / f"{src.stem}-{uuid.uuid4().hex[:8]}{src.suffix}"
    os.replace(src, target)


def _read_reply(path: Path) -> tuple[str, str]:
    """Read one inbox file without following a symlink or loading a huge body.

    `ok` is complete bytes that did not change during the read. `unstable` is still
    being written. `later` is empty or not valid UTF-8 yet. `symlink`, `oversize`,
    and `bad` are refused. `missing` means the file went away.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return "missing", ""
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EMLINK):
            return "symlink", ""
        return "later", ""
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return "bad", ""
        if info.st_size > _MAX_REPLY_BYTES:
            return "oversize", ""
        if info.st_size == 0:
            return "later", ""
        try:
            raw = os.read(fd, info.st_size)
        except OSError:
            return "later", ""
        try:
            after = os.fstat(fd)
        except OSError:
            return "later", ""
        if not stat.S_ISREG(after.st_mode) or after.st_size != info.st_size or len(raw) != info.st_size:
            return "unstable", ""
    finally:
        os.close(fd)
    try:
        return "ok", raw.decode("utf-8")
    except UnicodeError:
        return "later", ""


def _snapshot(data_dir) -> dict:
    """Tasks, routes, and which routes already have a reply or a failure."""
    tasks: dict[str, dict] = {}
    routes: dict[str, dict] = {}
    replied: set[str] = set()
    failed: set[str] = set()
    sent_at: dict[str, str] = {}
    anchor: dict[str, str] = {}
    anchor_rank: dict[str, int] = {}
    for ev in replay(data_dir, types=("task", "reply", "route")):
        if ev.get("type") == "route" and isinstance(ev.get("id"), str):
            routes[ev["id"]] = ev
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        rid = data.get("route_id")
        if ev.get("type") == "reply":
            if isinstance(rid, str):
                replied.add(rid)
            continue
        if isinstance(ev.get("id"), str):
            tasks[ev["id"]] = ev
        if not isinstance(rid, str):
            continue
        status = data.get("status")
        if status == "failed":
            failed.add(rid)
        if data.get("adapter") != ADAPTER:
            continue
        rank = 2 if status == "running" else 1 if status == "sent" else 0
        if rank and anchor_rank.get(rid, 0) < rank:
            anchor[rid] = ev["id"]
            anchor_rank[rid] = rank
        if status == "sent" and isinstance(ev.get("ts"), str):
            sent_at[rid] = ev["ts"]
    return {
        "tasks": tasks, "routes": routes, "replied": replied, "failed": failed,
        "sent_at": sent_at, "anchor": anchor,
    }


def _route_for(snap: dict, route_id: str, task: dict) -> dict:
    found = snap["routes"].get(route_id)
    if found is not None:
        return found
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    return {
        "id": route_id,
        "ts": task.get("ts"),
        "data": {
            "target": data.get("target") if isinstance(data.get("target"), str) else "",
            "text": data.get("text") if isinstance(data.get("text"), str) else "",
        },
    }


def _accept(path: Path, obj: dict, snap: dict, emit) -> str:
    """Apply one valid reply file. Returns 'processed' or 'quarantine'."""
    task_id = obj["task_id"].strip()
    route_id = obj["route_id"].strip()
    status = obj["status"].strip()
    summary = _plain(obj["summary"], _SUMMARY_LIMIT)
    text = _plain(obj["text"], _TEXT_LIMIT)
    if not _safe_id(task_id) or not _safe_id(route_id) or status not in _STATUSES:
        log.error("grokbot inbox quarantine %s: bad reply", path.name)
        return "quarantine"
    task = snap["tasks"].get(task_id)
    if task is None:
        log.error("grokbot inbox quarantine %s: unknown task_id %s", path.name, task_id[:80])
        return "quarantine"
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
    if task_data.get("route_id") != route_id:
        log.error("grokbot inbox quarantine %s: route_id does not match %s", path.name, task_id[:80])
        return "quarantine"
    route = _route_for(snap, route_id, task)
    anchor = snap["anchor"].get(route_id) or task_id
    if status == "failed":
        if route_id in snap["replied"] or route_id in snap["failed"]:
            return "processed"
        reason = summary or text or "Grok Bot 失败"
        reason = _plain(reason, _SUMMARY_LIMIT) or "Grok Bot 失败"
        emit(task_event(route, status="failed", ref=anchor, adapter=ADAPTER, error=reason, ts=_now()))
        snap["failed"].add(route_id)
        return "processed"
    if route_id in snap["replied"]:
        return "processed"
    body = text or summary
    done = task_event(route, status="done", ref=anchor, adapter=ADAPTER, ts=_now())
    emit(done)
    reply = reply_event(
        route, ref=done["id"], text=body, needs_decision=(status == "needs_decision"),
        adapter=ADAPTER, ts=_now())
    if summary:
        reply["data"]["summary"] = summary
    emit(reply)
    snap["replied"].add(route_id)
    snap["anchor"][route_id] = done["id"]
    return "processed"


def _retry_parse(path: Path, attempts: dict, retries: int) -> str:
    """Count one JSON failure. 'keep' leaves the file; 'quarantine' gives up."""
    attempts[path.name] = attempts.get(path.name, 0) + 1
    if attempts[path.name] >= retries:
        log.error("grokbot inbox quarantine %s: not valid JSON after %s tries", path.name, attempts[path.name])
        attempts.pop(path.name, None)
        return "quarantine"
    log.warning("grokbot inbox %s is not valid JSON yet (%s/%s)", path.name, attempts[path.name], retries)
    return "keep"


def poll_inbox(data_dir, cfg, emit) -> None:
    """Read new `*.json` reply files. Idempotent: a known task is emitted once."""
    root = inbox_dir(data_dir)
    if root.is_symlink():
        log.error("grokbot inbox %s is a symlink; not reading it", root)
        return
    if not root.is_dir():
        return
    attempts = _load_attempts(data_dir)
    snap = _snapshot(data_dir)
    retries = cfg.grokbot.parse_retries
    processed = _subdir(root, "processed")
    quarantine = _subdir(root, "quarantine")

    def park(path: Path, action: str) -> None:
        dest = quarantine if action == "quarantine" else processed
        if dest is None:
            log.error("grokbot inbox left %s in place", path.name)
            return
        try:
            _move(path, dest)
        except FileNotFoundError:
            return
        except OSError:
            log.exception("grokbot inbox could not move %s", path.name)

    files = sorted(
        p for p in root.iterdir()
        if p.suffix == ".json" and (p.is_symlink() or p.is_file())
    )
    for path in files:
        try:
            kind, raw = _read_reply(path)
            if kind in ("missing", "unstable"):
                continue
            if kind in ("symlink", "oversize", "bad"):
                log.error("grokbot inbox quarantine %s: %s", path.name, kind)
                park(path, "quarantine")
                continue
            if kind == "later" or not raw.strip():
                if _retry_parse(path, attempts, retries) == "quarantine":
                    park(path, "quarantine")
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                if _retry_parse(path, attempts, retries) == "quarantine":
                    park(path, "quarantine")
                continue
            attempts.pop(path.name, None)
            if not isinstance(obj, dict) or any(not isinstance(obj.get(name), str) for name in _FIELDS):
                log.error("grokbot inbox quarantine %s: malformed reply", path.name)
                action = "quarantine"
            else:
                action = _accept(path, obj, snap, emit)
            park(path, action)
        except Exception:
            log.exception("grokbot inbox skipped %s", path.name)
    try:
        _save_attempts(data_dir, attempts)
    except OSError:
        log.exception("grokbot inbox state was not written")


def _elapsed(ts: str, now: datetime) -> float | None:
    try:
        sent = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if sent.tzinfo is None or sent.utcoffset() is None:
        return None
    return (now - sent).total_seconds()


def _expire(data_dir, cfg, emit, *, now=None) -> None:
    """Fail grokbot tasks that have had no reply for `reply_timeout_s`."""
    now = _now() if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.astimezone()
    snap = _snapshot(data_dir)
    limit = cfg.grokbot.reply_timeout_s
    for route_id, ts in snap["sent_at"].items():
        if route_id in snap["replied"] or route_id in snap["failed"]:
            continue
        elapsed = _elapsed(ts, now)
        if elapsed is None or elapsed < limit:
            continue
        route = snap["routes"].get(route_id)
        if route is None:
            log.error("grokbot timeout for %s but the route is not in the log", route_id)
            continue
        anchor = snap["anchor"].get(route_id, route_id)
        log.error("grokbot timeout %s after %.0fs", route_id, elapsed)
        emit(task_event(route, status="failed", ref=anchor, adapter=ADAPTER,
                        error=TIMEOUT_REASON, ts=now))
        snap["failed"].add(route_id)


def tick(data_dir, cfg, emit, *, now=None) -> None:
    """Pick up reply files, then fail tasks that have waited too long."""
    poll_inbox(data_dir, cfg, emit)
    _expire(data_dir, cfg, emit, now=now)


def serve_inbox(data_dir, cfg, *, poll_s: float = 0.2, stop: threading.Event | None = None) -> None:
    """Watch the inbox until `stop` or Ctrl-C. Does not dispatch routes."""
    stop = stop or threading.Event()

    def emit(event: dict) -> None:
        append(data_dir, event, if_absent=True)

    log.info("grokbot inbox watching %s", inbox_dir(data_dir))
    while not stop.is_set():
        try:
            tick(data_dir, cfg, emit)
        except Exception:
            log.exception("grokbot inbox was not processed")
        if stop.wait(poll_s):
            break
