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
`AUDI27_GROKBOT_WEBHOOK_KEY` (environment or `.env`), never from config.toml.
The auth header is `[grokbot] header_name` / `header_value`, default
`Authorization: Bearer {key}`.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import threading
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


class WebhookError(Exception):
    def __init__(self, message: str, *, network: bool = False):
        super().__init__(message)
        self.network = network


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def inbox_dir(data_dir) -> Path:
    return Path(data_dir) / "inbox" / "grokbot"


def _now() -> datetime:
    return datetime.now().astimezone()


def _header(name: str, template: str, key: str) -> dict:
    return {name: template.replace("{key}", key)}


def post_webhook(url: str, payload: dict, *, header_name: str, header_value: str,
                 key: str, timeout_s: float) -> tuple[int, str]:
    """POST `payload`. One retry, and only when the failure is a network error.

    A non-2xx response is returned, not retried. Redirects are not followed.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **_header(header_name, header_value, key)}
    last: WebhookError | None = None
    for attempt in (1, 2):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with _OPENER.open(req, timeout=timeout_s) as resp:
                raw = resp.read()
                return resp.status, raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raw = e.read() if e.fp is not None else b""
            return e.code, raw.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as e:
            last = WebhookError(str(e) or e.__class__.__name__, network=True)
            if attempt == 1:
                continue
    assert last is not None
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
        if status < 200 or status >= 300:
            snippet = " ".join(body.split())[:80]
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


def _move(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    if target.exists():
        target = dest_dir / f"{src.stem}-{uuid.uuid4().hex[:8]}{src.suffix}"
    os.replace(src, target)


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
    summary = obj["summary"].strip()
    text = obj["text"].strip()
    if not task_id or not route_id or status not in _STATUSES:
        log.error("grokbot inbox quarantine %s: bad reply", path.name)
        return "quarantine"
    task = snap["tasks"].get(task_id)
    if task is None:
        log.error("grokbot inbox quarantine %s: unknown task_id %s", path.name, task_id)
        return "quarantine"
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
    if task_data.get("route_id") != route_id:
        log.error("grokbot inbox quarantine %s: route_id does not match %s", path.name, task_id)
        return "quarantine"
    route = _route_for(snap, route_id, task)
    anchor = snap["anchor"].get(route_id) or task_id
    if status == "failed":
        if route_id in snap["replied"] or route_id in snap["failed"]:
            return "processed"
        reason = summary or text or "Grok Bot 失败"
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
    if not root.is_dir():
        return
    attempts = _load_attempts(data_dir)
    snap = _snapshot(data_dir)
    retries = cfg.grokbot.parse_retries
    processed = root / "processed"
    quarantine = root / "quarantine"
    for path in sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ".json"):
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as e:
            if _retry_parse(path, attempts, retries) == "quarantine":
                try:
                    _move(path, quarantine)
                except OSError:
                    log.exception("grokbot inbox could not quarantine %s", path.name)
            else:
                log.warning("grokbot inbox could not read %s: %s", path.name, e)
            continue
        if not raw.strip():
            if _retry_parse(path, attempts, retries) == "quarantine":
                try:
                    _move(path, quarantine)
                except OSError:
                    log.exception("grokbot inbox could not quarantine %s", path.name)
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            if _retry_parse(path, attempts, retries) == "quarantine":
                try:
                    _move(path, quarantine)
                except OSError:
                    log.exception("grokbot inbox could not quarantine %s", path.name)
            continue
        attempts.pop(path.name, None)
        if not isinstance(obj, dict) or any(not isinstance(obj.get(name), str) for name in _FIELDS):
            log.error("grokbot inbox quarantine %s: malformed reply", path.name)
            action = "quarantine"
        else:
            action = _accept(path, obj, snap, emit)
        try:
            _move(path, quarantine if action == "quarantine" else processed)
        except FileNotFoundError:
            continue
        except OSError:
            log.exception("grokbot inbox could not move %s", path.name)
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
        except OSError:
            log.exception("grokbot inbox was not processed")
        if stop.wait(poll_s):
            break
