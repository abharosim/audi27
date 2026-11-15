"""Local dashboard: `audi27 web`. Stdlib only; binds to 127.0.0.1 by default.

Reading is always allowed. Two things write: saving a human verdict, and starting a replay run
(which may upload audio to a cloud backend and cost money). Both are POST + JSON and are refused
when the request comes from another origin, so a web page you happen to visit cannot trigger them.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import queue
import re
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import config as C
from . import evals as E
from .events import replay
from .route import list_routes, save_route_verdict
from .status import read_status
from .store import clean_json, parse_when, search
from .transcribers import TranscribeError

log = logging.getLogger("audi27.web")
PAGE = Path(__file__).with_name("dashboard.html")
ALL = ("ok", "suspect", "rejected")
CORRECT = "@correct"   # pseudo backend: re-run only the context correction over the stored ASR text


class Jobs:
    """Replay runs, one at a time, all on one thread (MLX models are bound to the thread that loaded them)."""

    def __init__(self, load_cfg):
        self.load_cfg = load_cfg
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.state: dict = {"running": False}
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, backend: str, label: str, force: bool) -> dict:
        with self.lock:
            if self.state.get("running"):
                raise ValueError("a replay run is already in progress")
            self.state = {"running": True, "backend": backend, "label": label, "ran": 0, "failed": 0, "todo": None}
        self.q.put((backend, label, force))
        return self.snapshot()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def _update(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def _loop(self) -> None:
        while True:
            backend, label, force = self.q.get()
            try:
                if backend == CORRECT:
                    stats = E.run_correct(self.load_cfg(), label, force=force, log=log.info,
                                          progress=lambda s: self._update(**s))
                else:
                    stats = E.run_eval(self.load_cfg(), backend, label, force=force, log=log.info,
                                       progress=lambda s: self._update(**s))
                self._update(running=False, done=True, **stats)
            except (TranscribeError, E.EvalError, C.ConfigError, OSError) as e:
                self._update(running=False, done=True, error=str(e))
            except Exception as e:  # keep the job thread alive whatever a backend does
                log.exception("replay run crashed")
                self._update(running=False, done=True, error=repr(e))


def _media_type(value: str | None) -> str:
    """The content type without parameters. `text/plain;charset=application/json` is `text/plain`."""
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def _loopback_host(value: str | None) -> bool:
    """True when this host is loopback, not a wildcard and not a rebound name."""
    if not value:
        return False
    host = value.strip().lower().rstrip(".")
    if host.startswith("["):
        end = host.find("]")
        name = host[1:end] if end > 1 else ""
    elif host.count(":") == 1:
        name = host.split(":", 1)[0]
    else:
        name = host
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def make_server(data_dir: str | Path, host: str = "127.0.0.1", port: int = 8727,
                config_path: str | None = None) -> ThreadingHTTPServer:
    root = Path(data_dir).resolve()

    def load_cfg() -> C.Config:
        C.load_dotenv()            # re-read each time: a key added to .env or a backend edited in
        cfg = C.load(config_path)  # config.toml applies to the next run without a restart
        cfg.store.data_dir = str(root)
        return cfg

    jobs = Jobs(load_cfg)

    def backends() -> list[dict]:
        cfg = load_cfg()
        names = list(dict.fromkeys(list(cfg.transcribe.backends) + list(cfg.backend)))
        out = []
        for name in names:
            opts = cfg.backend.get(name, {})
            kind = opts.get("type", name)
            out.append({"name": name, "type": kind, "model": opts.get("model", ""), "cloud": kind in ("openai", "xai")})
        if cfg.correct.enabled:
            model = cfg.corrector.get(cfg.correct.backend, {}).get("model", "") if cfg.correct.llm else "仅术语替换"
            out.append({"name": CORRECT, "type": "correct", "model": model or cfg.correct.backend, "cloud": False})
        return out

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet; the page polls constantly
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(clean_json(obj), ensure_ascii=False, allow_nan=False).encode()
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/status":
                    s = read_status(root)
                    s["spool"] = len(list((root / "spool").glob("*.wav")))
                    s["failed"] = len(list((root / "failed").glob("*.wav")))
                    self._json(s)
                elif url.path == "/api/days":
                    days = sorted((p.stem for p in (root / "transcripts").glob("*.jsonl")), reverse=True)
                    self._json({"days": days})
                elif url.path == "/api/records":
                    since = until = None
                    if q.get("date"):
                        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", q["date"]):
                            return self._json({"error": "date must be YYYY-MM-DD"}, 400)
                        since = parse_when(q["date"])
                        until = since + timedelta(days=1)
                    limit = min(int(q.get("limit", 2000)), 10000)
                    recs = search(root, q.get("q", "").split(), since, until, ALL, 0, limit)
                    self._json({"records": recs})
                elif url.path == "/api/eval/labels":
                    self._json({"labels": E.labels(root)})
                elif url.path == "/api/eval/compare":
                    r = E.compare(root, q.get("ref", ""))
                    r["total_rows"] = len(r["rows"])
                    r["rows"] = r["rows"][:300]
                    self._json(r)
                elif url.path == "/api/eval/matrix":
                    self._json(E.matrix(root, q.get("base")))
                elif url.path == "/api/eval/backends":
                    self._json({"backends": backends()})
                elif url.path == "/api/eval/plan":
                    planner = E.plan_correct if q.get("backend") == CORRECT else E.plan
                    _, stats = planner(root, q.get("label", ""), force=q.get("force") == "1")
                    self._json(stats)
                elif url.path == "/api/eval/job":
                    self._json(jobs.snapshot())
                elif url.path == "/api/routes":
                    limit = int(q.get("limit", 300))
                    if limit < 1 or limit > 1000:
                        raise ValueError("limit must be 1..1000")
                    self._json(list_routes(root, limit=limit))
                elif url.path == "/api/loop":
                    limit = int(q.get("limit", 200))
                    if limit < 1 or limit > 1000:
                        raise ValueError("limit must be 1..1000")
                    payload = list_loop(root, limit=limit)
                    payload["confirm"] = bool(load_cfg().confirm.enabled)
                    self._json(payload)
                elif url.path.startswith("/audio/"):
                    self._audio(unquote(url.path[len("/audio/"):]))
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, C.ConfigError) as e:  # ValueError includes EvalError
                self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_HEAD = do_GET  # noqa: N815

        def do_POST(self):  # noqa: N802
            try:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self._json({"error": "cross-origin request refused"}, 403)
                # The media type itself, not a parameter that happens to contain the
                # letters "application/json". A form post cannot set this type.
                if _media_type(self.headers.get("Content-Type")) != "application/json":
                    return self._json({"error": "send JSON"}, 415)
                length = int(self.headers.get("Content-Length") or 0)
                if length > 1_000_000:
                    return self._json({"error": "too large"}, 413)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                path = urlparse(self.path).path
                if path == "/api/eval/verdict":
                    rec_id = str(body.get("id") or "")
                    if not re.fullmatch(r"\d{1,20}", rec_id):
                        raise ValueError("bad id")
                    texts, marked = body.get("texts") or [], body.get("labels") or []
                    if not isinstance(texts, list) or not isinstance(marked, list):
                        raise ValueError("texts and labels must be lists")
                    self._json(E.save_verdict(root, rec_id, texts, marked, bool(body.get("cleared"))))
                elif path == "/api/eval/run":
                    backend = str(body.get("backend") or "")
                    if backend not in {b["name"] for b in backends()}:
                        raise ValueError(f"unknown backend '{backend}'")
                    label = E._check_label(str(body.get("label") or ("corrected" if backend == CORRECT else backend)))
                    self._json(jobs.start(backend, label, bool(body.get("force"))))
                elif path == "/api/routes/verdict":
                    route_id = str(body.get("id") or "")
                    if body.get("cleared") is True:
                        self._json(save_route_verdict(root, route_id, cleared=True))
                    elif not isinstance(body.get("correct"), bool):
                        raise ValueError("correct must be true or false")
                    else:
                        self._json(save_route_verdict(root, route_id, correct=body["correct"]))
                elif path == "/api/confirm":
                    # The listening socket has to be loopback, not only the Host
                    # header: a client can claim Host 127.0.0.1 on a wildcard bind.
                    # A page on another name that rebinds to 127.0.0.1 sends that
                    # name as Host, and Origin would match it.
                    bound = self.server.server_address[0] if self.server is not None else ""
                    if not _loopback_host(self.headers.get("Host")) or not _loopback_host(str(bound)):
                        return self._json({"error": "confirm is only accepted from localhost"}, 403)
                    site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
                    if site and site not in ("same-origin", "none"):
                        return self._json({"error": "cross-origin request refused"}, 403)
                    if not load_cfg().confirm.enabled:
                        return self._json({"error": "confirm gate is off"}, 403)
                    task_id = body.get("task_id")
                    decision = body.get("decision")
                    if not isinstance(task_id, str) or not isinstance(decision, str):
                        raise ValueError("task_id and decision must be strings")
                    from .confirm import record_decision
                    self._json(record_decision(root, task_id, decision))
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, C.ConfigError) as e:
                self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _audio(self, rel: str) -> None:
            base = (root / "audio").resolve()
            path = (base / rel).resolve()
            if base not in path.parents or path.suffix != ".wav" or not path.is_file():
                return self._json({"error": "not found"}, 404)
            data = path.read_bytes()
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):  # Safari needs range support to play audio
                if m.group(1):
                    a, b = int(m.group(1)), int(m.group(2) or len(data) - 1)
                else:
                    a, b = max(0, len(data) - int(m.group(2))), len(data) - 1
                b = min(b, len(data) - 1)
                if a > b:
                    return self._send(416, b"", "audio/wav", {"Content-Range": f"bytes */{len(data)}"})
                return self._send(206, data[a:b + 1], "audio/wav",
                                  {"Content-Range": f"bytes {a}-{b}/{len(data)}", "Accept-Ranges": "bytes"})
            self._send(200, data, "audio/wav", {"Accept-Ranges": "bytes"})

    return ThreadingHTTPServer((host, port), Handler)


def list_loop(root, *, limit: int = 200) -> dict:
    """Recent task / reply / brief / spoken / confirm events, newest first. Read-only."""
    rows = []
    for ev in replay(root):
        if ev.get("type") not in ("task", "reply", "brief", "spoken", "confirm"):
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if ev["type"] == "task":
            summary = " ".join(str(x) for x in (data.get("status"), data.get("target"),
                                                data.get("error") or data.get("text") or "") if x)
        elif ev["type"] == "reply":
            summary = data.get("text") or ""
        elif ev["type"] == "brief":
            summary = f"[{data.get('priority') or ''}] {data.get('summary') or ''}"
        elif ev["type"] == "confirm":
            summary = " ".join(str(x) for x in (data.get("decision"), data.get("task_id")) if x)
        else:
            flag = "interrupted" if data.get("interrupted") else "completed" if data.get("completed") else ""
            summary = " ".join(x for x in (flag, data.get("text") or "") if x)
        text = " ".join(str(summary).split())
        if len(text) > 240:
            text = text[:239] + "…"
        task_id = ev["id"] if ev["type"] == "task" else data.get("task_id")
        rows.append({
            "id": ev["id"], "ts": ev.get("ts"), "type": ev["type"],
            "ref": ev.get("ref"), "summary": text,
            "status": data.get("status") if ev["type"] == "task" else None,
            "task_id": task_id if isinstance(task_id, str) else None,
        })
    if limit > 0:
        rows = rows[-limit:]
    rows.reverse()
    return {"events": rows}


def serve(data_dir: str | Path, host: str, port: int, config_path: str | None = None) -> None:
    server = make_server(data_dir, host, port, config_path)
    print(f"audi27 dashboard: http://{host}:{server.server_address[1]}  (data: {Path(data_dir).resolve()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
