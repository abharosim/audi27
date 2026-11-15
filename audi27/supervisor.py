"""`audi27 supervise`: keep the recorder and the dashboard running.

`[turns] enabled`, `[route] enabled`, `[dispatch] enabled`, `[queue] enabled`,
`[speak] enabled`, `[confirm] enabled`, and `[screen] enabled` add those processes.
All default off. `[presence] enabled` does not add one: the recorder publishes
`data/presence.json` itself, and only when that flag is on. A child that exits is
started again on its own (1 s -> 30 s backoff). When data/restart.request appears,
every child is stopped cleanly and the supervisor re-executes itself, so a code
change (including to this file) is picked up. data/supervisor.json records pid,
start time, and restart count.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("audi27.supervisor")
PROJECT = Path(__file__).resolve().parent.parent


def _stop(procs: dict, grace: float) -> None:
    for p in procs.values():
        if p and p.poll() is None:
            p.send_signal(signal.SIGINT)  # the recorder flushes its last segment on SIGINT
    deadline = time.time() + grace
    for name, p in procs.items():
        if not p:
            continue
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            log.warning("%s did not stop in %.0f s; killing it", name, grace)
            p.kill()
            p.wait()


def supervise(data_dir: str | Path, specs: dict[str, list[str]], reexec, poll_s: float = 1.0,
              grace_s: float = 90.0) -> None:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    trigger = root / "restart.request"
    trigger.unlink(missing_ok=True)  # a request left over from before this start is already satisfied
    restarts = int(os.environ.get("AUDI27_RESTARTS", "0"))
    pyproject = PROJECT / "pyproject.toml"
    deps_mtime = pyproject.stat().st_mtime if pyproject.exists() else 0.0
    state = {"pid": os.getpid(), "started": time.time(), "restarts": restarts,
             "last_reason": os.environ.get("AUDI27_RESTART_REASON", ""), "crashes": {}}

    def save() -> None:
        tmp = root / "supervisor.json.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, root / "supervisor.json")

    save()
    procs: dict = {name: None for name in specs}
    backoff = {name: 1.0 for name in specs}
    next_start = {name: 0.0 for name in specs}
    started_at = {name: 0.0 for name in specs}

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    log.info("supervisor started (pid %d, restart #%d)", os.getpid(), restarts)
    try:
        while True:
            now = time.time()
            for name, cmd in specs.items():
                p = procs[name]
                if p is not None and p.poll() is not None:
                    if now - started_at[name] > 60:
                        backoff[name] = 1.0  # it ran fine for a while: not a crash loop
                    log.error("%s exited with code %s; restarting in %.0f s", name, p.returncode, backoff[name])
                    state["crashes"][name] = state["crashes"].get(name, 0) + 1
                    save()
                    procs[name] = None
                    next_start[name] = now + backoff[name]
                    backoff[name] = min(backoff[name] * 2, 30.0)
                if procs[name] is None and now >= next_start[name]:
                    procs[name] = subprocess.Popen(cmd)
                    started_at[name] = now
                    log.info("started %s (pid %d)", name, procs[name].pid)
            if trigger.exists():
                try:
                    reason = trigger.read_text(encoding="utf-8").strip()[:200]
                except OSError:
                    reason = ""
                trigger.unlink(missing_ok=True)
                log.info("restart requested: %s", reason or "(no reason given)")
                _stop(procs, grace_s)
                if pyproject.exists() and pyproject.stat().st_mtime != deps_mtime and shutil.which("uv"):
                    log.info("pyproject.toml changed; running uv sync")
                    subprocess.run(["uv", "sync", "--extra", "dev"], cwd=PROJECT)
                os.environ["AUDI27_RESTARTS"] = str(restarts + 1)
                os.environ["AUDI27_RESTART_REASON"] = reason
                reexec()
                return
            time.sleep(poll_s)
    except KeyboardInterrupt:
        log.info("supervisor stopping")
        _stop(procs, grace_s)


def child_specs(config_path: str | None, host: str, port: int, *, turns: bool = False,
                 route: bool = False, dispatch: bool = False, queue: bool = False,
                 speak: bool = False, confirm: bool = False,
                 screen: bool = False) -> dict[str, list[str]]:
    """Recorder and dashboard, plus whichever live loops have `[enabled] = true`."""
    base = [sys.executable, "-m", "audi27.cli"] + (["-c", config_path] if config_path else [])
    specs = {"run": base + ["run"], "web": base + ["web", "--host", host, "--port", str(port)]}
    if turns:
        specs["turns"] = base + ["turns", "run"]
    if route:
        specs["route"] = base + ["route", "run"]
    if dispatch:
        specs["dispatch"] = base + ["dispatch", "run"]
    if queue:
        specs["queue"] = base + ["queue", "run"]
    if speak:
        specs["speak"] = base + ["speak", "run"]
    if confirm:
        specs["confirm"] = base + ["confirm", "run"]
    if screen:
        specs["screen"] = base + ["screen", "run"]
    return specs


def main(config_path: str | None, data_dir: str, host: str, port: int, *, turns: bool = False,
         route: bool = False, dispatch: bool = False, queue: bool = False,
         speak: bool = False, confirm: bool = False, screen: bool = False) -> None:
    specs = child_specs(config_path, host, port, turns=turns, route=route,
                        dispatch=dispatch, queue=queue, speak=speak, confirm=confirm,
                        screen=screen)
    argv = [sys.executable, "-m", "audi27.cli"] + (["-c", config_path] if config_path else [])
    argv += ["supervise", "--host", host, "--port", str(port)]
    supervise(data_dir, specs, lambda: os.execv(sys.executable, argv))
