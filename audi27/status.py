"""Heartbeat file written by the recorder (data/status.json) and read by the dashboard."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

STALE_AFTER_S = 3.0


def write_status(root: Path, status: dict) -> None:
    tmp = root / "status.json.tmp"
    tmp.write_text(json.dumps(status), encoding="utf-8")
    os.replace(tmp, root / "status.json")


def read_status(root: Path) -> dict:
    """Return {"running", "age", "status"}; running means a fresh heartbeat from a live recorder."""
    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        age = time.time() - float(status["updated"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"running": False, "age": None, "status": None}
    running = age < STALE_AFTER_S and not status.get("stopped")
    return {"running": running, "age": round(age, 1), "status": status}
