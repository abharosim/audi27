#!/bin/bash
# Start the recorder and the dashboard under a supervisor. Ctrl-C stops everything.
# With `[turns] enabled = true` in config.toml, the live turn assembler is started too.
# With `[route] enabled = true`, the router is started too (it only displays; it sends nothing).
# `[dispatch] enabled`, `[queue] enabled`, and `[speak] enabled` start those processes.
# They default off: routes stay on screen until execution is turned on.
# `[presence] enabled` (default off) does not start a process. The recorder, which
# already has the frames, publishes data/presence.json only when that flag is on.
# `[confirm] enabled` (default off) starts `audi27 confirm run`.
# `[screen] enabled` (default off) starts `audi27 screen run`. It records only the
# frontmost app and the front window title, and only on macOS.
# To restart after a code change without touching this terminal:  touch data/restart.request
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv sync --extra dev
# Stop leftovers from an earlier start so the port and the microphone are free.
pkill -f "audi27(\.cli)? (web|run|supervise|turns|route|dispatch|queue|speak|confirm|screen)" 2>/dev/null && sleep 2 || true
(sleep 3; open http://127.0.0.1:8727 2>/dev/null) &
exec .venv/bin/audi27 supervise
