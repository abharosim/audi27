#!/bin/bash
# Start the recorder and the dashboard under a supervisor. Ctrl-C stops everything.
# With `[turns] enabled = true` in config.toml, the live turn assembler is started too.
# To restart after a code change without touching this terminal:  touch data/restart.request
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv sync --extra dev
# Stop leftovers from an earlier start so the port and the microphone are free.
pkill -f "audi27(\.cli)? (web|run|supervise|turns)" 2>/dev/null && sleep 2 || true
(sleep 3; open http://127.0.0.1:8727 2>/dev/null) &
exec .venv/bin/audi27 supervise
