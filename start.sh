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
# `[cue] enabled` (default off) starts `audi27 cue run`: a short sound as soon as
# a wake word is heard, before the turn closes and before route or dispatch.
# `[kws] enabled` (default off) does not start a process. The recorder runs a
# streaming keyword spotter on the frames it already captured and writes
# data/kws.live while that thread is up. `cue run` skips a trigger only when
# that file is live and a wake event already covers the utterance.
# To restart after a code change without touching this terminal:  touch data/restart.request
#
# `uv sync --extra dev` installs that extra and uninstalls every other one.
# sherpa-onnx is the `kws` extra. Always passing `--extra kws` would be safe on
# macOS (a wheel is published), but it downloads a native runtime on machines
# that leave the spotter off. Read `[kws] enabled` from config.toml instead, and
# pass `--extra kws` only then, so a start does not remove it.

kws_enabled() {
  # True when the last `enabled` in the `[kws]` table is true. Missing file, a
  # comment, or another section's `enabled = true` do not count.
  local file="$1" section="" line stripped flag=0
  [[ -f "$file" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    stripped="${line%%#*}"
    stripped="${stripped#"${stripped%%[![:space:]]*}"}"
    stripped="${stripped%"${stripped##*[![:space:]]}"}"
    [[ -z "$stripped" ]] && continue
    if [[ "$stripped" == \[*\] ]]; then
      section="${stripped:1:${#stripped}-2}"
      section="${section#"${section%%[![:space:]]*}"}"
      section="${section%"${section##*[![:space:]]}"}"
      continue
    fi
    [[ "$section" == "kws" ]] || continue
    if [[ "$stripped" =~ ^enabled[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*$ ]]; then
      if [[ "${BASH_REMATCH[1]}" == "true" ]]; then
        flag=1
      else
        flag=0
      fi
    fi
  done < "$file"
  [[ "$flag" == 1 ]]
}

kws_sync_args() {
  if kws_enabled "$1"; then
    echo "--extra dev --extra kws"
  else
    echo "--extra dev"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -e
  cd "$(dirname "$0")"
  command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  sync_args=(--extra dev)
  kws_enabled config.toml && sync_args+=(--extra kws)
  uv sync "${sync_args[@]}"
  # Stop leftovers from an earlier start so the port and the microphone are free.
  pkill -f "audi27(\.cli)? (web|run|supervise|turns|route|dispatch|queue|speak|confirm|screen|cue)" 2>/dev/null && sleep 2 || true
  (sleep 3; open http://127.0.0.1:8727 2>/dev/null) &
  exec .venv/bin/audi27 supervise
fi
