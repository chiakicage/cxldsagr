#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/../.." && pwd)"
REPRO_ROOT="$REPO_ROOT/3rdparty/ECHO/reproduction/cxldsagr"
export PYTHONDONTWRITEBYTECODE=1
for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    exec python3 -B "$TASK_ROOT/scripts/run_decode.py" "$@"
  fi
done
source "$REPRO_ROOT/env/activate.sh"
cd "$REPO_ROOT"
exec "$REPRO_ROOT/env/.venv/bin/python" -B "$TASK_ROOT/scripts/run_decode.py" "$@"
