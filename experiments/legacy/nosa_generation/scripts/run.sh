#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash experiments/legacy/nosa_generation/scripts/run.sh [RUN_ID]"
  echo "Replay the WITHDRAWN generation benchmark; this is NOT a GR benchmark."
  exit 0
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../../.."
export PATH="$PWD/.venv/bin:$PATH"
run_id="${1:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
output="experiments/legacy/nosa_generation/output"
if [[ -e "$output/log/$run_id" || -e "$output/data/$run_id" || -e "$output/profile/$run_id" ]]; then
  echo "Run already exists: $run_id; choose a new RUN_ID" >&2
  exit 2
fi
mkdir -p "$output/log/$run_id" "$output/data" "$output/profile"
run_logged() {
  local step="$1"
  shift
  "$@" 2> >(tee "$output/log/$run_id/$step.stderr.log" >&2) | tee "$output/log/$run_id/$step.stdout.log"
}
run_logged measure .venv/bin/python -m experiments.legacy.nosa_generation.src.measure \
  --output-dir "$output/data/$run_id"
