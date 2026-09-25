#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash experiments/nosa_gr_65536_1024/scripts/run.sh [RUN_ID]"
  echo "Capture actual KV prefix=65536 and extend Q=1024, then analyze nsys."
  exit 0
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
run_id="${1:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
output="experiments/nosa_gr_65536_1024/output"
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
mkdir -p "$output/profile/$run_id"
report="$output/profile/$run_id/nosa_gr_65536_1024"
run_logged capture nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output="$report" \
  .venv/bin/python -m experiments.nosa_gr_65536_1024.src.capture \
  --prefix-tokens 65536 --new-tokens 1024 --output-dir "$output/data/$run_id"
run_logged export nsys export --type=sqlite --output="$output/data/$run_id/nsys.sqlite" "$report.nsys-rep"
run_logged analyze .venv/bin/python -m experiments.nosa_gr_65536_1024.src.analyze \
  "$output/data/$run_id/nsys.sqlite" --output "$output/data/$run_id/analysis.json"

run_logged mfu .venv/bin/python -m experiments.nosa_gr_65536_1024.src.mfu "$output/data/$run_id"
