#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/nosa_indexer_pattern_65536_1024/scripts/threshold_sweep.sh [RUN_ID] --estimate-data-dir PATH [--baseline-threshold-pct NUMBER] [--bandwidth-gbps NUMBER] [--attention-mfu-scale NUMBER]"
  echo "Scan every sparse/dense classification threshold; minimize unhidden fetch and maximize hidden/fetch."
  echo "Requires an explicit valid estimate run; defaults to baseline 30%, bandwidth 50 GB/s and attention MFU scale 0.5. No GPU execution."
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="threshold_sweep_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then
  run_id="$1"
  shift
fi
arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --estimate-data-dir|--baseline-threshold-pct|--bandwidth-gbps|--attention-mfu-scale)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      arguments+=("$1" "$2")
      shift 2 ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
output="experiments/nosa_indexer_pattern_65536_1024/output"
for category in log data profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id; choose a new RUN_ID" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-threshold-sweep-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
claimed_categories=()
published=false
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do
      rm -rf -- "$output/$category/$run_id"
    done
    echo "Calculation failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
.venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.threshold_sweep \
  --run-id "$run_id" --output-dir "$staging/data" "${arguments[@]}" \
  2> >(tee "$staging/log/threshold_sweep.stderr.log" >&2) | tee "$staging/log/threshold_sweep.stdout.log"
mkdir -p "$output/data" "$output/log" "$output/profile"
for category in data log profile; do
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed threshold sweep: $output/data/$run_id"
