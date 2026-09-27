#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh [RUN_ID] [OPTIONS]"
  echo "Capture 65536+1024: dense QA-only64, dense full NOSA64, and actual sparse full NOSA64."
  echo "Options: --request-file PATH --model-path PATH --device DEVICE"
  echo "         --baseline-data-dir PATH (optional independent QA-only capture) --no-baseline"
  echo "Uses separate empty caches for the dense and sparse prefixes; no performance timing."
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="sparse_pattern_compare_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then
  run_id="$1"
  shift
fi
capture_arguments=()
analysis_arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --no-baseline) analysis_arguments+=("$1"); shift ;;
    --request-file|--model-path|--device|--baseline-data-dir)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      if [[ "$1" == --baseline-data-dir ]]; then
        analysis_arguments+=("$1" "$2")
      else
        capture_arguments+=("$1" "$2")
      fi
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
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-sparse-pattern-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
claimed_categories=()
published=false
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do
      rm -rf -- "$output/$category/$run_id"
    done
    echo "Run failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_logged() {
  local step="$1"
  shift
  "$@" 2> >(tee "$staging/log/$step.stderr.log" >&2) | tee "$staging/log/$step.stdout.log"
}
run_logged capture .venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.sparse_capture \
  --run-id "$run_id" --output-dir "$staging/data" "${capture_arguments[@]}"
run_logged analyze .venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.sparse_compare \
  "$staging/data" "${analysis_arguments[@]}"
mkdir -p "$output/data" "$output/log" "$output/profile"
for category in data log profile; do
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
