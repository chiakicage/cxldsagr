#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/nosa_indexer_pattern_65536_1024/scripts/run.sh [RUN_ID] [--block-budget 32|64] [--request-file PATH] [--model-path PATH] [--device DEVICE]"
  echo "Capture the exact 65536 + 1024 GR request on dense NOSA activations, then analyze unique selected KV blocks."
  echo "Requires the source request, CUDA, FlashInfer and the existing analysis dependencies."
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then
  run_id="$1"
  shift
fi
arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --request-file|--model-path|--device|--block-budget)
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
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-pattern-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
claimed_categories=()
published=false
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    # Remove only destinations claimed by this process; all source data/logs
    # remain in staging even if publication was interrupted midway.
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
run_logged capture .venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.capture \
  --run-id "$run_id" --output-dir "$staging/data" "${arguments[@]}"
run_logged analyze .venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.analyze "$staging/data"
# Publish only a successful capture and analysis. mkdir refuses concurrent reuse;
# retain staging until every copy succeeds so the EXIT trap can undo publication.
mkdir -p "$output/data" "$output/log" "$output/profile"
for category in data log profile; do
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
