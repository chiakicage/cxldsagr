#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_serving/scripts/cache_memory_audit.sh [RUN_ID] OPTIONS"
  echo "Separate cache-memory validation; no latency samples are published."
  .venv/bin/python -m experiments.gr_serving.src.cache_memory_capture --help
  exit 0
fi
run_id="echo_memory_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
for argument in "$@"; do
  option="${argument%%=*}"
  if [[ "$option" == --* && ( --output-dir == "$option"* || --profile-dir == "$option"* || --run-id == "$option"* ) ]]; then
    echo "The script owns --run-id, --output-dir and --profile-dir" >&2
    exit 2
  fi
done
output="experiments/gr_serving/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/echo-memory-${run_id}.XXXXXX")"
mkdir -p "$staging/log"
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do rm -rf -- "$output/$category/$run_id"; done
    echo "Failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
.venv/bin/python -m experiments.gr_serving.src.cache_memory_capture \
  --run-id "$run_id" --output-dir "$staging/data" --profile-dir "$staging/profile" "$@" \
  2> >(tee "$staging/log/capture.stderr.log" >&2) | tee "$staging/log/capture.stdout.log"
mkdir -p "$staging/profile"
for category in data log profile; do
  mkdir -p "$output/$category"
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed memory audit: $output/data/$run_id"
