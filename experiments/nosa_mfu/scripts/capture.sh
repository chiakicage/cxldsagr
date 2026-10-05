#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/nosa_mfu/scripts/capture.sh [RUN_ID] [CAPTURE_OPTIONS]"
  echo "Captures actual sparse NOSA inputs from an independent 64K prefix and 1K suffix."
  echo "Options include --device, --model-path, --request-file, --kernel-backend and --layers."
  .venv/bin/python -m experiments.nosa_mfu.src.capture_inputs --help
  exit 0
fi
run_id="kernel_inputs_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
for argument in "$@"; do
  option="${argument%%=*}"
  # argparse accepts unambiguous long-option prefixes, so reject those too.
  if [[ "$option" == --* && ( --output-dir == "$option"* || --run-id == "$option"* ) ]]; then
    echo "The script owns --run-id and --output-dir" >&2
    exit 2
  fi
done
output="experiments/nosa_mfu/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "/tmp/nosa-kernel-inputs-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do rm -rf -- "$output/$category/$run_id"; done
    echo "Failed (status $status); diagnostics outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
.venv/bin/python -m experiments.nosa_mfu.src.capture_inputs \
  --run-id "$run_id" --output-dir "$staging/data" "$@" \
  2> >(tee "$staging/log/capture.stderr.log" >&2) | tee "$staging/log/capture.stdout.log"
for category in data log profile; do
  mkdir -p "$output/$category"
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Captured model inputs: $output/data/$run_id"
