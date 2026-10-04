#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/nosa_kernel_mfu/scripts/run.sh [RUN_ID] [MEASURE_OPTIONS]"
  echo "Default bench requires --validation-receipt; --mode check and --mode profile are independent."
  echo "Measures native/Triton resident attention and pooled-score useful MFU on identical synthetic inputs."
  .venv/bin/python -m experiments.nosa_kernel_mfu.src.measure --help
  exit 0
fi
run_id="kernel_mfu_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
phase=bench
previous_option=
for argument in "$@"; do
  if [[ "$previous_option" == --mode ]]; then phase="$argument"; fi
  if [[ "$argument" == --mode=* ]]; then phase="${argument#*=}"; fi
  previous_option="$argument"
  option="${argument%%=*}"
  if [[ "$option" == --* && ( --output-dir == "$option"* || --profile-dir == "$option"* || --run-id == "$option"* ) ]]; then
    echo "The script owns --run-id, --output-dir and --profile-dir" >&2
    exit 2
  fi
done
output="experiments/nosa_kernel_mfu/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-kernel-mfu-${run_id}.XXXXXX")"
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
.venv/bin/python -m experiments.nosa_kernel_mfu.src.measure \
  --run-id "$run_id" --output-dir "$staging/data" --profile-dir "$staging/profile" "$@" \
  2> >(tee "$staging/log/measure.stderr.log" >&2) | tee "$staging/log/measure.stdout.log"
if [[ "$phase" == check ]]; then
  published=true
  echo "Independent check completed outside experiments: $staging/data"
  exit 0
fi
for category in data log profile; do
  mkdir -p "$output/$category"
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
