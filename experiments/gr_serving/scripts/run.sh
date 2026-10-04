#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_serving/scripts/run.sh [RUN_ID] MEASURE_OPTIONS"
  echo "Measures DeepSeek V3.2 before NOSA by default; forwards options to the measurement CLI."
  echo "GR_OUTPUT_ROOT overrides the output root (default: experiments/gr_serving/output)."
  .venv/bin/python -m experiments.gr_serving.src.measure --help
  exit 0
fi
run_id="gr_serving_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
for argument in "$@"; do
  option="${argument%%=*}"
  if [[ "$option" == --* && ( --output-dir == "$option"* || --run-id == "$option"* ) ]]; then
    echo "The script owns --run-id and --output-dir" >&2
    exit 2
  fi
done
output="${GR_OUTPUT_ROOT:-experiments/gr_serving/output}"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/gr-serving-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
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
.venv/bin/python -m experiments.gr_serving.src.measure \
  --run-id "$run_id" --output-dir "$staging/data" "$@" \
  2> >(tee "$staging/log/measure.stderr.log" >&2) | tee "$staging/log/measure.stdout.log"
if [[ "${GR_AUDIT_BEFORE_PUBLISH:-0}" == 1 ]]; then
  .venv/bin/python -m experiments.gr_serving.src.audit "$staging/data" \
    --expected-run-id "$run_id" --json "$staging/data/independent_audit.json" \
    >"$staging/log/audit.stdout.log" 2>"$staging/log/audit.stderr.log"
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
