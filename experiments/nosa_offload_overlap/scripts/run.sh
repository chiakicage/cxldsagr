#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/nosa_offload_overlap/scripts/run.sh [RUN_ID] [--profile] MEASURE_OPTIONS"
  echo "--profile captures CUDA/NVTX with nsys and computes actual GPU fetch/attention overlap."
  .venv/bin/python -m experiments.nosa_offload_overlap.src.measure --help
  exit 0
fi
run_id="nosa_overlap_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
profile=false
forward=()
for argument in "$@"; do
  if [[ "$argument" == --profile ]]; then profile=true; continue; fi
  option="${argument%%=*}"
  if [[ "$option" == --* && ( --output-dir == "$option"* || --run-id == "$option"* || --profiled == "$option"* ) ]]; then
    echo "The script owns --run-id, --output-dir and --profiled" >&2
    exit 2
  fi
  forward+=("$argument")
done
if [[ "$profile" == true ]]; then command -v nsys >/dev/null; fi
output="experiments/nosa_offload_overlap/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-offload-overlap-${run_id}.XXXXXX")"
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
command=(.venv/bin/python -m experiments.nosa_offload_overlap.src.measure --run-id "$run_id" --output-dir "$staging/data" "${forward[@]}")
if [[ "$profile" == true ]]; then
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite=false --output "$staging/profile/timeline" \
    "${command[@]}" --profiled \
    2> >(tee "$staging/log/measure.stderr.log" >&2) | tee "$staging/log/measure.stdout.log"
  nsys export --type=sqlite --output "$staging/data/timeline.sqlite" \
    "$staging/profile/timeline.nsys-rep" \
    2> >(tee "$staging/log/export.stderr.log" >&2) | tee "$staging/log/export.stdout.log"
  .venv/bin/python -m experiments.nosa_offload_overlap.src.analyze \
    --sqlite "$staging/data/timeline.sqlite" --work-intervals "$staging/data/work_intervals.json" \
    --output "$staging/data/overlap.json" --run-id "$run_id" \
    2> >(tee "$staging/log/analyze.stderr.log" >&2) | tee "$staging/log/analyze.stdout.log"
else
  "${command[@]}" \
    2> >(tee "$staging/log/measure.stderr.log" >&2) | tee "$staging/log/measure.stdout.log"
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
