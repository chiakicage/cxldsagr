#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/nosa_mfu/scripts/dense.sh [RUN_ID] [OPTIONS]"
  echo "--mode check|bench|profile (default: bench) --validation-receipt PATH --check-dir PATH"
  echo "--model-path PATH --request-file PATH --device DEVICE --warmup N --repeats N --peak-tflops N"
  echo "Resident dense H65536+A1024, chunk=1024, BF16, all 32 layers."
  echo "Check is independent and stays outside experiments. Bench does not invoke nsys."
  echo "Profile measures an independent benchmark first, then captures the same request with nsys."
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
mode=bench
check_dir=
receipt=
arguments=()
mfu_arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --mode|--check-dir|--validation-receipt|--model-path|--request-file|--device|--warmup|--repeats|--peak-tflops)
      if [[ $# -lt 2 || "$2" == --* ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --mode) mode="$2" ;;
        --check-dir) check_dir="$2" ;;
        --validation-receipt) receipt="$2"; arguments+=("$1" "$2") ;;
        --peak-tflops) mfu_arguments+=("$1" "$2") ;;
        *) arguments+=("$1" "$2") ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then echo "Invalid RUN_ID" >&2; exit 2; fi
if [[ "$mode" != check && "$mode" != bench && "$mode" != profile ]]; then echo "Invalid mode" >&2; exit 2; fi
if [[ "$mode" != check && -z "$receipt" ]]; then echo "Run check separately; --validation-receipt is required" >&2; exit 2; fi
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-dense-${mode}-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
run_logged() {
  local step="$1"
  shift
  "$@" 2> >(tee "$staging/log/$step.stderr.log" >&2) | tee "$staging/log/$step.stdout.log"
}
if [[ "$mode" == check ]]; then
  check_dir="${check_dir:-$staging/data}"
  run_logged check .venv/bin/python -m experiments.nosa_mfu.src.dense.capture \
    --mode check --output-dir "$check_dir" "${arguments[@]}"
  echo "Independent numerical acceptance: $check_dir/receipt.json"
  exit 0
fi
output="experiments/nosa_mfu/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then echo "Run already exists: $run_id" >&2; exit 2; fi
done
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do rm -rf -- "$output/$category/$run_id"; done
    echo "Run failed (status $status); diagnostics outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
benchmark_dir="$staging/data"
if [[ "$mode" == profile ]]; then benchmark_dir="$staging/benchmark"; fi
run_logged benchmark .venv/bin/python -m experiments.nosa_mfu.src.dense.capture \
  --mode bench --output-dir "$benchmark_dir" "${arguments[@]}"
if [[ "$mode" == profile ]]; then
  run_logged nsys_version nsys --version
  report="$staging/profile/nosa_dense"
  run_logged profile nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$report" \
    .venv/bin/python -m experiments.nosa_mfu.src.dense.capture \
    --mode profile --benchmark-data-dir "$benchmark_dir" --output-dir "$staging/data" "${arguments[@]}"
  cp -a -- "$benchmark_dir" "$staging/data/benchmark"
  run_logged export nsys export --type=sqlite --output="$staging/data/nsys.sqlite" "$report.nsys-rep"
  run_logged analyze .venv/bin/python -m experiments.nosa_mfu.src.dense.analyze \
    "$staging/data/nsys.sqlite" --output "$staging/data/analysis.json"
fi
run_logged mfu .venv/bin/python -m experiments.nosa_mfu.src.dense.mfu "$staging/data" "${mfu_arguments[@]}"
for category in data log profile; do
  mkdir -p "$output/$category"
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
