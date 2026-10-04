#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/nosa_baseline_performance/scripts/sparse.sh [RUN_ID] [OPTIONS]"
  echo "End-to-end NOSA sparse: prefix=65536, candidate=1024, chunk=1024, BF16, all 32 layers."
  echo "Options: --model-path PATH --request-file PATH --device DEVICE --kernel-backend native|triton"
  echo "         --mode check|bench|profile (default: bench) --validation-receipt PATH --check-dir PATH"
  echo "         --warmup N --repeats N --profile-repeats N --without-nsys --peak-tflops N --timeline-only"
  echo "Check runs separately outside experiments; bench reuses its receipt and does not profile."
  echo "Profile runs an independent benchmark and then the requested diagnostic process."
  echo "Offline MFU uses H200 dense BF16 peak 989 TFLOPS; other hardware requires --peak-tflops."
  echo "--timeline-only: independent benchmark + nsys root ranges only; no module wrappers/events or module MFU."
  echo "Successful results publish to output/{data,log,profile}/RUN_ID; failed runs stay in /tmp."
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
mfu_arguments=()
use_nsys=true
profile_mode=profile
mode=bench
check_dir=
receipt=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --mode|--check-dir|--validation-receipt)
      if [[ $# -lt 2 || "$2" == --* ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --mode) mode="$2" ;;
        --check-dir) check_dir="$2" ;;
        --validation-receipt) receipt="$2"; arguments+=("$1" "$2") ;;
      esac
      shift 2 ;;
    --without-nsys) use_nsys=false; shift ;;
    --timeline-only) profile_mode=timeline; shift ;;
    --peak-tflops)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      mfu_arguments+=("$1" "$2")
      shift 2 ;;
    --model-path|--request-file|--device|--kernel-backend|--warmup|--repeats|--profile-repeats)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      arguments+=("$1" "$2")
      shift 2 ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
  esac
done
if [[ "$mode" != check && "$mode" != bench && "$mode" != profile ]]; then echo "Unknown mode: $mode" >&2; exit 2; fi
if [[ "$mode" != check && -z "$receipt" ]]; then echo "Run check separately; --validation-receipt is required" >&2; exit 2; fi
if [[ "$mode" != profile && ( "$profile_mode" == timeline || "$use_nsys" != true ) ]]; then echo "Profiler options require --mode profile" >&2; exit 2; fi
if [[ "$profile_mode" == timeline && "$use_nsys" != true ]]; then
  echo "--timeline-only requires nsys; it cannot be combined with --without-nsys" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
if [[ "$mode" == check ]]; then
  check_staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-sparse-check-${run_id}.XXXXXX")"
  check_dir="${check_dir:-$check_staging/data}"
  .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.capture \
    --mode check --run-id "$run_id" --output-dir "$check_dir" "${arguments[@]}" \
    > >(tee "$check_staging/check.stdout.log") 2> >(tee "$check_staging/check.stderr.log" >&2)
  echo "Independent numerical acceptance: $check_dir/receipt.json"
  exit 0
fi
output="experiments/nosa_baseline_performance/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id; choose a new RUN_ID" >&2
    exit 2
  fi
done
if [[ "$mode" == profile && "$use_nsys" == true ]] && ! command -v nsys >/dev/null; then
  echo "nsys is required; --without-nsys retains CUDA-event module profiling only" >&2
  exit 1
fi
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-sparse-profile-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do
      rm -rf -- "$output/$category/$run_id"
    done
    echo "Run failed (status $status); diagnostics outside experiments: $staging" >&2
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
common=(--run-id "$run_id" --output-dir "$staging/data" --prefix-tokens 65536 --new-tokens 1024 --chunk-size 1024)
run_logged benchmark .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.capture \
  --mode benchmark "${common[@]}" "${arguments[@]}"
if [[ "$mode" == profile && "$use_nsys" == true ]]; then
  run_logged nsys_version nsys --version
  report="$staging/profile/indexer_block_sparse_profile"
  trace=cuda,nvtx,osrt
  if [[ "$profile_mode" == timeline ]]; then trace=cuda,nvtx; fi
  run_logged profile nsys profile --trace="$trace" --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$report" \
    .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.capture \
    --mode "$profile_mode" "${common[@]}" "${arguments[@]}"
  run_logged export nsys export --type=sqlite --output="$staging/data/nsys.sqlite" "$report.nsys-rep"
elif [[ "$mode" == profile ]]; then
  run_logged profile .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.capture \
    --mode profile "${common[@]}" "${arguments[@]}"
fi
if [[ "$profile_mode" == timeline ]]; then
  run_logged launch_report .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.launch_report "$staging/data"
else
  run_logged analyze .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.analyze "$staging/data"
  run_logged mfu .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.mfu "$staging/data" "${mfu_arguments[@]}"
  if [[ "$mode" == profile && "$use_nsys" == true ]]; then
    run_logged module_mfu .venv/bin/python -m experiments.nosa_baseline_performance.src.sparse.module_mfu "$staging/data" "${mfu_arguments[@]}"
  fi
fi
mkdir -p "$output/data" "$output/log" "$output/profile"
for category in data log profile; do
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
