#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CXLDSAGR_SM90_BACKEND="${CXLDSAGR_SM90_BACKEND:-native}"
python_bin="${NOSA_MOTIVATION_PYTHON:-$PWD/.venv/bin/python}"
args=("$@")
run_id=""
mode="bench"
for ((index=0; index<${#args[@]}; index++)); do
  case "${args[index]}" in
    --help|-h)
      exec "$python_bin" -m experiments.nosa_motivation.src.measure --help
      ;;
    --run-id) run_id="${args[index+1]:-}" ;;
    --run-id=*) run_id="${args[index]#*=}" ;;
    --mode) mode="${args[index+1]:-}" ;;
    --mode=*) mode="${args[index]#*=}" ;;
    --output-dir|--output-dir=*)
      echo "run.sh manages output directories; use the Python module for custom output." >&2
      exit 2
      ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Pass --run-id with letters, digits, underscores or hyphens." >&2
  exit 2
fi
base="$PWD/experiments/nosa_motivation/output"
if [[ "$mode" == "check" ]]; then
  base="${TMPDIR:-/tmp}/cxldsagr-checks/nosa_motivation"
fi
data_dir="$base/data/$run_id"
log_dir="$base/log/$run_id"
if [[ -e "$data_dir" || -e "$log_dir" ]]; then
  echo "Run ID already exists: $run_id" >&2
  exit 2
fi
mkdir -p -- "$log_dir"
if "$python_bin" -m experiments.nosa_motivation.src.measure \
  --output-dir "$data_dir" "${args[@]}" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"; then
  echo "Completed $run_id; data: $data_dir; logs: $log_dir"
else
  status=$?
  failed_logs="$(mktemp -d "${TMPDIR:-/tmp}/nosa-motivation-${run_id}-logs.XXXXXX")"
  mv -- "$log_dir/stdout.log" "$log_dir/stderr.log" "$failed_logs/"
  rmdir -- "$log_dir"
  echo "Run failed ($status); logs retained outside experiments: $failed_logs" >&2
  exit "$status"
fi
