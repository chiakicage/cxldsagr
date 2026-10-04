#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CXLDSAGR_SM90_BACKEND="${CXLDSAGR_SM90_BACKEND:-native}"
python_bin="${MOTIVATION_PYTHON:-$PWD/.venv/bin/python}"
run_id=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --help|-h)
      exec "$python_bin" -m experiments.deepseek_v32_motivation.src.measure --help
      ;;
    --run-id)
      run_id="${args[i+1]:-}"
      ;;
    --run-id=*)
      run_id="${args[i]#*=}"
      ;;
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
output_root="$PWD/experiments/deepseek_v32_motivation/output"
data_dir="$output_root/data/$run_id"
log_dir="$output_root/log/$run_id"
if [[ -e "$data_dir" || -e "$log_dir" ]]; then
  echo "Run ID already exists: $run_id" >&2
  exit 2
fi
mkdir -p -- "$log_dir"
if "$python_bin" -m experiments.deepseek_v32_motivation.src.measure \
  --output-dir "$data_dir" "${args[@]}" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"; then
  echo "Completed $run_id; data: $data_dir; logs: $log_dir"
else
  status=$?
  failed_logs="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-motivation-${run_id}-logs.XXXXXX")"
  mv -- "$log_dir/stdout.log" "$log_dir/stderr.log" "$failed_logs/"
  rmdir -- "$log_dir"
  echo "Run failed ($status); logs retained outside experiments: $failed_logs" >&2
  exit "$status"
fi
