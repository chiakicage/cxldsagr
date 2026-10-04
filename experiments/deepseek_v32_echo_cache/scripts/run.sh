#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CXLDSAGR_SM90_BACKEND="${CXLDSAGR_SM90_BACKEND:-native}"
python_bin="${ECHO_PYTHON:-$PWD/.venv/bin/python}"

run_id=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --help|-h)
      exec "$python_bin" -m experiments.deepseek_v32_echo_cache.src.capacity_probe --help
      ;;
    --run-id)
      run_id="${args[i+1]:-}"
      ;;
    --run-id=*)
      run_id="${args[i]#*=}"
      ;;
    --output-dir|--output-dir=*)
      echo "run.sh manages output/data and output/log; use the Python entry for custom output." >&2
      exit 2
      ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Pass --run-id with letters, digits, underscores or hyphens." >&2
  exit 2
fi
output_root="$PWD/experiments/deepseek_v32_echo_cache/output"
data_dir="$output_root/data/$run_id"
log_dir="$output_root/log/$run_id"
if [[ -e "$data_dir" || -e "$log_dir" ]]; then
  echo "Run ID already exists: $run_id" >&2
  exit 2
fi
mkdir -p -- "$log_dir"
"$python_bin" -m experiments.deepseek_v32_echo_cache.src.capacity_probe \
  --output-dir "$data_dir" "${args[@]}" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
