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
for ((index=0; index<${#args[@]}; index++)); do
  case "${args[index]}" in
    --help|-h) exec "$python_bin" -m experiments.nosa_motivation.src.profile --help ;;
    --run-id) run_id="${args[index+1]:-}" ;;
    --run-id=*) run_id="${args[index]#*=}" ;;
    --output-root|--output-root=*)
      echo "profile.sh manages output directories; use the Python module for custom output." >&2
      exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Pass --run-id with letters, digits, underscores or hyphens." >&2
  exit 2
fi
base="$PWD/experiments/nosa_motivation/output"
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" ]]; then
    echo "Run ID already exists: $run_id" >&2
    exit 2
  fi
done
log_dir="$base/log/$run_id"
mkdir -p -- "$log_dir"
if "$python_bin" -m experiments.nosa_motivation.src.profile "${args[@]}" \
  > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"; then
  echo "Completed profile $run_id; data: $base/data/$run_id; traces: $base/profile/$run_id"
else
  status=$?
  failed_logs="$(mktemp -d "${TMPDIR:-/tmp}/nosa-motivation-${run_id}-logs.XXXXXX")"
  mv -- "$log_dir/stdout.log" "$log_dir/stderr.log" "$failed_logs/"
  rmdir -- "$log_dir"
  echo "Profile failed ($status); logs retained outside experiments: $failed_logs" >&2
  exit "$status"
fi
