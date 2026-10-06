#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PATH="$PWD/.venv/bin:$PATH"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  exec python -m experiments.cache_manager_performance.src.recall --help
fi
run_id=""
mode=""
output_root=""
previous=""
for argument in "$@"; do
  case "$previous" in
    --run-id) run_id="$argument" ;;
    --mode) mode="$argument" ;;
    --output-root) output_root="$argument" ;;
  esac
  previous="$argument"
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || ! "$mode" =~ ^(check|bench)$ ]]; then
  echo "recall.sh requires --mode check|bench --run-id NAME" >&2
  exit 2
fi
if [[ -z "$output_root" ]]; then
  if [[ "$mode" == check ]]; then output_root="/tmp/cxldsagr-checks/cache_manager_recall";
  else output_root="experiments/cache_manager_performance/output"; fi
fi
if [[ -e "$output_root/data/$run_id" || -e "$output_root/log/$run_id" ]]; then
  echo "run ID already exists: $run_id" >&2
  exit 2
fi
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/cache-recall-run.XXXXXX")"
mkdir -p "$stage_dir/log/$run_id"
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "failed run retained outside experiments: $stage_dir" >&2; fi' EXIT
python -m experiments.cache_manager_performance.src.recall "$@" --output-root "$stage_dir" \
  >"$stage_dir/log/$run_id/stdout.log" 2>"$stage_dir/log/$run_id/stderr.log"
mkdir -p "$output_root/data" "$output_root/log"
mv "$stage_dir/data/$run_id" "$output_root/data/$run_id"
mv "$stage_dir/log/$run_id" "$output_root/log/$run_id"
rmdir "$stage_dir/data" "$stage_dir/log" "$stage_dir"
echo "$output_root/data/$run_id"
