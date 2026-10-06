#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
if [[ "${1:-}" == --help ]]; then
  exec python -m experiments.cache_manager_performance.src.measure --help
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
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ || "$mode" != check ]]; then
  echo "run.sh requires --mode check --run-id NAME; use profile.sh for top-k-to-MLA intervals" >&2
  exit 2
fi
if [[ -z "$output_root" ]]; then
  output_root="/tmp/cxldsagr-checks/cache_manager_performance"
fi
if [[ -e "$output_root/log/$run_id" || -e "$output_root/data/$run_id" ]]; then
  echo "run ID already exists: $run_id" >&2
  exit 2
fi
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/cache-manager-run.XXXXXX")"
log_dir="$stage_dir/log/$run_id"
mkdir -p "$log_dir"
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "failed run retained outside experiments: $stage_dir" >&2; fi' EXIT
python -m experiments.cache_manager_performance.src.measure "$@" --output-root "$stage_dir" \
  >"$log_dir/stdout.log" 2>"$log_dir/stderr.log"
mkdir -p "$output_root/data" "$output_root/log"
mv "$stage_dir/data/$run_id" "$output_root/data/$run_id"
mv "$log_dir" "$output_root/log/$run_id"
rmdir "$stage_dir/data" "$stage_dir/log" "$stage_dir"
echo "$output_root/data/$run_id"
