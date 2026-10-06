#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
if [[ "${1:-}" == --help ]]; then
  python -m experiments.cache_manager_performance.src.measure --help
  exit 0
fi
run_id=""
output_root="experiments/cache_manager_performance/output"
previous=""
for argument in "$@"; do
  if [[ "$previous" == --run-id ]]; then run_id="$argument"; fi
  if [[ "$previous" == --output-root ]]; then output_root="$argument"; fi
  previous="$argument"
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "profile.sh requires --run-id NAME" >&2
  exit 2
fi
if [[ -e "$output_root/profile/$run_id" || -e "$output_root/log/$run_id" || -e "$output_root/data/$run_id" ]]; then
  echo "run ID already exists: $run_id" >&2
  exit 2
fi
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/cache-manager-profile.XXXXXX")"
profile_dir="$stage_dir/profile/$run_id"
log_dir="$stage_dir/log/$run_id"
mkdir -p "$profile_dir" "$log_dir"
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "failed run retained outside experiments: $stage_dir" >&2; fi' EXIT
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false --output="$profile_dir/trace" \
  python -m experiments.cache_manager_performance.src.measure --mode profile "$@" --output-root "$stage_dir" \
  >"$log_dir/stdout.log" 2>"$log_dir/stderr.log"
nsys export --type sqlite --output "$stage_dir/data/$run_id/trace.sqlite" "$profile_dir/trace.nsys-rep" \
  >"$log_dir/export.log" 2>"$log_dir/export.stderr.log"
python -m experiments.cache_manager_performance.src.transition \
  --manager-run "$stage_dir/data/$run_id" \
  --run-id "${run_id}_transition" \
  --output-dir "$stage_dir/data/$run_id/analysis"
mkdir -p "$output_root/data" "$output_root/log" "$output_root/profile"
mv "$stage_dir/data/$run_id" "$output_root/data/$run_id"
mv "$log_dir" "$output_root/log/$run_id"
mv "$profile_dir" "$output_root/profile/$run_id"
rmdir "$stage_dir/data" "$stage_dir/log" "$stage_dir/profile" "$stage_dir"
echo "$output_root/data/$run_id"
