#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/indexer_block_sparse_profile/scripts/bottleneck.sh RUN_ID SOURCE_RUN_ID"
  echo "Measure CIS eager/graph and Nsight Compute counters; verify its source against a model run."
  echo "SOURCE_RUN_ID must be a current, completed model run. Uses CUDA_VISIBLE_DEVICES, cuda:0."
  echo "Requires project .venv and ncu. Successful results publish to output/{data,log,profile}/RUN_ID."
}
if [[ "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
run_id="$1"
source_run="$2"
for id in "$run_id" "$source_run"; do
  if [[ ! "$id" =~ ^[A-Za-z0-9_-]+$ ]]; then echo "Invalid run ID" >&2; exit 2; fi
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PYTHONDONTWRITEBYTECODE=1
output="experiments/indexer_block_sparse_profile/output"
source_data="$output/data/$source_run"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then echo "Run already exists: $run_id" >&2; exit 2; fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-bottleneck-${run_id}.XXXXXX")"
mkdir -p "$staging/data" "$staging/log" "$staging/profile"
published=false
claimed=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed[@]}"; do rm -rf -- "$output/$category/$run_id"; done
    echo "Failed (status $status); diagnostics outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_logged() {
  local step="$1"
  shift
  "$@" > "$staging/log/$step.stdout.log" 2> "$staging/log/$step.stderr.log"
}
run_logged gpu nvidia-smi -q
run_logged ncu_version ncu --version
common=(--run-id "$run_id" --output "$staging/data/cis_bottleneck.json")
run_logged cis .venv/bin/python -m experiments.indexer_block_sparse_profile.src.cis_bottleneck "${common[@]}"
run_logged ncu ncu --profile-from-start off --target-processes all --cache-control none --clock-control none \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
  --export "$staging/profile/cis" \
  .venv/bin/python -m experiments.indexer_block_sparse_profile.src.cis_bottleneck "${common[@]}" --counters
ncu --import "$staging/profile/cis.ncu-rep" --csv --page details \
  > "$staging/data/ncu.csv" 2> "$staging/log/ncu_export.stderr.log"
run_logged report .venv/bin/python -m experiments.indexer_block_sparse_profile.src.bottleneck_report \
  "$staging/data" --source-data-dir "$source_data" --cis-only
mkdir -p "$staging/data/sources"
cp --parents models/nosa/scoring.py \
  experiments/indexer_block_sparse_profile/src/{cis_bottleneck,bottleneck_report}.py \
  experiments/indexer_block_sparse_profile/scripts/bottleneck.sh "$staging/data/sources/"
for category in data log profile; do
  mkdir -p "$output/$category"
  mkdir "$output/$category/$run_id"
  claimed+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
