#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/run_parts_profile.sh MODE RUN_ID"
  echo "Independent five-layer Dense/DSA NVTX diagnostic, fixed budgets, original trace prefix."
  exit 0
fi
if [[ $# != 2 || ! "$2" =~ ^[A-Za-z0-9_-]+$ ]]; then exit 2; fi
case "$1" in dense_prefetch|sparse_sync) ;; *) exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
base="$PWD/experiments/gr_cache_serving/output"
for kind in data log profile; do
  if [[ -e "$base/$kind/$2" ]]; then echo "RUN_ID already exists"; exit 2; fi
done
env_dir="$PWD/3rdparty/ECHO/.venv"
export CUDA_HOME="${CUDA_HOME:-$env_dir/cuda-12.8}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$env_dir/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for include in "$env_dir"/lib/python3.12/site-packages/nvidia/*/include; do
  if [[ -d "$include" ]]; then export CPATH="$include${CPATH:+:$CPATH}"; fi
done
export DG_JIT_CACHE_DIR="$env_dir/deepgemm-jit"
export PYTHONDONTWRITEBYTECODE=1
stage="$(mktemp -d -t gr-parts.XXXXXXXX)"
mkdir -p "$stage/profile" "$stage/log"
echo "Staged logs: $stage/log"
ulimit -c 0
timeout --signal=TERM --kill-after=15s 1800s \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=false --force-overwrite=false \
  --output="$stage/profile/trace" \
  "$env_dir/bin/python" -m experiments.gr_cache_serving.src.profile_parts \
  --mode "$1" --output-dir "$stage/data" \
  --trace "$base/data/five_layer_fixed512_20261001_input_u64/requests.jsonl" \
  >"$stage/log/stdout.log" 2>"$stage/log/stderr.log"
nsys export --type=sqlite --output="$stage/profile/trace.sqlite" \
  "$stage/profile/trace.nsys-rep" >"$stage/log/export.log" 2>&1
test -s "$stage/data/summary.json"
test -s "$stage/profile/trace.sqlite"
for kind in data log profile; do
  mkdir -p "$base/$kind"
  mv -T --no-clobber "$stage/$kind" "$base/$kind/$2"
done
rmdir "$stage"
echo "Complete: $base/data/$2"
