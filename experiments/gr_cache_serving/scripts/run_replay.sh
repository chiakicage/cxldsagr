#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/run_replay.sh MODE RUN_ID [replay options]"
  echo "Full trace synchronous prototype measurement, not an online serving benchmark."
  echo "Uses native top-k, ECHO phase-snapshot fix, native cache policy, no heat override."
  echo "Logs stage outside experiments and are published only after successful completion."
  exit 0
fi
if [[ $# -lt 2 ]]; then
  echo "Expected MODE RUN_ID; use --help" >&2
  exit 2
fi
mode="$1"
run_id="$2"
shift 2
case "$mode" in resident|sparse_sync|echo_gr_adapted|dense_prefetch) ;;
  *) echo "Invalid mode" >&2; exit 2 ;;
esac
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, numbers, underscores or hyphens" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
base="$PWD/experiments/gr_cache_serving/output"
if [[ -e "$base/data/$run_id" || -e "$base/log/$run_id" ]]; then
  echo "RUN_ID already exists" >&2
  exit 2
fi
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
trace="$base/data/input_beauty_128_512_20260929_v2/requests.jsonl"
staging="$(mktemp -d -t echo-replay.XXXXXXXX)"
echo "Staged logs: $staging"
output_dir="$base/data/$run_id"
diagnostic_args=()
if [[ "${SPARSEGR_REPLAY_DIAGNOSTIC:-0}" == 1 ]]; then
  output_dir="$staging/data"
  diagnostic_args=(--diagnostic)
fi
ulimit -c 0
timeout --signal=TERM --kill-after=15s "${SPARSEGR_REPLAY_TIMEOUT_S:-1800}s" \
  "$env_dir/bin/python" -m experiments.gr_cache_serving.src.replay \
  --trace "$trace" --mode "$mode" --output-dir "$output_dir" "${diagnostic_args[@]}" "$@" \
  >"$staging/stdout.log" 2>"$staging/stderr.log"
if [[ "${SPARSEGR_REPLAY_DIAGNOSTIC:-0}" == 1 ]]; then
  echo "Diagnostic only: $staging"
  exit 0
fi
mkdir -p "$base/log"
mv -T --no-clobber "$staging" "$base/log/$run_id"
echo "Complete: $base/data/$run_id"
