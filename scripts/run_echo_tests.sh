#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash scripts/run_echo_tests.sh [resident|sparse_sync|echo_gr_adapted|dense_prefetch|all]"
  echo "Real-weight, three-layer GR correctness only; no benchmark or report output."
  echo "Requires ECHO/.venv with custom kernels and a matching CUDA toolkit."
  echo "Override CUDA_HOME or SPARSEGR_ECHO_MODEL when needed."
  echo "all also compares all three offload modes with a temporary resident reference."
  echo "SPARSEGR_ECHO_TEST_TRACE optionally selects GR JSONL with >=3 users and revisits."
  echo "Default logical top-k order is a correctness-only control, never a timing path."
  echo "SPARSEGR_ECHO_TEST_TOPK_ORDER=native disables it; native 64K is nondeterministic."
  echo "ECHO fused mode defaults to the pinned phase-snapshot kernel overlay."
  echo "SPARSEGR_ECHO_TEST_KERNEL_PATCH=native selects the original kernel (64K can hang)."
  echo "SPARSEGR_ECHO_TEST_TIMEOUT_S defaults to 300 seconds per backend."
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "Expected one optional mode; use --help" >&2
  exit 2
fi
mode="${1:-resident}"
case "$mode" in
  resident|sparse_sync|echo_gr_adapted|dense_prefetch|all) ;;
  *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
echo_env="$PWD/3rdparty/ECHO/.venv"
export CUDA_HOME="${CUDA_HOME:-$echo_env/cuda-12.8}"
if [[ ! -x "$echo_env/bin/python" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "Missing isolated ECHO Python or matching CUDA compiler; see experiment README." >&2
  exit 1
fi
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$echo_env/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cuda_packages="$echo_env/lib/python3.12/site-packages/nvidia"
for include_dir in "$cuda_packages"/*/include; do
  if [[ -d "$include_dir" ]]; then
    export CPATH="$include_dir${CPATH:+:$CPATH}"
  fi
done
export DG_JIT_CACHE_DIR="$echo_env/deepgemm-jit"
export PYTHONDONTWRITEBYTECODE=1
export SPARSEGR_RUN_ECHO_GPU_TESTS=1
export SPARSEGR_ECHO_TEST_TOPK_ORDER="${SPARSEGR_ECHO_TEST_TOPK_ORDER:-logical}"
export SPARSEGR_ECHO_TEST_KERNEL_PATCH="${SPARSEGR_ECHO_TEST_KERNEL_PATCH:-phase_snapshot}"
case "$SPARSEGR_ECHO_TEST_KERNEL_PATCH" in
  phase_snapshot|native) ;;
  *) echo "SPARSEGR_ECHO_TEST_KERNEL_PATCH must be phase_snapshot or native" >&2; exit 2 ;;
esac
case "$SPARSEGR_ECHO_TEST_TOPK_ORDER" in
  logical|native) ;;
  *) echo "SPARSEGR_ECHO_TEST_TOPK_ORDER must be logical or native" >&2; exit 2 ;;
esac
timeout_seconds="${SPARSEGR_ECHO_TEST_TIMEOUT_S:-300}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPARSEGR_ECHO_TEST_TIMEOUT_S must be a positive integer" >&2
  exit 2
fi
if ! command -v timeout >/dev/null; then
  echo "GNU timeout is required to bound GPU correctness checks." >&2
  exit 1
fi
echo "Correctness-only top-k order: $SPARSEGR_ECHO_TEST_TOPK_ORDER (no performance measurement)"
if [[ "$mode" == all ]]; then
  timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" \
    "$echo_env/bin/python" -m unittest discover \
    -s tests/integration -p test_echo_recall_address.py -v
  reference_dir="$(mktemp -d -t echo-gr-check.XXXXXXXX)"
  trap 'rm -rf -- "$reference_dir"' EXIT
  export SPARSEGR_ECHO_REFERENCE="$reference_dir/resident.pt"
  for backend in resident sparse_sync echo_gr_adapted dense_prefetch; do
    SPARSEGR_ECHO_MODE="$backend" timeout --signal=TERM --kill-after=10s \
      "${timeout_seconds}s" "$echo_env/bin/python" -m unittest discover \
      -s tests/integration -p test_echo_gr_prefix.py -v
  done
  exit 0
fi
export SPARSEGR_ECHO_MODE="$mode"
exec timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" \
  "$echo_env/bin/python" -m unittest discover \
  -s tests/integration -p test_echo_gr_prefix.py -v
