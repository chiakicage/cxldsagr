#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/run_tests.sh [cpu|gpu|all]"
  echo "Default: cpu. gpu/all require Hopper CUDA, nvcc, shared CUTLASS, TVM FFI, Triton and FlashInfer."
  echo "CLI checks use tiny weights."
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "Expected one optional mode; use --help" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
mode="${1:-cpu}"
if [[ ! "$mode" =~ ^(cpu|gpu|all)$ ]]; then
  echo "Invalid mode; use --help" >&2
  exit 2
fi
if [[ "$mode" == cpu || "$mode" == all ]]; then
  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
    models/nosa/tests models/deepseek_v32/tests/test_echo_adapter.py \
    models/deepseek_v32/tests/test_echo_cache.py models/deepseek_v32/tests/test_echo_dense.py \
    models/deepseek_v32/tests/test_echo_kernel.py \
    models/deepseek_v32/tests/test_echo_recall.py \
    models/deepseek_v32/tests/test_echo_index.py \
    operators/sm90/tests cache/tests executor/tests serving/tests GR/tests \
    tests/integration experiments/nosa_gr_65536_1024/tests \
    experiments/nosa_indexer_pattern_65536_1024/tests \
    experiments/gr_cache_serving/tests \
    experiments/indexer_block_sparse_profile/tests \
    experiments/nosa_kernel_mfu/tests \
    -q -rs -p no:cacheprovider
fi
if [[ "$mode" == gpu || "$mode" == all ]]; then
  .venv/bin/python -c 'import torch; import flashinfer; import triton; import tvm_ffi; torch.cuda.init(); from operators.sm90._native import build_info; build_info()'
  .venv/bin/python -m pytest models/nosa/tests operators/sm90/tests tests/integration \
    -k 'cuda or flashinfer' -q -rs -p no:cacheprovider
fi
