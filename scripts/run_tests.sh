#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/run_tests.sh [cpu|gpu|all]"
  echo "Default: cpu. gpu/all require Hopper CUDA, nvcc, shared CUTLASS, TVM FFI, Triton and FlashInfer."
  echo "CLI checks use tiny weights."
  echo "Optional checkpoint checks: NOSA_MODEL_PATH and NOSA_OFFLOAD_CHECKPOINT (NOSA-8B)."
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
    models/nosa/tests operators/nosa operators/deepseek_v32 operators/common \
    cache/tests executor/tests serving/tests GR/tests \
    models/deepseek_v32/tests/test_echo_model.py \
    models/deepseek_v32/tests/test_echo_block.py \
    models/deepseek_v32/tests/test_echo_infer.py \
    tests/integration experiments/nosa_gr_65536_1024/tests \
    experiments/nosa_indexer_pattern_65536_1024/tests \
    experiments/indexer_block_sparse_profile/tests \
    experiments/nosa_kernel_mfu/tests \
    experiments/nosa_offload_overlap/tests \
    experiments/deepseek_v32_echo_prefill/tests \
    -q -rs -p no:cacheprovider
fi
if [[ "$mode" == gpu || "$mode" == all ]]; then
  .venv/bin/python - <<'PY'
import torch

torch.cuda.init()
if torch.cuda.get_device_capability() != (9, 0):
    raise RuntimeError("GPU regression requires SM90/Hopper; unavailable hardware is not a skip")

import flashinfer
import triton
import tvm_ffi
from operators.nosa._native import build_info
from operators.deepseek_v32.indexer.echo import build_info as echo_build_info

build_info()
echo_build_info()
PY
  # Collect complete functional directories: GPU tests do not all contain
  # 'cuda' or 'flashinfer' in their names. Each numerical test is collected once.
  .venv/bin/python -m pytest models/nosa/tests operators/nosa \
    operators/deepseek_v32 operators/common tests/integration \
    cache/tests/test_sparse_token_cache.py \
    -q -rs -p no:cacheprovider
fi
