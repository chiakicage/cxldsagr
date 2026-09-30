#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/build_echo_backend.sh [--check]"
  echo "Build pinned ECHO SM90 kernels in ECHO/.venv; skip verified installed components."
  echo "Prerequisites: prepare_echo.sh, ECHO Python dependencies, isolated CUDA 12.8,"
  echo "and root 3rdparty/cutlass at the repository pin (prepare_3rdparty.py --init)."
  echo "--check validates the existing backend without installing or downloading."
  echo "CUDA_HOME may select an existing CUDA 12.8 toolkit; no system CUDA is changed."
  echo "Build logs and source checkouts stay under ignored ECHO/.environment/."
  echo "This does not run a model, GPU correctness test, or performance benchmark."
  exit 0
fi
check_only=0
if [[ $# == 1 && "$1" == --check ]]; then
  check_only=1
elif [[ $# != 0 ]]; then
  echo "Unexpected arguments; use --help" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
export ECHO_ROOT="$repo_dir/3rdparty/ECHO"
env_dir="$ECHO_ROOT/.venv"
python="$env_dir/bin/python"
records="$ECHO_ROOT/.environment"
echo_revision=bc1b75c1000010d0ac6f032ebaac283255c050b1
cutlass_revision=f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
fht_revision=7fd811c2b47f63b0b08d2582619f939e14dad77c
mla_revision=1408756a88e52a25196b759eaf8db89d2b51b5a1
export CUDA_HOME="${CUDA_HOME:-$env_dir/cuda-12.8}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$env_dir/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DG_JIT_CACHE_DIR="$env_dir/deepgemm-jit"
export PYTHONDONTWRITEBYTECODE=1
cd -- "$repo_dir"

fail() { echo "$*" >&2; exit 1; }
[[ -x "$python" && -x "$CUDACXX" && -x "$CUDA_HOME/bin/cuobjdump" ]] ||
  fail "Missing isolated ECHO Python, CUDA nvcc or cuobjdump; see experiment README."
[[ "$(git -C "$ECHO_ROOT" rev-parse HEAD)" == "$echo_revision" ]] || fail "Wrong ECHO revision."
[[ -z "$(git -C "$ECHO_ROOT" status --porcelain --untracked-files=no)" ]] || fail "ECHO has tracked source changes."
"$CUDACXX" --version | grep -q 'release 12.8,' || fail "This recipe requires CUDA 12.8."
"$python" - <<'PY'
import importlib.metadata as metadata
import sys
import torch
assert sys.version_info[:2] == (3, 12), sys.version
assert metadata.version('torch') == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
for name, expected in [('sglang', '0.5.3.post3'), ('flashinfer-python', '0.4.0'),
                       ('apache-tvm-ffi', '0.1.0b15')]:
    assert metadata.version(name) == expected, (name, metadata.version(name))
PY
cuda_packages="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia"
for include_dir in "$cuda_packages"/*/include; do
  if [[ -d "$include_dir" ]]; then export CPATH="$include_dir${CPATH:+:$CPATH}"; fi
done

component_ready() {
  "$python" - "$1" <<'PY'
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import torch
root = Path(os.environ['ECHO_ROOT'])
components = {
    'sgl-kernel': ('0.3.16.post2', 'sgl_kernel', ['fast_argtopk', 'fast_argmin_bounded'], root / 'sglang/sgl-kernel'),
    'deep-gemm': ('2.1.1+bc1b75c', 'deep_gemm', ['fp8_paged_mqa_logits_fused_v2', 'fp8_mqa_logits_fuse_prefetch'], root / 'DeepGEMM'),
    'fast-hadamard-transform': ('1.0.4.post1', 'fast_hadamard_transform', ['hadamard_transform'], root / '.environment/wheels/fast_hadamard_transform-1.0.4.post1-cp312-cp312-linux_x86_64.whl'),
    'flash-mla': ('1.0.0+1408756', 'flash_mla', ['flash_mla_sparse_fwd', 'flash_mla_with_kvcache', 'get_mla_metadata'], root / '.environment/sources/FlashMLA'),
    'ijson': ('3.5.1', 'ijson', ['items'], None),
}
try:
    version, module, symbols, source = components[sys.argv[1]]
    dist = metadata.distribution(sys.argv[1])
    assert dist.version == version
    imported = importlib.import_module(module)
    assert all(hasattr(imported, symbol) for symbol in symbols)
    if source is not None:
        assert json.loads(dist.read_text('direct_url.json'))['url'] == source.as_uri()
except (AssertionError, ImportError, OSError, TypeError, KeyError, metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
}

all_ready=1
for component in sgl-kernel deep-gemm fast-hadamard-transform flash-mla ijson; do
  if component_ready "$component"; then
    echo "Ready: $component"
  else
    echo "Needs build/install: $component"
    all_ready=0
  fi
done
if [[ "$check_only" == 1 ]]; then
  [[ "$all_ready" == 1 ]] || fail "Backend incomplete; rerun without --check after preparing prerequisites."
  echo "Backend imports and local installation provenance verified; GPU correctness not tested."
  echo "Known metadata mismatch: SGLang pins sgl-kernel 0.3.15; ECHO source is 0.3.16.post2."
  exit 0
fi

prepare_source() {
  local target="$1" url="$2" revision="$3"
  if [[ ! -e "$target" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$target"
    git -C "$target" checkout --detach "$revision"
  fi
  [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]] || fail "Wrong source revision: $target"
  [[ -z "$(git -C "$target" status --porcelain --untracked-files=no)" ]] || fail "Tracked changes: $target"
}
if [[ "$all_ready" != 1 ]]; then
  command -v uv >/dev/null || fail "uv is required."
  mkdir -p "$records/sources" "$records/wheels" "$records/logs"
  uv pip install --python "$python" cmake==3.31.6 scikit-build-core==1.1.0 wheel==0.48.0
  sgl_changed=0
  if ! component_ready sgl-kernel; then
    CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}" MAX_JOBS="${MAX_JOBS:-16}" \
      uv pip install --python "$python" --no-build-isolation \
      -Ccmake.define.SGL_KERNEL_ONLY_SM90=ON -Ccmake.define.ENABLE_BELOW_SM90=OFF \
      -Cbuild-dir=build-sm90 "$ECHO_ROOT/sglang/sgl-kernel" \
      2>&1 | tee "$records/logs/sgl-kernel-build-$(date -u +%Y%m%dT%H%M%S).log"
    sgl_changed=1
  fi
  if ! component_ready fast-hadamard-transform; then
    prepare_source "$records/sources/fast-hadamard-transform" https://github.com/Dao-AILab/fast-hadamard-transform.git "$fht_revision"
    FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE MAX_JOBS=2 \
      uv build --wheel --no-build-isolation --python "$python" --out-dir "$records/wheels" \
      "$records/sources/fast-hadamard-transform" \
      2>&1 | tee "$records/logs/fht-build-$(date -u +%Y%m%dT%H%M%S).log"
    uv pip install --python "$python" "$records/wheels/fast_hadamard_transform-1.0.4.post1-cp312-cp312-linux_x86_64.whl"
  fi
  if ! component_ready flash-mla; then
    prepare_source "$records/sources/FlashMLA" https://github.com/deepseek-ai/FlashMLA.git "$mla_revision"
    # This pin differs from both shared CUTLASS and sgl-kernel's own dependency.
    git -C "$records/sources/FlashMLA" -c submodule.recurse=false submodule update --init --depth 1 csrc/cutlass
    [[ "$(git -C "$records/sources/FlashMLA/csrc/cutlass" rev-parse HEAD)" == e94e888df3551224738bfa505787b515eae8352f ]] || fail "Wrong FlashMLA CUTLASS revision."
    FLASH_MLA_DISABLE_SM100=1 NVCC_THREADS=4 MAX_JOBS=4 \
      uv pip install --python "$python" --no-build-isolation "$records/sources/FlashMLA" \
      2>&1 | tee "$records/logs/flashmla-build-$(date -u +%Y%m%dT%H%M%S).log"
  fi
  if ! component_ready ijson; then uv pip install --python "$python" ijson==3.5.1; fi
  # sgl-kernel also owns deep_gemm files, so restore ECHO's package after replacing it.
  if [[ "$sgl_changed" == 1 ]] || ! component_ready deep-gemm; then
    [[ "$(git -C "$repo_dir/3rdparty/cutlass" rev-parse HEAD)" == "$cutlass_revision" ]] || fail "Prepare the shared root CUTLASS first."
    nested="$ECHO_ROOT/DeepGEMM/third-party/cutlass"
    [[ ! -e "$nested/.git" ]] || fail "Deinitialize nested ECHO CUTLASS before using the shared layout."
    mkdir -p "$nested"
    if [[ ! -e "$nested/include" && ! -L "$nested/include" ]]; then
      ln -s "$repo_dir/3rdparty/cutlass/include" "$nested/include"
    fi
    [[ -L "$nested/include" && "$(readlink -f "$nested/include")" == "$repo_dir/3rdparty/cutlass/include" ]] || fail "ECHO CUTLASS must link to the shared include directory."
    git -C "$ECHO_ROOT" config submodule.DeepGEMM/third-party/cutlass.active false
    git -C "$ECHO_ROOT" config submodule.DeepGEMM/third-party/cutlass.update none
    git -C "$ECHO_ROOT" -c submodule.recurse=false submodule update --init --depth 1 DeepGEMM/third-party/fmt
    [[ "$(git -C "$ECHO_ROOT/DeepGEMM/third-party/fmt" rev-parse HEAD)" == 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28 ]] || fail "Wrong fmt revision."
    DG_FORCE_BUILD=1 MAX_JOBS=2 uv pip install --python "$python" --no-build-isolation \
      --reinstall-package deep-gemm "$ECHO_ROOT/DeepGEMM" \
      2>&1 | tee "$records/logs/deepgemm-build-$(date -u +%Y%m%dT%H%M%S).log"
  fi
fi
for component in sgl-kernel deep-gemm fast-hadamard-transform flash-mla ijson; do
  component_ready "$component" || fail "Installed component validation failed: $component"
done
if check_output="$(uv pip check --python "$python" 2>&1)"; then
  printf '%s\n' "$check_output"
else
  printf '%s\n' "$check_output" >&2
  [[ "$check_output" == *"Found 1 incompatibility"* && "$check_output" == *'The package `sglang` requires `sgl-kernel==0.3.15`, but `0.3.16.post2` is installed'* ]] || exit 1
  echo "Expected upstream metadata mismatch retained; this is not a clean pip check."
fi
echo "Backend ready. Run bash scripts/run_echo_tests.sh all for real-weight correctness."
