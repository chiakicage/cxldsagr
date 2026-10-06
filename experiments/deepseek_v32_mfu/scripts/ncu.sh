#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash experiments/deepseek_v32_mfu/scripts/ncu.sh \
  --input <kernel_inputs_layer_N.pt> --kernel <kernel> [options]

Kernels: indexer-resident, indexer-offload, mla, recall
Options:
  --run-id ID          Fresh output run ID (default: UTC timestamp, kernel and PID)
  --warmups N          Unprofiled launches, including compilation (default: 2)
  --pool-slots N       Cold offload HBM pool size (default: 16384)
  --prefetch-limit N   Fused prefetch capacity, at most 8192 (default: 8192)
  --sets SET          full, source, or both (default: both)
  --help              Print this help without loading CUDA

Set CUDA_VISIBLE_DEVICES externally to select the profiling GPU; NCU may be
overridden with the NCU environment variable. Inputs come from measure.py
--save-kernel-inputs. Offload replays a cold historical pool; recall measures
an independent cold-union gather, not the original model's residual transfer.
Reports, metadata and separate stdout/stderr logs go to this experiment's
output/{profile,data,log}/<run_id>/. Existing run IDs are never overwritten.
Runs are staged outside experiments and published only after success; failures
print the temporary directory containing diagnostics.
EOF
}

input=""
kernel=""
run_id="${ECHO_NCU_RUN_ID:-}"
warmups=2
pool_slots=16384
prefetch_limit=8192
sets=both
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --input|--kernel|--run-id|--warmups|--pool-slots|--prefetch-limit|--sets)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --input) input="$2" ;;
        --kernel) kernel="$2" ;;
        --run-id) run_id="$2" ;;
        --warmups) warmups="$2" ;;
        --pool-slots) pool_slots="$2" ;;
        --prefetch-limit) prefetch_limit="$2" ;;
        --sets) sets="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$input" || -z "$kernel" ]]; then
  usage >&2
  exit 2
fi
case "$kernel" in
  indexer-resident) kernel_regex='deep_gemm::sm90_fp8_mqa_logits<' ;;
  indexer-offload) kernel_regex='echo_native::sm90_fp8_mqa_logits_fuse_prefetch<' ;;
  mla) kernel_regex='sparse_attn_fwd_kernel' ;;
  recall) kernel_regex='gather_records' ;;
  *) echo "Invalid kernel: $kernel" >&2; exit 2 ;;
esac
if [[ "$sets" != full && "$sets" != source && "$sets" != both ]]; then
  echo "--sets must be full, source, or both" >&2
  exit 2
fi
for number in "$warmups" "$pool_slots" "$prefetch_limit"; do
  if [[ ! "$number" =~ ^[0-9]+$ ]]; then
    echo "warmups, pool-slots and prefetch-limit must be nonnegative integers" >&2
    exit 2
  fi
done
if (( 10#$warmups < 1 || 10#$pool_slots < 1 || 10#$prefetch_limit > 8192 )); then
  echo "warmups/pool-slots must be positive and prefetch-limit must be in [0,8192]" >&2
  exit 2
fi
input="$(realpath -- "$input")"
if [[ ! -f "$input" ]]; then
  echo "Input capture does not exist: $input" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export DG_JIT_WITH_LINEINFO=1
ncu_bin="${NCU:-ncu}"
command -v "$ncu_bin" >/dev/null
if [[ -z "$run_id" ]]; then
  run_id="$(date -u +%Y%m%dT%H%M%SZ)_ncu_${kernel}_$$"
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run-id must contain only letters, digits, dots, underscores and hyphens" >&2
  exit 2
fi
base="experiments/deepseek_v32_mfu/output"
for category in data profile log; do
  if [[ -e "$base/$category/$run_id" || -L "$base/$category/$run_id" ]]; then
    echo "Run output already exists: $base/$category/$run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-echo-ncu-${run_id}.XXXXXX")"
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do rm -rf -- "$base/$category/$run_id"; done
    echo "Failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
data_dir="$staging/data"
profile_dir="$staging/profile"
log_dir="$staging/log"
mkdir "$data_dir" "$profile_dir" "$log_dir"
"$ncu_bin" --version >"$data_dir/ncu_version.txt" 2>"$log_dir/ncu_version.stderr.log"
"$ncu_bin" --list-sets >"$data_dir/ncu_sets.txt" 2>"$log_dir/ncu_sets.stderr.log"

common=(--profile-from-start off --kernel-name-base demangled
  --kernel-name "regex:$kernel_regex" --launch-count 1 --replay-mode kernel
  --cache-control all --clock-control none --import-source yes)
harness=(python -m experiments.deepseek_v32_mfu.src.kernel_profile
  --input "$input" --kernel "$kernel" --warmups "$warmups"
  --pool-slots "$pool_slots" --prefetch-limit "$prefetch_limit" --run-id "$run_id")
for capture in full source; do
  if [[ "$sets" != both && "$sets" != "$capture" ]]; then
    continue
  fi
  if [[ "$capture" == full ]]; then
    sections=(--set full --section PmSampling --section PmSampling_WarpStates)
  elif grep -Eq '^[[:space:]]*source[[:space:]]' "$data_dir/ncu_sets.txt"; then
    sections=(--set source --section SourceCounters)
  else
    # NCU 2026.1.1 does not ship a source set; its SourceCounters section
    # provides the requested per-source-line sampling directly.
    sections=(--section SourceCounters)
  fi
  command=("$ncu_bin" "${common[@]}" "${sections[@]}"
    --export "$profile_dir/${capture}_${kernel}"
    "${harness[@]}" --capture-label "$capture" --metadata "$data_dir/${capture}.json")
  printf '%q ' "${command[@]}" >"$data_dir/${capture}.command.txt"
  printf '\n' >>"$data_dir/${capture}.command.txt"
  "${command[@]}" >"$log_dir/${capture}.stdout.log" 2>"$log_dir/${capture}.stderr.log"
  if grep -Eiq 'No kernels (were )?profiled|No kernels found' \
      "$log_dir/${capture}.stdout.log" "$log_dir/${capture}.stderr.log"; then
    echo "NCU did not capture the requested kernel; inspect $log_dir" >&2
    exit 1
  fi
  if [[ ! -s "$profile_dir/${capture}_${kernel}.ncu-rep" ]]; then
    echo "NCU produced no report for $capture; inspect $log_dir" >&2
    exit 1
  fi
done
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "NCU reports: $base/profile/$run_id"
