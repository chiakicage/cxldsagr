#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: profile_q1_promotion.sh RUN_ID RECEIPT'
  exit 0
fi
if [[ $# != 2 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'Expected RUN_ID RECEIPT; see --help' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="$1"
receipt="$2"
base=experiments/deepseek_v32_echo_official/output
ncu=/opt/nvidia/nsight-compute/2026.1.1/ncu
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" ]]; then
    echo "Output already exists: $base/$category/$run_id" >&2
    exit 2
  fi
done
mkdir -p "$base/data/$run_id" "$base/log/$run_id" "$base/profile/$run_id"
cp -- "$0" "$base/data/$run_id/profile_q1_promotion.sh"
for method in baseline validation copy; do
  variant=candidate
  case "$method" in
    baseline) variant=baseline; kernel='regex:.*official_prefetch::promote_kernel.*' ;;
    validation) kernel='regex:.*validate_promotion_kernel.*' ;;
    copy) kernel='regex:.*copy_publish_kernel.*' ;;
  esac
  for collection in full source; do
    sections=(--set source --section SourceCounters)
    if [[ "$collection" == full ]]; then
      sections=(--set full --section PmSampling --section PmSampling_WarpStates)
    fi
    tag="${method}_${collection}"
    command=("$ncu" --target-processes all --profile-from-start off
      --replay-mode kernel --cache-control all --clock-control base
      "${sections[@]}" --kernel-name-base demangled --kernel-name "$kernel" --launch-count 1
      --export "$base/profile/$run_id/$tag"
      .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_promotion
      --mode profile --variant "$variant" --layout consecutive --occupied 0
      --receipt "$receipt" --output-dir "$base/data/$run_id/$tag")
    printf '%q ' "${command[@]}" > "$base/data/$run_id/$tag.command"
    printf '\n' >> "$base/data/$run_id/$tag.command"
    "${command[@]}" > "$base/log/$run_id/$tag.stdout.log" \
      2> "$base/log/$run_id/$tag.stderr.log"
    "$ncu" --import "$base/profile/$run_id/$tag.ncu-rep" --page details \
      > "$base/data/$run_id/$tag.details.txt"
  done
done
