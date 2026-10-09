#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: profile_q1_hint.sh RUN_ID HINT_RECEIPT [full|source]'
  echo 'Run the separate q1_hint_baseline check and clean bench first, using the same environment.'
  echo 'Select the GPU/CPU placement externally. Profiles one installed L0 PyTorch sum.'
  echo 'The source collection retains SASS counters; wheel line information is unverified.'
  exit 0
fi
if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'Expected RUN_ID HINT_RECEIPT [full|source]; see --help' >&2
  exit 2
fi
collection="${3:-full}"
if [[ "$collection" != full && "$collection" != source ]]; then
  echo 'Collection must be full or source' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
receipt="$(realpath -e -- "$2")"
cd -- "$script_dir/../../.."
run_id="$1"
base=experiments/deepseek_v32_echo_official/output
ncu=/opt/nvidia/nsight-compute/2026.1.1/ncu
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" ]]; then
    echo "Output already exists: $base/$category/$run_id" >&2
    exit 2
  fi
done
mkdir -p "$base/data/$run_id" "$base/log/$run_id" "$base/profile/$run_id"
cp -- "$script_dir/profile_q1_hint.sh" "$base/data/$run_id/profile_q1_hint.sh"
sections=(--set source --section SourceCounters)
if [[ "$collection" == full ]]; then
  sections=(--set full --section PmSampling --section PmSampling_WarpStates)
fi
# NCU prepends its target directory. Keep the application's checked library
# lookup unchanged while retaining NCU's other instrumentation variables.
application_env=(/usr/bin/env)
if [[ -v LD_LIBRARY_PATH ]]; then
  application_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
else
  application_env+=(-u LD_LIBRARY_PATH)
fi
# Follow env's exec into Python; the NVTX and kernel filters still select one sum.
command=("$ncu" --target-processes all --profile-from-start off
  --replay-mode kernel --cache-control all --clock-control base
  "${sections[@]}" --nvtx --nvtx-include 'q1_hint_baseline_layer_0/'
  --kernel-name-base demangled --kernel-name 'regex:.*reduce_kernel.*'
  --launch-count 1 --export "$base/profile/$run_id/$collection"
  "${application_env[@]}" .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_hint_baseline
  profile --layer layer_0 --receipt "$receipt"
  --output-dir "$base/data/$run_id/component")
printf '%q ' "${command[@]}" > "$base/data/$run_id/command.txt"
printf '\n' >> "$base/data/$run_id/command.txt"
"$ncu" --version > "$base/data/$run_id/ncu_version.txt"
"${command[@]}" > "$base/log/$run_id/stdout.log" 2> "$base/log/$run_id/stderr.log"
.venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_hint_baseline \
  verify-profile --layer layer_0 --receipt "$receipt" \
  --output-dir "$base/data/$run_id/component" \
  > "$base/data/$run_id/completion.json" 2> "$base/log/$run_id/completion.stderr.log"
"$ncu" --import "$base/profile/$run_id/$collection.ncu-rep" --page details \
  > "$base/data/$run_id/details.txt"
