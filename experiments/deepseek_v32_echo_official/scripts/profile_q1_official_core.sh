#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: profile_q1_official_core.sh RUN_ID RECEIPT LAYER zero|warm|empty pilot|source|full'
  echo 'Select the exclusive GPU/CPU placement externally to match the component receipt.'
  echo 'One original baseline core is captured; caller reset and readback are outside profiling.'
  exit 0
fi
if [[ $# -ne 5 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ! "$3" =~ ^[012]$ ]]; then
  echo 'Expected RUN_ID RECEIPT LAYER POLICY COLLECTION; see --help' >&2
  exit 2
fi
case "$4" in zero|warm|empty) ;; *) exit 2 ;; esac
case "$5" in pilot|source|full) ;; *) exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
receipt="$(realpath -e -- "$2")"
cd -- "$script_dir/../../.."
run_id="$1"
layer="$3"
policy="$4"
collection="$5"
tag="layer_${layer}_${policy}_${collection}"
base=experiments/deepseek_v32_echo_official/output
data="$base/data/$run_id/$tag"
log="$base/log/$run_id"
report="$base/profile/$run_id/$tag"
if [[ -e "$data" || -e "$report.ncu-rep" || -e "$log/$tag.stdout.log" ]]; then
  echo "Capture already exists: $tag" >&2
  exit 2
fi
mkdir -p "$base/data/$run_id" "$log" "$base/profile/$run_id"
mkdir "$data"
cp -- "$script_dir/profile_q1_official_core.sh" "$data/profile_q1_official_core.sh"
cp -- experiments/deepseek_v32_echo_official/src/q1_official_core_profile.py "$data/"
ncu=/opt/nvidia/nsight-compute/2026.1.1/ncu
pilot_metrics=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_aperture_sysmem_op_read.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max
sections=(--section LaunchStats --metrics "$pilot_metrics")
if [[ "$collection" == source ]]; then
  sections=(--section LaunchStats --section SourceCounters --section WarpStateStats
    --metrics "${pilot_metrics},smsp__sass_inst_executed_op_global_atom.sum")
elif [[ "$collection" == full ]]; then
  sections=(--set full --section PmSampling_WarpStates)
fi
# Restore the checked library lookup after NCU's target-directory injection.
application_env=(/usr/bin/env)
if [[ -v LD_LIBRARY_PATH ]]; then
  application_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
else
  application_env+=(-u LD_LIBRARY_PATH)
fi
command=("$ncu" --target-processes all --profile-from-start off
  --replay-mode kernel --cache-control all --clock-control base
  --import-source yes "${sections[@]}"
  --nvtx --nvtx-include 'q1_fused_prepare_complete_baseline/'
  --kernel-name-base demangled --kernel-name 'regex:.*fp8_paged_mqa_logits.*'
  --launch-count 1 --export "$report"
  "${application_env[@]}" .venv/bin/python -B -m
  experiments.deepseek_v32_echo_official.src.q1_official_core_profile
  --layer "$layer" --policy "$policy" --physical-device 1
  --receipt "$receipt" --output-root "$data/passes")
printf '%q ' "${command[@]}" > "$data/command.txt"
printf '\n' >> "$data/command.txt"
"$ncu" --version > "$data/ncu_version.txt"
"${command[@]}" > "$log/$tag.stdout.log" 2> "$log/$tag.stderr.log"
"$ncu" --import "$report.ncu-rep" --page details > "$data/details.txt"
