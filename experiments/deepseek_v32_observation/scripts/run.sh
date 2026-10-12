#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
run_id=""
gpu_id=0
group=all
checkpoint=/preset-models
warmup=5
repeats=31
inner=20
attention_slice=all
shard_index=0
shard_count=1
case_index=0

usage() {
    cat <<'EOF'
Usage: bash experiments/deepseek_v32_observation/scripts/run.sh --run-id ID [options]
  --gpu N              Physical GPU index (default 0)
  --group all|common|moe|attention (default all)
  --checkpoint PATH    Checkpoint root (default /preset-models)
  --warmup N           Independent API warmup calls (default 5)
  --repeats N          Timing samples (default 31)
  --inner N            API calls per graph sample (default 20)
  --attention-slice all|decode|extend (default all; group must be attention)
  --shard-index N      Attention workload shard index (default 0)
  --shard-count N      Number of attention shards (default 1)
All cases use independent check, bench, and NCU graph profile processes.
Bandwidth uses actual DRAM counters with warm caches, in one profiling pass.
Reusing a workload group/slice fails.
Separate groups may share one run ID on different GPUs; aggregate after all groups finish.
EOF
}
while (($#)); do
    case "$1" in
        --run-id) run_id=$2; shift 2 ;;
        --gpu) gpu_id=$2; shift 2 ;;
        --group) group=$2; shift 2 ;;
        --checkpoint) checkpoint=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --inner) inner=$2; shift 2 ;;
        --attention-slice) attention_slice=$2; shift 2 ;;
        --shard-index) shard_index=$2; shift 2 ;;
        --shard-count) shard_count=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { usage >&2; exit 2; }
[[ "$group" =~ ^(all|common|moe|attention)$ ]] || { usage >&2; exit 2; }
[[ "$attention_slice" =~ ^(all|decode|extend)$ ]] || { usage >&2; exit 2; }
[[ "$attention_slice" == all || "$group" == attention ]] || { usage >&2; exit 2; }
[[ "$shard_index" =~ ^[0-9]+$ && "$shard_count" =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
((shard_index < shard_count)) || { usage >&2; exit 2; }
[[ "$shard_count" == 1 || "$group" == attention ]] || { usage >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PATH="$repo_root/.venv/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=backend:native
task_python="$repo_root/.venv/bin/python"
task_output="$repo_root/experiments/deepseek_v32_observation/output"
task_data="$task_output/data/$run_id"
task_log="$task_output/log/$run_id"
task_checks="/tmp/cxldsagr-checks/deepseek_v32_observation/$run_id"
task_profile="$task_output/profile/$run_id"
if [[ "$group" == all ]]; then
    [[ ! -e "$task_data" && ! -e "$task_checks" && ! -e "$task_log" ]]
fi
mkdir -p "$task_data" "$task_log" "$task_checks" "$task_profile"
mkdir -p "$task_data/.claims"
task_claim=$group
if [[ "$group" == attention && "$attention_slice" != all ]]; then task_claim="attention_$attention_slice"; fi
if ((shard_count > 1)); then task_claim="${task_claim}_${shard_index}_of_${shard_count}"; fi
mkdir "$task_data/.claims/$task_claim"
if [[ "$attention_slice" == all && "$shard_count" == 1 ]]; then
    [[ ! -e "$task_data/$group" && ! -e "$task_checks/$group" ]]
fi

run_case() {
    local task_group=$1 task_key=$2
    shift 2
    if [[ "$task_group" == attention ]]; then
        local task_case_index=$case_index
        case_index=$((case_index + 1))
        if ((task_case_index % shard_count != shard_index)); then return; fi
    fi
    local task_common_args=(--group "$task_group" --checkpoint "$checkpoint" --device 0 "$@")
    "$task_python" -m experiments.deepseek_v32_observation.src.measure \
        --mode check "${task_common_args[@]}" --output "$task_checks/$task_group/$task_key" \
        >"$task_log/${task_group}_${task_key}_check.stdout" \
        2>"$task_log/${task_group}_${task_key}_check.stderr"
    "$task_python" -m experiments.deepseek_v32_observation.src.measure \
        --mode bench "${task_common_args[@]}" --warmup "$warmup" --repeats "$repeats" --inner "$inner" \
        --receipt "$task_checks/$task_group/$task_key/receipt.json" \
        --output "$task_data/$task_group/$task_key" \
        >"$task_log/${task_group}_${task_key}_bench.stdout" \
        2>"$task_log/${task_group}_${task_key}_bench.stderr"
    mkdir -p "$task_profile/$task_group/$task_key"
    /usr/local/cuda/bin/ncu --target-processes all --replay-mode kernel \
        --graph-profiling graph --cache-control none --clock-control none \
        --nvtx --nvtx-include 'regex:observation_hbm__.*/' \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__bytes.sum.peak_sustained_elapsed.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed \
        --export "$task_profile/$task_group/$task_key/graphs" \
        "$task_python" -m experiments.deepseek_v32_observation.src.profile capture \
        "${task_common_args[@]}" --warmup "$warmup" --inner "$inner" \
        --receipt "$task_checks/$task_group/$task_key/receipt.json" \
        --output "$task_data/$task_group/$task_key/profile_execution.json" \
        >"$task_log/${task_group}_${task_key}_profile.stdout" \
        2>"$task_log/${task_group}_${task_key}_profile.stderr"
    "$task_python" -m experiments.deepseek_v32_observation.src.profile parse \
        --execution "$task_data/$task_group/$task_key/profile_execution.json" \
        --report "$task_profile/$task_group/$task_key/graphs.ncu-rep" \
        --output "$task_data/$task_group/$task_key/hbm.json" \
        --replay-mode kernel-graph \
        --time-boundary 'NCU entire CUDA Graph GPU duration, no external event nodes; intrusive profile, separate from clean API timing' \
        >"$task_log/${task_group}_${task_key}_parse.stdout" \
        2>"$task_log/${task_group}_${task_key}_parse.stderr"
    echo "Completed $task_group/$task_key"
}

if [[ "$group" == all || "$group" == common ]]; then
    for s in 1 32 64 128 256 512 1024; do run_case common "s$s" --S "$s"; done
fi
if [[ "$group" == all || "$group" == moe ]]; then
    for s in 1 32 64 128 256 512 1024; do run_case moe "s$s" --S "$s"; done
fi
if [[ "$group" == all || "$group" == attention ]]; then
    if [[ "$attention_slice" == all || "$attention_slice" == decode ]]; then
        for b in 1 32 64 128 256 512 1024; do
            for h in 4096 16384 32768 65536; do
                run_case attention "b${b}_q1_h${h}" --B "$b" --Q 1 --H "$h"
            done
        done
    fi
    if [[ "$attention_slice" == all || "$attention_slice" == extend ]]; then
        for q in 32 64 128 256 512 1024; do
            for h in 4096 16384 32768 65536; do
                run_case attention "b1_q${q}_h${h}" --B 1 --Q "$q" --H "$h"
            done
        done
    fi
fi
if [[ "$group" == all ]]; then
    "$task_python" -m experiments.deepseek_v32_observation.src.aggregate --run-id "$run_id" --data-dir "$task_data"
fi
