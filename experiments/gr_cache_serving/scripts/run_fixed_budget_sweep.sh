#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/run_fixed_budget_sweep.sh prepare|status|run|report RUN_GROUP"
  echo "Fixed five layers, 64K/1K, 512 requests per population, Beauty seed 42."
  echo "72 GiB GPU ceiling; fixed 93-user HBM-only / 128-user Host cache; 66 GiB Host budget."
  echo "Offload device pools stay at 66624 tokens/layer, not optimized to use all HBM."
  echo "SPARSEGR_SWEEP_USERS / SPARSEGR_SWEEP_MODES select subsets; no overwrite or automatic retry."
  exit 0
fi
if [[ $# != 2 || ! "$2" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Expected phase and alphanumeric run group; use --help" >&2
  exit 2
fi
phase="$1"
group="$2"
case "$phase" in prepare|status|run|report) ;; *) echo "Unknown phase" >&2; exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
read -r -a users <<< "${SPARSEGR_SWEEP_USERS:-64 128 256 384 512}"
read -r -a modes <<< "${SPARSEGR_SWEEP_MODES:-resident sparse_sync echo_gr_adapted dense_prefetch}"
base="$PWD/experiments/gr_cache_serving/output/data"
python="$PWD/3rdparty/ECHO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
for population in "${users[@]}"; do
  case "$population" in 64|128|256|384|512) ;; *) echo "Invalid population" >&2; exit 2 ;; esac
done
for mode in "${modes[@]}"; do
  case "$mode" in resident|sparse_sync|echo_gr_adapted|dense_prefetch) ;;
    *) echo "Invalid mode" >&2; exit 2 ;; esac
done
if [[ "$phase" == status ]]; then
  exec python3 -m experiments.gr_cache_serving.src.sweep_status "$group" \
    --profile fixed_budget_512 --base "$base" --populations "${users[@]}" --modes "${modes[@]}"
fi
runs=()
for population in "${users[@]}"; do
  trace="$base/${group}_input_u${population}"
  if [[ "$phase" == prepare ]]; then
    "$python" -m experiments.gr_cache_serving.src.workload \
      --tokenizer /mnt/nfs/share/models/DeepSeek-V3.2 \
      --prefix-tokens 65536 --candidate-tokens 1024 --num-users "$population" \
      --count 512 --history-cache-users "$population" \
      --curve-dataset beauty --sampling weighted --arrival poisson --qps 1 --seed 42 \
      --output-dir "$trace"
    continue
  fi
  for mode in "${modes[@]}"; do
    run_id="${group}_u${population}_${mode}_r1"
    runs+=("$base/$run_id")
    if [[ "$phase" == run ]]; then
      SPARSEGR_REPLAY_TIMEOUT_S="${SPARSEGR_REPLAY_TIMEOUT_S:-3600}" \
        bash experiments/gr_cache_serving/scripts/run_replay.sh "$mode" "$run_id" \
        --profile fixed_budget_512 --trace "$trace/requests.jsonl" --layers 5 --host-users 128 \
        --hbm-budget-gib 72 --host-cache-budget-gib 66 --workspace-reserve-gib 8 \
        --non-torch-reserve-gib 2 --device-cache-tokens 66624
    fi
  done
done
if [[ "$phase" == report ]]; then
  uv run --no-project --python "$python" --with matplotlib==3.10.7 \
    python -m experiments.gr_cache_serving.src.report_capacity "${runs[@]}" \
    --output-dir "experiments/gr_cache_serving/report/$group"
fi
