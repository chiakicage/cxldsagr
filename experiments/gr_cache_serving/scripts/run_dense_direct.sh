#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/run_dense_direct.sh run|report RUN_GROUP"
  echo "Five layers, 64/128 users, fixed budgets, 512 requests: DSA vs GPU-direct Dense."
  exit 0
fi
if [[ $# != 2 || ! "$2" =~ ^[A-Za-z0-9_-]+$ ]]; then exit 2; fi
case "$1" in run|report) ;; *) exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
base="$PWD/experiments/gr_cache_serving/output/data"
runs=()
for users in 64 128; do
  modes=(sparse_sync dense_prefetch)
  if [[ "$users" == 128 ]]; then modes=(dense_prefetch sparse_sync); fi
  for mode in "${modes[@]}"; do
    id="${2}_u${users}_${mode}_r1"
    runs+=("$base/$id")
    flags=()
    if [[ "$mode" == dense_prefetch ]]; then
      flags=(--dense-prefetch-transport gpu_direct --dense-prefetch-schedule attention_window)
    fi
    if [[ "$1" == run ]]; then
      SPARSEGR_REPLAY_TIMEOUT_S=3600 bash experiments/gr_cache_serving/scripts/run_replay.sh \
        "$mode" "$id" "${flags[@]}" --profile fixed_budget_512 \
        --trace "$base/five_layer_fixed512_20261001_input_u${users}/requests.jsonl" \
        --layers 5 --host-users 128 --hbm-budget-gib 72 --host-cache-budget-gib 66 \
        --workspace-reserve-gib 8 --non-torch-reserve-gib 2 --device-cache-tokens 66624
    fi
  done
done
if [[ "$1" == report ]]; then
  uv run --no-project --python 3rdparty/ECHO/.venv/bin/python --with matplotlib==3.10.7 \
    python -m experiments.gr_cache_serving.src.report_capacity "${runs[@]}" \
    --allow-partial --output-dir "experiments/gr_cache_serving/report/$2"
fi
