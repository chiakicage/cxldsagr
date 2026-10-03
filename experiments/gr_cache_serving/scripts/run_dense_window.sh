#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/run_dense_window.sh run|report RUN_GROUP"
  echo "Five layers, fixed budgets, 64/128 users, 512 requests; paired dense schedules."
  echo "SPARSEGR_DENSE_INPUT_GROUP defaults to five_layer_fixed512_20261001."
  exit 0
fi
if [[ $# != 2 || ! "$2" =~ ^[A-Za-z0-9_-]+$ ]]; then exit 2; fi
case "$1" in run|report) ;; *) exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
base="$PWD/experiments/gr_cache_serving/output/data"
input_group="${SPARSEGR_DENSE_INPUT_GROUP:-five_layer_fixed512_20261001}"
runs=()
for users in 64 128; do
  # Reverse the pair order at the second size, without changing the request order.
  schedules=(layer_end attention_window)
  if [[ "$users" == 128 ]]; then schedules=(attention_window layer_end); fi
  for schedule in "${schedules[@]}"; do
    id="${2}_u${users}_${schedule}_r1"
    runs+=("$base/$id")
    if [[ "$1" == run ]]; then
      SPARSEGR_REPLAY_TIMEOUT_S=3600 bash experiments/gr_cache_serving/scripts/run_replay.sh \
        dense_prefetch "$id" --dense-prefetch-schedule "$schedule" --dense-prefetch-transport cpu_staging \
        --profile fixed_budget_512 --trace "$base/${input_group}_input_u${users}/requests.jsonl" \
        --layers 5 --host-users 128 --hbm-budget-gib 72 --host-cache-budget-gib 66 \
        --workspace-reserve-gib 8 --non-torch-reserve-gib 2 --device-cache-tokens 66624
    fi
  done
done
if [[ "$1" == report ]]; then
  uv run --no-project --python 3rdparty/ECHO/.venv/bin/python --with matplotlib==3.10.7 \
    python -m experiments.gr_cache_serving.src.report_dense_window "${runs[@]}" \
    --output-dir "experiments/gr_cache_serving/report/$2"
fi
