#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash experiments/nosa_framework_refactor/scripts/run.sh [RUN_ID] [cpu|gpu|all]"
  echo "Default mode: all. CUDA and NOSA weights are required for gpu/all."
  echo "NOSA_MODEL_PATH overrides the default /mnt/ssd-wlcb/chenkaiqi/NOSA-8B checkpoint."
  exit 0
fi
if [[ $# -gt 2 ]]; then
  echo "Expected RUN_ID and optional mode; use --help" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
run_id="${1:-$(date -u +%Y%m%d_%H%M%S)}"
mode="${2:-all}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ || ! "$mode" =~ ^(cpu|gpu|all)$ ]]; then
  echo "Invalid RUN_ID or mode; use --help" >&2
  exit 2
fi
output="experiments/nosa_framework_refactor/output"
for kind in log data profile; do
  if [[ -e "$output/$kind/$run_id" ]]; then
    echo "Run already exists: $run_id; choose a new RUN_ID" >&2
    exit 2
  fi
done
mkdir -p "$output/log/$run_id" "$output/data/$run_id" "$output/profile/$run_id"
run_logged() {
  local step="$1"
  shift
  "$@" 2> >(tee "$output/log/$run_id/$step.stderr.log" >&2) | tee "$output/log/$run_id/$step.stdout.log"
}
if [[ "$mode" == cpu || "$mode" == all ]]; then
  run_logged cpu env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
    models/nosa/tests cache/tests executor/tests serving/tests GR/tests \
    experiments/nosa_gr_65536_1024/tests experiments/legacy/nosa_gr_forward/tests \
    --basetemp "$output/data/$run_id/pytest" -q -rs -p no:cacheprovider
fi
if [[ "$mode" == gpu || "$mode" == all ]]; then
  run_logged gpu .venv/bin/python -m experiments.nosa_framework_refactor.src.gpu_smoke \
    --output-dir "$output/data/$run_id/gpu" --profile-dir "$output/profile/$run_id/gpu"
  checkpoint="${NOSA_MODEL_PATH:-/mnt/ssd-wlcb/chenkaiqi/NOSA-8B}"
  run_logged serving .venv/bin/python -m serving.run_gr --model-path "$checkpoint" \
    --device cuda --count 2 --num-users 1 --user-lengths 256 --item-lengths 128 --prefill-chunk-size 128
  run_logged generation .venv/bin/python -m models.nosa.infer --model-path "$checkpoint" \
    --device cuda --prompt '请用一句话解释 KV cache。' --disable-thinking \
    --max-new-tokens 4 --prefill-chunk-size 8
  run_logged analyze .venv/bin/python -m experiments.nosa_framework_refactor.src.analyze_cli \
    --log-dir "$output/log/$run_id" --output "$output/data/$run_id/real_cli_smoke.json"
fi
