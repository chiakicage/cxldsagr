#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export DG_JIT_WITH_LINEINFO=1
runner_source="$script_dir/profile_layers.sh"
runner_invocation=("$0" "$@")
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  python -m experiments.deepseek_v32_mfu.src.profile_layers --help
  exit 0
fi
run_id="${MFU_RUN_ID:-${ECHO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_mfu_layers3}}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid ECHO_RUN_ID" >&2
  exit 2
fi
for argument in "$@"; do
  option="${argument%%=*}"
  if [[ "$option" == --* && ( --run-id == "$option"* || --output == "$option"* || --nsys == "$option"* ) ]]; then
    echo "The script owns --run-id, --output and --nsys" >&2
    exit 2
  fi
done
base="experiments/deepseek_v32_mfu/output"
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" || -L "$base/$category/$run_id" ]]; then echo "Existing run: $run_id" >&2; exit 2; fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-layers3-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
echo "Staging: $staging"
published=false
claimed=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed[@]}"; do rm -rf -- "$base/$category/$run_id"; done
    echo "Failed (status $status); diagnostics: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$script_dir/runner_common.sh"
nsys --version >"$staging/log/nsys_version.txt" 2>"$staging/log/nsys_version.stderr.log"
command=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output "$staging/profile/layers3" \
  python -m experiments.deepseek_v32_mfu.src.profile_layers \
  --run-id "$run_id" --output "$staging/data" --nsys "$@")
"${command[@]}" >"$staging/log/stdout.log" 2>"$staging/log/stderr.log"
# Profiler exit status alone does not prove the target completed successfully.
validate_and_record_runner profile
mapfile -t captures < <(python - "$staging/data/result.json" <<'PYCODE'
import json, sys
for index, label in enumerate(json.load(open(sys.argv[1]))['nsys_capture_order'], 1):
    print(index, label)
PYCODE
)
analysis_args=()
for entry in "${captures[@]}"; do
  read -r capture label <<< "$entry"
  nsys export --type sqlite --output "$staging/data/capture_${capture}.sqlite" \
    "$staging/profile/layers3.${capture}.nsys-rep" \
    >"$staging/log/export_${capture}.stdout.log" 2>"$staging/log/export_${capture}.stderr.log"
  if [[ "$label" == graph_setup || "$label" == */extend_graph_setup ]]; then
    analysis_args+=(--graph-setup "$staging/data/capture_${capture}.sqlite")
  else
    analysis_args+=(--sqlite "$staging/data/capture_${capture}.sqlite")
  fi
done
python -m experiments.deepseek_v32_mfu.src.operator_report "${analysis_args[@]}" \
  --calls "$staging/data/operator_calls.json" --output-dir "$staging/data/analysis" \
  >"$staging/log/analysis.stdout.log" 2>"$staging/log/analysis.stderr.log"
python - "$staging/data/result.json" "$staging/data/analysis/analysis.json" <<'PY'
import sys
from experiments.deepseek_v32_mfu.src.run_contract import validate_operator_analysis
validate_operator_analysis(sys.argv[1], sys.argv[2])
PY
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $base/data/$run_id"
