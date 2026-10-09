#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export DG_JIT_WITH_LINEINFO=1
runner_source="$script_dir/gap_profile.sh"
runner_invocation=("$0" "$@")
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Minimal node profile: --trace-warmups N (default 1) warms each capture before restoring and measuring."
  python -m experiments.deepseek_v32_mfu.src.gap_profile --help
  exit 0
fi
run_id="${MFU_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_minimal_node_gap}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid MFU_RUN_ID" >&2
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
staging="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-minimal-node-${run_id}.XXXXXX")"
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
  --output "$staging/profile/minimal" \
  python -m experiments.deepseek_v32_mfu.src.gap_profile \
  --run-id "$run_id" --output "$staging/data" --nsys "$@")
"${command[@]}" >"$staging/log/stdout.log" 2>"$staging/log/stderr.log"
validate_and_record_runner profile minimal_node_model_scopes
mapfile -t captures < <(python - "$staging/data/result.json" <<'PY'
import json, sys
for index, label in enumerate(json.load(open(sys.argv[1]))['nsys_capture_order'], 1):
    print(index, label)
PY
)
for entry in "${captures[@]}"; do
  read -r capture label <<< "$entry"
  nsys export --type sqlite --output "$staging/data/capture_${capture}.sqlite" \
    "$staging/profile/minimal.${capture}.nsys-rep" \
    >"$staging/log/export_${capture}.stdout.log" 2>"$staging/log/export_${capture}.stderr.log"
done
python -m experiments.deepseek_v32_mfu.src.launch_gap \
  --profile-run "$staging/data" --output "$staging/data/gap_audit.json" \
  >"$staging/log/gap_audit.stdout.log" 2>"$staging/log/gap_audit.stderr.log"
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $base/data/$run_id"
