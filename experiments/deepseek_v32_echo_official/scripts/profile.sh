#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CXLDSAGR_SM90_BACKEND="${CXLDSAGR_SM90_BACKEND:-native}"
python_bin="${ECHO_OFFICIAL_PYTHON:-$PWD/.venv/bin/python}"
args=("$@")
run_id=""
captures=4
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --help|-h) exec "$python_bin" -m experiments.deepseek_v32_echo_official.src.profile --help ;;
    --run-id) run_id="${args[i+1]:-}" ;;
    --run-id=*) run_id="${args[i]#*=}" ;;
    --scheme|--scheme=*) captures=2 ;;
    --output-dir|--output-dir=*|--nsys|--nsys=*)
      echo "profile.sh owns --output-dir and --nsys" >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Pass a valid --run-id" >&2; exit 2
fi
base="$PWD/experiments/deepseek_v32_echo_official/output"
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" ]]; then
    echo "Existing run: $run_id" >&2; exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/official-profile-${run_id}.XXXXXX")"
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
nsys --version >"$staging/log/nsys_version.txt" 2>"$staging/log/nsys_version.stderr.log"
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end="repeat:$captures" \
  --output "$staging/profile/capture" \
  "$python_bin" -m experiments.deepseek_v32_echo_official.src.profile \
  --output-dir "$staging/data" --nsys "${args[@]}" \
  >"$staging/log/stdout.log" 2>"$staging/log/stderr.log"
"$python_bin" - "$staging/data/result.json" "$captures" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result['accepted'] and result['captures'] == int(sys.argv[2])
assert result['checked_requests'] >= result['captures']
PY
for ((capture=1; capture<=captures; capture++)); do
  nsys export --type sqlite --output "$staging/data/capture_${capture}.sqlite" \
    "$staging/profile/capture.${capture}.nsys-rep" \
    >"$staging/log/export_${capture}.stdout.log" 2>"$staging/log/export_${capture}.stderr.log"
done
"$python_bin" -m experiments.deepseek_v32_echo_official.src.profile_report \
  --run-dir "$staging/data" --output-dir "$staging/data/analysis" \
  >"$staging/log/analysis.stdout.log" 2>"$staging/log/analysis.stderr.log"
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $base/data/$run_id"
