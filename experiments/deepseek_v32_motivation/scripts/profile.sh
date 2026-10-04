#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CXLDSAGR_SM90_BACKEND="${CXLDSAGR_SM90_BACKEND:-native}"
python_bin="${MOTIVATION_PYTHON:-$PWD/.venv/bin/python}"
args=("$@")
run_id=""
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --help|-h) exec "$python_bin" -m experiments.deepseek_v32_motivation.src.profile --help ;;
    --run-id) run_id="${args[i+1]:-}" ;;
    --run-id=*) run_id="${args[i]#*=}" ;;
    --output-dir|--output-dir=*|--nsys|--nsys=*)
      echo "profile.sh owns --output-dir and --nsys." >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Pass --run-id with letters, digits, underscores or hyphens." >&2
  exit 2
fi
capture_counts="$("$python_bin" - "${args[@]}" <<'PY'
import json
import sys
from experiments.deepseek_v32_motivation.src.profile import parser

args = parser().parse_args(sys.argv[1:])
metadata = json.loads((args.reference_run / 'metadata.json').read_text())
schemes = 1 if args.scheme else 4
requests = 2 * schemes
setups = schemes if metadata['config'].get('enable_compute_graphs', False) else 0
print(requests, requests + setups)
PY
)"
read -r request_capture_count capture_count <<<"$capture_counts"
base="$PWD/experiments/deepseek_v32_motivation/output"
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" ]]; then
    echo "Existing run: $run_id" >&2; exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-motivation-profile-${run_id}.XXXXXX")"
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
  --capture-range=cudaProfilerApi --capture-range-end="repeat:$capture_count" \
  --output "$staging/profile/capture" \
  "$python_bin" -m experiments.deepseek_v32_motivation.src.profile \
  --output-dir "$staging/data" --nsys "${args[@]}" \
  >"$staging/log/stdout.log" 2>"$staging/log/stderr.log"
"$python_bin" - "$staging/data/result.json" "$request_capture_count" "$capture_count" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
assert result['accepted'] and result['captures'] == int(sys.argv[2])
assert result['nsys_captures'] == int(sys.argv[3])
assert result['exact_output_count'] > result['captures']
assert result['source_sha256'] != result['reference_source_sha256']
PY
for ((capture=1; capture<=capture_count; capture++)); do
  nsys export --type sqlite --output "$staging/data/capture_${capture}.sqlite" \
    "$staging/profile/capture.${capture}.nsys-rep" \
    >"$staging/log/export_${capture}.stdout.log" 2>"$staging/log/export_${capture}.stderr.log"
done
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $base/data/$run_id"
