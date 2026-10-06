#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export DG_JIT_WITH_LINEINFO=1
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  python -m experiments.deepseek_v32_mfu.src.measure --help
  exit 0
fi
if [[ "${ECHO_NSYS:-0}" != 0 ]]; then
  echo "Use scripts/profile_layers.sh for independent profiling" >&2
  exit 2
fi
mode="${ECHO_MODE:-bench}"
forward=()
while (($#)); do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --mode=*) mode="${1#*=}"; shift ;;
    *) forward+=("$1"); shift ;;
  esac
done
if [[ "$mode" != check && "$mode" != bench ]]; then
  echo "Mode must be check or bench" >&2; exit 2
fi
set -- "${forward[@]}"
for argument in "$@"; do
  option="${argument%%=*}"
  # argparse accepts unambiguous option prefixes as well as --option=value.
  if [[ "$option" == --* && ( --run-id == "$option"* || --output == "$option"* || --profile-dir == "$option"* || --nsys == "$option"* ) ]]; then
    echo "The script owns --run-id, --output, --profile-dir and --nsys; use ECHO_RUN_ID" >&2
    exit 2
  fi
done
run_id="${MFU_RUN_ID:-${ECHO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_mfu_layers3_64k_a128}}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Run ID must contain only letters, digits, dots, underscores and hyphens" >&2
  exit 2
fi
base="experiments/deepseek_v32_mfu/output"
if [[ "$mode" == check ]]; then
  base="${TMPDIR:-/tmp}/cxldsagr-checks/deepseek_v32_mfu"
fi
for category in data log profile; do
  if [[ -e "$base/$category/$run_id" || -L "$base/$category/$run_id" ]]; then
    echo "Run output already exists: $base/$category/$run_id" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-echo-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do rm -rf -- "$base/$category/$run_id"; done
    echo "Failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
command=(python -m experiments.deepseek_v32_mfu.src.measure
  --mode "$mode" --run-id "$run_id" --output "$staging/data" "$@")
"${command[@]}" >"$staging/log/stdout.log" 2>"$staging/log/stderr.log"
python - "$staging/data/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result['accepted'] and result['num_layers'] == 3
assert result['schema_version'] == 3 and result['mode'] in ('check', 'bench')
if result['mode'] == 'check':
    assert len(result['correctness']) == 13
    from pathlib import Path
    assert Path(sys.argv[1]).with_name('receipt.json').is_file()
else:
    assert not result['correctness'] and result['validation_receipt']
PY
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $base/data/$run_id"
