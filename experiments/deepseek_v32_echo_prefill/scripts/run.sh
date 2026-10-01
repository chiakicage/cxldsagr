#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  python -m experiments.deepseek_v32_echo_prefill.src.measure --help
  exit 0
fi
for argument in "$@"; do
  option="${argument%%=*}"
  # argparse accepts unambiguous option prefixes as well as --option=value.
  if [[ "$option" == --* && ( --run-id == "$option"* || --output == "$option"* || --profile-dir == "$option"* || --nsys == "$option"* ) ]]; then
    echo "The script owns --run-id, --output, --profile-dir and --nsys; use ECHO_RUN_ID/ECHO_NSYS" >&2
    exit 2
  fi
done
run_id="${ECHO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_echo_64k_1k}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Run ID must contain only letters, digits, dots, underscores and hyphens" >&2
  exit 2
fi
base="experiments/deepseek_v32_echo_prefill/output"
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
command=(python -m experiments.deepseek_v32_echo_prefill.src.measure
  --run-id "$run_id" --output "$staging/data"
  --profile-dir "$staging/profile" "$@")
if [[ "${ECHO_NSYS:-0}" == 1 ]]; then
  command=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
    --capture-range=cudaProfilerApi --capture-range-end=repeat:2
    --output "$staging/profile/extend" "${command[@]}" --nsys)
fi
"${command[@]}" >"$staging/log/stdout.log" 2>"$staging/log/stderr.log"
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $base/data/$run_id"
