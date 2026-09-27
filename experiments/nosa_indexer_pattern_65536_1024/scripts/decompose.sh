#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash experiments/nosa_indexer_pattern_65536_1024/scripts/decompose.sh [RUN_ID] [--pattern-data-dir PATH] [--bandwidth-gbps NUMBER]"
  echo "Split sink/local and query-aware unions, report overlap and additional fetch at 50 GB/s by default."
  echo "Reads existing arrays only; defaults to the 64-block baseline. No GPU execution."
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="components_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then
  run_id="$1"
  shift
fi
arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --pattern-data-dir|--bandwidth-gbps)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      arguments+=("$1" "$2")
      shift 2 ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
output="experiments/nosa_indexer_pattern_65536_1024/output"
for category in log data profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id; choose a new RUN_ID" >&2
    exit 2
  fi
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/nosa-components-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
claimed_categories=()
published=false
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do
      rm -rf -- "$output/$category/$run_id"
    done
    echo "Calculation failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
.venv/bin/python -m experiments.nosa_indexer_pattern_65536_1024.src.decompose \
  --run-id "$run_id" --output-dir "$staging/data" "${arguments[@]}" \
  2> >(tee "$staging/log/decompose.stderr.log" >&2) | tee "$staging/log/decompose.stdout.log"
mkdir -p "$output/data" "$output/log" "$output/profile"
for category in data log profile; do
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Calculated components: $output/data/$run_id"
