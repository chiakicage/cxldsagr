#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash experiments/gr_serving/scripts/profile.sh [RUN_ID] --latency-data DIR [--num-users N] [--device cuda:0] [--nsys]

Run only after the formal latency process terminates and its data is accepted.
Profiles NOSA layer 31 on the first three revisits using saved prefix/candidate geometry.
--num-users selects a saved population (default: 8 if present, otherwise smallest).
Each serialized/overlap sample constructs its own sparse prefix from empty cache,
then compares every profiled hidden value to its unprofiled control and formal HBM.
Native schema-3 intervals are sufficient; --nsys adds a supplemental raw capture
and SQLite export. The existing overlap SQLite CLI requires a third resident mode
and is not used to validate this two-mode diagnostic.

Choose CUDA_VISIBLE_DEVICES to select the same physical GPU as the formal run.
TMPDIR selects staging storage; prefer a shared-mount directory with free space.
Only accepted results are copied into experiments/gr_serving/output/{data,log,profile}.
Errors and partial results remain in the printed staging directory outside experiments.
90% is a per-sample claim gate, never a reason to discard a valid measurement.
HELP
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
run_id="gr_serving_nosa_profile_$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi

capture_nsys=false
forwarded=()
for argument in "$@"; do
  option="${argument%%=*}"
  if [[ "$option" == --nsys ]]; then
    if [[ "$argument" != --nsys ]]; then
      echo "Use --nsys without a value" >&2
      exit 2
    fi
    capture_nsys=true
  elif [[ "$option" == --output-dir || "$option" == --run-id || "$option" == --verify-only ]]; then
    echo "The script owns --run-id, --output-dir and --verify-only" >&2
    exit 2
  else
    forwarded+=("$argument")
  fi
done

output="experiments/gr_serving/output"
for category in data log profile; do
  if [[ -e "$output/$category/$run_id" ]]; then
    echo "Run already exists: $run_id" >&2
    exit 2
  fi
done
if [[ "$capture_nsys" == true ]]; then command -v nsys >/dev/null; fi
staging="$(mktemp -d "${TMPDIR:-/tmp}/gr-serving-profile-${run_id}.XXXXXX")"
mkdir -p "$staging/log" "$staging/profile"
published=false
claimed_categories=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for category in "${claimed_categories[@]}"; do rm -rf -- "$output/$category/$run_id"; done
    if [[ -f "$staging/log/profile.stderr.log" ]]; then cat "$staging/log/profile.stderr.log" >&2; fi
    echo "Failed (status $status); diagnostics remain outside experiments: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command=(.venv/bin/python -m experiments.gr_serving.src.profile
  --run-id "$run_id" --output-dir "$staging/data" "${forwarded[@]}")
if [[ "$capture_nsys" == true ]]; then
  export GR_SERVING_NSYS_CAPTURE=1
  nsys --version >"$staging/log/nsys.version.log" 2>"$staging/log/nsys.version.stderr.log"
  command=(nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
    --force-overwrite=false --output "$staging/profile/nosa" "${command[@]}")
else
  export GR_SERVING_NSYS_CAPTURE=0
fi
"${command[@]}" \
  2>"$staging/log/profile.stderr.log" | tee "$staging/log/profile.stdout.log"

# A profiler wrapper's exit status alone does not authenticate target completion.
.venv/bin/python -m experiments.gr_serving.src.profile \
  --run-id "$run_id" --output-dir "$staging/data" --verify-only \
  >"$staging/log/verify.stdout.log" 2>"$staging/log/verify.stderr.log"
if [[ "$capture_nsys" == true ]]; then
  nsys export --type sqlite --force-overwrite=false --output "$staging/data/profile.sqlite" \
    "$staging/profile/nosa.nsys-rep" \
    >"$staging/log/nsys_export.stdout.log" 2>"$staging/log/nsys_export.stderr.log"
fi
for category in data log profile; do
  mkdir -p "$output/$category"
  mkdir "$output/$category/$run_id"
  claimed_categories+=("$category")
  cp -a -- "$staging/$category/." "$output/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed: $output/data/$run_id"
