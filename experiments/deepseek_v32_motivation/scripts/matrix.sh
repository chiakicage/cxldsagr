#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."

usage() {
  cat <<'EOF'
Usage: bash experiments/deepseek_v32_motivation/scripts/matrix.sh --run-id ID [--dry-run]

Run independent check and bench processes for each H x A point, in order:
  H = 4096, 16384, 65536; A = 128, 256, 512, 1024.
Use four methods, P=65536, NH=16777216, 16 users, two sequential rounds,
chunk=1024, seed=42 and compute islands. GPU/CPU affinity comes from the caller.
Each point has its own check receipt. Stop on the first failure; do not retry
or overwrite an existing run. --dry-run prints commands without running them.
EOF
}

run_id=""
dry_run=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --run-id)
      if (($# < 2)); then usage >&2; exit 2; fi
      run_id="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Pass --run-id with letters, digits, underscores or hyphens." >&2
  exit 2
fi

experiment="$PWD/experiments/deepseek_v32_motivation"
checks="${TMPDIR:-/tmp}/cxldsagr-checks/deepseek_v32_motivation"
histories=(4096 16384 65536)
candidates=(128 256 512 1024)

for history in "${histories[@]}"; do
  for candidate in "${candidates[@]}"; do
    point="${run_id}_h${history}_a${candidate}"
    for path in "$checks/data/${point}_check" "$checks/log/${point}_check" \
      "$experiment/output/data/${point}_bench" "$experiment/output/log/${point}_bench"; do
      if [[ -e "$path" ]]; then
        echo "Run path already exists: $path" >&2
        exit 2
      fi
    done
  done
done

for history in "${histories[@]}"; do
  for candidate in "${candidates[@]}"; do
    point="${run_id}_h${history}_a${candidate}"
    common=(--history-tokens "$history" --candidate-tokens "$candidate"
      --sparse-pool-tokens 65536 --host-arena-tokens 16777216
      --num-users 16 --rounds 2 --chunk-size 1024 --seed 42 --compute-graphs)
    check=(bash "$script_dir/run.sh" --mode check --run-id "${point}_check" "${common[@]}")
    bench=(bash "$script_dir/run.sh" --mode bench --run-id "${point}_bench" "${common[@]}"
      --validation-receipt "$checks/data/${point}_check/receipt.json")
    if ((dry_run)); then
      printf '%q ' "${check[@]}"; printf '\n'
      printf '%q ' "${bench[@]}"; printf '\n'
    else
      "${check[@]}"
      "${bench[@]}"
    fi
  done
done
