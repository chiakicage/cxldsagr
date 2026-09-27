#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash experiments/cpu_dram_bandwidth/scripts/run.sh [RUN_ID] [options]
  --threads LIST         Threads per NUMA node (default: 8,16,24,32,48,96)
  --node-sets LIST        Semicolon-separated node sets (default: '0;1;0,1')
  --mib-per-thread N      MiB per array per worker (default: 128; three arrays)
  --warmup N             Warmup repetitions per operation (default: 2)
  --reps N               Timed repetitions per operation (default: 7)
  --passes N             Full array traversals per repetition (default: 4)
  --ops LIST             read,nt-write,nt-copy,nt-triad (default: all four)
  --imc                  Require Intel IMC counters and capture DRAM clock evidence
Requires Linux x86 AVX-512, gcc, pthreads, numactl, and Python 3.
Runs configurations serially, pins workers, and first-touches local memory.
EOF
}
if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
run_id="$(date -u +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then run_id="$1"; shift; fi
threads="8,16,24,32,48,96"
node_sets="0;1;0,1"
mib=128
warmup=2
reps=7
passes=4
ops="read,nt-write,nt-copy,nt-triad"
imc_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --threads) threads="$2"; shift 2 ;;
    --node-sets) node_sets="$2"; shift 2 ;;
    --mib-per-thread) mib="$2"; shift 2 ;;
    --warmup) warmup="$2"; shift 2 ;;
    --reps) reps="$2"; shift 2 ;;
    --passes) passes="$2"; shift 2 ;;
    --ops) ops="$2"; shift 2 ;;
    --imc) imc_args=(--imc); shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid run ID: $run_id" >&2; exit 2
fi
output="experiments/cpu_dram_bandwidth/output"
data="$output/data/$run_id"
log="$output/log/$run_id"
if [[ -e "$data" || -e "$log" || -e "$output/profile/$run_id" ]]; then
  echo "Run already exists: $run_id" >&2; exit 2
fi
mkdir -p "$data/cases" "$data/source" "$log" "$output/profile/$run_id"
cp experiments/cpu_dram_bandwidth/src/*.py experiments/cpu_dram_bandwidth/src/dram_bench.c \
  experiments/cpu_dram_bandwidth/scripts/run.sh "$data/source/"
run_logged() {
  local label="$1"
  shift
  "$@" 2> >(tee "$log/$label.stderr.log" >&2) | tee "$log/$label.stdout.log"
}
run_logged compile gcc -O3 -march=native -std=c11 -pthread -Wall -Wextra \
  "$data/source/dram_bench.c" -lm -o "$data/dram_bench"
run_logged metadata python3 -m experiments.cpu_dram_bandwidth.src.measure metadata \
  --output "$data/system.json"
if [[ ${#imc_args[@]} -gt 0 ]]; then
  run_logged clock python3 -m experiments.cpu_dram_bandwidth.src.probe_imc \
    --output "$data/clock_counters.json"
fi
IFS=',' read -r -a thread_counts <<< "$threads"
IFS=';' read -r -a node_configs <<< "$node_sets"
for nodes in "${node_configs[@]}"; do
  for count in "${thread_counts[@]}"; do
    label="nodes_${nodes//,/_}_threads_${count}"
    run_logged "$label" python3 -m experiments.cpu_dram_bandwidth.src.measure run \
      --binary "$data/dram_bench" --label "$label" --nodes "$nodes" \
      --threads-per-node "$count" --mib-per-thread "$mib" --warmup "$warmup" \
      --reps "$reps" --passes "$passes" --ops "$ops" "${imc_args[@]}" \
      --output "$data/cases/$label.json"
  done
done
run_logged summary python3 -m experiments.cpu_dram_bandwidth.src.measure summarize --data-dir "$data"
