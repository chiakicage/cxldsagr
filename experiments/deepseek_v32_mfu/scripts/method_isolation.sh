#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage: method_isolation.sh --request PATH [--model PATH]

Run H65536+A1 with one cache method per fresh Python process on GPU0/CPU0-7.
The explicit saved request must contain the seed42 H64K history and its first
A128 suffix token. All four checks finish first, with HBM as the CPU reference.
Each method then has its own clean benchmark, minimal node profile, and
operator profile, bound to its own check and benchmark.

Fixed configuration: real layers0-2; history chunk1024; extend chunk1;
P=NH=65600; cold full-extend and compute graphs; one method warmup;
three prefill samples; five extend samples; one minimal trace warmup.
The model defaults to /preset-models. Set MFU_RUN_ID to a fresh cohort ID.

The runner sets the five execution variables of the HINT/FREE formal cohort.
Other tracked execution overrides are rejected, including allocator overrides.
PATH is inherited and receives the same orchestrator and child venv prefixes
as shape_matrix.sh. No JIT cache is cleared and no provider is preloaded.

Checks remain under ${TMPDIR:-/tmp}/cxldsagr-checks/deepseek_v32_mfu.
Successful children publish their own output/{data,log,profile}/<child-id>.
The cohort manifest publishes only after all 16 children complete:
  output/data/<cohort-id>/manifest.json
A failed child stops execution, retains earlier completed child runs, and
leaves incomplete diagnostics outside experiments. CPU cohort audit and
report generation are separate steps; this entrypoint does not publish figures.
EOF
}
invocation=("$0" "$@")
request=""
model=/preset-models
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --request|--model)
      if (($# < 2)) || [[ "$2" == --* || -z "$2" ]]; then
        echo "Missing value for $1" >&2; exit 2
      fi
      if [[ "$1" == --request ]]; then request="$2"; else model="$2"; fi
      shift 2 ;;
    --request=*) request="${1#*=}"; shift ;;
    --model=*) model="${1#*=}"; shift ;;
    *) echo "Unsupported or cohort-owned option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [[ -z "$request" || ! -f "$request" || -z "$model" ]]; then
  echo "An existing --request and a nonempty --model are required" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0 DG_JIT_WITH_LINEINFO=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
python - <<'PY'
import os
prefixes = ('CXLDSAGR_', 'DG_', 'DJ_', 'FLASHINFER_', 'CUTE_DSL_', 'TRITON_',
            'PYTORCH_', 'OMP_', 'MKL_', 'OPENBLAS_')
expected = {'CUDA_VISIBLE_DEVICES': '0', 'DG_JIT_WITH_LINEINFO': '1',
            'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
            'TRITON_PTXAS_BLACKWELL_PATH': '/usr/local/cuda/bin/ptxas'}
actual = {key: value for key, value in os.environ.items()
          if key.startswith(prefixes)
          or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'CUDA_LAUNCH_BLOCKING')}
if actual != expected:
    raise ValueError('Unexpected execution environment overrides: ' + repr(actual))
PY
run_id="${MFU_RUN_ID:-$(date -u +%Y%m%dT%H%M%S%NZ)_method_isolation_h65536_a1}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid MFU_RUN_ID" >&2; exit 2
fi
base="$PWD/experiments/deepseek_v32_mfu/output"
check_base="${TMPDIR:-/tmp}/cxldsagr-checks/deepseek_v32_mfu"
methods=(hbm echo serial_sparse dense_prefetch)
require_fresh() {
  local root="$1" id="$2" category
  for category in data log profile; do
    if [[ -e "$root/$category/$id" || -L "$root/$category/$id" ]]; then
      echo "Run output already exists: $root/$category/$id" >&2; exit 2
    fi
  done
}
require_fresh "$base" "$run_id"
for method in "${methods[@]}"; do
  require_fresh "$check_base" "${run_id}_${method}_check"
  for phase in bench profile operators; do require_fresh "$base" "${run_id}_${method}_${phase}"; done
done
staging="$(mktemp -d "${TMPDIR:-/tmp}/deepseek-isolation-${run_id}.XXXXXX")"
mkdir "$staging/data" "$staging/log" "$staging/profile"
published=false
claimed=()
finish() {
  local status=$?
  if [[ "$published" != true ]]; then
    for path in "${claimed[@]}"; do rm -rf -- "$path"; done
    echo "Method cohort stopped (status $status); diagnostics: $staging" >&2
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python - "$staging/data" "$base/data/$run_id" "$request" "$run_id" "$model" \
  "$script_dir" "${invocation[@]}" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

staging, published, request_path, run_id, model, scripts, *invocation = sys.argv[1:]
staging, published, request_path, scripts = map(Path, (staging, published, request_path, scripts))
request = json.loads(request_path.read_text())
if (request.get('stable_prefix_tokens') != 65536
        or request.get('candidate_suffix_tokens') != 1
        or len(request.get('input_ids', [])) != 65537
        or request.get('token_source', {}).get('seed') != 42
        or request['input_ids'][-1] != 111090):
    raise ValueError('Request does not match the H64K seed42 single-token workload')
prepared = staging / 'request.json'
prepared.write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
runner_sources = {str(path.relative_to(Path.cwd())): digest(path)
                  for name in ('method_isolation.sh', 'runner_common.sh', 'run.sh',
                               'gap_profile.sh', 'profile_layers.sh')
                  for path in (scripts / name,)}
manifest = {
    'schema_version': 1,
    'kind': 'deepseek-v32-method-isolation',
    'run_id': run_id,
    'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'configuration': {
        'prefix_tokens': 65536, 'extend_tokens': 1, 'num_layers': 3,
        'chunk_size': 1024, 'extend_chunk_size': 1,
        'sparse_pool_tokens': 65600, 'host_arena_tokens': 65600,
        'workspace_query_tokens': 1024, 'hbm_cache_budget_bytes': 24 * 2**30,
        'dram_cache_budget_bytes': 64 * 2**30,
        'extend_residency': 'cold', 'compute_graphs': True, 'extend_graph': True,
        'warmups': 1, 'prefill_repeats': 3, 'repeats': 5, 'trace_warmups': 1,
        'seed': 42, 'physical_device': '0', 'cpu_affinity': list(range(8)),
        'model': str(Path(model).resolve()),
        'method_isolation': 'fresh-process-one-method-v1',
    },
    'request': {'path': str(published / 'request.json'), 'sha256': digest(prepared),
                'source_path': str(request_path.resolve()), 'source_sha256': digest(request_path)},
    'runner': {'source': str((scripts / 'method_isolation.sh').relative_to(Path.cwd())),
               'sha256': digest(scripts / 'method_isolation.sh'),
               'invocation': invocation, 'sources': runner_sources,
               'environment': {key: os.environ.get(key) for key in
                               ('PATH', 'CUDA_VISIBLE_DEVICES', 'DG_JIT_WITH_LINEINFO',
                                'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                'TRITON_PTXAS_BLACKWELL_PATH')}},
    'methods': {method: {} for method in ('hbm', 'echo', 'serial_sparse', 'dense_prefetch')},
    'execution_order': [],
}
(staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
shape=(--model "$model" --request "$staging/data/request.json" --physical-device 0
  --prefix 65536 --extend 1 --chunk-size 1024 --extend-chunk-size 1
  --sparse-pool-tokens 65600 --host-arena-tokens 65600 --workspace-query-tokens 1024
  --hbm-cache-budget-gib 24 --dram-cache-budget-gib 64 --compute-graphs --extend-graph
  --extend-residency cold --warmups 1 --prefill-repeats 3 --repeats 5 --seed 42)

run_child() {
  local method="$1" phase="$2" runner="$3"; shift 3
  local id="${run_id}_${method}_${phase}" started root="$base"
  if [[ "$phase" == check ]]; then root="$check_base"; fi
  local child_command=(env "MFU_RUN_ID=$id" taskset -c 0-7 bash "$script_dir/$runner"
    "${shape[@]}" --method "$method" "$@")
  started="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  echo "$method: $phase"
  "${child_command[@]}" >"$staging/log/${id}.stdout.log" 2>"$staging/log/${id}.stderr.log"
  python - "$staging/data/manifest.json" "$method" "$phase" "$id" "$root" "$started" \
    "${child_command[@]}" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

from experiments.deepseek_v32_mfu.src.run_contract import validate_completed_result

manifest_path, method, phase, run_id, root, started, *invocation = sys.argv[1:]
manifest_path, root = Path(manifest_path), Path(root)
manifest = json.loads(manifest_path.read_text())
directory = (root / 'data' / run_id).resolve(strict=True)
result = validate_completed_result(
    directory / 'result.json', expected_mode=phase if phase in ('check', 'bench') else 'profile',
    expected_profile_detail='minimal_node_model_scopes' if phase == 'profile' else None,
)
if (result['schema_version'] != 4 or result['methods'] != [method]
        or result['run_id'] != run_id or result['request_sha256'] != manifest['request']['sha256']):
    raise ValueError('Child identity differs from its cohort method/request/run binding')
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
runner_path = directory / 'runner.json'
runner = json.loads(runner_path.read_text())
if runner['run_id'] != run_id or runner['invocation'] != invocation[6:]:
    raise ValueError('Child runner does not bind the invoked shell command')
row = {
    'run_id': run_id, 'directory': str(directory), 'order': len(manifest['execution_order']),
    'result_sha256': digest(directory / 'result.json'),
    'runner_record_sha256': digest(runner_path), 'invocation': invocation,
    'started_at_utc': started,
    'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files': {
        category: {str(path.relative_to(root / category / run_id)): digest(path)
                   for path in sorted((root / category / run_id).rglob('*')) if path.is_file()}
        for category in ('data', 'log', 'profile')
    },
}
if phase == 'check':
    row.update(receipt_path=str(directory / 'receipt.json'),
               receipt_sha256=digest(directory / 'receipt.json'))
if phase in manifest['methods'][method]:
    raise ValueError('Duplicate child phase')
manifest['methods'][method][phase] = row
manifest['execution_order'].append({'method': method, 'phase': phase, 'run_id': run_id})
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
}

for method in "${methods[@]}"; do
  reference=()
  if [[ "$method" != hbm ]]; then
    reference=(--hbm-check-receipt "$check_base/data/${run_id}_hbm_check/receipt.json")
  fi
  run_child "$method" check run.sh --mode check "${reference[@]}"
done
for method in "${methods[@]}"; do
  receipt="$check_base/data/${run_id}_${method}_check/receipt.json"
  benchmark="$base/data/${run_id}_${method}_bench"
  run_child "$method" bench run.sh --mode bench --validation-receipt "$receipt"
  run_child "$method" profile gap_profile.sh --validation-receipt "$receipt" \
    --benchmark-run "$benchmark" --trace-warmups 1
  run_child "$method" operators profile_layers.sh --validation-receipt "$receipt" \
    --benchmark-run "$benchmark"
done
python - "$staging/data/manifest.json" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
methods = ('hbm', 'echo', 'serial_sparse', 'dense_prefetch')
expected = [(method, 'check') for method in methods]
expected += [(method, phase) for method in methods for phase in ('bench', 'profile', 'operators')]
actual = [(row['method'], row['phase']) for row in manifest['execution_order']]
if actual != expected or any(set(manifest['methods'][method]) != {'check', 'bench', 'profile', 'operators'}
                             for method in methods):
    raise ValueError('Method cohort is incomplete or execution order changed')
for source, expected_sha in manifest['runner']['sources'].items():
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Cohort runner changed during execution')
manifest['completed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
for category in data log profile; do
  mkdir -p "$base/$category"
  mkdir "$base/$category/$run_id"
  claimed+=("$base/$category/$run_id")
  cp -a -- "$staging/$category/." "$base/$category/$run_id/"
done
published=true
rm -rf -- "$staging"
echo "Completed method cohort: $base/data/$run_id/manifest.json"
