# Shared publication checks and literal child-process provenance.
# The caller defines script_dir, staging, runner_source, runner_invocation and command.
runner_sha256="$(sha256sum "$runner_source" | cut -d ' ' -f 1)"
runner_helper_sha256="$(sha256sum "$script_dir/runner_common.sh" | cut -d ' ' -f 1)"
runner_started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
runner_shell_pid=$$

validate_and_record_runner() {
  local expected_mode="$1" expected_detail="${2:-}"
  python - "$staging/data/result.json" "$expected_mode" "$expected_detail" \
    "$runner_source" "$runner_sha256" "$script_dir/runner_common.sh" \
    "$runner_helper_sha256" "$runner_started_at_utc" "$runner_shell_pid" \
    "${#runner_invocation[@]}" "${runner_invocation[@]}" "${command[@]}" <<'PY'
import datetime
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

from experiments.deepseek_v32_mfu.src.run_contract import validate_completed_result

(result_path, mode, detail, source, source_sha, helper, helper_sha, started,
 shell_pid, invocation_count, *arguments) = sys.argv[1:]
result = validate_completed_result(
    Path(result_path), expected_mode=mode, expected_profile_detail=detail or None
)
for name, expected in ((source, source_sha), (helper, helper_sha)):
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError("Runner source changed during execution")
invocation_count = int(invocation_count)
invocation, command = arguments[:invocation_count], arguments[invocation_count:]
proc = Path('/proc') / shell_pid
stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
record = {
    'schema_version': 1,
    'run_id': result['run_id'],
    'source': str(Path(source).resolve().relative_to(Path.cwd())),
    'sha256': source_sha,
    'helper_source': str(Path(helper).resolve().relative_to(Path.cwd())),
    'helper_sha256': helper_sha,
    'invocation': invocation,
    'command': command,
    'command_text': shlex.join(command),
    'shell_process': {
        'pid': int(shell_pid),
        'proc_start_ticks': int(stat[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'started_at_utc': started,
        'target_validation_completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    },
    'target_process': result.get('process_provenance'),
    'environment': {
        key: os.environ.get(key)
        for key in ('CUDA_VISIBLE_DEVICES', 'PATH', 'DG_JIT_WITH_LINEINFO',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'TRITON_PTXAS_BLACKWELL_PATH', 'PYTORCH_ALLOC_CONF',
                    'PYTORCH_CUDA_ALLOC_CONF', 'CXLDSAGR_SM90_BACKEND')
    },
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
}
with Path(result_path).with_name('runner.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
PY
}
