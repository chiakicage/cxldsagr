# T-007 historical numerical reference relocation

The [file index](t007_reference/evidence.json) preserves only the original
numerical dependencies of the unfinished
[NOSA indexer dispatch task](../../system/nosa/nosa_indexer_dispatch_plan.md).
All six files, totaling 26,548,253 bytes, have been copied and independently
rehashed against their originals. The index records the file identities. Do not
remove the original published run until its replacement publication and complete
three-version comparison have been accepted.

These are historical inputs from
`nosa_motivation_poolscan_sm90_20261004_01`, source
`93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79`.
Relocation does not rerun a diagnostic, validate the refactored implementation,
or preserve the old run as a current performance result. Original metadata,
commands, source identities and numerical payload bytes remain unchanged.

## Retained files

For each row, the original path is
`experiments/nosa_motivation/output/data/nosa_motivation_poolscan_sm90_20261004_01/`
plus the relative path below. The retained path is
`docs/agents/acceptance/unified_runtime_20261005/t007_reference/` plus the same
relative path. The index records both absolute paths, byte counts and file
SHA-256 values. The retained payloads are ignored local acceptance data; only
the index and this explanation are versioned.

| Relative path | Dependency |
| --- | --- |
| `metadata.json` | Original accepted run and native-build records |
| `source_manifest.json` | Original 118-file source identity |
| `measurements.jsonl` | Original 128 measurement rows, including HBM request 0/16 references |
| `workload/requests.jsonl` | Original 32 requests and exact history/candidate tokens |
| `numerical/hbm/000000.pt` | Independent HBM output for request 0 |
| `numerical/hbm/000016.pt` | Independent HBM output for request 16 |

The frozen descriptor reviewer checks all 32 workload and 128 measurement
rows before selecting the two HBM records. Retaining those original manifests
is therefore necessary. No other output tensor, profile, attention-API data or
old performance report is retained for this dependency. The workload file's
SHA-256 is distinct from the logical `workload_sha256` in original metadata.

## Reopening the frozen reviews

The three original reviewers remain under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/`. Their bytes and historical commands are
unchanged. Their hashes at relocation are in the file index. Each reviewer
hard-codes a module-level `FORMAL` path and has no reference-root CLI option.
After copy completion, a caller can import the original file, change only its
in-memory `FORMAL` value, then call `review` with the existing diagnostic input.

| Reviewer directory | `review` argument | Required CPU affinity |
| --- | --- | --- |
| `nosa_hbm_prequeue_review_20261004_01` | `Path` of an existing diagnostic output directory under `nosa_hbm_prequeue_diagnostic_20261004_01/output/` | CPU16–23 |
| `nosa_guarded_copy_control_review_20261005_01` | Original run-ID string | CPU0–7 |
| `nosa_copy_descriptor_results_review_20261005_02` | Original run-ID string | CPU0–7 |

Use the original project's Python environment with `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`PYTHONDONTWRITEBYTECODE=1`. Bind the process to the exact affinity above. The
following Python fragment shows the path override; `reviewer_path` and
`diagnostic_input` come from the corresponding existing reviewer and run:

```python
import hashlib
import importlib.util
import json
from pathlib import Path

reference_root = Path(
    "/mnt/ssd-wlcb/chenkaiqi/cxldsagr/docs/agents/acceptance/"
    "unified_runtime_20261005/t007_reference"
)
index = json.loads((reference_root / "evidence.json").read_text())
assert index["status"] == "copied_exact"
reviewer_path = Path(reviewer_path).resolve()
identity = next(
    row for row in index["reviewer_identity_at_relocation"] if row["path"] == str(reviewer_path)
)
assert hashlib.sha256(reviewer_path.read_bytes()).hexdigest() == identity["sha256"]
spec = importlib.util.spec_from_file_location("t007_frozen_review", reviewer_path)
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
reviewer.FORMAL = reference_root
result = reviewer.review(diagnostic_input)
```

Preserve the caller, original reviewer hash, overridden path, input argument,
environment and returned result in a new reopening record outside the original
receipt directories. Do not overwrite old receipts or reinterpret a reopened
historical review as new runtime or performance acceptance.

The guarded and descriptor reviewers import the pinned prequeue helper for
history/native-record checks. Those helper calls receive the formal metadata
explicitly and do not read that helper's `FORMAL`. Their independent frozen
driver, prototype, source snapshots and diagnostic output remain at their
original external paths; this relocation does not move or replace them.

The path mapping and reviewer call signatures were inspected statically. No
reviewer has been rerun as part of this relocation.
