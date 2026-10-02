# Independent GR serving acceptance audit

The maintained entry is `experiments/gr_serving/src/audit.py`, with focused CPU
checks in `experiments/gr_serving/tests/test_gr_serving_audit.py`. It supersedes
the former `/tmp` draft. This document describes an acceptance procedure; no
performance result is accepted merely because its implementation/tests pass.
Runs 01–03 were withdrawn/incomplete, run 04 did not launch, and root stopped
run 05 without publication when the user expanded the experiment scope.

## Run after successful measurement

From the repository root, substitute the successfully completed run ID:

```bash
.venv/bin/python -m experiments.gr_serving.src.audit \
  experiments/gr_serving/output/data/RUN_ID \
  --expected-run-id RUN_ID \
  --json experiments/gr_serving/output/data/RUN_ID/independent_audit.json
```

The destination must be new. The auditor writes it exclusively and only after
all selected gates pass. The current sequential run uses the command above and
does not read any heat curve or archived access file. `--expected-trace-directory`
is required for industrial
runs and names the existing access-only dataset that the user selected. It is
optional for historical Beauty runs; an industrial invocation must add
`--expected-trace-directory GR/generated/industrial_10m_pv_share_t4096_seed42`.
Incomplete/running metadata fails before reference
loading. The default `--reference-check cpu` loads every saved HBM tensor on CPU,
one at a time. It never executes models, invokes CUDA APIs, attaches to running
processes, or changes numerical source/data artifacts. It temporarily uses one
CPU torch thread for tensor inspection and restores the previous thread count.

`--reference-check existence` is an optional preliminary structural check. It
writes status `structural_only` and exits 2; it cannot produce accepted status.
`--expected-source-sha256` optionally supplies an independently recorded expected
identity. Otherwise, the digest comes from metadata and is checked against the
complete source manifest, current source files and saved source copies. Run this
before changing any source covered by the measurement snapshot. Markdown edits
remain outside the numerical source freeze.

## Dynamic experiment contract

Each accepted run has one fixed history/candidate geometry. The auditor derives
its expected ordered matrix from validated `parameters.models` and
`parameters.users`, using the known four schemes for each selected model. It
allows only DeepSeek replay/NOSA and positive integer populations, including
1024, rejects duplicates and unknown cases, and preserves the
runner's canonical model order and configured population order. A per-run audit
states its actual coverage; it does not itself certify that all separately
requested history lengths have been completed.

An actually empty revisit scope is accepted only when metadata contains the
strict boolean `parameters.allow_empty_revisits=true` and the independently
replayed access sequence also contains no revisits. Its count must be zero and all
latency statistics must be null; a fabricated zero latency fails. The optional
case fields `observed_users`, `revisits` and `max_revisits_observed` must match
the verified workload when present.

`parameters.requests` is an explicit positive total-request count. With
`parameters.max_revisits=null`, all requested accesses are required. Weighted
sampling has no per-user caps or forced user coverage. An explicit non-null
`max_revisits` excludes the
first visit; for population N, expected requests then become
`min(requests, N * (max_revisits + 1))`. Counts are computed before
reading case/observed counts, so omissions cannot redefine the expected matrix.
For two models, four schemes each, all seven populations, cap 32 and maximum
8 revisits, this means 56 cases and 1608 rows per history length: 9 requests for
one user and 32 for each other population. No request count or source hash is
hard-coded in the auditor. The current small run uses 16 users, exactly 32
requests per case, sequential `0..15` repeated twice, no heat distribution and
no revisit cap. Two models with four schemes each give 8 cases and 256 measured
requests. The geometry is 65536 history tokens plus 128 candidate tokens, with
cache caps of 4 GiB HBM and 64 GiB DRAM. The earlier industrial 1024/4096 run was
stopped during input generation, before publication, when the user requested
this smaller run; its workload mode remains supported for later use.
These budgets remain separate from model weights and ordinary activations. The
auditor reads the budgets from parameters; it does not impose these values on
other runs.

The workload config records explicit `sampling`, `heat_dataset` and `heat_field`.
`sampling="sequential"` requires no revisit cap or external access-trace file,
normalizes both heat fields to null and uses explicit IDs with equal placeholder
weights. The heat metadata has no source path/dataset, `heat_sha256` is null, and
per-user probability is null because there is no probabilistic sampling. The
auditor directly requires `user_id=request_id % N` and
`visit_index=request_id // N`, so 16/32 implies exactly 16 first visits and 16
revisits. No random sampler or heat-file read is performed for this mode.

For weighted sampling, measurement
parameters may leave `heat_field=null`; the auditor resolves the dataset default
(`beauty/interaction_count` or `industrial_10M/pv_share`). The original Beauty
manifest schema without `heat_field`, and old weighted configs without
`sampling`, remain supported. Industrial and sequential workload
manifests require the additional `access_trace_sha256` over canonical JSON records
containing `request_id`, `user_id`, `visit_index`, `previous_request_id` and
`timestamp`. This hash has different semantics from metadata's
`access_trace_sha256`, which identifies the raw copied `access_trace.csv` bytes.

History and candidate lengths, seed, chunk size and both cache byte budgets come
from run parameters. Backend descriptions must agree with the execution geometry;
all row budgets must equal the run's byte/GiB caps. Full candidate hidden shapes
are `[candidate_tokens,7168]` for DeepSeek replay and
`[candidate_tokens,4096]` for NOSA. The known scope remains ten independently
copied DeepSeek dense blocks plus endpoint layers, or the full 32-layer NOSA.
The known output difference is preserved: DeepSeek computes the last-token LM
head, whereas NOSA returns candidate hidden states.

## Independent gates

- Recompute every complete input/prefix/candidate token hash; validate immutable
  per-user history, changed successive candidates, contiguous visit counters,
  previous-request links and exact metadata pairing across all four schemes.
- Verify the signed workload configuration and request identities, including
  `max_revisits` and the explicit `context_limit`. Check the context manifest's
  default/effective generation limits and override flag against the requested
  geometry. The context declaration is not model-quality validation.
- Replay seeded heat sampling without importing GR: original recorded positive
  weights, sorted user IDs, one `random.Random(seed)` draw per request, and a
  freshly built cumulative distribution over eligible users. A user leaves after
  its first visit plus its revisit allowance. Cap eligibility and actual stopping
  count are independently enforced. This creates no forced first visits or user
  quotas; cold users may remain unvisited when the global request cap is reached.
- Independently replay sequential mode using integer remainder and quotient.
  Enforce all requested accesses, exact uniform placeholder weights, null heat
  identity/probabilities and the explicit synthetic-population declaration.
  Retain the same token, phase, allocation, LRU and numerical checks as weighted
  runs. A second-round cache miss stays a revisit.
- For the selected industrial dataset, validate its manifest's dataset, field,
  population/sampling seeds, schedule and population metadata against each
  workload. Verify the request/user CSV hashes, then match all eight access CSV
  columns and all nine population CSV columns, including unvisited users.
  Independently derive heat ranks, first/last visits and reuse distances. The
  run's copied `access_trace.csv` must have the same raw SHA-256 as both metadata
  and the archived requests CSV. Recheck all these file identities at the end
  and include their hashes in the audit JSON. No token content or model quality
  is attributed to these access-only files.
- Require every measured request to have exactly one finite, zero-error,
  zero-tolerance, exact BF16 all-hidden correctness record with the full expected
  shape. Load all expected HBM reference files on CPU and check shape, dtype and
  finite elements; aggregate their file hashes in the audit JSON.
- Check every reserved, post-truncate and pre-cleanup allocation against both caps.
  Derive whole-session capacity as the minimum nonzero-tier cap divided by its
  session reservation. Reuse distance determines every hit; last-visit ranking
  determines the exact victim, retained users and total reservations. Revisit
  misses remain revisits and stay in their latency statistics.
- Independently recompute all/first/revisit summaries, inclusive descriptive
  percentiles, phase sums, hit/miss/eviction counts and allocation samples. Match
  every per-request CSV value to its raw measurement, and every summary CSV value
  to verified JSON. SVG existence is checked; rendered inspection is separate.
- Verify source manifest membership and all current/saved bytes, aggregate SHA-256,
  recorded gitlinks against actual submodule HEADs, and absence of tracked
  submodule changes. For weighted mode, verify the current heat-curve data digest.
  Recheck source and
  metadata identity at completion, then record evidence-file hashes.

The numerical audit correlates persisted correctness records and HBM references.
Non-HBM output tensors are not saved, so it is not an independent model rerun.
Cache allocations are sampled boundaries/reservations, not continuous process
memory peaks. CUDA allocator peaks remain a separate recorded measurement.

## CPU verification and remaining work

The maintained focused suite passed 86 checks on 2026-10-02:

```bash
.venv/bin/python -m pytest -q experiments/gr_serving/tests/test_gr_serving_audit.py
```

It covers the former rejection gates plus all three history geometries,
metadata-derived matrices, bounded/unbounded input rejection, missing cases,
backend geometry mismatch, exhausted users, hot repeats before cold first visits,
global cap termination, coherent re-signed but wrong schedules, incorrect observed
maxima, CDF boundary semantics and explicit context overrides. CPU reference tests
use only two tiny tensors. A separate read-only design check found independent
CDF replay equal to the GR Fenwick implementation on 1600 Beauty traces across
four populations, 100 seeds and four cap choices. Strict replay/eligibility
checks are retained; numerical boundary uncertainty is not treated as a pass.
The final three checks cover explicit permission for an empty revisit scope,
null statistics, strict boolean flag typing and optional case observation fields.

The sequential extension covers 16/32 accesses, both models' eight-case matrix,
4/64 GiB budgets, exactly one first visit and one revisit per user, retained
revisit classification after LRU eviction, refusal to call heat replay, and
rejection of reordered/truncated/excess accesses, invalid counters, heat
probabilities and metadata, capped runs and external heat traces. Weighted
manifest compatibility is retained.

The generated full 65536+128-token sequential workloads also passed this audit's
workload gates for both NOSA and DeepSeek: 32 requests per model, 16 users, all
input/prefix/candidate token hashes, exact cyclic visits and explicit no-heat
metadata. Both use access SHA-256
`c3511ba7f3fb306def87447c0b7d7c545e253f5f113fb98288987612befedda9`.
Those preflight artifacts are temporary CPU workload fixtures, not measured GPU
results or numerical model acceptance.

The industrial extension covers dataset default resolution, uncapped 1024/4096
scope, 64/1024 GiB budgets, signed access identity, required archive selection,
copied CSV identity, and rejection of re-signed CSV changes to users, visit flags,
prior accesses, reuse distances, timestamps, counts and probabilities. It retains
legacy Beauty compatibility. A separate CPU-only check independently replayed
all 4096 accesses from the archived 1024-user probabilities and compared every
request/user CSV field using temporary small-token audit fixtures. The result
has 751 visited users, 3345 revisits and maximum 129 visits / 128 revisits. Source
dataset hashes are:

```text
manifest.json     60b44103dc61440271aa4c6387e3f3c0273a4361c5795e0f76e26ef85c390db6
requests_1024.csv f2a507dd546c33f5b9d9774a50824945e44f3dfce5709052864b20e7a5888452
users_1024.csv    7af5779fe5710cf28f2ed94c32a13bf22a52225fdeaeb87add432e973600dff9
```

This is access-data and audit-logic validation, not GPU experiment acceptance.
The real 64K token workload and persisted correctness/reference artifacts must
still pass the complete post-measurement audit. Request JSONL is read one row at
a time; full token arrays are discarded after hashing and only request metadata
is retained across the audit.

Root still must run this audit against each accepted formal run, inspect rendered
figures, publish selected reports and README conclusions, perform the required
NOSA native internal-overlap profile after latency measurement, and use the
current repository Research Supervisor skill to update research state. An
accepted audit JSON covers only the stated per-run gates.
