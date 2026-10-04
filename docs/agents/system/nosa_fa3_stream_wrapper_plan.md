# Isolated FA3 stream-wrapper candidate

Status, 2026-10-04: root applied the selected function and two focused tests.
The production-source targeted check passed all 11 cases with ordinary loaders;
the fresh full32 integration gate and independent CPU receipt audit passed.
Earlier isolated GPU checks passed
11/11 for each variant, all six process runs and both CPU audits passed, and
candidate wall medians were lower in all three pairs by 0.043–0.117 ms (0.28–0.77%).
Those timing results retain their isolated source/binary identities. The initial
baseline check had 6 passes and 5 FlashInfer loader failures before graph execution;
that harness failure and its correction remain recorded below. Production Python,
native sources, headers, build caches and selected libraries stayed frozen during
the isolated A/B. The later production change is recorded separately at the end.
This plan does not resume GR budget traces.

## Isolated validation contract and scope

The only candidate implementation is
`operators/nosa/attention/device_only/_fa3.py::_launch_validated_workspace`,
staged outside the repository. Its source obtains the current stream once,
inside the existing explicit CUDA device context, then uses that same stream
for the public `tvm_ffi.use_raw_stream` context and every existing workspace
`record_stream` call. The original function is restored when the isolated
process context exits. That phase used no production patch or native rebuild.

The injected function uses the original function's module globals, including
its original `_module()` and native build identity. Baseline keeps the original
function object. Both variants retain output allocation, the empty-query path,
mask/bias selection, module lookup, complete native argument order and return.
Public input/workspace guards, capture rejection, alias/capacity checks, native
validation/repair, layer stream checks, weight/precision/config checks, allocator
entry/exit validation, finite flags, pre-commit host decision, transaction
rollback/discard, lease synchronization and output lifetime remain unchanged.
The current stream is observed afresh on every call; no stream is cached across
calls or borrowed across owners.

The source review identified two stream observations in each original FA3
workspace call: one implicit observation inside `use_torch_stream()` and one
explicit observation for `record_stream`. The current HBM diagnostic exercised
32 such calls per candidate. Removing the duplicate is an unmeasured hypothesis.
Its cProfile call tree was incomplete; cumulative entries are not expected
savings. No subtraction of profiled API spans from unprofiled wall time is valid.

## Staged artifacts and identities

All external paths below share this root:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_fa3_stream_wrapper_20261004_01/`.

| File | Purpose |
| --- | --- |
| `baseline_function.py` | Exact source of the current production function |
| `candidate_function.py` | Only the changed Python function; compiled with original module globals |
| `identity.json`, `bundle.json` | Function, helper, input and complete staging-file hashes |
| `source_manifest.json`, `native_build_before.json`, `native_artifacts_after_warmup.json` | Frozen 114-source manifest, dependency identity and allowed existing libraries from the completed current HBM diagnostic |
| `gr_request.json` | Exact first sequential GR request from that diagnostic |
| `profile_hbm_overlay.py` | Existing eight-thread diagnostic helper, with untimed output/selection persistence and a pre-execution frozen-request check |
| `driver.py` | Function injection, existing-library-only loader, compiler rejection, checks and bounded helper entry |
| `flashinfer_cache_only.py`, `test_flashinfer_cache_only.py` | Exact-path FlashInfer cache loader and four CPU mock tests for accepted/rejected loads and restoration |
| `test_stream_overlay.py` | Two workspace stream/lifetime/exception checks and the existing rollback scenario adapted only to HBM |
| `launch.py` | Canonical GPU5/NUMA1 environment, dry-run preview and serial process ordering |
| `compare.py` | CPU comparison of every sample, cross-process outputs, selections and declared trace scopes |

Production module SHA256 is
`64f699cf8d11d0551f70808fb733577235fda12f61b9efa8bbc4395a62158653`.
Baseline function SHA256 is
`b5082068941dfe229ab3b24108b3b110b2f34286a5eb5bf323ac3b75313cf333`;
candidate function SHA256 is
`9367e39d4501183f67acaed08ab924e6a49327b21ceeb52f1e46634e7080c74e`.
The production source digest remains
`02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`.
Candidate reports must record this baseline identity plus the separate overlay
identity; the unchanged source digest alone does not identify the executed
candidate. `identity.json` records every helper edit. No timed model code is
added to save outputs: hidden is saved after the timing loop, and exact selection
IDs/masks are saved in the helper's separate selection replay.

The driver routes TVM C++ load requests only to matching, hash-verified binaries
in the frozen inventory. Missing names fail. TVM build/inline-build entry points
and compiler/ninja subprocesses are rejected; compiler version/dependency
inspection remains allowed. `FLASHINFER_DISABLE_JIT=1` remains enabled. Installed
FlashInfer's `JitSpecNvcc.try_load()` accepts only AOT artifacts and deliberately
returns `None` for existing JIT artifacts; its inherited `build_and_load()` then
rejects those artifacts under this flag. The revised external driver patches only
`JitSpecNvcc.build_and_load()` to call the original `JitSpecNvcc.load()` with the
exact frozen `rope` or `silu_and_mul` path after name, path, size and SHA256 checks.
Unknown modules and changed artifacts fail. Each accepted load is recorded in
the run's overlay receipt, and the loader method is restored on context exit.
The initial bundle, preflight, plan and reported failure receipt are retained
under external `revision_history/initial_preflight/`; they are harness history,
not valid timing results. Both variants use the same revised loader. No cache is
cleared or rebuilt to make a check pass. Native dependency identity is checked
before and after execution. The allocator adapter must report `private_cpp`
with no fallback before execution and at completion.

## Exact execution sequence

Run from the repository root. The following preparation commands are CPU-only;
`verify` hashes files without importing Torch or initializing CUDA:

```bash
nosa_stream_stage=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_fa3_stream_wrapper_20261004_01
python3 "$nosa_stream_stage/driver.py" verify
python3 "$nosa_stream_stage/launch.py" checks --variant baseline --dry-run
python3 "$nosa_stream_stage/launch.py" checks --variant candidate --dry-run
python3 "$nosa_stream_stage/launch.py" pairs --dry-run
```

Only after root authorizes GPU execution, run these commands in order:

```bash
python3 "$nosa_stream_stage/launch.py" checks --variant baseline
python3 "$nosa_stream_stage/launch.py" checks --variant candidate
python3 "$nosa_stream_stage/launch.py" pairs
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$nosa_stream_stage/compare.py"
```

The launcher binds physical GPU5, UUID
`GPU-a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`, CPUs48–55 and NUMA memory node1.
It checks for the expected idle device before each child and stops on failure.
Its environment contains exactly one repository `.venv/bin` prefix,
`CXLDSAGR_SM90_BACKEND=native`, OMP/MKL/OpenBLAS=8, `PYTHONDONTWRITEBYTECODE=1`,
and canonical
`PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8`.
Deprecated CUDA/HIP allocator overrides and no-cache overrides are unset.
The helper sets PyTorch intra-op threads to 8; inter-op is left unchanged and
recorded. No clock, GC, allocator-policy or cache-reset change is made.

### Focused checks: 11 per variant

Four existing workspace checks cover varying geometry, retained independent
outputs, bounded allocations and prelaunch rejection of bad capacity, backend,
layout, capture and scratch aliasing. Two staged checks exercise the changed
workspace branch on two distinct nondefault streams with explicit serialization,
verify every original scratch tensor's `record_stream` association, retain outputs
across workspace reuse, inspect native FFI stream binding, and verify Torch/FFI
restoration after a native exception before successful reuse.

Four existing graph checks cover parameter mutation, precision mutation,
captured norm-policy mutation, and one late-layer HBM nonfinite-query case
spanning both prefix and candidate failure. A third staged check reuses the
existing tiny-model attention-failure rollback scenario with scheme `hbm`;
the original rollback test uses dense prefetch and is outside this candidate
check. The driver
selects only scheme `hbm`, layer1, field `query` from that parameterized finite
test; it does not run its full scheme/field/layer matrix. Any skip fails this
focused invocation. Tests print to the terminal and create no persistent test
report or result log. CUDA0 is the only visible device, so this bounded run does
not claim multi-device validation.

### Full32 controlled A/B

The existing helper constructs an independent empty HBM session in each process,
builds H65536 in C1024 chunks, and executes A128 candidate request0 with compute
graphs. The generated request must equal the frozen request before prefill/timing.
Its SHA256 is
`c121bc06d8cfe94c319f257c03a8993ad13fb1efb91692ad013ed5b42c8deefa`.
Checkpoint path, seeds, graph query sizes, allocation policy, warmup and workload
are identical across variants. Existing per-execution weight and cache guards
remain active; checkpoint provenance retains the helper's metadata/hash and
shard-stat boundary, not a new hash of every weight byte.

Each process runs the helper's one complete candidate warmup followed by all
31 synchronized candidate timing samples. The six fresh processes run serially:

| Pair | Process order | Run IDs |
| --- | --- | --- |
| 1 | baseline, candidate | `fa3_stream_baseline_pair1_20261004_01`, `fa3_stream_candidate_pair1_20261004_01` |
| 2 | candidate, baseline | `fa3_stream_candidate_pair2_20261004_01`, `fa3_stream_baseline_pair2_20261004_01` |
| 3 | baseline, candidate | `fa3_stream_baseline_pair3_20261004_01`, `fa3_stream_candidate_pair3_20261004_01` |

The complete process command appears in `launch.py pairs --dry-run` and each
run's `launch.json`. If interrupted, a never-started process can be invoked with
`launch.py one --variant <baseline|candidate> --pair <1|2|3>` after root checks
state; an existing run is never overwritten or automatically retried.

Each run stores `data/`, `profile/`, `log/`, `launch.json` and `overlay.json` under
external `runs/<run_id>/`. The helper keeps cProfile and Chrome capture separate
from unprofiled samples. Complete hidden and all 32 selection IDs/masks are saved
outside timing for CPU bitwise A/B comparison. The helper's exact replay checks,
32-layer selection checks, source/dependency freeze and native-library checks
remain active. This is full32 candidate validation on one independent history
per process, not the multi-user/four-scheme correctness suite or formal serving
measurement. Failure artifacts stay outside `experiments/` and supply no result.

## Acceptance and decision

`compare.py` checks every raw sample, exact request, checkpoint identity, overlay
identity, private allocator provider, unchanged source, zero eager fallback,
32 scopes and 32 active scopes for each declared API, zero uncorrelated activities,
and 64 graph launches. Baseline/candidate hidden bytes for all 128 tokens and all 32 selection
IDs/masks must agree exactly, including hidden and selection agreement among the
baseline processes. Every process must match the frozen integrated source digest,
its variant’s selected function hash and the frozen checkpoint identity.

Report complete-candidate wall and event medians separately for each process
pair, retaining all 31 samples including the first. Host-submit includes the
execution lease's internal drain; it is not launch-only CPU cost. Do not pool
93 samples per variant as independent process replications or pair sample
indices from separate processes. Profile spans and cProfile times remain
intrusive diagnostics, not wall-overhead estimates or isolated API references.

Three same-direction process-pair medians are a screening condition, not
automatic promotion. Root reviews the size and consistency of any benefit,
correctness and source/trace evidence. An inconclusive or regressing result does
not justify a production edit. Existing published results stay intact; a later
production promotion requires affected-experiment replacement under project
rules. No indexer/weight/allocator/finite-guard optimization is bundled here.

## Completed isolated validation, 2026-10-04

The revised frozen bundle SHA256 is
`e6a85027046dc08f1e3ec28d8e1be0861ec36959e8c11c9441806f4023dcb6a8`.
The initial baseline attempt had 6 passes and 5 `MissingJITCacheError` failures
at FlashInfer's loader, before graph execution. After the exact-path cache-loader
correction, its four CPU mock tests passed; root ran the same focused GPU set and
reported baseline 11/11 in 2.80 s and candidate 11/11 in 2.86 s, with no skips.
Both used the frozen `rope` and `silu_and_mul` binaries. No native artifact was
rebuilt, and the candidate function itself did not change during this correction.

All six independent processes listed above exited 0. Each retained all 31 timing
samples, including the first, for 93 samples per variant and 186 overall. The
initial pairs launcher stopped before candidate pair 2 because its idle check
saw a stale utilization sample; root confirmed no GPU process and zero current
utilization before starting that never-started process with the frozen `one`
command. The remaining order stayed C2/B2/B3/C3. A separate GPU4 correctness check
ran in the idle gap before B3; no completed process or timing sample was repeated.

| Process pair | Baseline wall median (ms) | Candidate wall median (ms) | Wall reduction | Baseline event median (ms) | Candidate event median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, B1/C1 | 15.147159 | 15.057928 | 0.5891% | 15.117120 | 15.028288 |
| 2, C2/B2 | 15.190257 | 15.073695 | 0.7673% | 15.158816 | 15.042624 |
| 3, B3/C3 | 15.160615 | 15.117471 | 0.2846% | 15.129248 | 15.088992 |

The frozen `compare.py` passed, and a separate CPU audit recomputed the statistics
from every raw sample and the attribution from every raw Chrome trace. Across
all six processes, saved hidden bytes for all 128 tokens and saved selection
IDs/masks for all 32 layers agree exactly. Every run's 114-file source snapshot,
selected function identity, checkpoint metadata/shard-stat identity, and actual
24-file warmup/final native inventory match their frozen references. The final
driver verification also passed. Every trace independently contains 803 kernels,
161 copy/memset activities, 64 graph launches, 32 active scopes for each of the
four declared APIs, and zero uncorrelated activities. The 228 activities outside
API scopes remain outside those scopes; they were not dropped from the full trace.

This is a small same-direction median improvement in three process pairs.
Its absolute wall reduction varies from 0.043144 to 0.116562 ms. Within-process
wall IQRs range from 0.023467 to 0.043281 ms; all untrimmed maxima range from
16.237700 to 17.862704 ms. These observations do not establish a tail-latency
benefit or a broad workload effect. The three process pairs remain the replication
units; no 93-sample independence assumption, sample-index pairing, new numerical
proximity threshold, or profiled-time subtraction was used. Root decides whether
to promote the narrowly scoped change before any affected-experiment reruns.

All receipts are under the external staging root, outside `experiments/`:

- `comparison.json`, SHA256
  `dba781fcfa0268b612c66f7df2d914decc51291636485689fcd88b495919b6fe`.
- `review/independent_audit.py`, SHA256
  `7deeed82ca2f929024c7b0f6f6ef56b8da8b349d9f1c5fbc22b867cd1674f672`.
- `review/independent_audit.json`, SHA256
  `2dac0750c4ddc707ba28ddac06d265edaaaed345e3a4af929d81e57b0edf2377`.
  This receipt retains all raw sample arrays, per-process dispersion, tensor
  byte identities, recomputed trace metrics and hashes of each run's launch,
  overlay, source/native inventory, trace and output artifacts.

The immutable preparation identity still says `prepared_not_executed`; the run
receipts and this completion record carry execution status. Published experiment
results and production code remain unchanged by this isolated validation.

## Production integration, 2026-10-04

Root selected the narrow change after the isolated audits and applied
`promotion/production.patch`, SHA256
`45acc22f02923596e88dc91df5855402a3a57387d12db7bf834cafa3699a512d`.
Only `_launch_validated_workspace` changed in runtime code. Two focused stream
and exception tests were added to the existing attention workspace test module;
all prior runtime/test functions and the native argument list remain unchanged.
Ruff and the prepared patch's AST/applicability checks passed.

The promoted production `_fa3.py` SHA256 is
`d56c29942291580bb39281858fe98908a42ee501df460ca700a7a1bcb7949254`.
Root recorded integrated source digest
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
The Python file contributes to the native FA3 build key, so the production gate
uses its ordinary loader and permits a fresh cache entry. The old binaries are
protected; no isolated function or cache-loader adapter is injected.

Root ran the prepared 11-case current-source invocation on GPU5: all 11 passed,
with one dependency deprecation warning, in 72.77 s (exec session `13003`, exit 0).
This verifies the integrated stream tests, existing workspace guards, selected
graph guards and HBM rollback scenario; the pytest duration is not a performance
measurement. The test invocation and fresh full32 command remain in external
`promotion/verification_commands.md`.

The fresh full32 wrapper passed (root exec session `80809`, exit 0).
Its path is external `promotion/full32_gate.py`, SHA256
`b60b48acebadf901a9f290a5f19497ad662499e7365b8233b7c1fb2cb96d89a1`.
It asserts the promoted `_fa3.py` hash, snapshots fresh source/dependency identity,
protects the prior 24 native binary files, and records actually mapped post-gate
libraries. It retains the existing fixed four-scheme correctness gate: four eager
HBM references and 16 graph requests, including strict complete hidden equality
and session/graph-replay checks. Its original correctness preflight keeps Torch
intra-op at 1 while pinned registration remains 8; no timing result is claimed.
Fresh output is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_fa3_stream_full32_20261004_01/`.
The CPU review found no stale old-source precondition: fresh source is checked
against its own entry snapshot, while only the prior native file bytes are held
fixed.

The gate completed all four eager reference requests and 16 graph requests with
finite complete hidden tensors satisfying `atol=0, rtol=0`. Its graph replay,
retained-output, session-length, visit and cache semantics checks passed; the
allocator remained `private_cpp` with no fallback and an unchanged configuration
key. This is the fixed-serving H65536/A128 full32 gate, not the ordinary A1024
checkpoint gate or a performance measurement. Complete tensors were not saved
by this gate for another independent byte comparison; numerical acceptance is
the successful strict assertion sequence in the unchanged, hash-verified driver.

The independent CPU audit verified every saved and current source file, recomputed
the native build identity using the recorded compiler PATH, checked all 24
protected old binary files, and rehashed all 29 actually mapped binaries. The
114-file source digest is the promoted `a73ad32…` identity above. Only `_fa3.py`
differs from the isolated runtime-source manifest. Both current source-derived
native module keys were present in the mapped inventory:

| Module | Current key | Binary SHA256 |
| --- | --- | --- |
| Resident FA3 | `5ad6b53673ce7d58` | `035dbdf9437e038d9b02c1581443df8a7158e665b395b9f1535b1fce2e355b75` |
| Fused offload | `335cc5c52e5efbb2` | `ef80c51c2eecd8d209f6928c26f010f6dc8c37f2857d64d8da01788d8e56b000` |

Fresh `evidence.json` SHA256 is
`865979653553a92e5c5f74dfa51b062a418edb537448b66c1f513dc0a78aa852`.
The external `promotion/integration_audit.json` SHA256 is
`092c62677712bd247824ba402c649587e4c3f8e6ed7fcd7c4e1590808bf3e96c`.
Affected-experiment replacement remains separate work. The isolated median
improvement above is not presented as a timing result for these new binaries.
