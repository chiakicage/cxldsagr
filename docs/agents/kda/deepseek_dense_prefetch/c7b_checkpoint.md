# C7b prototype checkpoint

Status: isolated generic and real source-0 pair GPU correctness passed.
Two exclusive source-0 component timing sweeps passed their exactness and
provenance checks. Cap 264 retained a modest primary pair improvement under an
independent sample-order seed; no C7b production source has changed. The separate
physical activity profile and root's independent SQLite audit passed. C7b remains
a component diagnostic; all-layer, helper/lookahead and full serving gates remain
open, and root has deferred production cap integration.

Root reviewed the [task](c7b_task.md), [draft](c7b_draft.md) and
[implementation plan](c7b_implementation_plan.md), and authorized `/tmp` CPU
implementation. The approved order is generic exact coverage first, then real
Q128 source-0 projection graph and official MLA serial/concurrent controls.
Other individual matrix diagnostics are conditional. Promotion later requires
all actual layer kinds and complete lookahead/full serving, since resource
demand differs across source layers and residual branches.

## Current isolated source

Workspace: `/tmp/deepseek_c7b_transport_20261004_01`.

- `baseline/kv_transfer.cu` and `.py` are unchanged copies of the original
  generic transport. `source_snapshot.json` records their original paths/hashes.
- `capped.cu`: the 128-thread per-row copy body now has an int64 grid-stride
  loop, retaining per-row bounds/trap, ordinary uint4/byte loads and actual
  pointer/width alignment checks. A separate template specialization adds
  diagnostic row/unit counters; it is never a performance kernel.
- `transport.py`: lazy, independently named FFI modules for original/capped
  sources; default uses original; positive-int cap validation rejects bool and
  malformed values. Empty valid input returns before mapping/launch.
- `probe.py`: `prepare` and `check` modes only. The byte oracle compares all
  host/device backing bytes, including guards, sentinel and candidate tail.
  It covers 1,145 dtype/count/cap/offset/ID cases. Repeated host IDs retain
  distinct destination copies. Twelve invalid-ID cases use fresh subprocesses
  after a successful warm invocation, with the bad final row beyond the first
  grid-stride iteration for capped tests.

Current SHA-256s:

| File | SHA-256 |
|---|---|
| `capped.cu` | `8fd42316230eccba651837782d5968f643c72b79ba02739265da0e1f954d0f9a` |
| `transport.py` | `d86c7c54efedb889d02f5c82917240e226be193c9979beedb72f7ff283412e4b` |
| `probe.py` | `d57d2d8de834836a15b81f34caabd3e66d01682e076f5208fefd63aeea502f9f` |

CPU `prepare_02/prepared.json` verifies 59 distinct count/cap partitions,
cap rejection and no initialized CUDA/native import. Ruff check/format passes
for the two prototype Python files. These are preparation checks only.

## Independent source review

The cache reviewer and root found no blocking source issue for a future
correctness screen. A grid-stride row belongs to one CTA, and no cross-record
barrier is needed for distinct destination IDs. Bounds and alignment remain
inside every iteration. Diagnostic vector/byte-unit counters match actual
dispatch and leave unused counter columns zero. The largest counter tensor
is about 302 MB and belongs only to untimed validation.

The numeric reviewer checked the pair timing design: record start on main,
make gather wait on start, and record end on main only after the common
compute/gather join. Serial adds a main wait on gather-ready before compute;
CPU enqueue order alone is not a cross-stream makespan. A projection-only
graph replay excludes production replay-input copies and must be labeled
accordingly. Keep fresh output references and verify after each measured
block, outside the timing window.

## Generic GPU correctness

Run `check_02` in the isolated workspace completed with exit 0 (session 88006,
`CUDA_VISIBLE_DEVICES=2`, PyTorch device name NVIDIA H200). It passed all 1,145
byte/guard and unchanged-host cases, 229 for each cap `None,33,66,132,264`.
The capped diagnostic specialization checked 876,903 rows and 927,125,628
logical copied bytes, with exactly one dispatched copy unit per expected unit.
All twelve fresh invalid-ID subprocesses produced the expected device failure
after successful warm launches. The reviewed source hashes remained unchanged.
These counters establish dispatched work, not measured PCIe traffic or cache
behavior. No timing was collected, and GPU 2 was released after completion.

The preceding `check_01` exited before its first native launch because direct
Python invocation did not put the existing virtual environment's `ninja` on
`PATH`. `check_02` used the same source with `.venv/bin` prepended to `PATH`;
the earlier environment failure is not numerical evidence. Both records remain
temporary engineering artifacts outside `experiments/`.

## Remaining work

1. Keep C7b as a component diagnostic pending the combined C8 full profile.
   No isolated gather time or overlap percentage promotes production.
2. Any future selected cap requires full helper lifetime checks and root's combined
   source/full-model/serving acceptance before publication changes.

The C6 diagnostic and independent overlap review are complete in the
[profile checkpoint](../../system/deepseek_motivation_graph_profile_checkpoint.md).
They establish zero observed matrix overlap, not the hardware cause.

## Real source-0 pair harness prepared

Temporary `pair.py` SHA-256 is
`7c18327de794ca09054cb3e6e2f115f9289b4a8dc6766dabd7a05a1822533a68`.
CPU `pair_prepare_03/result.json` passed with unchanged source and no CUDA
initialization; Ruff check/format passed. Root's independent source review
accepted setup, fresh-output ownership, protected-storage checks, production
callback boundary and the event DAG. It required dependency hashes before and
after execution and explicit connection/launch-blocking/allocator environment
values, including null; the final source includes those additions. Root granted
GPU 2 for correctness only. `pair_check_01` completed as session 50065 with
exit 0 and released GPU 2. This run collected no timing.

The default compute component invokes the production projection callback,
including static input copies/start fill and the fixed-pool transient branch,
then the complete official sparse MLA adapter. The actual model source for pair
correctness, both timing sweeps and the profile is
`/tmp/deepseek-motivation-c7_hint-frozen-a44l_2uk`; it is the accepted C7+hint
implementation, not the C6 implementation. Exact selection, resident history
and current source-0 embedding inputs are constructed from request 16 of the
accepted C6 workload. The canonical workload identity `7e4c737a...5284ae` is
distinct from the request JSONL file SHA-256 `167a0af2...f71355`; both are checked.
The first preparation attempt confused those two identities and failed on CPU
before any GPU work; the corrected version validates each independently.

The proposed correctness phase contains 150 serial/concurrent/component cases
over the five caps and fixed/four-slab address regimes. Gather payloads are
actual source-0 history values, copied into four independent pinned allocations;
they represent the 72 MiB transfer shape rather than source-1 activations.
Current MLA reads an independent resident allocation. The frozen production
precision configurator disables matmul TF32 before setup and verifies the same
policy after execution. The harness records actual imported Python and mapped
native-library identities before and after the controls, requiring every prior
hash to remain unchanged, without claiming every cached CUDA cubin executed.

All 150 exact controls passed: 50 fixed-address cases and 100 four-slab cases.
Before them, three complete projection/MLA repetitions matched the independent
eager candidate output byte for byte. Query shape is `[128,128,576]`, exact
selection `[128,2048]`, and independent resident KV `[65664,576]`. The tests
checked projection fields, MLA outputs, all protected input/weight bytes,
unchanged host slabs and the complete gathered destination including sentinel
and tail guards. All four host allocations and four destination allocations
are distinct; the current resident KV aliases none of the gather destinations.

The CPU saved-record audit verified the unique complete 150-case key set,
fixture-file SHA-256, source identity, unchanged 1,916 imported interface files
and 83 mapped libraries, and TF32-disabled precision before/after execution.
The hardware property reports 62,914,560 bytes of L2; each source slab contains
75,497,472 bytes. This size comparison does not prove a cache miss policy or
actual PCIe bytes. Connection count, launch blocking, allocator settings,
module loading, TF32 override and cuBLAS workspace environment values were all
unset (recorded as null). Results are temporary engineering artifacts at
`/tmp/deepseek_c7b_transport_20261004_01/pair_check_01/result.json` and
`saved_record_audit.json`; no experiment publication has been changed for C7b.

## First exclusive component timing

Root held the other agents and granted GPU 2 exclusively for `pair_measure_01`.
Session 28335 exited 0 and released the device. The same reviewed source used
five warmups per variant and 30 measured samples in three randomized ten-sample
blocks, for 1,500 complete `(regime, cap, mode, block, repeat)` keys. Every retained
MLA output, projection and protected input/weight/host check passed, as did full
destination guard checks at block boundaries. Source/dependency identities and
TF32-disabled precision were unchanged. All samples and CPU analysis are in
the run's `result.json`, `samples.csv`, `summary.csv` and `analysis.json` under
the temporary workspace; analysis does not initialize CUDA.

GPU joined medians in ms for the primary gather-first pair:

| CTA cap | Fixed addresses | Four source slabs |
|---|---:|---:|
| Original unrestricted | 1.804704 | 1.803584 |
| 33 | 3.687696 | 3.782272 |
| 66 | 1.978288 | 2.066032 |
| 132 | 1.799264 | 1.800752 |
| 264 | 1.766688 | 1.766688 |

Cap 264 is the only primary confirmation candidate: all six per-block medians
improve, and means improve from 1.806682 to 1.767005 ms (fixed) and 1.804407
to 1.767211 ms (four slabs). CPU joined medians track those changes and are
about 0.010 ms longer than the GPU spans. The gain is about 2%, and is not yet independently
confirmed. The cap makes isolated gather slower, from about 1.501 to 1.530 ms;
serial pairs also regress. Caps 33 and 66 regress substantially. Cap 132's
primary change is inconclusive, with a worse mean in the four-slab observation.

Compute-first is diagnostic: cap 132 medians are 1.571488/1.573136 ms and cap
264 1.608016/1.603920 ms, compared with unrestricted 1.794224/1.798672 ms.
That alternate order does not select a production candidate. Neither GPU events
nor CPU wall time identify useful internal copy/compute overlap. Full lookahead,
all source-layer types and complete requests remain outside this measurement.

Actual rotating-address reuse distances within each measured block are 2, 4
and 6 gather calls (38, 120 and 38 occurrences respectively), because each
variant uses the same ten-sample slab sequence. Fixed-address reuse distance is
one. The four-slab label describes distinct rotating allocations; it does not
promise a cold cache or establish PCIe bytes. The analysis retains this exact
distribution alongside the per-sample slab/order fields.

## Independent confirmation

Root granted a second exclusive GPU 2 window for `pair_measure_02`, using the
same source and protocol with explicit seed 8429 instead of 7419. Session 44906
exited 0 and released GPU 2. All 1,500 keys, exact output/storage checks,
source/dependency identities and precision checks passed. Both runs preserve
their exact invocation and random seed in `command.json`.

Cap 264 versus original unrestricted gather-first:

| Metric (ms) | Fixed original | Fixed cap 264 | Four slabs original | Four slabs cap 264 |
|---|---:|---:|---:|---:|
| GPU joined median | 1.806176 | 1.770624 | 1.806016 | 1.770448 |
| CPU joined median | 1.815966 | 1.780223 | 1.815891 | 1.779995 |
| GPU joined mean | 1.806797 | 1.776206 | 1.807735 | 1.770459 |
| CPU joined mean | 1.816631 | 1.786612 | 1.817625 | 1.780166 |

All six confirmation block medians improved, making twelve of twelve across
the two runs. A fixed-address cap-264 GPU sample of 1.924992 ms remains in the
mean and all saved data. The approximately 2% median improvement is confirmed
only for this source-0 component pair. Isolated gather remains about 1.531 ms
and serial pairs about 1.866 ms, both regressions relative to the original.
Compute-first measurements remain diagnostic and did not select the candidate.

## Physical activity profile and independent audit

Separate `profile_pair.py` SHA-256 is
`e4e032d401cf3438d60221ff5246a2112e67173b60436e0fd3502f2e5587c4e1`.
CPU `profile_prepare_01/result.json` verified 42 unique cases, accepted timing
identities and unchanged sources without initializing CUDA. Root reviewed the
driver and granted a separate exclusive GPU 2 window. `profile_capture_01`
completed as session 77824 with exit 0 and released GPU 2. All 42 complete exact
controls, source/dependency identities and precision checks passed.

The capture includes original/cap-264 serial, gather-first and compute-first
controls in both address regimes with three samples each, plus six cap-132
compute-first diagnostic samples. It reuses the unchanged fixture and event DAG
with CUDA timing disabled. NVTX brackets only each complete joined invocation;
fixture restores and exact byte audits lie outside those ranges. The raw trace
still retains those other activities for audit. Nsight Systems used CUDA graph
node tracing, CUDA/NVTX/OSRT tracing, no CPU sampling/context-switch trace, and a
cudaProfilerApi capture range. Exact capture/export commands are preserved in
`profile_capture_01/command.json`. The raw report is
`/tmp/deepseek_c7b_transport_20261004_01/profile_capture_01_trace.nsys-rep`;
SQLite and analyses are under `profile_capture_01/`. This profile includes joined
pair controls only; isolated gather and compute controls remain in both timing
sweeps.

The CPU analyzer `audit_profile.py` attributes every physical activity, verifies
same-thread runtime correlation against each NVTX range, and saves the raw schema,
all activities, per-sample intervals and kernel inventory. The raw trace contains
3,612 activities: 2,982 kernels, 420 memcpy and 210 memset. The 42 pair ranges
contain 2,142 kernels and 42 memcpy. Each pair contains one gather, one start fill,
48 projection graph kernels, one official MLA kernel and one hidden-input D2D
copy. Each projection has seven matrix primaries: five DeepGEMM FP8 kernels, one
absorption BMM and one FP32 index-head-weight matrix operation. The same 48 graph
node IDs appear in all 42 replays. Every serial pair has zero gather/main kernel
intersection and finishes gather before the first main activity.

Root independently rebuilt all per-sample interval metrics and 14 summary
groups directly from SQLite, conserved all 3,612 activities, verified runtime
attribution, and rehashed 95 source files. Its evidence is
`profile_capture_01/independent_profile_audit_root.json`, produced by
`independent_profile_audit_root.py` (SHA-256
`38edfd8c1f77d8d01990f2cc74f7a33bc72720941e21d8806427c047ef1c8104`).
The SQLite SHA-256 is
`e2f288aa89ccd0ecdf3f295839575ffde8db78bd795b2d79745932d4fa4e90ab`.

Primary gather-first whole-kernel interval medians, in microseconds:

| Addresses / cap | Gather | Intersection with projection graph | Intersection with matrix primaries | Intersection with MLA |
|---|---:|---:|---:|---:|
| Fixed / original | 1470.018 | 0 | 0 | 0 |
| Fixed / 264 | 1501.410 | 72.192 | 19.008 | 0 |
| Four slabs / original | 1470.723 | 0 | 0 | 0 |
| Four slabs / 264 | 1511.747 | 105.825 | 43.008 | 0 |

All six original gather-first samples have zero graph/matrix/MLA intersection;
their nonzero main-stream intersection is only input preparation. All six cap-264
gather-first samples have positive graph/matrix intersection and zero MLA
intersection. Compute-first controls are variable and remain diagnostic. For
example, two cap-132 samples have much longer MLA windows, while the other four
do not; no sample is dropped. Marginal medians must not be added because their
values can come from different samples. Profile timings are separate from the
unprofiled timing sweeps and do not replace those latency results.

The profile can identify gather/matrix/MLA kernel intervals and their
intersections. Those intervals do not locate each kernel's internal useful
copy/compute work. A positive intersection alone will not be labeled proof of
useful internal overlap, nor replace complete helper/lookahead/serving acceptance.
