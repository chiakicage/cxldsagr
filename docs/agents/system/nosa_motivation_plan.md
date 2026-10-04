# NOSA motivation implementation and acceptance

Started 2026-10-04. The requested efficiency outcome remains incomplete.
Current evidence is summarized first; earlier checkpoints retain their
historical source identities and validation boundaries.

## Selected-source measurement and profile acceptance, 2026-10-04

The promoted FA3 reserved-workspace stream wrapper passed its fresh full32
gate, followed by formal `nosa_motivation_sm90_20261004_03`, matching
`nosa_motivation_profile_sm90_20261004_03` and independent
`nosa_attention_reference_sm90_20261004_02`. All three passed independent
audits at source
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`
on GPU5/CPUs48–55/NUMA1/register8. The formal run includes 128 exact complete
candidate outputs; the independent reference checks 2112 captured calls and
their source/output linkage. Register8 is measured for this selected source,
not merely a candidate setting. No Q128 native candidate was selected.

The accepted package is published and exact superseded-output cleanup is complete. The
[fixed experiment report](../../../experiments/nosa_motivation/README.md) and
[fixed publication record](nosa_motivation_publication.md) provide the final
installed report and cleanup receipt; the
[command handoff](nosa_motivation_selected_source_commands.md) preserves actual
IDs, placement and measurement limits. The earlier `_02` and `02b5d0...`
records below do not describe the selected-source run.

For HBM requests 0/16, full rebuilt-request FA3-composition MFU ratios are
97.99%/99.29%, but matched candidate-only ratios are 75.93%/72.50%. The candidate
gap includes required nonmatrix work and timing-method differences; it is not
all removable CPU overhead. Candidate MFU, tail stability and compute/IO
dominance remain unresolved. The formal serial/async total-latency ratio is
0.973566, and all 96 applicable async layer samples fail both 90% overlap
thresholds. Valid failed performance gates remain reported; the overall goal
is not complete.

The separate GR budget H4K/H16K/H64K `_02` families at `8dc00e...` are fully
published after independent formal/profile audits and exact old-output cleanup.
See the [GR execution receipt](nosa_gr_selected_source_execution.md). Its
admission ledger and sampled storage do not establish physical HBM capacity,
and those results do not replace fixed P/NH evidence.

## Earlier integrated correctness checkpoint, 2026-10-04

The integrated production private allocator adapter passed the full 32-layer
fixed-serving graph gate at source digest:

```text
02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b
```

The gate reused the previously accepted v2 driver with register8 on physical
GPU5, CPU and memory NUMA1. CUDA reported `NVIDIA H200`, SM90; NVML reported
`NVIDIA M403` for UUID `GPU-a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`.
It rebuilt independent empty histories with H65536/A128, chunk1024,
P65536/NH16777216, for two users over two visits. All 16 graph outputs across
HBM/dense-prefetch/serial-sparse/overlap matched four eager reference outputs
bitwise. Real host misses, exact replay counts, zero eager fallbacks, output
lifetime, retained history/indexer state, candidate discard and owner cleanup
passed. Register8 remains a validated candidate, not the formal measured default.

The process used `private_cpp` without fallback. Source, allocator configuration,
adapter build identity and binary file hash matched before and after the gate.
Adapter fingerprint:

```text
21785fc27ce306846630270eae3af5ea2b31026dbc6bfa42be99c47c410a729e
```

Loaded binary SHA-256:

```text
36bfb148e301b5cb2ad5a2d3b89895c69d6f208c1796929714849e00887f7171
```

The full build identity, loaded path, initial mapping records and frozen source
snapshot are in `/tmp/nosa_integrated_allocator_full32_20261004/evidence.json`
and its adjacent `source_manifest.json`/`source/`. The wrapper checked the mapped
path before execution and binary file hashes afterward; strict before/after
mapped-inode verification is outside this gate. It did not monkeypatch global
PyTorch APIs or edit runtime source. Direct-host and owned H64K/A1K checks were
unchanged and were not repeated. Separate allocator parity and engineering
measurements belong in the [allocator guard record](nosa_allocator_guard.md).

This earlier gate supplied no new-source performance result and left the
[remeasurement queue](nosa_motivation_remeasurement_queue.md) open at that time.
The selected-source measurement/profile/API sequence above has since completed;
its evidence does not change this gate's original digest or validation boundary.

## Earlier short-prefix/register8 checkpoint, 2026-10-04

Short-prefix deferred selection and the allocator permitted-set hoist passed
correctness acceptance at source digest:

```text
6c75e3bacb8f1975cdb7d4718dd30a8fe1a4d8c7c87d9d50bfc36c1486455828
```

The short path preserves the original standalone Triton/native score schedule
for `count < 2047`; it does not expand the joint-native scoring predicate. All
31 affected Q1024 prefix lengths (1K through 31K) passed exact byte comparisons,
and the 32K model boundary retained the existing checked branch. Validation
passed 86 CPU checks, 142 new GPU operator/model tests, 354 existing native
regressions, and 18 targeted Compute Sanitizer initcheck cases with zero errors.
Actual PTX/SASS checks passed all 59 compiled guarded-score specializations:
the finite flag is the only pre-branch global read, and a false flag exits before
score input reads or output writes. These checks do not measure serving latency.

Register8 uses the following explicit configuration before process startup:

```text
PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
```

It remains a validated candidate configuration, not the formal measured default.
The register8 gates used physical GPU4, CPU and memory NUMA1. CUDA properties
reported `NVIDIA H200`, capability `(9, 0)`; NVML reported `NVIDIA M403` for the
same UUID `GPU-f8efa4fd-5a9d-cdde-c13b-0e230e2082af`. PyTorch was
`2.12.1+cu130`, revision `7269437d655783a26cba32aa88195b741ff496aa`.
Preflight checked effective allocator settings and CUDA attribute 91. Precision
was `highest`, TF32 disabled, with BF16 reduced-precision reduction enabled.
Allocator configuration keys remained unchanged after every gate.

- Thirteen existing direct-host native tests passed, including partial history,
  stripe ownership, exact fetch unions/bytes and serial/fused output comparisons.
- Full fixed-serving acceptance used every original checkpoint layer (32),
  H65536/A128, chunk1024, P65536/NH16777216, and two users over two visits with
  changed candidates. One eager HBM backend supplied four reference outputs;
  all 16 outputs from independent graph HBM/dense-prefetch/serial-sparse/overlap
  backends matched them bitwise. Exact replay counts, zero eager fallbacks,
  returned-output lifetime, retained history/indexer lengths and candidate
  discard passed. Successful
  runner unbinding and backend close also checked owner/session/lease cleanup.
- Each offload session held actual pinned history K/V of 2 GiB. Revisits hit
  retained histories and forced main-KV host reads: dense fetched 2 GiB per
  revisit; serial and overlap fetched 208,502,784 and 128,942,080 bytes for the
  two users. This prevents a fully resident revisit from standing in for host
  consumption acceptance.
- The separate existing owned-cache H65536/A1024 test propagated resident and
  offload prefixes independently through all 32 layers. Every extend hidden
  element matched bitwise, maximum absolute error zero; its existing tolerance
  was 0.016.

Evidence and the frozen post-gate source snapshot are in
`/tmp/nosa_register8_acceptance_stage_20261004/` (`handoff.md`, `evidence.json`,
`source_manifest.json`, `source/`). Short-path operator/sanitizer evidence is in
`/tmp/nosa_short_deferred_stage_20261004/handoff.md`. The temporary full-graph
driver needed one fixture correction: load weights outside inference mode so
parameter versions remain trackable, then run inference normally. The complete
gate was rerun successfully; no runtime source fix was needed. Failed setup
attempts are excluded from acceptance.

Accepted formal run `nosa_motivation_sm90_20261004_02` still describes its original
source digest
`6e3dfd17a86dd87be4ec89f0bfccc9bb25a52c5af773f5a5f2ed0753ba5c3e0c`.
That measured implementation deferred only `count >= 2047`; it cannot establish
the short-path or register8 performance. Preserve its measurements, source
snapshot and report materials until a replacement passes the complete measurement
and reporting gates. No performance result or compact-snapshot adapter acceptance
is supplied by this correctness checkpoint.

## Contract

Reproduce the DeepSeek motivation workload configuration with all 32 actual NOSA
checkpoint layers: H=65536, A=128, history chunk=1024, 16 users, two sequential
rounds, P=65536 historical HBM tokens per layer and NH=16777216 host history
tokens. Compare HBM-only, dense prefetch, synchronous sparse fetch and asynchronous
sparse fetch under the same inputs and capacity semantics. Preserve complete NOSA
selection, CIS, causal attention and independent session state. HBM-only uses a
session LRU with P historical token quota. Offload uses NH host admission and
finite per-layer P storage, plus explicitly accounted transient execution space.
Do not substitute existing budget-mode shared full-address staging for P pools.

NOSA currently returns candidate hidden states; DeepSeek additionally evaluates
the last-token head. Record this distinction, or add a matching head to all NOSA
schemes before claiming the same output computation. BF16 NOSA is not FP8 DeepSeek.

Performance gates require measured evidence, not merely implemented options:

1. Independent empty-cache prefix construction and every candidate hidden output
   checked against HBM. Include interleaved users, history changes, transient
   discard, errors, finite capacity admission, and asynchronous lifetime fences.
2. An unprofiled complete trajectory for all four schemes with identical warmup,
   inputs and timing boundaries; preserve all valid evidence until replaced.
3. A matching diagnostic trace and matrix API benchmark. Report effective matrix
   FLOPs (projection, compressed score, sparse QK/PV) and explicitly named
   projection-only diagnostics. Compare wall MFU and API MFU on matching work;
   no dense attention FLOPs may stand in for sparse work.
4. Identify and reduce CPU launch gaps and non-compute/IO overhead based on trace
   evidence. Re-run correctness and affected performance after implementation
   changes. CUDA Graph setup and memory must be explicitly accounted, with no
   silent eager fallback or mutable cache state inside pure computation graphs.
5. For asynchronous sparse, verify unique host reads, internal copy/compute
   intervals (page envelope and nonempty stripe windows), and overall latency
   against full-query union synchronous sparse. Execution-window intersection
   alone does not establish useful overlap.
6. Record allocated, reserved, device-used observations and actual pinned
   allocation capacities. NH alone is a token quota, not a claim that all NH
   sessions fit a 512 GiB DRAM planning budget; 32-layer NOSA main KV alone at
   NH=16777216 consumes 512 GiB before metadata if completely populated.

## Ownership

- Root: pure-compute launch optimization, integration, complete acceptance.
- `experiment`: new `experiments/nosa_motivation` measurement/report entrypoints.
- `cache`: fixed capacity backend and persistent residency/lifetime support.
- `hbm_audit`: real-checkpoint baseline, profile and correct FLOP accounting.

Existing extensive unrelated worktree changes are preserved. Model/operator
changes must identify dependent experiments and mark them pending remeasurement;
old performance numbers cannot be relabeled as new implementation results.

## Initial authoritative observations

No NOSA motivation directory or live GPU job existed on entry. Eight SM90-class
M403 devices were idle, with 143771 MiB reported per card. CUDA device properties
and checkpoint identity still need to be captured with actual runs. The previous
shared NOSA correctness work explicitly did not measure its final serving
performance and has no finite slots. The last DeepSeek motivation publication is
`motivation_c10_20261004_u16_r2_01`; its numeric performance cannot validate NOSA.

## Implementation checkpoint, 2026-10-04

The fixed backend and experiment entrypoints now exist. Offload uses per-layer
logical-offset K/V pages with independent per-head session tags. Complete
histories and prefix chunks must be multiples of 64 and H<=P; a candidate is
transient, with no main-KV host write. NH admission is a global quota with lazy
per-session backing. This is explicitly a direct-mapped page policy.

Executed correctness evidence before the compute-graph integration:

- New/default offload operator suite: 56 GPU tests passed on GPU0, including
  hit/conflict/tail behavior, page-zero padding invalidation, dense copy streams
  and submission failure followed by retry.
- Fixed backend full NOSA checkpoint: 32 layers, H65536/A128, four schemes,
  independent histories and users0/1/0/0 passed on GPU2 (pytest39.78s, not a
  latency result). All candidate hidden elements were bitwise identical to HBM;
  repeated hits had zero main-KV H2D and candidates had zero main-KV D2H.
- Native shorter-history conflict tests passed, including H8192 -> H4096/A81
  -> H8192. Dense and CPU-reference suffix writes now invalidate tags belonging
  to overwritten longer histories.

Subsequent compute-graph work splits each layer into projection (normalization,
QKV/RoPE/CIS) and finish (output projection, normalization, MLP), with eager cache,
indexer and main attention. The model copies final graph hidden output so later
replays cannot mutate previously returned results. Capture tracks weight/version,
precision and configuration identity. Static allocation and private reserved
segments are accounted separately; close returns inactive graph pool segments.
Graph setup failure releases traceback-held private aliases before cleanup.

Temporary diagnostic evidence is not a formal experiment publication:

- Old generic HBM candidate median98.773ms had an expensive full-GC allocator
  guard twice per lease. cProfile located293ms of its333ms instrumented request
  in that guard, not a directly subtractable unprofiled overhead estimate.
- Incremental generation-aware GC checks preserve new/private pool rejection;
  hot complete allocator validation was observed at median0.0272ms in a
  controlled process with582578 tracked objects. Owned graph pools are explicitly
  registered; foreign private pools remain rejected.
- Fixed HBM eager candidate median26.885ms; pure-compute graph median20.496ms
  on the same full-checkpoint random-token diagnostic. The graph profile has
 12.244ms of device active union in a28.487ms hull. Its full compute API union
 11.902ms includes helper work; it is not an isolated raw-matmul API denominator.
  The candidate wall/API ratio remains about58%, so the requested performance
  outcome is NOT complete.
- Root's independent full32-layer HBM graph check reconstructed H65536 from an
  empty cache and compared all A128 hidden elements bitwise with eager, then
  checked five replay results and previously returned output lifetime. Its
  cold eager timing included compilation and cannot support a pre/post speedup.

Work remaining at that checkpoint: finish graph exceptional cleanup checks,
remove duplicate native attention validation without weakening rejection,
combine offload Q/newK/newCIS
finite checks without removing nonfinite rejection, then freeze sources and run
full four-method numerical/performance acceptance and matching profiles.

## Earlier deferred-validation checkpoint

The wrapper/finite-helper version passed the full 32-layer checkpoint comparison
for all four graph schemes at H65536/A128, users 0/1/0/0, against independently
constructed eager HBM histories. Every candidate hidden element was bitwise
equal. Candidate main-KV D2H was zero; conflicting-user recalls fetched data and
an immediate repeated user fetched no historical main KV. This acceptance used
runtime digest `246fda3791b89a73acb79b615e1debc889b6aca447a41cd8e1c53aee5b939cf7`;
it predates the following deferred-validation change.

Graph-enabled fixed serving now retains one GPU finite flag per layer and one
reduction result in explicitly accounted graph-owned static storage. The model
checks their conjunction before committing a step or returning an output.
The native asynchronous ranked preparation/selection endpoint guards all
downstream kernels. Failed input leaves derived records/ranking unchanged and
emits IDs=-1/mask=false, avoiding reads of uninitialized ranking. Ordinary
synchronous APIs preserve their immediate rejection and output-byte contract.
Offload may construct unpublished append records before the host observes its
flag; step rollback preserves committed history and indexer records. Small
resident prefixes retain the existing synchronous preparation check. Cache and
IO transactions remain outside CUDA Graph capture.

New validation so far: existing graph/checked-native suite 68 passed; native
asynchronous suite 21 passed, including both score/selection and fused/pruned
paths; graph failure/owner suite 25 passed across four schemes, Q/K/CIS, early
and later layers, candidate H8192 and prefix start4096. The latter checks unchanged
committed history/indexer, aborted pending cursors, release and exact recovery.
Final H64K four-method checkpoint validation is being repeated for this source.

The allocator guard now uses CPython 3.12's C-level referrer traversal of MemPool
and every subclass for mandatory full scans, retaining incremental young-generation
checks and the old conservative fallback. Its selected regression suite passed
281 tests; real MemPool/no_split subclass probes reject pools in all generations.
Controlled full-scan validation dropped from roughly39ms to7ms; this is a guard
diagnostic, not an end-to-end performance result.

The complete H64K+A128 workload has a different efficiency balance from candidate
alone. Before deferred validation, three new-empty-history backend samples had
median2387.084ms and49.393% useful matrix MFU. The matching complete compute API
union2059.315ms included helpers. Independent shape-weighted matrix APIs plus
separately measured fused attention gave a2133.969ms composition estimate in one
diagnostic; subsequent raw API sampling showed clock/run variation. These are
engineering diagnostics outside experiments, omit runner admission/input upload,
and cannot replace the formal trace or a final matched reference.

The first deferred version failed the final multi-user test: graph HBM user 1
first differed in selected blocks at query_start20480/layer5. The new predicate
had expanded the joint native path below2047 compressed keys, replacing the
established Triton scoring path for short1024-query prefix chunks. This changed
rounded selection and later hidden states. The original count>=2047 predicate
has now been restored in the model and native endpoint. Deferred validation
applies only within the original joint-native dispatch; short-prefix finite
checks remain synchronous. The failed version's performance diagnostics are
invalid for conclusions. Its native same-path tests did not establish equivalence
to the prior Triton path; the full independent-prefix multi-user test is required
again before any performance publication.

A transient external worktree/source staging change was observed and the source
subsequently returned to the current implementation. No team agent reported
performing that swap; source hashes and current authoritative files must be
checked around acceptance. Do not restore the entire worktree or overwrite the
large independently staged DeepSeek changes.

## Earlier acceptance checkpoint before the short-prefix update

The corrected dispatch-preserving runtime passed full 32-layer independent-prefix
HBM versus all four graph schemes at H65536/A128 for users 0/1/0/0. All candidate
hidden tensors matched byte for byte. Conflict recall, immediate same-user hits,
candidate main-KV D2H=0, graph coverage and retained lengths also passed. Runtime
digest: `ec453e213f755712a774204b859bf6fdfc1d8f56ca3343ef84885e748f33e9d4`.
Evidence remains in
`/tmp/nosa_full_graph_preserved_dispatch_acceptance_20261004/evidence.json`.
The final H64K failure/owner suite passed 25 tests, the checked/asynchronous native
suite passed 70, and the global CPU regression passed 3012 tests with 1130 GPU or
optional skips and 58 subtests. These are engineering checks, not experiment
latency results.

Formal attempt `nosa_motivation_sm90_20261004_01` was stopped before publication
because its report compared the graph-wrapped resource plan with the base cache
plan and omitted graph charges from the shared-storage equality. Its partial
measurements remain outside experiments and cannot support conclusions. The
report/profile fix now checks the base plan separately and charges graph static
allocation plus private reserved capacity; 110 experiment tests and an independent
cache review passed.

The complete 32-layer small pipeline measurement
`nosa_pipeline_graph_smoke_20261004_01` passed at H4096/A128, P4096/NH8192,
two users and two rounds, with all 16 outputs exact. Its artifacts are under
`/tmp/nosa_pipeline_graph_smoke_20261004_01`; this is harness validation only.
Matching profile replay is in progress. Before the formal replacement, strengthen
warmup auditing to require actual main-KV recall rather than total H2D, which also
includes the indexer's boundary K copy.

The independent matrix reference still needs attention QK/PV coverage. Add both
raw materialized selected-QK/PV BMM and independent resident FA3 on actual captured
operands, with complete layer/invocation coverage. Report duplicated K/V and padded
matrix work in the raw reference. Beating that slower materialized reference alone
does not establish the requested efficiency; retain the FA3-based composition and
the measured activity/gap inventory as separate evidence. The goal remains active.

## CPU regression and source-provenance checkpoint, 2026-10-04

The final `bash scripts/run_tests.sh cpu` rerun completed with script exit 0:
3121 passed, 1272 skipped, 1 warning and 58 subtests in 76.24 seconds. Results
were reported in the terminal; this checkpoint creates no test output files.
The skipped cases do not establish GPU or optional-path acceptance.

The minimal experiment provenance fix adds `nosa_guarded_buffers.cuh` to the
current native source inventory in `indexer_block_sparse_profile` and stamps
new captures with source inventory version 2. Older model-layout inventories
remain accepted only when all 13 recorded source hashes match the verified
pre-guard include closure at Git revision
`4a02aef12dcf60d78625242856c2196e518b7315`; removing the header and version from
current metadata cannot bypass this check. The historical flat-SM90 validation
branch is unchanged. All 631 affected experiment tests and Ruff passed. Rebuilding
the published native and Triton MFU phases with their recorded peak parameter
produced unchanged results. No report or measurement artifacts were replaced.

The fix changes experiment capture/analysis provenance and its tests only.
The motivation runtime source digest remains
`02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`.
This CPU checkpoint supplies no new performance result and does not promote
the staged Group4 candidate. The accepted measurement and outstanding
remeasurement scope above remain unchanged.
