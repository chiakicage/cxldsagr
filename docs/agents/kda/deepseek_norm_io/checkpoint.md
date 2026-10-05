# Norm I/O investigation checkpoint

Date: 2026-10-04. Current status: temporary plain typed-I/O prototype passed
independent source review, strict screen and complete component validation.
The owned-output fused prototype also passed its strict screen and complete
component validation. The complete-API benchmark and its independent audits
passed. The production BF16 adapter passed focused GPU correctness and live
runtime identity checks. The combined C8 checkpoint/output, graph-memory and
lifecycle gate also passed; its scope and independent saved-record audit are
in the C8 validation record（Git `934485b:docs/agents/system/deepseek_motivation_c8_validation.md`）.
Formal C8 serving/profile measurements remain deferred.
The separate packed norm candidate remains unimplemented.
The plans are
[draft.md](draft.md) and [implementation_plan.md](implementation_plan.md).

## Temporary plain prototype

Root subsequently authorized CPU-only implementation under
`/tmp/deepseek_norm_adapter_20261004/`. `local_plain.py` overrides only the
vendor kernel method; it inherits the official Float32 constructor and launch
method. X/W/Y atoms use their own dtype widths with the same four logical
values. The probe checks each static TV/register layout against the canonical
Float32 partition during compilation. All seven checks passed for each of the
three initial screen specializations.

`probe.py prepare` completed as `prepared_02` without importing FlashInfer or
CuTe, instantiating the vendor class, or initializing CUDA. It loaded all 13
real FP32 norm vectors and checked 72 CPU fixtures, including the KV latent
row stride of 576, offset views, epsilon-sensitive rows and unrounded sums.
The local scalar arithmetic AST matches the installed vendor kernel, and the
source delta is saved in `prepared_02/kernel_source_delta.diff`. Ruff,
`py_compile` and `--help` passed; these do not establish DSL compilation or
numerical correctness.

The original `probe.py` CLI supports only `prepare` and `screen`. The screen first
compares local FP32 input/output against the public CuTe baseline, then BF16
input with FP32 diagnostic output before BF16 output. It preserves compiler
errors and rejects any bit mismatch. The wrapper owns outputs and explicitly
packs noncontiguous or insufficiently aligned inputs in their original dtype.
At that initial gate, full numerical/lifetime/graph validation, fused outputs
and timing were still pending; subsequent validation is recorded below. The
implemented row-strided path uses explicit packing. No installed or production
source changed.

After independent CPU source review passed, root granted GPU 0 for the strict
screen only. `independent_source_review.json` records that review.
`screen_01/result.json` records exact agreement for all 64,512 elements in each
of FP32-input/FP32-output control, BF16-input/FP32-output diagnostic and
BF16-input/BF16-output. The shape was `[9,7168]`, weight was the actual source-0
input norm FP32 vector, epsilon was `1e-6`, and all inputs were contiguous and
aligned. All values were finite, caller input/weight stayed unchanged and each
output owned its allocation. No input packing occurred in this screen.

All three specializations compiled with the unchanged Float32 constructor,
vec_size=4, PDL enabled and a 28,688-byte launch shared-memory budget. Observed
register layout was `((4,14),1,1):((1,4),0,0)`, and complete TV/register maps
matched the canonical FP32 maps. Actual device identity was NVIDIA H200,
150,121,545,728 total bytes. Source hashes were unchanged. Process 1806180 in
exec session 99218 exited 0 and released GPU 0. Separate logs are
`screen_01.stdout.log` and `screen_01.stderr.log`; stderr was empty.

Returned PTX paths were unavailable despite requested retention and are marked
unavailable; no PTX instruction claim follows. This initial screen does not
validate other widths, weights, row counts, epsilon values, packing paths,
retained-output/stream/graph behavior or complete-model outputs. No timing or
performance inference was made.

| Temporary source | SHA-256 |
|---|---|
| `local_plain.py` | `1d8c69b6db227670af7fb6831769ea425e8c415bd8bd886a0886f30593fa1353` |
| `probe.py` | `6e2873b9d7de5d87332f4203a1971310c98eac4e47e1dfe1a2f05baaa1f25dfe` |

## Broader and fused preparation

During root's C7a+hint formal trace, all CUDA work was held. New temporary
`validation.py` and `local_fused.py` were prepared on CPU; the accepted plain
kernel and initial `probe.py` remain unchanged. `broad_prepared_04/prepared.json`
records 60 CPU shape/epsilon fixtures, no FlashInfer/CuTe import and CUDA
uninitialized. Ruff and `py_compile` passed. This preparation preceded the
independent review and fused GPU results recorded below.

The fused subclass inherits the vendor Float32 constructor and overrides its
launch/kernel to accept two independent output pointers. CPU AST checks prove
that the launch matches the vendor after removing those two arguments and that
the FP32 input conversion, h=x+r, square/reduction, epsilon/rsqrt and final
multiplication expressions match. The saved-output cast is separate; h stays
FP32 through normalization. Twelve complete TV/register-map checks are planned
at compilation. Initial fused scope is width 7168 with the unchanged 57,360-byte
Float32 launch budget; the actual constructor must confirm it on the GPU.

The broader CLI supports `prepare`, `screen` and `validate`. Validation requires
an exact-source passing screen record and rechecks the initial numerical gates.
It covers all 13 plain norm vectors or six input/post-attention fused vectors,
Q=1/9/121/128/1024, four epsilons, BF16/FP32 input combinations, contiguous,
row-strided and offset inputs, and FP32 diagnostics before BF16 outputs. It
checks complete input/weight backing storage and independent output allocations.
Non-default-stream tests retain earlier outputs while overwriting inputs;
Q128 graph tests replay changed inputs, preserve a clone across replay, check
output pointer identity and record full graph-private reserved segments plus
allocated/reserved/device-used snapshots. Those snapshots are diagnostic, not
physical-capacity acceptance. Full serving graph integration remains separate.

Full validation audits the complete Cartesian-product case keys rather than
only reporting executed rows. Plain requires 3,120 unique matrix cases, fused
2,880, with equal FP32-diagnostic and BF16-output counts. The three initial
gates are reported separately. The full-matrix completion flag stays false
until every required case passes. Lifetime checks require 36 plain or 48 fused
cases, covering the input dtype combinations as well as layouts/output types.

| New temporary source | SHA-256 |
|---|---|
| `local_fused.py` | `4154101d90dc55f824e2498eaa6b56c6d1f21f0f26fed9b9a7bf0f77d1e1c565` |
| `validation.py` | `2b3e755aff332c13e507cf7344e4dcd797cb5b64bc7894a1b01bcbb0c867428e` |

The broad/fused CPU source review passed at the prior harness identity
`2ea19b2b...`. The first broad plain run, `plain_validate_01`, then stopped on
a fixture ownership bug after three initial and four matrix output cases:
inference-mode tensors omit `_base` tracking, so the offset fixture was rebuilt
over compact storage and failed a storage-bounds check. It reported no numerical
mismatch and kept `completed_full_numerical_matrix=false`. Process 1824729 in
session 24881 exited 1 and released GPU 0. This is a failed harness run, not a
kernel rejection or broad validation result.

The harness now reconstructs an explicit flat typed view over the complete
underlying CPU storage before upload, preserving guards and offset. An additional
36 CPU cases under inference mode check backing capacity and reconstructed
values. `broad_prepared_04` passes without CUDA; the fixture-only delta passed
independent review in `independent_backing_delta_review.json`. Both kernel
sources remain unchanged.

## Broad plain result and fused screen

`plain_validate_02/result.json` passed all 3,120 unique matrix cases, split into
1,560 FP32 diagnostics and 1,560 BF16-output cases, comparing 3,468,410,880
normalized elements across those repeated fixtures. The three initial gates
are separate. All 36 retained-output/non-default-stream/graph cases passed.
Twelve width/input/output specializations compiled, with actual row-stride and
offset packing exercised. Source hashes stayed unchanged and stderr was empty.
Process 1828482, session 37011, exited 0. This is component validation, not
full-model or physical-capacity acceptance.

`fused_screen_01/result.json` then passed its three initial gates: FP32 X/R to
FP32 Y/saved, BF16 X/R to FP32 Y/saved, and BF16 X/R to BF16 Y/saved. Each gate
matched 64,512 normalized and 64,512 saved elements. All twelve canonical
layout checks passed per specialization. Source hashes stayed unchanged and
process 1830485, session 77990, exited 0. The test used actual source-0
post-attention norm weight at Q9/D7168 and epsilon `1e-6`.

Under root's same correctness-only grant, `fused_validate_01/result.json`
passed all 2,880 matrix cases, split into 1,440 FP32 diagnostics and 1,440 BF16
outputs. Both normalized and saved outputs compared 5,297,209,344 elements
across those repeated fixtures; three initial gates are separate. All 48
lifetime/graph cases passed, covering all input dtype combinations, layouts
and output dtypes. Eight specializations compiled. Source hashes stayed
unchanged and stderr was empty. Process 1830964, session 3800, exited 0 and
released GPU 0. Returned PTX paths remained unavailable. No timing or
production integration has been performed.

Root independently audited both saved full results in
`independent_full_result_audit.py` / `.json`: complete expected matrix and
lifetime key sets, all zero bit/value errors, owned storage, counts and all
18 source hashes per result passed. Its initial BF16 `max_ulp` field assumption
was corrected because only FP32 comparison records include ULP; the rerun
passed. That record/source audit used no GPU.

## Complete-API benchmark preparation

`benchmark.py` is separate from all four frozen implementation/validation
sources. Its current SHA-256 is
`98daa59211e240cdb8869ab71c8340c353ae90085f431c9acb0039ffab4bd7dc`.
`benchmark_prepared_04/prepared.json` passed CPU preparation with 16 selected
fixtures and 64 planned variant/execution checks. Ruff, `py_compile` and CLI
help passed; CUDA remained uninitialized and FlashInfer/CuTe were not imported.
Independent CPU review passed in `independent_benchmark_source_review.json`.
The subsequent GPU driver check passed as recorded below. No timing has run,
and no GPU window is currently held by this agent.

The default matrix uses actual source-0 input, query-A and KV-A norm weights
for plain widths 7168/1536/512, plus source-0 post-attention fused width 7168;
Q128/1024, BF16, contiguous/row-strided input, epsilon `1e-6`. The KV fixture
retains the actual 576-element row stride. All selected fixtures must match
passing full-validation keys. The complete ordinary model API is the baseline.

Eager calls include allocation, conversion and packing. Owned graph calls
include caller-to-static copies, complete API replay and a fresh clone of each
returned output. Captured outputs remain private. Full caller/alternate/weight
backing storage is guarded from before capture through post-sample checks;
actual returned outputs survive changed-input calls. Graph checks preserve immutable pre-capture
static-storage guards, verify disjoint ownership and place captured outputs inside
the recorded graph-private segments. Every expected check and sample key is
audited. Before/after exactness records are retained separately.

Wall and CUDA-event samples are separate; event recording is absent from wall
samples. The default protocol uses five warmups and 30 repetitions, alternating
baseline/local order. Compilation, capture, correctness, resource observations
and optional profiler passes are excluded from timed samples. Resource records
report full output storage, graph-private reserved/active bytes and process
allocated/reserved/device-used observations. Other fixture/executor objects
coexist, so these observations do not establish isolated or serving capacity.
See [benchmark_plan.md](benchmark_plan.md) for commands and promotion gates.


Independent review caught a static-guard checking gap before CUDA execution:
the first driver copied the current graph backing storage into its expected
state before each check, which could adopt padding corrupted during preceding
samples. The final driver holds immutable full-owner snapshots from before
warmup/capture. The reviewer injected corruption into four CPU BF16/FP32
row-strided/offset fixtures and confirmed that the revised expectations detect
it after a valid logical input copy. A later metadata-only correction identifies
both plain and fused constructors/launches and removes the inherited initial-
screen-only flag. Neither change modified a validated kernel or arithmetic.

`benchmark_check_01/result.json` then passed the root-authorized GPU 0 check:
64 unique fixture/execution/variant keys, both check phases exact, including
32 owned graph cases with private-pool containment and static-storage guards.
Four local specializations compiled. The record contains zero timing samples,
`source_changes={}`, and empty stderr. Session 13698 / PID 1853063 exited 0 and
released GPU 0. Result SHA-256:
`30a32afdfe5eafc12cfc206d67be3b0b6b87067efb65572328b8d98d9727828f`.
Graph-profile's independent result audit passed in
`independent_benchmark_check_audit.json`: 64 keys, 128 check phases, 32 graph
cases and 16 distinct baseline/local pool pairs. Root then granted the quiet
component measurement below.

## Complete-API component result

`benchmark_measure_01/result.json` completed under root's exclusive GPU 0 grant;
session 90159 / PID 1856028 exited 0 and released the device. Result SHA-256:
`946f63dea505c95732dc4bfca1c23df7fde020fa44c66b3d6c90771a2e6a16dd`.
It retains 32 measurement keys, 3,840 raw samples, 64 separate profiler traces,
and 64 before/after exactness records. Source hashes stayed unchanged. The
independent audit `independent_benchmark_measure_audit_graph_profile.json`
reconstructed every key/order, recomputed all summaries, and rehashed/read all
64 traces. It identified 184 kernel activities and 60 separate GPU memcpy
activities; copy/clone work was not omitted from complete API samples.

Local wall medians were lower in all 32 comparisons, ranging from 1.025 to
2.250 times baseline/local speedup. Event medians were also lower in all 32,
ranging from 1.016 to 2.294 times. Selected wall medians in microseconds:

| Operation | Q | Layout | API | Baseline | Local | Ratio |
|---|---:|---|---|---:|---:|---:|
| Plain D7168 | 1024 | contiguous | eager | 60.721 | 45.620 | 1.331 |
| Plain D7168 | 1024 | contiguous | owned graph | 73.946 | 41.666 | 1.775 |
| Plain D512 | 1024 | stride 576 | eager | 52.237 | 49.064 | 1.065 |
| Plain D512 | 1024 | stride 576 | owned graph | 34.139 | 32.286 | 1.057 |
| Fused D7168 | 1024 | contiguous | eager | 115.937 | 57.735 | 2.008 |
| Fused D7168 | 1024 | contiguous | owned graph | 153.946 | 68.406 | 2.250 |
| Fused D7168 | 1024 | row-strided | eager | 124.843 | 78.376 | 1.593 |
| Fused D7168 | 1024 | row-strided | owned graph | 184.768 | 129.204 | 1.430 |

The complete CPU summary is
`benchmark_measure_01/analysis_graph_validate/summary.csv` / `summary.json`,
generated by `summarize_benchmark_graph_validate.py`. It includes every row,
separate p10/p90 distributions, trace kernel/memcpy counts and contextual graph
storage. These single-run component fixtures use real checkpoint weights and
synthetic inputs. Owned graph timings include static copies and owned output
clones; production compute-island graphs have separate lifetime/boundary rules.
No full-model, serving speedup, physical-capacity or PTX-instruction claim follows.

Before production changes, both timed `nonmatrix.py` and
`operators/flashinfer.py` were verified byte-identical to their copies under
`/tmp/deepseek-motivation-c7_hint-frozen-a44l_2uk`. The mapping and hashes are
retained in `benchmark_original_source_mapping_graph_validate.json`, so old
timing identities remain reconstructible after live paths change.

## Production integration and focused acceptance

The source plan is [integration_plan.md](integration_plan.md). Root accepted
component evidence and authorized narrow BF16 production integration. The new
`operators/deepseek_v32/norm/` reuses the measured plain/fused class ASTs exactly;
temporary trace printing/file writes became a local compile-time layout check.
The normal model path selects eligible nonempty 2D BF16 SM90 inputs at actual
widths, with aligned contiguous or `width + 64` row-strided layout and FP32
weights. FP32, CPU/autograd and unsupported shapes/layouts retain the ordinary
adapter. Fresh output allocations and current-stream TVM FFI execution remain
call-local. The separate packed SiLU helper consumes existing packed gate/up
storage for Cache-C3's MLP integration.

Fingerprint
`8b8a3e6128f23d6d65f6aca3993723c09c74fc9cf0c4da75f21742f1a5630b26`
covers 380 source labels plus versions. The process compiled-callable memoizer
keys this identity and specialization; a supported compile-scoped hook writes
it into live IR. Direct `cute.compile` sets `no_cache=True`; no DSL disk-cache
claim is made. `production_prepared_01.json` confirms identical measured class
ASTs and equal standalone/post-compiler-import fingerprints without CUDA
initialization. Root independently checked the ASTs and source design.

In the root-granted GPU 0 run, all 71 focused tests passed with zero skips in
4.77 seconds: exact direct-vendor norm comparisons, full input guards, output
retention, stream/graph execution, fallback behavior and packed SiLU checks.
The driver additionally checked all four live IR source attributes. The final
temporary report assertion then used `runtime['cute']` instead of the collector's
`runtime['cute_jit']` and raised `KeyError`. Session 23051 exited 1 and GPU 0 was
released. This was a postprocessing-only error; its failed status is preserved
in `production_gpu_check_01.json`, SHA-256
`95add2dc14be7232e52a0eaedc78d631f592c23cef9f7ffda77e1be65f154b70`.

The result had already saved complete runtime and IR evidence. The corrected
CPU audit `production_gpu_check_01_runtime_audit_graph_validate.json` verified
four exact seven-field specialization keys, all four live fingerprint attributes,
MLIR bytecode identities, zero timing samples and 381 unchanged unique source
paths. Root independently repeated that saved-data audit in
`independent_production_gpu_audit_root.json`. No production source changed and
no GPU rerun was needed to resolve the report-field error.

Root's separate global CPU regression exited 0 with 2,778 passed tests, 58
subtests and 1,005 explicit skips. The production source/test files remain
frozen for root's combined checkpoint and serving gates; this agent holds no
CUDA window.

## Existing evidence

- Vendor Triton public and no-FMA variants failed the first normalized FP32
  screen despite matching BF16 rounding. They are not candidate replacements.
- Direct unchanged-class CuTe with BF16 X / FP32 W / FP32 Y failed compilation
  with `profile of input tuples doesn't match: ((4, 2), 8)` at y multiplication.
- The separate instance-only copy_bits 128->64 variant failed `cute.copy`
  predicate shape verification. No BF16 candidate kernel executed.
- Both direct FP32 controls matched all 64,512 values exactly. The two screen
  processes finished and GPU 0 was released. No timing was performed.
- Temporary evidence remains under `/tmp/deepseek_norm_probe_20261004/`:
  `cute_mixed_screen_01/result.json`,
  `cute_mixed_copywidth_screen_01/result.json` and `cute_mixed_rejection.md`.
  Missing returned PTX paths are marked unavailable; no PTX instruction claim
  is made for them.

CPU source inspection found no load/store hook in the monolithic vendor
methods. A local kernel-method override can reuse constructor/launch/reduction
helpers, but a constructor-only subclass cannot supply typed copies or owned
fused outputs. The plan explicitly acknowledges this local maintenance scope.

CPU-only arithmetic checks confirmed exact BF16-to-FP32 widening into a packed
output and demonstrated that `torch.add(BF16,BF16,out=FP32)` rounds too early.
No GPU behavior is inferred from those CPU checks.

The independent cache agent inspected the installed PyTorch source commit
`7269437d655783a26cba32aa88195b741ff496aa`: TensorShape.cpp lines 278–285 and 311,
and CUDA Shape.cu fast paths at 595/624 and fallback at 657–664. Different input
and output dtypes select per-input widening copies, so a packed fused wrapper
can save one output conversion launch, not both input casts. Fetched source
copies are `/tmp/pytorch_tensor_shape_for_norm_review.cpp` and
`/tmp/pytorch_shape_for_norm_review.cu`.

KernelWiki was queried on CPU for mixed-type CuTe RMSNorm. Returned pages were
FP4/FP8 quantization work, which does not satisfy this task's exact BF16/FP32
contract; no API support or performance inference was taken from those pages.
The installed source and actual compiler rejections remain the direct evidence.

## Reviewed source identities

| Source | SHA-256 |
|---|---|
| `models/deepseek_v32/nonmatrix.py` | `69384604478ebc22c67784f0fa150b1c42f7a61a4a05723fe5ab6c3d3848c867` |
| Installed `flashinfer/norm/kernels/rmsnorm.py` | `b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0` |
| Installed `flashinfer/norm/kernels/fused_add_rmsnorm.py` | `f6f4a7d9c88996f26c33ed2661f82026e4ee3e747bf12c57febe9f9e68e61435` |
| Installed `flashinfer/norm/utils.py` | `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a` |

Installed files are under `/root/.venv/lib/python3.12/site-packages/`.
These identities cover the review, not a full future build manifest. No
production or installed source was changed for this investigation.
