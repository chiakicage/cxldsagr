# Separate candidates: activation scale layout and call-local reuse

Status: CPU source review and plan only. No SFA or projection-sharing candidate
has been implemented or run. The current baseline is the integrated handwritten
Triton T1 quantizer, whose accepted source and validation evidence are recorded
in [checkpoint.md](checkpoint.md). Root is measuring formal C9 and will select
the next candidate after its matching profile. The earlier native candidates
and compiled helper are historical evidence, not this plan's baseline.
Root schedules GPU work and owns full-model/report updates.

## Contract and source evidence

Reduce redundant activation-scale transformations in the existing packed/shared
MLP while preserving separate gate/up weights and GEMMs, current activation
quantization, SiLU/down arithmetic, source identity checks and one-use prepared
activation lifetime. The public `quantize_fp8_activation` result remains owned,
contiguous FP8 data plus contiguous FP32 scales. CPU behavior and grouped MoE
paths retain their current contracts.

Pinned DeepGEMM 057ca596 exposes the public function
`transform_sf_into_required_layout`; its source is
`3rdparty/DeepGEMM/csrc/apis/layout.hpp`. For the current recipe `(1,128,128)`,
activation SFA takes the FP32 1-by-128 path in
`csrc/jit_kernels/impls/smxx_layout.hpp::get_mn_major_tma_aligned_tensor`.
Weight SFB takes the 128-by-128 layout-validation path. These transpose launches
therefore transform activation scales, not weight scales.

For logical `[M,G]` SFA, where `G=ceil(K/128)`, the no-copy path requires exact
element strides `(1,align(M,4))`. Four FP32 elements satisfy the 16-byte TMA
stride alignment. Merely being noncontiguous is insufficient; another layout
can trigger an allocation and `copy_`. The public call for a gate input is:

```python
sfa = deep_gemm.transform_sf_into_required_layout(
    row_major_scales, M, K, (1, 128, 128), is_sfa=True
)
```

Use the actual padded K consumed by GEMM. Do not call an unexported convenience
function or change the GEMM recipe. Confirm the installed public API identity
and return layout in the first screen.

The historical packed/shared trace cited in [semantics_review.md](semantics_review.md)
contains three SFA transpose launches per full MLP: two with G=56 for gate/up
and one with G=144 for down. This is motivation for a new measurement, not a
performance claim for the current Triton T1 baseline.

## Draft: two separately evaluated changes

**S1, reuse the official transform.** In the existing `fp8_linear` path that
returns a prepared activation, quantize once, invoke the official SFA transform
inside the gate call, and pass the transformed tensor to gate GEMM. Retain that
same tensor in `_PreparedFP8Activation`; up consumes it once through the
existing source/pointer/version checks. Both public GEMMs should take the exact
layout no-copy branch. The source row-major scale allocation should become
unreferenced after the transform has been enqueued safely on the current stream.
Prepared FP8 data/transformed scales and packed gate/up output remain released
before down, as in the accepted MLP implementation.

The expected fresh trace signature is one gate SFA transform, none for up,
and the existing down SFA transform: three becomes two. The explicit helper's
Python/C++ validation and launch cost must stay inside the gate and full-MLP
timing boundary. Ordinary linears and public quantizer outputs keep their
existing layout. A transform-count reduction alone does not justify promotion.

**S2, direct internal SFA emission, optional after S1.** Add a separate private
Triton quantization entry or explicit internal output mode for the prepared
gate/up path. It writes the exact same FP32 scale bits at
`scales[group * align(M,4) + row]`, with logical shape `[M,G]` and the required
strides. FP8 data remains contiguous. The public quantizer continues to use its
contiguous-output entry. Compare S2 against both the accepted baseline and S1;
never infer their cumulative gain by adding isolated deltas.

First restrict this mode to the same aligned BF16 inference domain as the
prepared-activation option. Keep all other public dtype/layout/tail behavior
in the accepted generic path. Do not expand direct emission to down, ordinary
projections or grouped paths in this candidate. That expansion would require
its own consumer and performance matrix.

With S2, gate/up should have zero SFA transforms; down still has one. A
column-major scale store can be less coalesced than the current group-major
store, so fewer launches may still lose overall. Compare current group ordering
before proposing a different warp mapping. Avoid changing numerical arithmetic
or dispatch geometry in the same initial S2 candidate.

## Storage, stream and graph requirements

- Record logical bytes and actual backing-storage bytes separately. For M>0,
  a minimal `[M,G]` tensor with strides `(1,align(M,4))` requires
  `4 * ((G-1)*align(M,4)+M)` bytes; a full padded `[G,align(M,4)]` owner instead
  allocates `4 * G * align(M,4)` bytes. Measure `untyped_storage().nbytes()`
  and allocator deltas; do not substitute either formula for observed capacity.
- For S2, choose and document one allocation contract before implementation.
  A full padded owner with defined zero padding is the straightforward first
  candidate. Every required initialization, including any extra launch, belongs
  in API timing. Guard the full owner and verify logical writes plus the
  specified padding. Do not write beyond a minimal strided allocation.
- Preserve current-stream ordering for quantization, transformation and both
  GEMMs. No cross-call global scale buffer or storage reuse is introduced.
  Allocator stream lifetime must cover the transform's read of row-major scales
  and both GEMMs' reads of the transformed scales.
- Capture Triton quantization and the helper/GEMMs only after warmup. Compare
  changed-input replay using independent baseline/candidate graph pools; retain
  owned output clones across further replays. Report graph capture time,
  allocated/reserved deltas and retained pool storage separately.
- Preserve existing prepared-activation consumption, source mutation rejection,
  exception cleanup, packed-output alias rules and release-before-down checks.
  Do not weaken those checks to accommodate a layout change.

## Executable work sequence and acceptance

1. Freeze the accepted Triton T1 baseline, including quantizer and packed MLP
   source, DeepGEMM Python/native helper identities and generated GEMM/layout
   artifacts. Record the actual source-0/1/2 checkpoint tensors or synthetic
   input identity, M/K/N, precision, device and flags.
2. Build S1 in a temporary module under
   `/tmp/deepseek_linear_sfa_reuse_20261004/`; leave production unchanged.
   Reuse the accepted quantizer/MLP drivers and assertions, adapting their
   `prepare`, `screen` and `bench` commands only as needed. Their
   record format must retain raw paired samples and fail closed on missing or
   mismatched correctness evidence.
3. CPU-review helper dispatch and ownership; run existing CPU metadata tests.
   After a GPU grant, compare logical FP8 bytes and FP32 scale bits, then exact
   gate/up and full MLP output bytes against the accepted Triton T1 baseline.
   A layout-only change has no numerical tolerance allowance. Include M=1,2,3,4,
   7,8,15,16,17,63,64,65,121,128,129 and 1024 for layout correctness, and actual
   K=7168/G=56 and down K=18432/G=144. Exercise G=1 and awkward small dimensions
   in direct helper checks without forcing unsupported GEMM shapes.
4. Verify helper pointer/stride identity and collect a short untimed profiler
   trace. Check complete MLP launch attribution and the expected SFA counts;
   all other GEMMs/activation kernels must retain their accepted identity.
   Compare actual allocator storage and prepared-object release with baseline.
5. Measure gate+up including allocation/quantization/transform, complete
   packed/shared MLP including SiLU/down, and separately complete graph
   input-copy/replay/owned-output APIs. Use BF16 M=1,121,128,1024 with actual
   hidden/intermediate dimensions 7168/18432 and checkpoint source layers 0–2
   where available. Report wall-completion and joined CUDA-event durations,
   warmup/repeat counts and all raw within-block candidate/baseline pairs.
6. Randomize pair order with a recorded seed. Confirm any improvement in a
   separate exclusive window with an independent seed. A primary full-MLP
   regression or unstable gain blocks promotion regardless of transpose count.
   Keep small-M and eager/graph boundaries separately visible.
7. Only after S1's review and measured decision, implement and screen S2 as a
   new source identity. Re-run affected quantizer bit/guard/lifetime cases and
   all S1 consumer checks; include aligned checkpoint-width fixtures because
   the original exhaustive fixture corpus deliberately uses offset guards.
8. Root integrates the selected candidate and runs focused production
   dense/grouped/packed MLP checks, actual checkpoint correctness and combined
   H64K four-scheme serving validation. Repeat affected full-serving/MFU
   measurements before replacing reports. Preserve
   existing valid publication materials until the replacement passes and is
   published. Component timing does not complete the motivation objective.

Planned commands below become executable only after the temporary driver exists;
`<gpu>` is supplied by root's scheduling grant. Each run uses a fresh output.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  /tmp/deepseek_linear_sfa_reuse_20261004/driver.py prepare \
  --variant official_sfa_reuse --output /tmp/deepseek_linear_sfa_reuse_20261004/s1_prepare_01.json

CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=8 .venv/bin/python \
  /tmp/deepseek_linear_sfa_reuse_20261004/driver.py screen \
  --variant official_sfa_reuse --output /tmp/deepseek_linear_sfa_reuse_20261004/s1_screen_01.json

CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=8 .venv/bin/python \
  /tmp/deepseek_linear_sfa_reuse_20261004/driver.py bench \
  --variant official_sfa_reuse --screen /tmp/deepseek_linear_sfa_reuse_20261004/s1_screen_01.json \
  --warmups 20 --repeats 100 --seed 20261004 \
  --output /tmp/deepseek_linear_sfa_reuse_20261004/s1_bench_01.json
```

## Broader alternative S3: private TMA-ready quantization for dense linears

Root requested this broader assessment after the initial S1/S2 plan. S3 is an
alternative candidate with a larger consumer matrix; implementing S1 and S2 is
not a prerequisite. Keep their narrower variants available for attribution if
S3 regresses. No SFA variant is implemented or GPU-validated in this document.

A private `quantize_for_gemm` may produce contiguous FP8 data and TMA-ready
FP32 scales for all aligned dense `fp8_linear` calls. The public quantizer's
owned contiguous scale ABI remains unchanged. Prepared gate/up activations
retain the private strided scales once; ordinary linears release their scales
after their consuming GEMM has been enqueued on the same stream. Preserve the
public/grouped fallback for all unsupported input types, strides, widths and
empty shapes. Do not route grouped MoE through the new dense contract.

CPU safetensors-header inspection of `/preset-models` confirms the following
FP8 weights in each source layer 0, 1 and 2. Shapes are `[N,K]`:

| Consumer | Weight shape | Activation K | Current calls per block |
| --- | --- | --- | --- |
| `wq_a` | `[1536,7168]` | 7168 | 1 |
| `wq_b` | `[24576,1536]` | 1536 | 1 |
| `wkv_a` | `[576,7168]` | 7168 | 1 |
| `index_wq` | `[8192,1536]` | 1536 | 1 |
| `index_wk` | `[128,7168]` | 7168 | 1 |
| `wo` | `[7168,16384]` | 16384 | 1 |
| MLP gate/up | two independent `[18432,7168]` weights | 7168 | 2 GEMMs, 1 shared quantization |
| MLP down | `[7168,18432]` | 18432 | 1 |

These nine dense GEMM consumers use eight quantizations after the existing
gate/up sharing. All four K values fit the aligned BF16 domain, and all N
values satisfy the dense API's 64-row alignment. In particular, `wkv_a` has
576 rows, which needs a partial 128-row weight scale block but no 64-row output
padding. SFA remains activation scaling with `(1,128,128)` recipe semantics.
The nine transform sites are source-derived; this assessment has not counted
them in a new full-block profile.

`kv_b_proj` is FP8 in the checkpoint but is explicitly dequantized to BF16 for
BMM and is outside this change. Index head weights, embedding and LM head are
BF16 checkpoint tensors; the current index head uses FP32 `F.linear`. Indexer
post-RoPE quantization has a separate operator/ABI and is also outside S3.

S3 must retain the storage, stream and graph gates above, and add these steps:

1. Freeze the accepted Triton T1 quantizer before changing only its private scale
   storage path. Specialize or measure the row/group addressing needed by
   `sfa[group * align(M,4) + row]`; T1's contiguous full-group branch currently
   stores at a flattened group offset without this division. Keep its launch
   geometry and FP8 conversion/reduction arithmetic unchanged.
2. Compare private logical scale bits with the public contiguous result and
   verify the required exact strides, actual storage capacity and padding
   contract. Call the installed official layout helper and check identical
   pointer/strides with no transpose or copy launch. Preserve output guards
   and generic/public dispatch coverage in the accepted quantizer screen.
3. Add exact consumer-output checks for every table row, all three source
   layers and M=1,121,128,1024, plus the small-M/padding matrix already specified
   for S2. Exercise ordinary output allocation, packed gate/up views, prepared
   lifetime, exceptions, alternate streams and changed-input graph replay.
4. Time the private quantizer, all complete dense API shapes, full projection,
   output projection and full MLP, including all allocations, initialization,
   graph input copies and owned outputs. Retain eager and borrowed-graph
   diagnostics separately. Run the whole-block/serving checks after component
   acceptance; a reduced transpose count is not sufficient for promotion.

The expected trace change is zero SFA transpose/copy launches at the nine
aligned dense sites. A private wrapper cannot promise that count without
checking the actual DeepGEMM no-copy branch and all shapes at runtime.

## Historical evidence and payoff boundary

CPU extraction script:
`/tmp/deepseek_linear_sfa_review_20261004/extract_historical_evidence.py`.
The resulting `historical_sfa_evidence_01.json` in the same directory has SHA256
`b0858dc9b70eea3b59eed39269f1bafea1bf88da5d5e56532c7ca6085568d255`.
It records source hashes, checkpoint header hashes/metadata, and all nine input
trace paths/hashes. Header identity does not hash tensor payloads.

Each saved `graph_bench_01` packed/shared full-MLP borrowed-replay trace has nine
kernels, including three SFA transforms: gate/up with G=56 and down with G=144.
Across source layers 0–2, their summed profiled durations are:

| M | Three SFA kernels, total microseconds | Fraction of summed kernel durations |
| --- | --- | --- |
| 121 | 8.351–8.544 | 5.62–5.75% |
| 128 | 8.128–8.320 | 5.58–5.61% |
| 1024 | 9.856–10.145 | 1.21–1.25% |

These are historical single-profile kernel durations from the compiled-helper
implementation. They exclude allocation, ownership costs and launch gaps, and
are neither predicted API savings nor native-v2 results. Removing just the
second gate/up transform targets about 2.05–2.56 microseconds in these traces;
the down transform alone is about 4.06–5.31 microseconds. This supports measuring
the broader candidate, but does not establish its speedup. Full serving/MFU
improvement still requires new paired measurements with accepted code.

## Later independent S4: share immutable projection inputs within one call

Source inspection shows that `project_positions` passes the same normalized
`x` to `wq_a`, `wkv_a` and `index_wk`, and the same normalized `qr` to `wq_b`
and `index_wq`. Explicit local sharing could reduce five projection
quantizations to two: two fewer K=7168 calls and one fewer K=1536 call. Keep all
five weight tensors and GEMMs independent, in their original order. Do not
cache quantized activations across projection calls, graph replays or sessions.

This is a separate candidate from S3. After S3, SFA transforms are already
absent, so do not add their old durations to S4's potential savings. Before S3,
sharing data alone would leave one official SFA transform per GEMM; sharing
both data and transformed scales could eliminate the redundant transforms.
No full-projection trace was used for this assessment, so the projection
opportunity is a source-derived call count without a latency estimate.

The current gate/up prepared object permits one consumer. Preserve that
contract; use explicit local handles or a separate bounded projection lifetime
if three consumers are required. The gate must check every returned `Projected`
field byte-for-byte against the accepted Triton T1 baseline, unchanged input
bytes, one call's exact consumer counts, source mutation rejection where version
counters exist, release after final consumption, exception cleanup, independent
graph pools and retained outputs across changed inputs. Profile and time the
complete projection API before any whole-serving promotion.

## Bounded follow-up after the C9 profile

The 2026-10-04 CPU review confirms these implementation sites:

| Site | Current role | Smallest relevant change |
| --- | --- | --- |
| `operators/deepseek_v32/linear/fp8.py:111` and `:185` | Single-use prepared activation; raw scales passed separately to each official GEMM | S1 retains the officially transformed SFA with the existing data |
| `operators/deepseek_v32/linear/quantization.py:58` and `_quantization_kernel.py:62` | Owned row-major output allocation and flattened scale store | S2/S3 add a private layout mode without changing public outputs or T1 arithmetic |
| `models/deepseek_v32/echo_model.py:247` and `:453` | CheckpointLinear already forwards prepared options; projection has repeated immutable inputs | S4 pair variants need call-local plumbing and eligibility checks, not a new reusable-handle API |
| `models/deepseek_v32/compute_graphs.py:186` and `:317` | Captures the same projection; replay updates static inputs and returns borrowed graph outputs | Recheck changed-input replay, retained owned outputs and graph capacity |

**S4q, share only `qr`.** Let `wq_b(qr)` return its existing prepared handle
and consume it once at `index_wq(qr)`. Keep the intervening Q split/BMM,
`wkv_a`, normalization and RoPE operations in their current order. None writes
`qr` in the inspected source. Release the handle immediately after `index_wq`.
This reduces the five projection quantizations to four, removing one K=1536
quantization. With raw scales it still performs five SFA transformations.

**S4x, share one pair of `x` consumers.** Let `wkv_a(x)` return a handle and
consume it at `index_wk(x)`. This spans fewer operations than starting at
`wq_a(x)` and keeps the original GEMM order. Release the handle before the
indexer normalization/RoPE/quantization. This independently removes one
K=7168 quantization. Combining S4q and S4x gives five to three quantizations;
all five GEMMs remain. If combined with accepted S1 behavior, each pair also
shares its transformed SFA. Measure that combination directly.

Both pair variants must check that both consumers use FP8 checkpoint weights,
BF16 SM90 inference, supported 2D input strides and aligned K/N. A BF16
checkpoint fallback rejects prepared options today, so an unconditional
`return_quantized=True` would break supported callers. Preserve the ordinary
path for ineligible consumers and small CPU fixtures. The two consumers may
have different N; only their K and the exact source object must agree.

The current handle rejects repeat consumption and mutation of normal tensors,
but inference tensors have no version counter. Correctness there relies on
the inspected immutable local dataflow and fixed current-stream ordering.
Retaining a handle across unrelated calls or a later graph replay is outside
the contract. Exceptions unwind the local handles; no global cache is added.
The `quantized` plus `return_quantized` combination is explicitly rejected, so
it cannot be used to chain all three `x` consumers. A full five-to-two variant
requires a separately reviewed bounded lifetime and is not the smallest next
candidate.

Longer live ranges can increase peak storage even when allocations are removed.
For row-major scales the pair handles retain `M*(K+4*K/128)` logical bytes:
1,584 bytes per token for `qr`, and 7,392 for `x`. These source-derived sizes
exclude allocator rounding and any TMA padding. Record actual eager peak and
graph-private allocated/reserved storage; do not infer memory savings from
the reduced call count.

For direct SFA emission, the private T1 scale offset is
`(group % G)*align(M,4) + group//G`. Its accepted `[32,128]` tile, two warps,
one stage and explicit division stay fixed in the first storage candidate.
The official SM90 helper allocates minimal `empty_strided` storage and writes
logical elements only. Thus a full padded owner with defined zero padding is
an optional private contract, not an upstream requirement. Before coding,
choose either that conservative owner (including initialization in timing) or
the exact upstream minimal-stride contract (guard logical writes and gaps;
never write nonexistent trailing padding). Neither choice is selected here.

Reuse the public quantizer suite, dense/grouped consumer suite and packed-MLP
lifetime/stream/graph checks. For S4, add focused comparisons of all six
`Projected` fields, input bytes and independent weights against frozen T1;
exercise both `normalized` branches, nonzero/changing positions, source
layers 0–2 and M=1/121/128/1024. Reuse the existing compute-graph checkpoint
tests for borrowed versus owned outputs and delayed writeback. Same-version
eager/graph agreement alone does not establish cross-version exactness.

Timing scopes stay separate: S1/S2 use complete gate+up and full MLP; S4 uses
the complete projection including normalization, all five GEMMs, the BMM,
both RoPE pair calls, indexer processing and all six outputs. S3 additionally covers each
affected dense API and output projection. Eager allocation and graph
input-copy/replay/owned-output costs belong in their respective complete API
measurements; borrowed replay is diagnostic. Use fresh randomized paired runs
and an independent confirmation window. Select only after C9 shows the
relevant scale-transform/quantization contribution, and remeasure serving/MFU
after any promotion. No new audit framework is needed.
