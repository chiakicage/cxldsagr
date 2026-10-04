# Norm I/O implementation plan

Status: temporary plain/fused kernels passed complete numerical/lifetime
matrices, complete-API timing and independent evidence audits. The measured
BF16 implementation is now integrated under `operators/deepseek_v32/norm/` and
`nonmatrix.py`; focused production tests and runtime identity checks passed.
See [checkpoint.md](checkpoint.md) for the temporary postprocessing-only error
and its independently audited resolution. [integration_plan.md](integration_plan.md)
tracks the production gate. Full-checkpoint/serving acceptance remains with root;
the separate packed norm candidate remains unimplemented.
Read [task.md](task.md) and [draft.md](draft.md) before implementation.

## Stage 1: freeze and prepare on CPU

1. Create `/tmp/deepseek_norm_adapter_20261004/` for independent candidates.
   Save hashes of the current nonmatrix adapter, vendor plain/fused kernels,
   imported reduction helpers, installed Torch/CuTe versions and probe source.
   Keep the earlier direct-CuTe and Triton rejections as separate records.
2. Implement a probe CLI with modes `prepare`, `screen`, `validate`, `measure`
   and candidates `packed_vendor`, `out_only`, `local_plain`, `local_fused`.
   `prepare` must not import CuTe, instantiate the vendor class or initialize
   CUDA. Root grants CUDA windows separately; the mode is not permission.
3. Reuse the actual FP32 norm-vector loader and strict bit comparator from the
   prior temporary probe. Add the real KV row-stride-576 layout, offset views,
   awkward row counts, epsilon-sensitive magnitudes, cancellation, mixed
   magnitudes and BF16 pairs with nonrepresentable FP32 sums.
4. State every planned fallback. Packed and direct outputs must not alias
   caller input. Store complete owning-storage footprints and shape/stride
   metadata; do not count only view numel when outputs share backing storage.
5. Independently review the prepared wrappers and source delta before GPU use.

## Stage 2: smallest unchanged-vendor screens

Run only in a correctness window explicitly assigned by root.

1. `packed_vendor`: allocate one FP32 pair, perform two explicit widening
   copies, call the unchanged official fused norm on its contiguous halves,
   and cast the complete pair to BF16 once. Keep a diagnostic access path to
   the post-call FP32 normalized/summed halves before conversion. Start at
   Q=9,D=7168 and actual source-0 post-attention norm weight.
2. Compare both FP32 halves against independently allocated official baseline
   inputs, then both BF16 halves. Reject on the first bit mismatch. Verify
   unchanged callers, disjoint output extents and whole-pack retention.
3. `out_only`: directly compile the unchanged Float32 constructor with Float32
   X/W and BF16 Y. Keep default layout/copy_bits. Require exact FP32 control and
   BF16 output; a compile failure is an unsupported candidate, not a reason to
   silently modify the body under this candidate name.
4. The optional public `out=owned_FP32_activation` alias screen is separate
   and BF16-input-only. Check the actual constructor is cluster_n=1 for the
   tested width; validate output and original input lifetime before broadening.
5. Record emitted kernel/copy counts. Do not infer one kernel from a single
   Python `cat`, cast or vendor call. No component timing at this stage.

## Stage 3: minimal local plain kernel

1. Implement a local subclass that reuses the official Float32 constructor and
   plain `__call__` launch method, overriding only `kernel`. If the DSL does not
   support that inherited dispatch, use a local launch wrapper with an explicit
   source delta; do not dynamically rewrite the vendor source.
2. Reuse vendor configuration/reduction/predicate helpers where valid. Define
   distinct X/W/Y copy atoms with the same logical vector element count and
   operand-specific predicates. Retain canonical FP32 TV and register layout,
   the scalar FP32 sequence, barriers, PDL and launch shared-memory budget.
3. Emit compile-time layout evidence for input, weight, reduction and output
   partitions. Prove logical indices reach the same FP32 register coordinates
   as the official control. A retile is acceptable only with that proof and
   exact numerical validation; no reduction rearrangement is authorized.
4. First compile Float32-input/weight/output control, then BF16-input/FP32-output
   diagnostic, then BF16 output at Q=9,D=7168. Preserve exact compiler failures.
   Reject immediately if the normalized FP32 comparison drifts.
5. Broaden to all actual weights, widths, row counts and epsilons. Cover direct
   row-strided input or a declared BF16 packing fallback. No final-norm fusion,
   weight conversion, different reduction or occupancy tuning in this stage.

## Stage 4: owned fused outputs

Proceed only after plain kernel layout/math acceptance and a new source review.

1. Reuse the official fused Float32 constructor, geometry and imported
   reduction. Add a small launch wrapper because the official in-place ABI
   has no independent output pointers. Keep separate owned Y and saved outputs.
2. Build typed X/R/W/Y/saved copies and predicates. Convert X/R to FP32 in
   canonical registers, add once, retain the unrounded h through normalization,
   and independently cast h for the saved residual store. Never use the saved
   BF16 result as the normalized operand.
3. Use a diagnostic specialization with FP32 Y and FP32 saved output. Compare
   normalized FP32, summed FP32 and both final BF16 results to the existing
   public CuTe path. Include cancellation and BF16 rounding-boundary cases.
4. Preserve the original FP32 input behavior and own all outputs. Do not fuse
   final norm or alter residual arithmetic as part of this candidate.

## Stage 5: lifecycle and complete API measurement

1. For each passing candidate, verify retained outputs survive subsequent
   calls, input overwrites and non-default stream execution. Confirm no write
   outside an output view affects the other view. Capture/replay both residual
   branches, change inputs, retain one output and audit private graph pools.
2. For packed output, specifically test graph v2 `pair.saved`: the normalized
   half may remain resident because its saved half shares backing storage.
   Measure full private reserved segments, static allocation, PyTorch allocated/
   reserved and device-used snapshots. Reject capacity claims based only on
   smaller tensor views or fewer allocation calls.
3. Only after all exactness/lifetime cases pass, request a quiet timing window.
   Compare full wrapper allocation, conversion, packing/fallback, vendor/local
   launch and completion with alternating candidate/baseline order, warmups
   and repeated samples. Report CUDA-event and CPU wall results separately.
   Include the real strided KV layout. Save raw samples and profiler evidence
   for actual launch counts; no bound-based serving speedup estimate.
4. Compare graph replay separately: fewer GPU nodes may reduce execution work
   while the same projection/finish host replay launches remain. Include graph
   setup/capacity separately from measured replay.
5. Root owns promotion, real-checkpoint comparison of all candidate hidden and
   logits, runner fault/lifetime tests and quiet full serving/MFU runs with new
   source snapshots/run IDs. A failed/unhelpful candidate stays in temporary
   investigation evidence, not published experiment results.

## Implemented commands and next gate

The frozen `probe.py` implements the initial plain prepare/screen path;
`validation.py` implements broad plain/fused validation. Completed commands and
run identities are recorded in the checkpoint. Future benchmark runs use the
separate driver, preserving those validated source identities. CPU preparation
from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python \
  /tmp/deepseek_norm_adapter_20261004/benchmark.py prepare \
  --checkpoint /preset-models --output /tmp/norm_benchmark_prepare_new
```

After independent source review and root's correctness grant assigning
`NORM_GPU`, check both variants and execution boundaries:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES="$NORM_GPU" OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_norm_adapter_20261004/benchmark.py check \
  --checkpoint /preset-models --output /tmp/norm_benchmark_check_new
```

`measure` requires the exact-source full validation records plus a passing
`--check-record` for the same benchmark subjects and execution plan. It runs
only in a subsequent root-assigned quiet window. The complete commands and
measurement semantics are in the benchmark plan. Root's ordinary combined
real-checkpoint acceptance remains the production integration gate.
