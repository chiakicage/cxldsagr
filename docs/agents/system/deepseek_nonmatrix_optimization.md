# DeepSeek nonmatrix optimization

Status: implementation, three-layer acceptance and report publication completed,
2026-10-03. Results and reproduction commands are in the
[experiment report](../../../experiments/deepseek_v32_echo_prefill/README.md).

The user requests FlashInfer replacements for ordinary operators, removing
Hadamard following vLLM's implementation, and KDA optimization of quantization.
The non-GR acceptance workload is the actual checkpoint layers 0–2 with
sequential hidden/residual propagation, 65536 prefix tokens and 1024 extend
tokens. No full61 or GR performance claim follows from this workload.

## Implementation and acceptance

- Reuse FlashInfer 0.6.18 RoPE, FP32 RMSNorm/fused residual RMSNorm, SiLU and
  deterministic exact top-k. Preserve FP32 checkpoint norm weights and the
  pre-rounding residual sum. Include dtype conversion, packing and helper cost.
- Share a per-call FP32 cosine/sine table for main MLA and indexer RoPE;
  preserve model-derived YaRN frequencies and the two pairing conventions.
- Indexer quantizes BF16 post-RoPE Q/K directly. Hadamard remains only as a
  reference helper, with no production invocation. vLLM v0.11.2's
  [Indexer.forward](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/models/deepseek_v2.py#L828)
  and the current main implementation use this direct path. This source
  observation is not a task-quality evaluation or proof of exact equivalence.
- Quantization contract, draft, plan and candidate records live in
  [the KDA component](../kda/deepseek_quantization/task.md). Preserve FP8
  data/scale bits, including UE8M0 and unrounded FP32 scales; promote only
  measured candidates. FlashInfer's available cutile quantizer cannot compile
  for this installed Hopper toolchain and is not an accepted backend.
- Candidate and fresh control sources were frozen with identical current shared
  cache, workload and budget semantics. Control uses the original nonmatrix
  operations. Both were measured on the same physical H200, with explicit
  source/backend identity. Superseded results were removed after the replacement
  report passed acceptance and was published.
- Check independent empty-cache resident/offload prefixes and all extend
  hidden; separately report cross-version hidden/logit and selection changes.
  Removing a quantization rotation is a semantic change; same argmax is not
  sufficient to establish task quality. Do not widen existing model tolerances.
- Profile four annotated captures separately from uninstrumented wall time.
  Require complete call coverage and conservation of kernel count/time;
  expose casts and launch overhead without summing overlapping API/wall costs.

## Operator acceptance

Temporary checks are not paper experiments. Norm and SiLU have passed CPU/GPU
reference tests; exact top-k has matched saved three-layer inputs and subsequent
MLA bitwise. RoPE passed long-position, sliced-layout and suffix-preservation
tests (8 cases); FP32 fused arithmetic can change BF16 rounding at a few values.
The preliminary 1024-token four-rotation API, including table construction and
packing, measured 433.920 → 59.088 microseconds (10 warmups, 30 samples, GPU 1).
This temporary diagnostic is separate from the accepted checkpoint measurements
below. KDA accepted the exact q2 indexer quantizer; its source SHA256 is
`083b9671efe7f5dee7e71752dda3255b2d41ee45f906e48876771db5804411ea`.
Linear activation quantization continues to use the compiled official DeepGEMM
helper; q2 has not replaced that distinct contract.

## Accepted runs and measurement boundary

| Purpose | Run ID |
| --- | --- |
| Fresh control | `20261003_echo_layers3_nonmatrix_control_01` |
| Accepted implementation | `20261003_echo_layers3_nonmatrix_candidate_02` |
| MLA NCU | `20261003_echo_ncu_nonmatrix_mla_02` |
| Resident indexer NCU | `20261003_echo_ncu_nonmatrix_indexer_resident_02` |
| Offload indexer NCU | `20261003_echo_ncu_nonmatrix_indexer_offload_02` |

Control and candidate ran sequentially on physical GPU 5, H200 SXM / SM90 /
132 SM, UUID `a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`, at default clocks.
Both use the same request/checkpoint identity, 1024-token chunks, 16,384 usable
sparse-pool tokens per layer, 66,560 host-arena tokens, 1024 query-workspace
tokens, 5 GiB HBM / 64 GiB DRAM cache budgets and prefetch configuration.
Dependencies include Torch 2.12.1+cu130, Triton 3.7.1, FlashInfer 0.6.18,
DeepGEMM `057ca596` and FlashMLA `ba89a346`.

The timed call includes embedding, checkpoint layers 0–2 in sequence, three
dense MLPs, and final norm / LM head for the last token. All extend hidden values
are verified separately. Each mode constructs its prefix from an independent
empty cache; extend repetitions restore identical prefix residency. Synchronized
wall timings exclude loading, compilation, restoration and numerical checks.
There is one warmup, three prefix samples and five extend samples per mode.

| Mode / phase | Control median ms | Candidate median ms |
| --- | ---: | ---: |
| Resident / prefix | 1176.2025 | 850.6642 |
| Resident / extend | 24.2997 | 18.3442 |
| Offload / prefix | 2025.6879 | 1773.9631 |
| Offload / extend | 51.5112 | 36.2451 |

The five control offload-extend samples are 51.5112, 51.7155, 52.2868, 43.9949
and 41.4298 ms; candidate samples are 36.3524, 36.2451, 36.2892, 36.1652 and
36.1651 ms. The control has substantial variability. The 1.421× median ratio
describes these sequential samples from one request, without a confidence
interval or a claim of stable speedup across workloads.

Four separate annotated captures per run provide exclusive kernel attribution.
Nonmatrix scopes excluding explicit cache and matrix-call scopes decrease from
668 to 208 kernels / 10.201065 to 4.548206 ms for resident extend, and 669 to
209 kernels / 10.263430 to 4.610033 ms for offload extend. Offload indexer
quantization decreases from 72 kernels / 0.486659 ms to 6 / 0.020320 ms;
Hadamard decreases from 144 / 1.525125 ms to zero. Exact top-k and projection
helpers remain the largest nonmatrix costs.

These sums include casts, trig construction, packing and exclusive parent
remainders, but are neither wall latency nor GPU-busy union. Activation
quantization and other helpers inside matrix-call scopes remain there; fused
indexer prefetch is also outside the explicit cache group. Explicit offload
extend cache cost is essentially unchanged, 2.707624 → 2.710060 ms. Changed
selections alter cache accesses: prefix H2D is 397,967,616 → 402,087,168 bytes,
and extend H2D is 11,719,296 → 11,802,240 bytes. Whole-call offload improvement
cannot be attributed entirely to nonmatrix kernels or called a cache-policy
optimization.

The three fresh NCU runs use saved candidate inputs and retain full / source
reports. Full-report replay durations for MLA, resident indexer and offload
indexer are respectively 880.352, 1057.536 and 1951.840 microseconds; tensor
pipe active is 68.986%, 54.909% and 28.660%. These are isolated replay
observations, separate from synchronized model wall time and useful-FLOP MFU.
They also differ from the temporary q2 quantizer NCU diagnostics.

## Numerical and instrumentation acceptance

Each accepted run passes all eight internal checks bitwise: ordinary versus
annotated prefix / extend logits and all extend hidden, plus resident versus
offload hidden and logits from independent empty-cache prefixes. Candidate
MLA also passes the independent FP32 QK / softmax / PV check over all
201,326,592 output elements across three layers, with TF32 disabled and the
unchanged `atol=0.004`, `rtol=0.02` tolerances.

Cross-version outputs are not equivalent. Across all 7,340,032 extend-hidden
elements, maximum absolute difference is 0.34375 and relative L2 is 0.01505245;
50,532 elements exceed the original `atol=0.02`, `rtol=0.01`. Across all
129,280 logits, maximum absolute difference is 0.09375 and relative L2 is
0.00792472; 9,129 exceed that tolerance. Argmax is unchanged, which does not
establish task quality. Removing Hadamard changes the quantized indexer input;
the complete change set also changes floating-point arithmetic, propagated
activations, selections and cache accesses. The comparison does not attribute
all drift to one operation. Quantizer bitwise exactness for identical inputs
does not establish equivalence of these different whole-model paths.

Candidate01 was excluded after a harness issue: reconstructed annotated hidden
was normalized after `model.forward()` returned, outside inference mode.
The grad-enabled dispatch selected eager normalization instead of the
FlashInfer FP32 path. It passed tolerance but was not bitwise equal. Adding
`@torch.inference_mode()` to `profile_layers.annotate` fixed the dispatch;
model/operator sources were unchanged and the workload was rerun as
candidate02. Hidden reconstruction occurs after `cudaProfilerStop()`, outside
the measured profile, and checks every value in `[1024, 7168]`.

Both source manifests contain 1,218 entries and match their snapshots and
frozen trees. Candidate02's runtime audit verifies FlashInfer
native-library/source/build hashes and four live CuTe norm identities, with FP32
compile keys at widths 1536, 512 and 7168, including fused width 7168. Each run's four captures
conserve raw SQLite kernel count/time and the exclusive ledger, with zero
unattributed kernels, call-count mismatches or API time outside scope.
Candidate has no Hadamard calls.

At publication, 31 of the 32 production files in the candidate manifest match
the current checkout exactly. The sole later difference is an `echo_infer.py`
constructor guard requiring `max(query reservation, chunk, extend_chunk) <=
slots`. It is outside the timed region and does not trigger for this measured
1024-query / 16,384-slot workload. The saved manifest remains the source
identity of the published measurement.

Validation includes 30 quantizer tests, 40 model-plus-quantizer checks,
49 norm/RoPE/model/instrumentation checks, 69 top-k-related GPU checks and
2,202 global CPU tests. Final report tooling checks passed: 16 tests covering
nonmatrix comparison, backend provenance and layer profiling, plus 24 comparison
audit tests. These overlapping suites are not added into a unique-test total.
The report links the accepted data and audit artifacts; ignored raw data remain
under `experiments/deepseek_v32_echo_prefill/output/{data,log,profile}/<run_id>/`.
This completion covers the declared three-layer workload, not full61, a trained
three-layer model, GR performance or task-quality acceptance.
