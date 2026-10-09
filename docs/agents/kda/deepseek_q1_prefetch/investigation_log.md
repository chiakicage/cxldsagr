# Investigation log

## Official source and policy

The checkout is clean for tracked upstream files at
`bc1b75c1000010d0ac6f032ebaac283255c050b1`. The official entry is
`DeepGEMM/csrc/apis/attention.hpp:380`; the host launch is in
`csrc/jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp:333`; the CUDA template is
`deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:413`.
The CUDA header SHA256 is
`b6fe25e6c7cfff65b1e8ee92e14e18acde5a25ed910868b3606f9866d0ce0cac`.

The official kernel hardcodes 64 temporary records per query at line 571.
The strict predicate is `logit > decode_topk_logits[q]`, and positions must be
less than `context_len - 1`. SGLang initializes the hint to zero and updates it
with an EMA decay of 0.5 after exact top-k. This is a separate decode policy,
not the prefill global coarse-bin certified subset. The device pool pointer is
an ABI input; the kernel writes prefetched bytes to temporary staging and
encodes mappings as `len(device_pool) + staging_slot`.

The official scheduler partitions 256-token segments across all SMs. The Q1
specialization uses 132 CTAs on this device, 768 threads per CTA, 83 prefetch
pipeline stages, and 230,468 bytes of dynamic shared memory. Descriptor data
types, strides, page layout, 128-byte swizzling, and L2 256-byte promotion are
transcribed from the original host launcher. No kernel body or upstream file
is changed.

## Source bridge compatibility

- TVM FFI compiles the original CUDA templates without importing the legacy
  Torch 2.8 binary or SGLang. The existing mainline DeepGEMM stays installed.
- The official paged and clean headers declare different dynamic shared-memory
  types under the same symbol. Upstream JIT compiles them separately. The
  bridge therefore uses `official_decode.cu` and `official_decode_clean.cu` as
  separate translation units in one extension.
- Both calls explicitly enter `torch.cuda.device()` and
  `tvm_ffi.use_torch_stream()`. Default-stream success alone is insufficient;
  the harness also tests a delayed side-stream update and CUDA Graph replay.
- The raw bridge accepts a mapping of length `hostN` or `hostN + 1`. The
  additional upstream negative-ID sentinel is unnecessary when all initialized
  history table entries are valid nonnegative host IDs. Host ID zero is valid
  locally. Root/reviewer own the separate staging-to-persistent-pool adapter.
- Initial compilation lacked the virtual environment's `ninja` on PATH; the
  launch environment now explicitly prepends `.venv/bin`. This is an
  environment correction, not an automatic runtime fallback.

## Validation progress

The first complete three-layer functional matrix passed exact score bits,
exact top-k values/IDs and saved selections, cold/warm/partial/empty/small and
learned-threshold staging, buffer canaries, side-stream execution, and changed
graph replay. Its provenance was intentionally rejected because device-scope
and physical-tail checks were strengthened during setup. No result file was
published for that attempt. The final source-bound run is
`q1_official_prefetch_check_20261008_03`; its result is recorded in the
checkpoint after completion.

An independent reviewer also exercised the bridge through the new pool
adapter on Q1/N257, including host ID zero and exact staged/promoted BF16
records. That adapter acceptance is separate from this three-layer source
bridge receipt and from full-model acceptance.

## Follow-up directions after isolated H64K/A1 publication

The accepted separate-process cohort is `deepseek_h64k_a1_isolated_20261008_01`.
Its original official-core and preparation node details are now selected in
`experiments/deepseek_v32_echo_official/report/decode_gap/`.
The stateless preparation-fusion component benchmark has small positive paired
results; complete-model promotion remains pending its strict native-bound gate.

A read-only graph-lifetime review found that persistent packed history is a
possible separate experiment, not a storage-free rewrite of the accepted packer.
For H65536/A1, retain distinct 8,659,200-byte buffers per layer/graph and refresh
the projected suffix after index_cache_write. Gross retained payload over L0–L2
would be 25,977,600 bytes. Existing temporary graph allocations can be reused;
retention may increase actual private reserved memory. Current graph identity
binds addresses/lengths, not indexer history content, and prefix snapshots omit
indexer bytes. Same-address history rebuilding therefore needs explicit content
invalidation before packed reuse. Any candidate must bind an immutable prefix
owner, reject/invalidate rewritten histories, account for multiple graph output
variants, and preserve unknown-drain owner retention. The measured 15.808 us
packing activity sum is an opportunity size, not a predicted wall-time saving.
No persistent packing candidate has been implemented.

The fixed official Q1 kernel already aggregates reservations per warp. In
`sm90_fp8_paged_mqa_logits.cuh`, the miss ballot/popcount and leader atomic are
at lines 788–801; the task-level budget observation is at lines 843–848.
The observed 33,471 counter value counts attempted records, not atomic
instructions. A local two-iteration aggregation or CTA aggregation could change
when later tasks see the exhausted budget and hence the realized attempt
count, even if each submitted mask is accounted exactly. No retained NCU capture
establishes atomic contention as the bottleneck. Profile the original kernel
first; CTA-wide barriers would be unsafe across its warp-specialized roles.
The third-party rule prohibiting modification/copying of the official kernel
continues to apply. No algorithm change follows from this hypothesis review.
