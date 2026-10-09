# Draft before implementation

The local Q1 ECHO path currently invokes the prefill fused kernel on one CTA.
The published diagnostic reports 2.690281 ms for its three layer invocations,
whereas official normal decode uses paged fused MQA. These workload and
residency states differ, so their timing ratio is not an equal-workload speedup.

The first candidate directly includes
`3rdparty/ECHO/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`
and instantiates the official Q1/H64/D128/DMLA576 template. A small independent
CUDA/TVM-FFI launcher creates identical TMA descriptors, invokes the official
scheduler and kernel, and cleans unused score columns. No Torch C++ ABI or
SGLang import is needed. The module name and C++ filename will avoid the
existing `echo_*.cu*` fingerprint, keeping the prefill baseline unchanged.

The official decode kernel is not the local prefill policy. It compares scores
against a per-query FP32 threshold, initialized to zero in SGLang and updated
with EMA decay 0.5 after exact top-k. It excludes `context_len - 1` from host
prefetch, reserves at most 64 staging slots, and sets host mappings to
`device_pool_size + staging_slot`. It does not evict persistent slots or
update a device-to-host journal. The official consumer must separately gather
or reset those temporary mappings. Review of that consumer proceeds in
parallel; this candidate will not masquerade as a production pool adapter.

Main risks: shared-memory/TMA launch ABI errors; fingerprint omissions;
unmapped pinned host memory; score differences between old and new official
DeepGEMM versions; duplicated token aliases not supported by the official
non-atomic per-token miss check; saturated staging selection depends on GPU
scheduling; main-environment compiler compatibility; unsafe host sentinel
indexing; and stale staging mappings across graph replay.

Implementation order:

1. Record contract and executable plan before writing code.
2. Implement an independent raw ABI loader and native launcher using unchanged
   upstream includes. Restrict the initial bridge to Q1, H64, D128, page 64,
   DMLA576, and the actual H200 SM count.
3. Add an experiment harness that constructs exact official state, packs
   current inputs, and validates score/top-k and staging/canaries. Each replay
   restores the caller-owned initial state, outside clean timing events.
4. Validate on GPU 1/CPU 8–15, then use a separate timing process with matching
   source/input/native identities. Report scratch bytes and timing boundaries.

The first implementation uses the already verified page packing convention.
The official SGLang `_set_k_and_s_triton_kernel` is a later bridgeable source
entry if packing is material; any such reuse must be identified and included
in the full wrapper boundary. A persistent packed cache is outside this
candidate and requires model/cache accounting.

Commands: use the check and evaluation commands in `task.md`, with
`CUDA_VISIBLE_DEVICES=1`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`,
`PYTHONDONTWRITEBYTECODE=1`, and `taskset -c 8-15 .venv/bin/python`.
Artifacts belong to the official experiment. Failed attempts are cleaned and
retained only as concise KDA investigation facts, not experiment results.
