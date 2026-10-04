# Aligned-host diagnostic plan

Status: helper preparation selected on 2026-10-04. The GPU4 allocation-only probe
has reproduced 16-byte residues modulo32. Prepare the sibling driver and CPU
checks, then schedule the bounded GPU diagnostic. No native or production runtime
changes are included.

Test whether host pointer alignment explains the current per-PC result: each
`0xf750`/`0xf770` host load executes 14,368 full-warp instructions, with 258,624
theoretical versus 229,888 ideal sectors, or 18 versus 16 per instruction. Both
current reports reproduce this; corresponding HBM stores have 16/16 sectors.
Use the same frozen halves2 library, request 16/layer 16 input, Q128/H65536,
selection, CIS, owner tags, queue, math, and saved output hash.

**Smallest comparison.** Two arms, selected before collection:

| Arm | Host K/V supplied to the frozen call |
| --- | --- |
| Original | Fresh tensors produced by unchanged `screen.execute_case`: `.contiguous().pin_memory()`; pass them unchanged. |
| Aligned32 | Same original tensors, copied exactly into deterministic 32-byte-aligned pinned views. |

A sibling diagnostic driver temporarily wraps `NosaFetchWorkspace.run`, as the
existing NVTX driver does. It changes only the two host arguments in Aligned32,
forwards every other argument and return unchanged, and restores the method on
all exits. Frozen helpers and production files stay untouched. Existing checks
accept contiguous pinned BF16 views; native code maps `host.data_ptr()` through
`cudaHostGetDevicePointer`, so a nonzero storage offset requires no native change.

For each 32 MiB record tensor, allocate a BF16 owner with **16 extra bytes**,
checking its 16-byte base alignment explicitly. Let
`skip = (-owner.data_ptr()) % 32`; require `skip` to be 0 or 16, and expose exactly
the original element count at `skip / 2`, reshaped to `[65536, 2, 128]` with strides
`[256, 128, 1]`. Check CPU and mapped-device pointers modulo 16/32/64/128. Do not
truncate payload, exploit unexposed allocator slack, force an original misaligned
view, or retry allocations to select favorable addresses. If original pointers
are already aligned, report that the original amplification did not reproduce;
an additional deliberately misaligned arm needs a separate decision.

**Receipts and lifetime.** For both original and supplied tensors record logical
bytes, requested owner bytes, storage bytes/base pointer, data pointer, storage
offset, mapped base/view pointers, pinning, shape/stride/dtype, full payload hashes,
and padding guards. Snapshot `torch.cuda.host_memory_stats()` before each case,
at wrapper entry, after owner allocation, after GPU completion, and after release.
Retain aligned owners through the unchanged execute/finalize path and confirmed
workspace/device synchronization; then drop owners/views and run GC. On uncertain
CUDA completion, stop and retain owners through process teardown. No cache flush
or allocator-setting change is needed. Verify released tensor/view/storage owners
with weakrefs after confirmed GPU completion. Preserve raw host statistics, but
do not use their active counters as a return-to-baseline gate: the allocation
probe observed those counters growing across reuse of already released blocks.
Report distinct live backing and cached allocator-owned bytes separately. Preserve
the existing 33,554,432-byte GPU allocated/reserved cleanup baseline.

**Capacity tradeoff.** Each logical K/V tensor is exactly 33,554,432 bytes.
Installed `CachingHostAllocator.h` uses `PowerOf2Ceil(size)`; adding even 16 bytes
requests a 64 MiB allocator block. Adding 31 or 127 bytes has the same rounding
cost and does not avoid it. The aligned K/V owners therefore cost **128 MiB actual
rounded blocks**, versus 64 MiB for originals. With the wrapper retaining the
originals, the two sets can occupy **192 MiB of distinct known rounded backing**,
before other host state. Verify owner identities and bucket sizes against
allocator-owned statistics; preserve active counters only as raw observations.
`storage.nbytes()` alone omits rounding. Cached bytes may remain after release.
This is an alignment diagnostic, not a capacity-safe runtime promotion. External
aligned allocation/registration would change allocator ownership and needs its
own design and capacity acceptance.

**Allocation evidence.** GPU4/NUMA1 probe `pinned_alignment_gpu4_02.json` in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_independent_warps_ncu_review_20261004/`
has SHA256 `645832c06630ffb6989654d867971e5ee84f91590ff64b9b47014f34d6255381`.
Three sequential fresh K/V tensor pairs reused two 32 MiB blocks. Every pointer
was 16 modulo32/64/128/256/4096; tensor weakrefs died after release and device
allocated/reserved remained zero. Allocator-owned `allocated_bytes.current`
stayed64 MiB while raw `active_bytes.current` grew64/128/192 MiB. The latter
cannot describe simultaneous live physical backing in this probe. This is fresh
allocation evidence; the old NCU target's actual data pointers remain unknown.

**Bounded execution.** Reuse frozen input/identity verification,
`execute_case`, exception-frame-dropping finalization, and `finish_case`. First
prepare/freeze/review the sibling helper with CPU checks. Run one standalone
validation per arm. Each application invocation retains one original halves1
control on original buffers, two fresh halves2 warmups using the selected arm,
and one untraced halves2 target; all four calls retain exact saved output, fixed
FP32 tolerance 0.016, payload/suffix, 14,712,832 unique bytes, queue/cursor, native
identity, and cleanup checks. Add receipts without weakening those checks.

Collect one strict application-replay NCU report per arm on the root-selected
GPU with the same NUMA/CPU/allocator environment. Request only per-PC executed
instructions/true threads, theoretical/ideal global sectors, and TEX sysmem read
misses. Confirm selectable metric names in CPU preparation; if software metrics
require `SourceCounters`, use that section plus the TEX metric. Record actual
replay count and every invocation; no full/PM/timed collection or API timing is
needed. Use one constant target NVTX range and one matched action, as before.

For each host pointer at 16 modulo 32, predict 18 theoretical versus 16 ideal
sectors per instruction; at 0 modulo 32, predict 16/16. Aligned32 should yield
229,888 sectors per host LDG and 459,776 total TEX sysmem read-miss sectors while
preserving 14,368 instructions, 459,776 true thread instructions per LDG, and all
payload checks. Attribute counters through reported `launch__function_pcs` and
verified SASS; preserve raw unit labels and compute excess from the two sector
metrics. Failure of that prediction limits the alignment explanation. Neither
the sector result nor intrusive kernel duration establishes overlap or serving
performance; those require a separately selected, freshly validated follow-up.
