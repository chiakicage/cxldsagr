# ECHO native cache implementation checkpoint

Date: 2026-10-03. Status: native P0/P3 implementation and operator verification complete; shared-model integration pending. No new performance claim.

## Frozen inputs

Before cache edits, the actual dirty indexer source hashes and every shared CUTLASS header hash were saved in [echo_cache_native_before.json](echo_cache_native_before.json). The complete existing diff against HEAD is [echo_cache_native_before.patch](echo_cache_native_before.patch). The base revision is `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`. These include the ongoing MFU optimization; HEAD alone is not its source identity.

The read-only official checkout is `3rdparty/ECHO`, revision `bc1b75c1000010d0ac6f032ebaac283255c050b1`, with clean `git status --short` before inspection. No official source or nested dependency was modified.

## Native ABI agreed with shared cache

- Global host records `[NH,576]`; physical records `[P+1,576]`, slot zero reserved.
- Session page table: int32 global 64-token page numbers; logical token i maps to `table[i//64]*64+i%64`.
- Live `host_to_device`: int32, MISSING=`INT_MAX`, transient CLAIMED=`-1`; reverse map int64 over global IDs.
- `free_slots`: priority-sorted permutation of slots 1..P, including occupied victims; preparation does not evict.
- `allocation_log`: reusable int64 `[P+1]`, indexed by physical slot, value global host ID or MISSING.
- `counter`: uint32 reservation attempts; `prefetch_stats`: int64 `[3]` actual copies, actual evictions, rejected reservations. Stats, not overshooting counter, determine traffic.
- `history_length`: initialized main-KV logical prefix; current suffix is excluded even though index-K is visible.
- `max_prefetch <= min(8192,P-Q)`; reject Q>P before launching.

## Official behavior sources

- `sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py:520-559`: sort the full priority array; clear allocation log/counter; fused prefetch; update mean hint; post_alloc; update priority.
- `sglang/python/sglang/srt/mem_cache/memory_pool_host.py:1432-1444`: assignment of FIFO clock then increment; restore padding priority sentinel.
- Native claim/copy logic is checked against the pinned DeepGEMM header. Local race-safe CAS invalidation, explicit host-history bound, global page translation, actual counters and sentinel exclusion are correctness/storage adaptations.

## Preservation boundary

WGMMA mathematics, compute launch geometry, register redistribution, histogram threshold and prefetch-score selection remain unchanged. The kernel-wiki skill was consulted for Hopper constraints; cache changes do not adopt an unrelated register or GEMM rewrite. Numerical parity is rechecked on the changed implementation.

## Changed implementation validation

The native methods share one `claim_and_copy_warp` helper between fused indexer and isolated metadata/transport verification. The supplied allocation log is physical-slot indexed, and exact statistics are copied records, actual victim invalidations and reservations denied by cap. A rejected reservation restores CLAIMED with CAS. There is no pre-eviction or initial h2d snapshot.

On GPU 1 (Hopper SM90, NVIDIA M403), the command `PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q operators/deepseek_v32/indexer/tests/test_echo_cache_ops.py operators/deepseek_v32/indexer/tests/test_echo_indexer.py` passed **24 tests**. Numerical coverage includes 64K+1K indexer logits bit-for-bit against the resident official DeepGEMM path; this is operator verification, not full-model numerical or performance acceptance.

Transport coverage: empty/full/partial-free pools; repeated and invalid global IDs; all-hit and zero-cap calls with no eviction; exact 11-allocation prefix despite 48 requested capacity; concurrent cap exhaustion; no residual CLAIMED; record contents and both map directions; slot-zero preservation; pending suffix exclusion via a nonidentity, noncontiguous page table; supplied host-write event dependency on a nondefault consumer stream.

The normal-domain differential imports only the AST function definitions of the pinned official `_cuda_graph_allocator_post_alloc_kernel` (`allocator.py:822`) and `update_priority_when_use` (`memory_pool_host.py:1432`), runs them on GPU, and compares free bitmap, priorities and clock against the new native finalize for empty, partial and full allocation logs. This is independent official implementation evidence rather than a handwritten copy of the local algorithm.

Native finalize and protect advance the clock in a separate stream-ordered kernel after all data-parallel CTAs finish reading it, including empty calls. Release checks both global-ID ownership and the expected physical slot, does not advance the clock, and does not modify other users' maps. These helpers own no scratch allocations. Full exact-recall/append and shared-pool budgeting remain the cache/model integration's responsibility.
