# C6 accepted component checkpoint

Run ID: `deepseek_sparse_recall_c6_20261004_01`. C6 is promoted for root's combined
full-request acceptance. This is an exact-cache component result, not a published
motivation experiment or proof of the full first-visit/MFU objective. Production
source is frozen. All GPU work used the authorized quiet GPU 2 window and exited.

## Final implementation and safety

The existing C3 all-resident certificate path remains preferred. For an exact
private int32 contiguous selection on a shared SparseTokenCache, C6 handles an
uncertified history H <= P. Other geometries and public IDs keep checked ensure.
The limit-1 boundary falls back so normalization still occurs between events.

A logical bitmap deduplicates every valid ID; candidate bits are counted without
host translation. Classification protects selected hits at t, writes full-P miss
flags, and commits protection clock t+1. One miss-count scalar read preserves
zero-miss proof fields and avoids speculative map_generation changes. For real
misses, append_owner is cleared, wait_host establishes copy dependencies, stable
argsort of priority[1:] chooses victims, and a prefix scan preserves logical miss
order even with nonmonotonic host pages. The cache increments map_generation at the mutation boundary; compaction then
tombstones victims before the generic record gather.
Publication follows that gather, installs new maps, and commits allocation clock
t+2. The final selection remap does not mutate metadata. Zero-miss calls still
commit both events. Successful publication updates the existing recall counter;
failed copies cannot appear as successful recall or H2D bytes.

The native helper owns no tensors. Shared free_slots, miss_scratch, allocation_log,
union/count and existing counter storage are reused under the exclusive operation
lease with no pending prefetch. The stable-sort result becomes chosen int64 IDs
in-place; prefix/misses/chosen storage aliasing is rejected. Generic
operators/common/kv_transfer.py retains model-provided width/dtype and its existing
ordinary uint4/byte source loads; no new host-cache policy is introduced. New persistent storage is zero; existing pool metadata execution reservation
and indexer workspace remain in the planning ledger. Component allocation deltas
below do not establish reserved/device-used or full-process capacity compliance.

Independent lifecycle review: [lifecycle_review.md](lifecycle_review.md). Its four
failure-ordering/accounting findings were fixed before accepted GPU validation.
The independent 20,000-state CPU allocator audit remains ordering evidence only.

## Validation

- CPU fallback/cache/native-ABI checks: 29 passed, 22 CUDA skips under an explicitly
  CPU-only invocation; four extra alias/short-prefix rejection checks passed.
- Targeted GPU integration after final fixtures: 97 passed.
- Full affected suites: 272 passed in 31.25 seconds, one upstream deprecation
  warning. These include the full C5 selection tests and C3 resident helpers.
- New C6 differentials compare complete record buffers, maps, priority/free state,
  both clocks, append/resident/lease proofs, map_generation, metrics and IDs.
  Coverage includes uint8 width7, FP32 width7, BF16 width576, fragmented and
  nonmonotonic host pages, partial-free stable ties, alternating owners, H<P,
  H=P with A>0, padding/empty/all-hit uncertified calls, rollover, stream changes,
  pending host writes, and isolated invalid private IDs.
- Fault injections at wait, sort, post-gather, pre-publication and post-publication
  remap boundaries match the checked path's completed events, proof mutations,
  counters and other-session records. Releasing the failed user and recalling
  the other user succeeds. Existing pending-prefetch, transaction, transient
  ownership and rollback suites also passed.

The first targeted invocation had 18 passing cases and one fixture failure: its
isolated child omitted native_metadata and correctly took checked ValueError,
so it could not test the intended device trap. The fixture was corrected before
the 97- and 272-case accepted invocations. No kernel numerical failure was hidden.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q cache/tests operators/deepseek_v32/indexer/tests models/deepseek_v32/tests/test_pool_prefetch.py
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python /tmp/deepseek_c6_recall_probe.py --profile
```

## Component timing and CUDA activity

Device NVIDIA H200 / SM90; PyTorch 2.12.1+cu130; CUDA 13.0. Record BF16 x576.
Every measured case restores the same committed snapshot, then writes the same
GPU-only candidate tail outside timing. Complete ensure API wall time includes
its internal miss-count synchronization and a final explicit GPU synchronization.
Five warmups, 25 alternating checked/native samples; profiles are three additional
single-call captures with restoration outside the trace. Every case first passes
strict full-state/output/metric comparison. These synthetic unions do not claim
the model's access distribution or full-request latency.

| P / H / A / Q | Prior displacement | Checked median ms | C6 median ms | Checked GPU us | C6 GPU us | Launches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65536 / 65536 / 128 / 128 | 0 | 1.023729 | 0.192222 | 204.995 | 23.968 | 80 → 5 |
| 65536 / 65536 / 128 / 128 | 32768 | 2.744119 | 1.113463 | 1086.660 | 815.628 | 135 → 21 |
| 65536 / 65536 / 128 / 128 | 65536 | 3.464231 | 1.834823 | 1800.668 | 1537.955 | 133 → 21 |
| 65536 / 65536 / 1024 / 1024 | 65536 | 3.587850 | 1.931295 | 2065.246 | 1626.202 | 133 → 21 |
| 193 / 137 / 35 / 128 | 150 | 1.835004 | 0.362330 | 229.556 | 44.842 | 119 → 12 |

The P193 case releases 70 competitor records before recall: 94 misses consume
those 70 free slots and evict exactly 24 live records in both paths. The 64K
all-displaced Q128 case selects 64,431 unique IDs: 124 candidate IDs are resident,
and exactly 64,307 historical records are recalled/evicted. Its native gather
occupies 1,444.567 us of 1,537.955 us total kernel activity; the remaining cost is
now mostly actual transport. Q1024 recalls all 65,536 history records once.

For Q128/K2048 all-displaced, peak allocated delta fell from 13,049,856 to
4,569,600 bytes; retained output is 1,048,576 bytes in both. For Q1024/K2048,
82,845,696 to 11,042,304 bytes, with 8,388,608 retained output bytes. These are
warmed PyTorch allocated deltas, not allocator-reserved or device-used peaks.
The small P193 native case has a 0.362330 ms median but 0.485441 ms mean; raw
samples retain the variation. No full-model saving is inferred from these probes.

Artifacts: `/tmp/deepseek_c6_recall_probe.py`,
`/tmp/deepseek_sparse_recall_c6_20261004_01.json`, adjacent Chrome traces,
`/tmp/deepseek_c6_resource_usage.txt`. Engineering artifacts remain outside
experiment deliverables. Source/run identities are retained below.

## Compiled resources

Read-only cuobjdump of the measured module reports the following. Every new kernel
uses 256 threads and has zero stack/local bytes. No cooperative launch, inter-CTA
wait or dynamic register redistribution is used. CTA barriers in reductions and
local bitmap aggregation are reached by every thread.

| Kernel | Registers/thread | Static shared bytes |
| --- | ---: | ---: |
| union clear | 24 | 1024 |
| direct union | 18 | 1024 |
| CTA-local union | 20 | 1024 |
| classify | 22 | 2048 |
| compact/tombstone | 32 | 2048 |
| publish | 32 | 1024 |
| final map | 17 | 1024 |

The local-union path adds at most 16,384 dynamic shared bytes; it activates only
for at least 1M selection elements and at most 4096 bitmap words. At P65536/A1024
it uses 8,324 dynamic bytes. Other measured Q128 shapes use the direct union.

## Source identities and parent handoff

- `cache/sparse_token_cache.py`: `011e6990e9a496b1f6af5ea1ae0afc1d466731ca370978d98daa6f8218d5abfb`
- `operators/deepseek_v32/indexer/cache_ops.py`: `8df0c741ecc343537729f5862790b9f3ac1520e9da792aef5ce6e2b22bad0224`
- `operators/deepseek_v32/indexer/csrc/echo_sparse_recall.cuh`: `ddad6791b1205aa931689805018ff86a285601788c620fe4fc592846e74fd39e`
- `operators/deepseek_v32/indexer/csrc/echo_indexer.cu`: `623094b95ea7e81e5784d027ca98c045a6376b5ce141d9efaf562d23b021d1f0`
- `cache/tests/test_sparse_recall_metadata.py`: `f6ec16ea97a95add7f4b06b995c516ced5c2075867d80ba43826e5430cfc8929`
- `operators/deepseek_v32/indexer/tests/test_echo_cache_ops.py`: `6082b82e2264b7901721531794824290be82ac8b4208a29e077e75b4489ae99b`

Source matches the independent review. The measured native module is
`cxldsagr_echo_indexer_154b815ff6e9d7c3`; helper code is included by the existing
model-local source/header fingerprint. Root owns full-model numerical checks,
complete serving trajectories, current-source MFU analysis and replacement of
affected experiment reports. The existing reports have not been replaced by
these component results.
