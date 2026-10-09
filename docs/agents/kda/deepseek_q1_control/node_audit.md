# HBM graph-node and top-k audit

This read-only audit uses local matrix `deepseek_h64k_a1_official_20261008_03`,
combined report `engine_mfu_decode_official_20261008_03`, and official HBM profile
`h64k_decode_hbm_profile_20261008_05`. It preserves the published results and
does not add GPU measurements. Raw CUPTI rereading and interval arithmetic are
recorded in `/tmp/cxldsagr_hbm_node_audit_20261008.json`; the helper is
`/tmp/cxldsagr_hbm_node_audit_20261008.py`.

## Node accounting

Both selected steps launch one CUDA Graph. Local replay contains 203 actual GPU
activities (187 kernels, 15 memcpy, one memset); official replay contains 141
(140 kernels, one memset). The L0–L2 windows contain 198 and 136 activities.
Thus the same 62-activity difference occurs within the layer window. These are
GPU activities, not 62 additional Python launches or a count of every CUDA
Graph event/empty/dependency node.

The following regrouping follows captured owners and names. It changes no
published category or timing.

| Work group | Official activities | Local activities | Difference |
| --- | ---: | ---: | ---: |
| Projection, norms and indexer projections | 75 | 105 | +30 |
| Indexer setup, packing, MQA and causal tail | 6 | 21 | +15 |
| Complete top-k scope | 12 | 12 | 0 |
| Indexer/main-KV cache writes | 6 | 21 | +15 |
| Complete main-attention API | 9 | 9 | 0 |
| Output projection and MLP | 27 | 30 | +3 |
| Shared final norm overlapping the window edge | 1 | 0 | −1 |
| Total | 136 | 198 | +62 |

The official shared final norm overlaps the layer window by only 0.320 us.
Its full kernel duration is not charged to this window. The extra local MLP
node per layer comes from separate gate/up GEMMs; merging weights is not an
equivalent zero-cost implementation change.

All 15 local memcpy activities are D2D, not host KV traffic. Per layer they
write 128 B of index K, 4 B of index scale, 1,152 B of main KV, a 4 B
host-to-device mapping and an 8 B reverse mapping: 3,888 B in total. Their
activity-time sum is 13.888 us. `index_cache_write` owns six copies (5.856 us);
resident `cache_write` owns nine copies (8.032 us), plus three arange and three
int-cast kernels (5.824 us). The latter five operations per layer implement
the resident append in `cache/sparse_token_cache.py`; they are real state
updates, not rollback-offset backups. HBM does not mutate prefetch hints.

Other local GPU-control work includes page64 packing (15.681 us), explicit Q
repeat for split attention (11.104 us), rotary/KV concatenation (7.872 us),
and small metadata creation. Control is a classification, not proof that work
can be removed. For example Q repeat belongs to the complete attention API,
and packing supplies the official paged MQA layout.

## Idle intervals and profiling boundary

The local window is 1,073.347 us versus 1,014.975 us. Busy-union time differs
by only +3.814 us; uncovered GPU time differs by +54.558 us. Local has 150
gaps totalling 68.832 us (median 0.416 us, maximum 1.088 us); official has 98
gaps totalling 14.274 us (median 0.128 us, maximum 0.544 us). The largest local
group is 52 gaps within projection work, totalling 24.352 us. Other internal
groups include indexer 8.000 us, MLP 7.264 us, resident append 4.928 us, and
top-k 4.608 us.

The gap difference therefore reflects both more gaps and longer observed
gaps. Adjacency does not establish a dependency bottleneck or its cause.
These are node-level NSYS traces on different physical GPUs and framework
versions, with different tokens and warmup/collection boundaries. No paired
profiler-on/off experiment isolates relative inflation. Removing a node may
reduce replay scheduling work, but this trace does not predict clean wall
savings by adding its duration and adjacent gaps.

## Top-k is a different operation sequence

The published kernel-family sums are 141.121 us locally and 103.968 us
officially. Official arange (2.272 us) is classified as GPU control; complete
top-k scopes instead sum to 141.121 and 106.240 us. Both have four kernels per
layer. Local scope envelopes, including internal gaps, total 145.729 us;
official envelopes total 107.296 us.

| Local operation, three layers | Activity time us |
| --- | ---: |
| FlashInfer FilteredTopKUnified, SMALL tie selection | 79.873 |
| FinalizeTopKIndices: sort selected IDs ascending | 25.344 |
| StableSortTopKByValue: stable descending value sort | 32.992 |
| Nonfinite-ID mask | 2.912 |

Official fused `topk_transform_decode_kernel` totals 95.200 us, followed by
arange, comparison and padded-row masked fill. Its source writes selected
positions through atomic counters and directly maps them to the page table;
it does not perform the local full deterministic value/index ordering. The
official call also operates on different logits. A direct replacement would
not preserve this project's exact-top-k contract.

Both selection cores launch one 1,024-thread CTA and use 128 KiB dynamic
shared memory. Local core reports 64 registers/thread and 15,760 B static
shared memory; official reports 26 and 11,536 B. These launch attributes do
not establish occupancy, stalls or a benefit from changing the core. The
local core is already faster in these traces; its 61.248 us postprocessing
is the clearer bounded candidate surface.

Installed FlashInfer `topk.cuh` explicitly derives
`deterministic_selection = deterministic || tie_break != None`. With SMALL,
setting API `deterministic=False` retains the same deterministic Filtered
kernel specialization while skipping the independent index-sort finalize;
`sorted_output=False` also skips the independent value sort. A local fused
sort/mask can then restore exactly `(FP32 ordered value descending, ID
ascending)`, preserving value bits and the original nonfinite-ID contract.
This is a candidate, not an accepted optimization.

## Candidates and required evidence

1. Keep the original Filtered+SMALL core and fuse its three postprocessing
   kernels into one. Current surface: 61.248 us and nine kernels per three
   layers; replacement would remove six nodes, but its sort cost must be
   measured. Validate all-equal ties, signed zeros, -inf padding, changed
   graph inputs, actual L0–L2 scores and k boundaries bitwise before timing.
2. Fuse resident append record/map publication into one kernel per layer,
   preserving transaction state. Current five-node append activity totals
   13.856 us plus 4.928 us internal gaps across three layers. This would
   remove 12 nodes; no clean speedup is yet established. Index K/scale writes
   are a separate six-copy, 5.856 us surface that may accept fused/direct
   producer writes with explicit lifetime and aliasing validation.
3. Prepare fixed-prefix graph geometry outside capture: token/rotary
   positions, trig table, causal ends, block table and MQA schedule. The
   graph already binds history/query geometry, but prepared storage must be
   owned, budgeted and invalidated with that identity. Scores, selected IDs,
   KV contents, maps and dynamically changed token results remain live work.

Component gains require fresh independent check, complete-API balanced A/B
timing and actual native identity. Production promotion additionally requires
the affected full-model and experiment gates; the current reports remain
valid only for their frozen implementations.
