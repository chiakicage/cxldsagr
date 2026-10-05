# NOSA sparse fetch / attention overlap: KDA task contract

This task follows NVlabs KDA `ef6ce617693ef0782b3ecb9f37e39bbf10226a90` at
`/root/.local/share/kda`; implementation belongs to this project.

## Objective and interfaces

Target NOSA BF16 65536 prefix + 1024 queries, 32 Q heads, 2 KV heads, D128 and
64-token / 64-block selection. Every sample must reach >=90% for both page-
envelope and true stripe-window overlap with softmax, while unprofiled complete
latency is below a freshly measured whole-query sparse-union serial control.

The explicit model interface remains full NOSA `attention_mode="sparse"`,
`cache_backend="offload"`, `offload_query_tile_size=128`,
`offload_fetch_ctas=96` and `offload_overlap=True`. Query tile size groups
first-use byte counters only, never splits attention. CUDA requires native
SM90 BF16/D128/GQA16; unsupported paths and capture fail explicitly. CPU is a
correctness reference. CIS and compressed records remain resident; incremental
compression uses the device append and at most 31 historical boundary tokens.

## Implementation contract

A GPU planner deduplicates `(KV head, block)` pages. Each unique 64-token
page is partitioned into eight disjoint 8-token stripes. Fetch leaders atomically
claim `(page, stripe)` tasks and broadcast through a shared slot and 96-thread
barrier. Each historical 16-byte K/V vector has one owner and one `.cv` host
load; all reuse is in HBM. Different CTAs may fetch disjoint stripes of a page.

Each stripe's writers complete stores, thread fences and the subgroup barrier
before the leader's `atom.acq_rel.gpu.global.add.u32` completion. The RMW chain
joins all CTAs; TMA acquire observes ready=8, then an async-proxy fence precedes
TMA. Empty tail stripes have no reads, payload or trace but still complete the
chain. Only the last completion counts the page's logical bytes.

All CTAs compute persistent FA3. By default, up to 96 CTAs additionally use producer warps
1–3 for fetch; warp 0 retains TMA and both consumer warpgroups retain attention.
Only two KV heads with `ceil(queries / 8) * KV_heads == 256` use matching
head-1-then-head-0 fetch/compute priority. Each head retains page-zero-first,
descending-block fetch and native compute cost ordering/logical-batch ties.
Other shapes use block-major fetch and the original attention scheduler.
Producer/consumer limits remain 24/240, guarded against the actual
64512-register CTA pool; cooperative occupancy ensures concurrent residency.

Native initialization merges full-capacity metadata reset, historical page-zero
padding and strided suffix staging. First-use planning remains a separate,
stream-dependent launch. The strong serialized control uses the same new
initialization, fetches the exact whole-query sparse union once and then runs
the original full-query FA3. Complete-call timing includes initialization,
planning, compaction, prepare/sort/main/repair, launch gaps, output handling and
completion dependencies. Selections, CIS, masks and per-query arithmetic remain
unchanged. One full logical-address HBM staging layer is shared across layers;
queue, staging and scratch reuse wait for previous caller completion.

## Independent evidence

Let S be the union of validated nonempty stripe-copy windows, E the union
of page envelopes, and M the union of instrumented softmax-update windows.
Each envelope equals min(stripe starts) through max(stripe ends); E can contain
gaps absent from S. Report `duration(S ∩ M) / duration(S)` and
`duration(E ∩ M) / duration(E)` independently. Both must be >=0.9 for every
profiled sample in every layer, checked before aggregation; medians alone do
not pass. Neither ratio is assumed to bound the other.

Stripe timestamps exclude task claiming/decoding and consumer ready polling.
They start after the barrier that excludes task claiming/decoding and before
the barrier releasing host loads. They end after stores, thread fences and
the subgroup barrier, before completion RMW and counters.
Concurrent windows are unioned once. Every expected nonempty stripe occurs
exactly once with its historical payload, and stripes partition each page.
Page envelopes must match stripe min/max. Softmax excludes QK/PV MMA, so these
ratios are not full attention hiding or PCIe/CXL wire occupancy. Logical KV
payload does not prove physical link bytes. Device-globaltimer and Nsight
absolute clocks are not mixed. Nsight separately checks one fused main per
sample, preparation classification and zero serial fetch/attention overlap.

## Accepted evidence and source ownership

The accepted implementation build key is `484e32532ba74fb6`. See the
[2026-09-30 checkpoint](checkpoint.md) for run IDs, results, correctness and
measurement limits; future changes require new acceptance and run identities.

Source ownership is [backing](../../../../cache/host_backing.py),
[NOSA adaptation](../../../../models/nosa/cache/offload.py),
[workspace](../../../../operators/nosa/attention/offload/api.py),
[native preparation](../../../../operators/nosa/attention/offload/csrc/nosa_offload.cu) and
[fused kernel](../../../../operators/nosa/attention/offload/csrc/nosa_offload_fused.cu).
Correctness tests remain in module test directories; failed diagnostics remain
in `/tmp`. Unaffected resident reports retain their source and measurement identity.
