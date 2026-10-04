# Shared Group8 union with original M128 row halves

Status: rejected for promotion, 2026-10-04. Staged operator and full-model
correctness passed, but all three formal overlap samples failed the fixed 0.90
gate. The pre-implementation contract below is retained as the executed plan.
No integration is authorized by this candidate's results.

Progress: static/compile gates and `same_input_all32_gpu4_02` passed. All 64
candidate outputs exactly match saved Group8 hashes, with FP32, unique-byte,
async stripe/envelope and cleanup acceptance. Root independently reopened the
evidence and allocated GPU4 for the unchanged full32 fixture through wrapper
`55340fab37d99290d48abb3c804bcb64ca3037b472600b6d13d96b29ecf1a13d`.
The independent full32 run then passed with all 16 graph outputs exact; evidence
SHA256 is `8f30cc3b94423cf029fc9973da2cd8237639e29a3e20bbc243a839048fd5dd8f`.
Root allocated GPU0/NUMA0 for the prepared performance matrix. The fixed
performance matrix then failed all three candidate overlap samples: 0.710582,
0.728228 and 0.695460. Async-half CUDA/wall medians were 0.632576/0.669274 ms,
versus original serial 0.644640/0.669534 ms. See [checkpoint.md](checkpoint.md)
for the decision identity and measurement boundary.

The previous Group4 candidate failed the full32 fixed numerical contract and every representative overlap sample. Original Group8 V-first was then rejected: representative async0.758784ms versus serial0.608864ms, with all overlap samples below 0.90. These are rejected-candidate engineering observations, not accepted experiment results. This candidate starts from current repository Group8 sources and does not include either rejected change.

Stage: `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_union_halves_candidate_20261004/`.

Ownership: experiment agent owns CUDA kernels, native `_fa3.py`/`_fused.py` and native-operator tests. Cache agent owns Python workspace/offload API, model resource planning, overlap analyzer and their tests. HBM-audit agent reviews the frozen native source and contract independently. Root owns later integration and experiment provenance.

The agreed native interface is `build_info(compute_halves=1)` and `_module(compute_halves=1)` in the two native loaders. Existing TVM forward arguments remain unchanged. Compile-time specialization2 is selected only for actualQ128/Hkv2. Workspace/resource configuration uses `q128_compute_halves=2` with default1; selection grouping always remains8. Original page/member/count tensors remain keyed to original unions. Trace records use `attention_group_size=8` and effective `attention_compute_halves=1|2`.

This round ends at staged implementation, CPU/static review and compilation guards. GPU4/NUMA1 correctness requires the coordinated allocation after static review; no GPU performance is authorized yet. Every executed run must record exact source, helper, selected build and mapped native identities. Failed artifacts remain outside `experiments/`.


Root authorized exactly one staged implementation after rejecting V-first. The future constructor/build vocabulary must distinguish `selection_group_queries=8`, `compute_queries_per_half=4`, `compute_halves=2`, `query_heads_per_kv=16` and `compute_rows=64`. Do not name this route Group4, set `FA3_GROUP=4`, pass `q128_group_size=4`, or rebuild four-query unions. A possible explicit opt-in is `q128_compute_halves=2`; the public selection group remains8 and default compute halves remains1. Restrict the specialization to Q128/Hkv2; other calls, including Q1024 history chunks, retain the original Group8 module.

For original union `u = query_group*Hkv + kv_head`, make work item `w=2*u+half`. Its64 rows correspond to original M128 rows `original_row=64*half+row`. Therefore:

| Use | Required value |
|---|---|
| Original union query start | `selection_query_base=8*(u/Hkv)` |
| This half's Q loading start | `compute_query_base=selection_query_base+4*half` |
| Query membership bit | `query_in_union=4*half+row/16` |
| Physical output query | `selection_query_base+query_in_union` |
| Physical output Q head | `16*(u%Hkv)+row%16` |
| K/V/CIS head and ready identity | `u%Hkv`, and `block*Hkv+u%Hkv` |
| Page/member/count index | `u`, with the original512-int allocated row stride |
| Trace math work identity | `w`, not `u` |

Use one shared decoder for work/union/half coordinates in scheduler, producer, variant, epilogue and trace. If the existing eight-field coordinate tuple is retained, its last field may carry `work_id`; every place currently treating that field as a union batch must explicitly decode `u=work_id/2`. `kh` remains the physical KV head, `klen=counts[u]*64`, and page addresses use `u*Capacity`. Never derive physical head as `work_id%Hkv`.

Prepare remains exactly the original32 unions for Q128/Hkv2, including page order, eight membership bits, even-page padding, count, overflow and fallback decisions. Both halves traverse the entire same original page list in the same descending128-token tile order, including masked columns; neither half filters out pages selected only by its sibling. The all-common shortcut must still test membership `0xff` and the original selection-group query start. In the masked path, add the half offset to the membership/causal query index; merely changing `query_start` to the compute-half start is insufficient because it would lose the original membership bit and shortcut behavior.

The Q TMA descriptor remains16×64. Issue four queries beginning at `compute_query_base`, with destination `dhalf*compute_rows*64 + local_query*16*64`. The64×128 BF16 query payload is16384 bytes, which must equal the query barrier's transaction count. K/V descriptors,128-token tiles and two pipeline stages remain unchanged. The epilogue writes only the half's64 rows. If either half detects nonfinite output, mark both original four-query fallback bins, as original Group8 does; otherwise repair scope could change numerics even in the other half. Zero-store and overflow handling cover disjoint half rows, then the unchanged full repair kernel runs once after all cooperative work finishes.

All 64 CTAs must compute one real four-query half. They share the existing single unique host queue, physical ready flags, first-use metadata and byte counters. The96 fetch workers remain in each attention CTA. Each physical historical vector still has exactly one `.cv` host read and each page has eight disjoint stripes. The two halves independently TMA-read K/V from HBM and stage CIS in shared memory; the resulting duplicate HBM-to-shared work is expected and must be reported separately from unchanged unique host bytes. No dedicated fetch CTA is introduced.

Counts, pages, members and fallback allocations remain sized for32 original unions, not64 compute works. For Q128, scheduling can expand each original logical union to its two consecutive halves without another order buffer; existing Q1024 ordering stays in the unmodified original path. Trace math uses `fetch_slots+w*64+tile`, so Q128 reserves4096 math rows instead of2048 and moves the stripe offset accordingly. At maxQ128 this costs65536 additional bytes; accepted maxQ1024 storage already covers the Q128-only specialization. The trace schema must expose selection group8 and compute halves2; it must not reuse a Group4 label.

The one-consumer scheduler and160-participant query-empty barrier are required, with no peer-consumer barrier. Keep fused producer/consumer limits24/240 initially, but obtain actual selected-kernel register allocation and occupancy before launch; demand33792 alone does not establish an adequate CTA pool. A matching serial-half control may use the upstream one-consumer register limits, with its own guard. All build keys must include selection group, compute halves, compute rows and actual native source/header identity, and must keep the original Group8 control binary separately addressable.

The arithmetic argument is stronger than for the GQA8 remap: this preserves the original M64 atom's row positions, physical16-head ordering, paired-page sequence, masks, FP32 reductions and P/PV traversal. The original updater parameter remains2. It still is not proof of bitwise equality: template specialization, layout selection, scheduler and compiler changes require measurement.

The pre-implementation plan and stopping gates are:

1. Freeze the above contract and a separate original-Group8 source snapshot. Keep the rejected Group4 implementation out of dispatch and reject a four-query selection-union build. V-first is closed; implement this one candidate in the isolated stage only.
2. Add the explicit coordinate decoder and narrow compute-half specialization while leaving original prepare and control paths unchanged. Verify that every `(query, Q head)` has exactly one half owner, both halves reference identical original union rows, and resource/trace planning occurs before allocation. Check the selected shared/register and query barrier geometry by compilation; do not launch a potentially waiting CTA configuration.
3. Before full-model work, run the accepted captures with original Group8 hash verification first. **Every same-input candidate output must reproduce the saved Group8 hash exactly**, across all32 layers and both serial-half and async-half routes. FP32 tolerance is an additional check, not permission to accept a nonexact result. Stop on the first nonexact output, even if0.016 passes. Also verify unique bytes and disjoint stripe identities with poisoned misses.
4. Only after exact same-input acceptance, run the independent full32 fixture with two users/two visits, empty-prefix controls, effective pre-allocation split settings and cleanup checks. Retain the existing full-hidden atol=rtol=0.016 contract and report exactness separately; retained-output lifetime checks remain exact. Do not alter a threshold after failure.
5. Only after correctness passes, measure31 complete-API repeats against original serial Group8 and the matching serial-half control, plus three actual interval samples. Every page-envelope and stripe-copy ratio must reach0.90, and complete async latency must beat original serial Group8. Report duplicated HBM/shared work separately. Broader performance measurement and integration require those gates; no candidate is promoted from source plausibility or isolated timing.
