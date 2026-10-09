# Initial compute-bank collection result

Moving initial compute-bank preparation outside active collection does not
remove the long HBM gap. The measured idle is 67.648 us, compared with
67.584 us when the same uninspected bank was prepared inside collection.
Both have 416 ns median gaps. No cause or optimization is established.

| Initial bank / replay | L0-L2 span us | Busy us | Idle us | Median gap ns |
| --- | ---: | ---: | ---: | ---: |
| Inside collection / warmup | 1078.817 | 1012.673 | 66.144 | 416 |
| Inside collection / measured | 1056.001 | 988.417 | 67.584 | 416 |
| Outside collection / warmup | 1072.963 | 1005.539 | 67.424 | 416 |
| Outside collection / measured | 1059.235 | 991.587 | 67.648 | 416 |

The new run is `q1_compute_collection_profile_20261008_01`, compared with
`q1_compute_inspector_profile_20261008_01` on GPU0/CPU0-7. All-four-method
warmup/cache history and later full-extend inspection remain unchanged.
The first of 13 ranges contains exactly one completed
`compute_bank_already_prepared` marker enclosing one device synchronization,
with zero graph/capture calls and zero GPU activities. The missing initial
bank-node history is intentional; no prefill graph attribution is claimed.

`q1_compute_collection_audit_20261008_01/windows.json` verifies both runs'
complete 197-node HBM signatures and native ownership, their 192 layer nodes,
and the equal clipped union of every intersecting process activity. Outside-
collection CPU graph launch times are 499.151/319.003 us.

Independent audit
`/tmp/cxldsagr-checks/compute_collection_independent_audit_20261008_01.json`
verifies 1,370 production sources, 6,434 concrete runtime/source files,
unchanged execution/receipt/benchmark/input identity, and all eight saved
output tensors against the inside-collection control and independent check.
Base measurement files are unchanged; the explicit wrapper is replaced.
All 13 profiler/exporter identities match; the only launcher-environment
difference between these controls is the run ID. The audit JSON and script
are also copied into the analysis run directory.

The complete formal lifecycle still differs from the earlier reduced HBM
controls. The latter also used GPU1/CPU8-15, separate JIT caches and some
different quantization binaries, as recorded in
`/tmp/cxldsagr-checks/formal_vs_timer_runtime_20261008_01.json`.
Offload preparation additionally creates host pools, but it occurred outside
collection, so the existing trace does not establish pinned-allocation
history or allocator state as the cause. Further work must preserve these
uncertainties and use an explicitly matched comparison.
