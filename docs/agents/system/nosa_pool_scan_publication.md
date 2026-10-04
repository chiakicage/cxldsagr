# NOSA pool-scan integration publication

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

The native pool-referrer provider is integrated. Its affected fixed P/NH and
ordinary GR measurements have been replaced after correctness and measurement
review. The current results are in the [fixed report](../../../experiments/nosa_motivation/README.md)
and [旧 GR 实验（已结束）](experiment_organization.md#retired-gr-serving). The overall efficiency
goal remains open: candidate efficiency, remaining tails and compute/IO dominance
are unresolved, and fixed-path async is slower than sync.

The researcher has paused additional H4K/H16K GPU measurements. Their fresh
formal/profile runs had already completed and passed before that instruction;
their valid results are retained. Further measurements are limited to H64K.

## Implementation and acceptance

The fixed runtime contains 118 files, with source identity
`93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79`.
The optional CPython provider uses authenticated runtime/header ABI details and
retains dynamic inspection lookup, retry, fallback and exception behavior.
Unexpected initialization or callback exceptions propagate. GC remains enabled
with unchanged thresholds; the implementation does not mutate GC lists. The
pool-referrer provider is separate from the allocator-snapshot provider.

Native build fingerprint:
`09ba1d834970359cf92e17c028ab5aa7801199d7055212e3ce36635077eaf2ae`.
Binary SHA256:
`ff048610ed0a446f4f7ddf82caa6bada67d6617f0fe1301c7a4c9e1791aa8e01`.
ABI SHA256:
`e0537a6cc6c16867894a36bff1d84bc54b53cce64de99c5a0886b34b3bdc8079`.

Production CPU regression passed 647 tests, with one CUDA-only skip. The separate
CUDA lifecycle run passed 30 tests with no skips, including that skipped case.
Nine real empty-MemPool combinations checked rejection, release and restored
admission. Full32 fixed correctness compared four eager references and 16 graph
requests; ordinary GR compared another 16 candidate outputs. All hidden
comparisons were exact. These are correctness gates, not performance samples.

The integration receipt is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_pool_scan_integration_acceptance_20261004_01/production_integration.json`,
SHA256 `3d5ba5b38d192d360b3cf990d458fac63b36e169eb7a71f58e1768af11b64c96`.
The [optimization contract](nosa_pool_scan_optimization.md) retains the rejected
candidates and earlier intervention boundaries.

## Fresh performance families

| Scope | Current run ID |
| --- | --- |
| Fixed formal | `nosa_motivation_poolscan_sm90_20261004_01` |
| Fixed profile | `nosa_motivation_poolscan_profile_sm90_20261004_01` |
| Independent APIs | `nosa_attention_poolscan_sm90_20261004_01` |
| GR formal | `gr_nosa_poolscan_{h4k,h16k,h64k}_20261004_01` |
| GR profile | `gr_nosa_poolscan_{h4k,h16k,h64k}_profile_20261004_01` |

All 128 fixed formal outputs were reopened, including 96 exact non-HBM/HBM
comparisons. HBM revisit hits remain 0/16 with 31 total evictions; each offload
method has 16/16 revisit hits and no evictions. First/revisit means are:

| Method | First mean ms | Revisit mean ms |
| --- | ---: | ---: |
| HBM | 2338.155622 | 2321.742774 |
| Dense | 2685.632117 | 71.594266 |
| Sync | 2664.894168 | 29.320605 |
| Async | 2758.952741 | 31.404843 |

Sync/async latency ratios are 0.9655446274 for all requests and 0.9336332267
for revisits. These ratios indicate slower async execution. Revisit candidate
H2D is 34,363,932,672 payload bytes for dense and 4,718,657,536 for each sparse
method; payload is not physical bus traffic. All latency samples remain reported.

The fixed profile passed 24 timelines, 12 internal records, 36 exact outputs and
384 layer samples. All 96 applicable async samples failed both 90% gates.
Page-envelope and nonempty-stripe min/median/max ratios are
0.621540/0.713207/0.790677. Another 96 async samples have no fetch and null ratios.
These measurements describe internal work intervals; they are not a complete
compute/IO attribution of ordinary request wall time.

Dense20/sync22/async22 remain the largest revisit candidate samples, at
90.496473/46.396237/48.826513 ms. Against the preceding formal run they are
13.58–14.16 ms shorter, but remain about 21.7–23.4 ms above their revisit medians.
The all-128 descriptive comparison is retained in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_poolscan_publication_root_20261004_01/formal_old_new_descriptive.json`.
Its matched workload/useful-work/cache fields do not isolate system drift or
establish uniform speedup. Whole-case native route counters do not locate or
time individual tail scans.

The independent API comparison uses fresh captures and timings. Its exact
numerators, reference composition and full-request/candidate ratios are in the
[API table](../../../experiments/nosa_motivation/report/nosa_motivation_poolscan_sm90_20261004_01/api_comparison.json).
For request0/16, full-request MFU proximity is 100.227379%/102.020880%;
candidate-only proximity is 75.118264%/73.599107%. Candidate wall times are
16.203399/16.642680 ms and the independent FA3 compositions are
12.171712/12.248864 ms. The full-request ratios can exceed 100% because the
reference sums independently measured medians. Independent API medians are not
an observed full request, a performance ceiling, or quantities to subtract
into removable host overhead. The CPU API review passed all 2,112 attention
calls and exact FA3 receipts; raw QK/PV validation samples three batch rows per
call. Its SHA256 is
`94d0ba81a64646c1a9a0bdbb7393d836fd130d5ebcdd1e11a27e42c28e0f2cfe`.

The six GR families contain 1776 requests and 1332 exact non-HBM/HBM comparisons.
HBM has the lowest all-request mean in 21/21 groups. Overlap is below serial in
21/21 all-request means and 3/17 nonempty revisit groups. All nine internal
overlap samples fail 90%. The full32 budget paths keep compute graphs disabled
and use append/truncate semantics; fixed P/NH uses transient candidates and
pure-compute graphs. Their latency, capacity and memory-budget claims remain
separate. GR active-allocation admission does not certify a whole-process
physical HBM cap.

## Publication, reproduction and replacement

The root publication evidence is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_poolscan_publication_root_20261004_01/`.
`sealed_publication.json` binds reviewed stages, source maps, accepted reviews,
dependency preservation and exact old-root inventories. `publication_receipt.json`
records installed hashes, checked links, the 26 removed roots and three GR
report trees replaced in place. Current raw data stay under their fresh run IDs;
report assets and READMEs refer to those runs.

Required follow-up inputs from superseded fixed families are preserved as a
433-file, 5,969,502,739-byte subset at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_poolscan_retained_followup_inputs_20261004_01/`.
`mapping.json`, `transfer.jsonl` and `reopen_receipt.json` bind source names,
payload hashes, source aliases and copy/reopen verification. The mapping SHA256
is `b65e0efbdc6e12a1a6c5d1d8bbe4cb2a69199484a91464f0b3c3f512d93d020b`;
the one-shot reopen receipt SHA256 is
`c635940898f65db8c5c8fe9d4bce40e84f50b572c8cefa60db6876446653f10b`.
Existing consumer receipts are unchanged. They describe historical evidence and do not certify
new runtime performance. Superseded raw families are not retained wholesale.

The GR cross-version workload auditor uses 42 byte-identical fresh request and
workload files through
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_poolscan_gr_finalizer_20261004_01/workload_alias_reopen_receipt.json`,
SHA256 `3dea100fb9472a70fad9e2956fd9064f48f9500f4f7135ef542c011e9df7196a`.
Both sides were verified before cleanup; the original auditor records remain
historical. Current GR reproduction material is in
`experiments/gr_serving/output/data/gr_nosa_poolscan_publication_20261004_01/`.

The initial API launcher wrapper aborted before reaching Python. It produced no
measurement, API destination or residual GPU process. The retry corrected only
target-process recognition and retained the frozen measurement argv/run ID.
Separate wrapper receipts are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_pool_scan_integration_acceptance_20261004_01/`.
The failed wrapper receipt was not overwritten or counted as an API sample.

## Next bounded diagnostic

The prepared H65536/A128 HBM prequeue test compares ordinary, event-only,
delayed-control and prequeue calls on requests 0 and 16 in separate processes.
It uses unchanged candidate computation and checks whether the full pre-check
DAG is queued before the start event executes. Finite validation, transaction
completion and lease drain remain intact. The plan and driver are at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_hbm_prequeue_diagnostic_20261004_01/`.
At this publication checkpoint no diagnostic GPU run has occurred. A shorter
prequeued event interval would establish sensitivity to this intervention;
it would not be an ordinary serving speedup or a proof of compute/IO dominance.

Subsequently, both requests completed and passed independent review; see the
[diagnostic results](nosa_hbm_prequeue_diagnostic.md). The publication receipt
above records the files and state at publication time, before this follow-up.
