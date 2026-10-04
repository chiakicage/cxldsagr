# NOSA pool-scan optimization contract

This work follows the completed
[candidate diagnostic](nosa_candidate_efficiency_followup.md), for research
items 2.1 and 4.1. The reviewed native pool-referrer provider is installed at
the 118-file source identity
`93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79`.
Production CPU/CUDA and both full-checkpoint gates passed. Fresh fixed formal,
profile and API families and all six GR families have been accepted and
published. The [publication record](nosa_pool_scan_publication.md) identifies
the new evidence and exact replacement scope. Historical candidates below
retain their original identities and limits. The partial tail reductions do
not establish uniform speedup or complete compute/IO dominance.

## Measured problem

Matched full-trace diagnostics contain 36.471005/36.012608/36.528546 ms of
`gc.get_referrers` work in dense20/sync22/async22 candidate tails. Direct GC
callback time is separate. The probe's sealed receipt is
`682ae6868a778918492be9bbbfbd6e89394837ef372e881929253ba0ba13324b`.
These durations identify a host cost. The first intervention below measures a
partial tail reduction, without isolating the scan's removable share. The
original formal events were not instrumented.

The current incremental guard already avoids complete scans until a later
generation collection, frozen-count change or inspection identity change.
Every guard still checks fresh allocator settings and segment ownership.
The remaining target is the complete scan itself, without suppressing GC or
moving required work outside the measured request.

## Required behavior

- Reject observable live Python MemPool wrappers, including empty pools and
  ordinary-generation subclasses. Preserve the existing frozen-generation
  exclusion and documented external-pool observability limits.
- Keep the allocator configuration, foreign segment, owned graph-pool and
  environment checks. Do not cache a successful observation after rejection.
- Preserve subclass-inventory retry and audit event/error behavior. Unsupported
  interpreter, replaced inspection primitives, active collection and reentrancy
  need a conservative path. Native traversal must hold the GIL.
- Bind private interpreter fields to a verified runtime/header ABI. A version
  string and compiled offsets alone do not authenticate a patched interpreter.
- Keep automatic GC settings unchanged in model runs. A CPU fixture that
  disables automatic GC equally for both implementations is only a traversal
  microbenchmark, with that condition disclosed.

## External candidates

The first prototype uses `PyUnstable_GC_VisitObjects`. It visits the frozen
generation too, regresses on a large frozen heap, and did not establish audit
equivalence. It is not ready for integration. Its original material is retained
in `.cxldsagr-tmp/nosa_pool_visitor_investigation_20261004_01/` under the workspace
parent.

The second prototype traverses only ordinary generations through pinned private
CPython fields. It filters concrete types before invoking candidate
`tp_traverse`, while retaining the original Python helper code object and retry.
Source SHA256 is `f51165fb28590d0ba11a1b829e80d65d34cdc571dfbc3e702310f4ee77df02a1`;
binary SHA256 is `52dc5209df431e8f7a3eee23741d409d1955e6567f7bed7ad776da2bbf5926b4`.

Its isolated CPU full-guard fixture uses fixed allocator metadata and real GC
stamps, with automatic GC disabled for both implementations. For 182,763
ordinary objects, forced-full median time changes from 5.379097 to 2.950307 ms;
after another 100,000 retained lists, from 5.810605 to 3.192465 ms. A large
frozen heap leaves the complete guard near 7.6 ms in both versions because the
unchanged freeze-count stamps still traverse it. These are 41-sample fixture
medians, not model or serving performance.

Independent review found a counterexample: a `sys.settrace` callback can change
a non-pool referrer's type to Pool after the C scan and before the Python `any`
filter. The original result list retains that object; the filtered list omits
it. The original helper returns true and the candidate false, with no collection
for the outer stamp to detect. A separate ordinary SIGALRM-handler case also
changes a compatible object's class at the return boundary, without tracing or
collection: the original complete guard rejects, while the candidate admits
with the Pool object still alive. These CPU cases use a synthetic Pool type;
they establish a failure of the candidate's general type-inspection contract,
not a GPU MemPool construction test. Thus the second candidate is not accepted.

A post-return GC/finalizer case changes raw-helper behavior by retaining fewer
non-pool referrers. In that case the complete guard observes the collection,
retries, and rejects the newly created pool. This lifetime difference is not a
demonstrated complete-guard false acceptance.

Prototype and counterexample material are retained in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_pool_referrers_native_20261004_01/`
and sibling `nosa_pool_referrers_review_20261004_01/`. Successful bounded cases
do not override the confirmed false acceptance. The independent review SHA256 is
`3ccd502703708218fc91097638a6765228cef9b2a9e2dc4ecf42e6ec6719d91f`;
its receipt is `7d1b4d789aeb595fefc29f87c8eb5a223fd1cec27e261fe3becf79854f53fa5f`.

The third prototype remains external in sibling
`nosa_pool_referrers_native_20261004_02/`. It retains the original
scan for tracing, profiling, monitoring, custom Python signal handlers and
changed inspection primitives, with checks before and after callbacks. Default
signal actions, ignored signals and an authenticated default interrupt handler
can use the accelerated path. Review nevertheless found a dynamic lookup gap:
the transformed helper skipped the original `gc.get_referrers` attribute lookup,
and its fallback always used the captured module. This version is not selected.

The successor in sibling `nosa_pool_referrers_native_20261004_03/` preserves
that lookup before star-argument expansion. A binder returns the accelerated
entry only for the authenticated builtin; other resolved callables are returned
unchanged. Already-bound builtin calls keep one audit event and continue with
an unfiltered scan when a callback changes supported conditions. Thirteen
focused independent CPU records passed, including the original trace/signal
cases, pending-call handling, alias lookup and getter/argument evaluation order.
This is bounded semantic acceptance for an external intervention, not production
integration or a performance claim. It does not extend the existing non-atomic
guarantee to arbitrary future cross-thread allocator/type mutation or falsified
GC metadata. Conservative monitoring detection can keep the fallback active
after a tracing tool has been disabled.

The runtime ABI is pinned to the actual statically linked uv Python executable,
not a separately installed `libpython` file. The loader verifies that executable,
the symbol hosts and the 335-header closure before importing the native module.
Unsupported identity selects the original helper. The controlled intervention
requires authenticated acceleration and refuses silent fallback.

| Successor evidence | SHA256 |
| --- | --- |
| C source | `f9288e23daa1034b748739c02b4bf30e9dfe7c61a25e28c3716dfb77fa47da3e` |
| Binary | `e050128068eaa7ef7d9a8b6be02a1859dc8b836f3f75c2075e445435b4b6bb9a` |
| Loader/harness | `5b24fe6990fbd894d3d96e156db66dfe2cbb911276c37684779b22a6414499bc` |
| Independent successor review | `f56eca06b5636b9c053738de4b68fd1c5fecfba33fdd0b9a8f61ed1152f6f277` |
| Independent intervention-driver review | `d2d9c9a1ab22838b21731fefc9c4c377f15af463559314c32da04499b0d6d84c` |

The intervention driver and launcher are prepared in sibling
`nosa_pool_scan_intervention_20261004_01/`. Both arms import identical prototype
material; only helper routing differs. The driver pins C/binary/loader hashes,
preserves automatic GC settings, checks every saved hidden output and requires
accelerated calls during each actual native `run_case`, beyond warmup. These
checks accompany the timing results below; case-wide counters do not identify
individual scan durations.

The actual CUDA empty-pool gate `nosa_native_pool_guard_sm90_20261004_01` has
since passed on GPU5/H200 with CPUs 48–55 and NUMA node 1. All nine combinations
of real `torch.cuda.MemPool(no_split=True)` subclass depth 0/1/2 and ordinary
generation 0/1/2 were rejected on the filtered native path. Each object was
released and default admission restored. The gate recorded 19 filtered calls,
zero fallback/unfiltered calls and unchanged allocated/reserved device bytes;
it executed no tensor workload or performance measurement. Evidence is in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_native_pool_guard_sm90_20261004_01/gate.json`.

The first baseline/native full-trace pair completed with both processes exiting
zero and all 128 outputs in each arm exact to the accepted outputs. Independent
integrity checks passed 6,118 comparisons; an independent arithmetic review
passed 9,692 checks. All 256 observations and 128 matched pairs are retained in
sibling `nosa_pool_scan_intervention_analysis_20261004_01/`. Its final receipt
SHA256 is `2a6a7a86fc5198d4296eb59519474b72ba00e5247f6551984d60843b97880fd7`.

| Tail request | Baseline extend ms | Native extend ms |
| --- | ---: | ---: |
| Dense 20 | 104.600 | 90.800 |
| Sync 22 | 60.356 | 46.532 |
| Async 22 | 63.451 | 49.468 |

The same request IDs remain maxima, about 22–24 ms above the native revisit
medians. Offload revisit means improve by 0.772–0.866 ms, while differences of
medians are +0.056/+0.222/+0.074 ms. Offload first-visit extend means increase
by +0.595/+0.582/+0.550 ms; 47 of 48 paired first-visit candidates are slower
in native. HBM first/revisit extend means increase by +0.063/+0.144 ms, and
native request 24 becomes the HBM maximum at 18.651 ms. The descriptive tail
screen flags no new native-only request, which does not mean no new maxima.

Native records 19 filtered calls and no fallback/unfiltered calls; the actual
four `run_case` intervals contain 2/4/3/3 filtered calls. GC remains enabled
with thresholds 700/10/10 and recorded freeze count 375. One sequential pair
does not isolate order or system drift and does not establish uniform or net
speedup. The second pair reversed arm order with unchanged driver/prototype,
using fresh `native_02` then `baseline_02` labels. Both processes exited zero
and each passed all 128 exact-output comparisons.

| Second-pair tail request | Baseline extend ms | Native extend ms | Native minus baseline ms |
| --- | ---: | ---: | ---: |
| Dense 20 | 104.646 | 91.295 | -13.351 |
| Sync 22 | 60.184 | 46.343 | -13.841 |
| Async 22 | 63.128 | 49.777 | -13.351 |

The partial tail reduction recurs with reversed order. Offload revisit extend
means improve in both pairs. In the second pair, first-visit extend mean deltas
are -0.240/-0.076/+0.057 ms, unlike the uniformly positive first-pair deltas.
Pooling the two repeats descriptively gives revisit extend mean deltas of
-0.810/-0.860/-0.838 ms and first-visit deltas of +0.177/+0.253/+0.304 ms.
Typical and complete-request effects remain mixed. Repeated IDs are not
independent samples, and reversal does not remove system drift. All 512
observations and 256 matched pairs are retained in sibling
`nosa_pool_scan_intervention_analysis_20261004_02/`; independent arithmetic
review passed 31,625 checks with zero discrepancies. Saved-record integrity
passed 12,256 checks. The combined analysis receipt SHA256 is
`cc918be091ba55e607c1f434eac7a75de0835db5a5db002a7a1f44cdad41f93f`.

These observations support preparing an integrated candidate for fresh
acceptance. They do not promote that implementation or establish uniform/net
speedup. The external integration memo is summarized in the
[follow-up record](nosa_candidate_efficiency_followup.md). Production and
published experiment assets remain unchanged.

The integrated candidate is now being prepared externally in sibling
`nosa_pool_scan_integrated_candidate_20261004_01/stage/`. It replaces AST
substitution with a literal call expression, keeps the original retry loops,
and adds optional authenticated loading plus fixed/GR provenance. Its loader,
literal-helper semantics and reporting adapters require new CPU review before
repository integration and GPU acceptance. Earlier prototype checks do not
cover these new setup and lifecycle paths.

## Integration and measurement sequence

The installed source is the repaired v2 loader with the literal helper call,
unchanged outer retries, exact static-CPython/header authentication and explicit
original-provider fallback. Unexpected setup/callback errors now propagate;
the rejected v1 broad exception catches were never installed. Independent CPU
review found no open issue in its bounded cases, including the original trace,
signal and pending-work counterexamples. This is not a general raw-referrer
lifetime-equivalence guarantee. The experimental adapters also bind provider,
binary, ABI and case counters, including the shared helper in fixed-family
runtime/orchestration comparisons.

The source installation receipt is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_pool_scan_integration_acceptance_20261004_01/production_integration.json`,
SHA256 `3d5ba5b38d192d360b3cf990d458fac63b36e169eb7a71f58e1768af11b64c96`.
It records all 18 changed/new files and preserves their prior bytes. The model
review receipt is `3c08360a552059076bba9b87e4923d111669c0f9d673b8fc1c405e6a1eaa47ba`;
the amended experiment receipt is
`f195a7150e05870753ec565a37778206afa191d7950c097e400dd1c13286110a`.
The two experiment READMEs label the pending replacement; all published assets
and corresponding raw outputs remain intact.

1. Resolve supported-path false negatives with targeted differential cases;
   preserve original failed candidates. Check the final source hash, relevant
   allocator tests and CPU fast/fallback paths before model execution.
2. Run an external intervention on the accepted complete workload, with the
   same GPU/CPU placement, method order, warmups and preceding cache state.
   Retain all requests, full hidden comparisons, actual transfers, graph replay
   and lifecycle checks. Use an ordinary uninstrumented run for performance;
   separately record scan windows if attribution still requires it.
3. Select a runtime change only if complete-request measurements support it.
   A faster forced-full fixture does not establish a serving benefit or a tail
   guarantee. Loader/runtime provider receipts must distinguish acceleration
   from fallback.
4. If integrated, update actual source/build dependency coverage, including C
   source files in GR manifests. Check fixed and ordinary shared-resource callers
   to determine affected experiments. Keep published families until accepted
   replacements are available, following the repository's publication rules.

Owned A1024, resident/module, pattern and cold-union reports are not automatically
affected by a shared-serving guard change. Decide from actual call paths. API
compute inputs and boundaries must also be checked before deciding whether an
existing independent reference remains applicable. Do not infer an efficiency
pass by subtracting instrumented components from an uninstrumented request.
