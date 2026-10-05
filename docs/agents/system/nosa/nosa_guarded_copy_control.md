# NOSA H64K matched guarded-copy control

The shared eager/captured control does not justify production promotion.
Captured copies have lower candidate medians under the matched stronger
contract in both requests, but the differences vary from 0.257480 to 0.038759 ms.
Against the original class, the median reductions are only 0.076418 and
0.015620 ms. These short sequential comparisons do not establish a stable net
serving benefit or isolate guard CPU cost. The subsequent
[local descriptor-reuse diagnostic](nosa_copy_descriptor_result.md) completed
both requests with exact outputs and approximately 0.1 ms lower medians against
the matched control. It remains external; no full-serving benefit is established.

This follows the first captured-copy pilot（Git `934485b:docs/agents/system/nosa_captured_copy_pilot.md`） and
addresses research item 4.1, with links to 2.1 and 4.2. H4K/H16K performance work
remains paused. Production runtime sources and published experiment reports
remain unchanged.

The six original formal numerical-reference files needed by the frozen reviewers
are retained separately with their original bytes and hashes. See the
[reference relocation and reopening instructions](../../acceptance/unified_runtime_20261005/t007_reference.md).
Historical commands below remain unchanged; reopening uses only an in-memory
`FORMAL` path override. The relocation does not rerun these diagnostics.

## Controlled change and execution

The external `GuardedCopyGraphs` shares sequence, cache-owner, input/predecessor
identity and geometry, mode, stream, reentry/poison, failure and close logic
between E and C. Both retain independent owned inputs, predecessor references
and deterministic setup finish-output priming. An immutable mode selects only
whether the 62 predecessor copies execute eagerly or inside the projection
graphs. Position and attended-result copies remain eager. The original class B
retains its original contract and setup.

| Arm | Input copies per 32-layer forward | Projection / finish replays |
| --- | --- | --- |
| B: original | 127 eager, 0 captured | 32 / 32 |
| E: guarded eager | 127 eager, 0 captured | 32 / 32 |
| C: captured | 65 eager, 62 captured | 32 / 32 |

Six independent H65536/A128/C1024 processes ran B/E/C for request0, then C/E/B
for request16. Each built history from an empty cache and used three warmups
and seven measured candidate calls. The driver retained the first pilot's
ordinary complete `extend_candidate` boundary, including finite decision,
public-output clone, transaction/discard, lease drain and exit allocator guard.
It also saved each construction-request output. CPU-only independent analysis
ran on CPU0–7 with CUDA hidden and one intra-op thread, separate from the target
GPU5/CPU48–55/NUMA1 allocation. No profiler, delay, retry, GC suppression or
threshold change was introduced.

The same diagnostic history clones, repeated-candidate trajectory and outside
timing recording limits apply as in the first pilot. E/C compares copy placement
under a shared stronger contract; E/B also changes setup/reference/collector
conditions and cannot isolate guard CPU time. Planned reverse order and retained
samples do not turn these runs into randomized or independent paired samples.

## Results

All values are milliseconds; all seven measured samples enter each summary.

| Request | Arm | Mean | Median | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | B | 19.468172 | 15.318158 | 15.155731 | 37.053567 |
| 0 | E | 15.498359 | 15.499220 | 15.336201 | 15.666337 |
| 0 | C | 15.226438 | 15.241740 | 15.108320 | 15.354731 |
| 16 | B | 19.611734 | 15.433952 | 15.292054 | 37.360318 |
| 16 | E | 15.486476 | 15.457091 | 15.382961 | 15.616229 |
| 16 | C | 15.416691 | 15.418332 | 15.272892 | 15.529745 |

Median differences for request0/request16 are C−E = −0.257480/−0.038759 ms,
E−B = +0.181062/+0.023139 ms, and C−B = −0.076418/−0.015620 ms. These are
differences of descriptive medians, not isolated recoverable CPU costs.

B has measured_00/measured_02 tails of 37.053567/22.903019 ms for request0 and
37.360318/23.095758 ms for request16. Each brackets one filtered pool call.
E/C have no filtered call in their candidate sample loops. The absence also
occurs with eager copies and cannot be assigned specifically to captured
placement. This short-loop difference cannot prove durable scan avoidance or explain scan duration; all
tails remain in the means. The much larger baseline tails in the first pilot
also must not be substituted into this matched comparison.

All six target processes passed exact-output, history/state, replay/copy/mode,
zero-HBM-transfer, allocation-counter, source/native/checkpoint and CPU checks.
Independent review reopened all 60 saved warmup/measured outputs and six saved
construction outputs with bit-exact agreement, plus each 118-file source snapshot.
All 60 sample device allocation-counter deltas were zero. Launch timestamps also
verified the planned six-process order without overlap. History byte/pointer
equality remains a target assertion. All arms retain the same static/private
graph accounting, and successful closes pass inherited graph-pool release.
The external class's 49 CPU cases test control flow; neither these nor the HBM
pilot replace all-four-scheme CUDA lifecycle/checkpoint acceptance.

## Evidence and next bounded candidate

Raw code, exact v1 diff, CPU receipts, plan, commands, launch receipts and saved
outputs are in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_guarded_copy_control_20261005_01/`.
Run IDs are `nosa_control_{baseline,guarded_eager,captured}_r{0,16}_20261005_01`.
Independent reviews are in sibling `nosa_guarded_copy_control_review_20261005_01/`.

| Identity | SHA256 |
| --- | --- |
| Unchanged production runtime | `93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79` |
| Shared external class | `43b5e6022c5d5755ce0ed236af33dc9bdab1c58be0dcefd1a0bd4e23bfc42bf1` |
| Driver | `ecb925e9cd1442e3cf4d1a6b31dac7ee218992ec70e5f7928c77d09bdf8d2b88` |
| Commands | `24685c661126bba59291160d4de3912a2a693b614902fb35082ad07abbf16bd9` |
| Source review | `583639a2c1806a40c761a8001cb65b93673f74e970edb226097e94bc1d227189` |
| Static release review | `6c65838b5b633435c9fb40dfe201fbc6743304c6843551fa053443f01b5707b9` |
| Independent analyzer | `462c39051ac49ed89714254cb80e77e7e72b06e55d02ce82fd299eda8f62e4f2` |
| Consolidated independent review | `ab27da91d1348d4b87705368e779f32eb9e161f39dbdec3b09f237b0d6be1403` |

The descriptor-reuse candidate is external preparation only. It may reuse
metadata within one `_validate_inputs` invocation, but must still check object,
data pointer, shape, stride, dtype/device, cache/stream and predecessor
identity at their existing points. It must not cache across eager attention,
skip post-attention checks, alter position ownership or merge this change with
kernel/copy-policy changes. Source counts are not timing estimates. Any measured
benefit requires another correctly scoped comparison; no result above applies
to that candidate or to an unrun production implementation.
