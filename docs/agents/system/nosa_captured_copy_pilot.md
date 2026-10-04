# NOSA H64K captured-copy pilot

The external prototype is not promoted. Moving 62 inter-layer copy submissions
into existing projection graphs preserved exact outputs in the bounded pilot,
but did not consistently reduce ordinary complete-candidate median latency.
The first request improved by 0.158010 ms; the second changed by +0.002172 ms.
The differing tail behavior does not establish that the prototype removes pool
scans or improves the complete serving trajectory. Production sources and
published experiment results remain unchanged.

This addresses research item 4.1, with links to 2.1 and 4.2. It follows the
[prequeue diagnostic](nosa_hbm_prequeue_diagnostic.md), which established
submission sensitivity without isolating removable CPU cost. Further performance
work remains limited to H64K; H4K/H16K are paused.

## Implementation and measured boundary

`CapturedInputCopyGraphs` retains independent owned hidden/residual inputs.
Each successor projection graph copies the preceding finish graph's outputs
before normalization. This preserves the predecessor tensors when fused norm
overwrites its arguments. Setup primes predecessor finish outputs before
successor warmup/capture. Ordered layer execution, cache validation ownership,
query/position/predecessor identity, stream and failure guards constrain replay.
Eager attention, cache work, transfers, finite decisions and lease drains are
unchanged.

Each 32-layer model forward still uses 32 projection and 32 finish replays.
The prototype moves 62 hidden/residual copies into capture and retains 65 eager
input copies, versus 127 in the original class. The GPU copy work remains.
Both arms charge 896,828,416 B static allocation and 6,106,906,624 B private
graph reservation for Q=128/1024; 128 captured graphs cover both shapes. These
are graph allocation categories, not a whole-process peak or physical HBM cap.

Four independent processes used GPU5, CPU48–55, NUMA1 and the accepted checkpoint,
precision, native builds and 118-file runtime snapshot. Each rebuilt a full
H65536 history from an empty cache with C1024, then ran the same A128 candidate
three warmups and seven measured times. Process order was baseline0,
prototype0, prototype16, baseline16. The driver timed the original synchronous
`extend_candidate` call, including public-output cloning, finite decision,
transaction/discard, lease drain and exit allocator guards. No delay, profiler,
event/check wrapper, retry, GC suppression or sample deletion was used.

The driver saved all 40 warmup/measured candidate hidden tensors before numerical
validation. The four construction-request outputs were checked but not saved.
History/pointer checks, output saving and metadata writes were outside timing.
Both arms held 2,223,619,840 B of additional exact-history GPU clones. This
repetition/recording process differs from the formal 16-user trajectory and can
change GC phase, memory/cache conditions and scan placement. Recorded allocation
and provider-counter brackets surround the call but do not time individual scans.

## Results

All values below are milliseconds. All seven measured samples enter each summary.

| Request | Arm | Mean | Median | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Original | 33.839085 | 15.416903 | 15.250091 | 123.045208 |
| 0 | Captured copies | 15.270372 | 15.258893 | 15.114703 | 15.399645 |
| 16 | Original | 33.205173 | 15.330098 | 15.260413 | 118.773616 |
| 16 | Captured copies | 15.336977 | 15.332270 | 15.199263 | 15.466505 |

The request0 median ratio is 0.989751. Request16 medians are effectively equal.
The original arm has measured_00/measured_02 tails of 37.079517/123.045208 ms
for request0 and 37.023268/118.773616 ms for request16. Each tail brackets one
filtered pool call. Neither prototype has a filtered call in its measured
brackets. These counts do not identify a scan's duration, explain the entire
tail, or prove permanent avoidance. Means therefore cannot establish the
prototype's net effect on the formal workload.

The four target processes completed all exact-output, history/state,
replay/copy-count, zero-HBM-transfer, source/native/checkpoint and CPU-environment
checks. All 40 saved outputs passed independent bit-exact comparison and each
118-file saved source snapshot was independently verified. All 40 warmup/measured
device alloc/free/retry/OOM deltas were zero; successful
close also passed the inherited graph-pool release check. Independent review
reopened all saved candidate hidden payloads and saved source files. Construction
hidden equality and history byte/pointer equality remain target-run assertions
because those payloads were not saved.
The prototype's 18 CPU contract tests cover control flow, not actual CUDA
ordering or every failure/rollback path. This pilot does not replace all-four
fixed-scheme checkpoint and lifecycle acceptance.

## Evidence and next step

Raw inputs, frozen external code, CPU receipts, commands, launch receipts and all
40 output tensors are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_captured_input_copy_prototype_20261004_01/`.
Run IDs are `nosa_copy_{baseline,prototype}_r{0,16}_20261004_01`.
Independent source/driver and result reviews are under sibling
`nosa_captured_copy_pilot_review_20261004_01/`.

| Identity | SHA256 |
| --- | --- |
| Unchanged production runtime | `93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79` |
| External prototype | `73687b1e64852f65567576287d1314cf7ba8c1f258c51273ac9aaeec60903639` |
| Driver | `a2a27de5d15ed163a7f2a59768509d104055f8d442f00c725a28d39b59aa7bdd` |
| Commands | `a2a29ae3f2bba5451427e6387b961ee0ef540d25653c085e4d30e23366e6bdd5` |
| Static release review | `71cf7e5fcf1113f016133c135137f86d6033b35c18a6e5eab0b34d38189075cf` |
| Independent analyzer | `36840a5c18005220b8138f9b93161ed615b4758f0e0227f8c6946b6c5b9425a8` |
| Consolidated independent review | `bd0119032757e4a4f184f777703e7d7bb348f368034d9554c9b79388e949b8c3` |

The next bounded comparison should hold the prototype's guards and setup
constant while changing eager versus captured predecessor copies. Keep an
unmodified baseline for any claim of net benefit. This can test whether added
guards offset submission savings; the current results do not quantify either
cost. Further copy/position changes remain candidates pending that evidence.

That [matched eager/captured control](nosa_guarded_copy_control.md) is now
complete. It found small, variable net median differences and is not promoted;
its six new runs and identities are separate from this first pilot.

A read-only impact/acceptance map is in sibling
`nosa_captured_copy_acceptance_map_20261005_01/production_impact_acceptance.md`
(SHA256 `468ad47339031058f1c4aca511ae369c86de8dd40733f489f20eaba753f4fd53`).
All four fixed schemes share this graph class. Ordinary GR formal/profile
entrypoints keep it disabled; no low-history GR rerun follows from this pilot.
Future integration must handle dynamic copy counters consistently in immutable
plan metadata and pass actual CUDA chain/drain, late-capture cleanup, short-prefix
rollback and full32 all-four-scheme checks before replacing affected fixed
formal/profile/API measurements. No such promotion or replacement is scheduled
from the current result.
