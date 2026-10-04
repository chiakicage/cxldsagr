# H64K HBM candidate prequeue diagnostic

Both independent-process diagnostics completed and passed CPU review. Supplying
the unchanged pre-check candidate DAG before device execution reduced its CUDA
event interval in all 14 measured pairs. This establishes sensitivity to prior
submission under this intervention. It does not isolate removable CPU launch
cost, measure an ordinary serving speedup, or establish compute/IO dominance.

## Inputs and timing boundary

The full 32-layer checkpoint uses H65536/A128 and the exact saved HBM requests 0 and 16
from `nosa_motivation_poolscan_sm90_20261004_01`. Each process independently
constructs an empty-cache history, then repeatedly executes its candidate
against that retained history. It does not replay the complete four-method,
16-user process trajectory. The unchanged 118-file runtime identity is
`93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79`.

Four arms each have three warmups and seven measured samples in fixed,
counterbalanced orders: ordinary, event-only, delayed control and prequeue.
Request 16 reverses the corresponding orders. Both delayed arms use the same
calibrated GPU delay within a process. Control waits for that delay before
submitting the candidate; prequeue submits while it is pending. Every prequeue
sample proves its start event is still incomplete at the validation boundary.

The event interval includes embedding, all 32 layers, final normalization and
the original public-output clone. It ends on entry to `DeferredValidation.check`.
Finite reduction/host decision, transaction completion, execution-lease drain
and exit allocator checks still run unchanged, but are outside this event
interval. Prequeue wall time includes the pending delay and is not compared
as a serving speedup. No profiler, GC setting change or production modification
was used.

## Results

Times are milliseconds. Medians describe repeated identical requests, not
independent workload samples.

| Request | Ordinary wall | Event-only wall | Control event | Prequeue event | Paired event delta median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 15.385808 | 15.436919 | 14.721472 | 12.917824 | -1.785600 |
| 16 | 15.420108 | 15.349558 | 14.712832 | 12.905472 | -1.811520 |

All seven pairs per request have shorter prequeue intervals. Request 0 deltas
range from -2.213728 to -1.709568 ms; request 16 from -1.966624 to -1.732480 ms.
No samples were removed or replaced. Event-only/ordinary wall ratios are
1.003322 and 0.995425; prequeue/control delay-duration ratios are 0.999940 and
1.000070. Both predeclared factor-1.05 checks compare medians and pass for each
process. They do not assert small overhead in every sample. Delay medians are
approximately 49.23 ms and 48.15 ms respectively.

The independent reviews reopened all 80 saved BF16 hidden tensors and found
them bit-exact against the current formal HBM outputs. They reopened all 118
saved source files and checked ordering, calibration, graph replay counts,
allocation counters, zero HBM host transfers and recorded native identities.
The target separately checked the current source files before and after execution.
All 56 measured samples have zero device allocation/free/retry/OOM counter
changes. The sample loops each recorded three filtered pool-referrer calls and
zero other routes/errors. Each process has two calls in warmups and one in
measured repeat 5. Per-candidate counters identify those brackets, not the scan
intervals or durations.

The two measured brackets retain substantial wall tails. Request 0's repeat-5
prequeue wall is 84.686882 ms, with 71.915708 ms after check entry including
the remaining delay/device wait; adjacent same-arm post-check spans are
50.648281/50.700349 ms. Request 16's repeat-5 event-only wall is 37.603702 ms,
with a 24.475655 ms post-check span versus adjacent 2.931052/3.078731 ms.
Its pre-check event interval is 14.948768 ms. These tails are largely after
check entry and coincide with the counted candidate brackets, but do not
separately time or causally attribute the scan. The prequeue arm still contains
a wall-tail sample.

The target checks history and derived-record bytes, pointers and retained state
after every sampled candidate call. Those checks passed, but history payloads were not separately
saved for independent reopening. The 2,223,619,840 bytes of diagnostic GPU
history clones and their comparisons are outside timing and shared by all
arms. Their presence changes allocation/cache conditions from formal serving.
Outside-window clock observations do not certify constant clocks within a call.

## Evidence and next action

Driver, plan, complete metadata, logs and output tensors are at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_hbm_prequeue_diagnostic_20261004_01/`.
Run IDs are `nosa_hbm_prequeue_r0_20261004_01` and
`nosa_hbm_prequeue_r16_20261004_01`. The frozen driver SHA256 is
`9426cc931c9c8d5e4ec4a55d350a519c4cc22b38a234084b776b21ee236357dd`.
The CPU analyzer and independent reviews are at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_hbm_prequeue_review_20261004_01/`.

| Evidence | SHA256 |
| --- | --- |
| Request 0 metadata | `ca3f2bbc9edbcda84ad829a2ec4efe2e9219b2cb4efa0ead6d5c5a501c7a65e4` |
| Request 16 metadata | `3870da947e97a475678f52fbadad08c0102057662685a9e7e6d3e1594e51823e` |
| Request 0 independent review | `af56f084a4582e4ef295f2e6d8980394d6fbaa34f9b0dfeb2881d21f7d9929a9` |
| Request 16 independent review | `c2d1bbc3580ac215593ed7998bec2730dc22aa1a78a29abe7edb501fd6116576` |
| Frozen analyzer | `ecb7de9ccab109f1093ba64922628fd4ec16e08037ea533c01277d9726ddd420` |

The result supports testing a concrete reduction in eager graph-input copies
and submissions. A prototype must preserve in-place residual semantics,
graph-buffer ownership, accounting, finite validation and transaction lifetime.
It needs paired ordinary complete-candidate timing and exact checkpoint checks
before production integration. The current formal reports remain valid for the
unchanged implementation; no new implementation candidate has been measured.
Further measurements stay within H64K, as requested.
