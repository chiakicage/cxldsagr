# Formal graph inspector on/off: result

The formal inspector scope did not reproduce the new formal HBM trace's
66.528 us layer-window idle. Both inspected and plain graphs, each containing
two external event nodes, remain at 17.952–20.224 us idle with 96 ns median
gaps. This control does not identify the cause of the remaining difference.
The next authorized control varies those two graph event nodes directly.

The current CUB production version is fixed across both arms on GPU1/CPU8–15,
with H=65,536, A=1, token 111090, three checkpoint layers, HBM cache and
`return_hidden=False`. Both graph constructions run under collection; the
`inspect` arm uses the existing `FullExtendGraphCapture` scope/finalize and
the plain arm only inserts the same two graph-body timing events. Both receive
five replay warmups with collection stopped and one replay warmup per arm
inside the final collection interval. Instance creation order is unchanged.

Independent check `q1_inspector_boundary_check_20261008_01` passed bitwise
eager/graph checks in both arms, cross-arm prefix/logits and two changed-token
replays. The receipt and outputs are under
`/tmp/cxldsagr-checks/q1-inspector-boundary/` with that ID. Both check and the
separate no-NSYS benchmark explicitly load the same fixed NSYS CUPTI provider,
without a subscriber added by this harness. The provider and its runtime ABI
match the formal inspector's path/hashes. These are controlled comparisons
within that explicit-provider process, not normal-model latency claims.

The 50-pair AB/BA run `q1_inspector_boundary_bench_20261008_01` measured full
graph medians of 1.584144/1.582576 ms for plain/inspect. The paired
inspect-minus-plain median is -2.480 us, with 30/50 faster inspected pairs;
AB and BA medians are -1.024/-6.496 us. Wall medians are
1.951014/1.951732 ms, with opposite-sign order effects. No large positive
inspector latency penalty is established by this run. Do not subtract these
numbers from the earlier production version/process's measurements.

NSYS `q1_inspector_boundary_profile_20261008_01` has three independently saved
collection intervals: plain construction, inspected construction, and two
warmups plus four formal AB/BA replays. The treatment executed: plain records
one `cudaGraphGetNodes`; inspect records 184 such calls, 9,050
`cudaGraphNodeGetType` calls and 319 capture-info calls versus plain's 227.
Each construction has one begin/end capture pair. The final interval has six
GraphLaunch calls and no capture APIs.

| Order / arm | L0–L2 span us | Busy us | Idle us | Gaps | Median gap ns | Maximum gap ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AB / plain | 1017.919 | 997.695 | 20.224 | 140 | 96 | 832 |
| AB / inspect | 1015.583 | 997.054 | 18.529 | 141 | 96 | 864 |
| BA / inspect | 1006.495 | 988.543 | 17.952 | 141 | 96 | 768 |
| BA / plain | 1020.895 | 1001.823 | 19.072 | 141 | 96 | 736 |

`q1_inspector_boundary_audit_20261008_01` resolves the inspected replay's
complete membership against native capture IDs, same-process clone lineage
(398 recorded clone edges) and executable graph ID. Each replay has 197 GPU
activities; the 192 layer-owned activities are assigned through the native
inspected ledger. Both arms have identical complete ordered GPU signatures,
all 17 kernel launch fields and D2D byte counts. The inspect template contains
181 kernels, 15 copies, one memset and two type-7 event nodes.

A shared final norm overlaps 288–448 ns of the layer-window tail. All
intersecting GPU activities are clipped to the window and their union is
verified equal to the selected-node union; the overlap fills no idle gap.
Full-graph NSYS event medians are 1.734144/1.727424 ms and include profiler
effects. No raw graph-edge ledger was added to the plain arm, so this result
does not prove equal DAG edges or attribute a difference to one inspector API.
Published reports and production source were not changed by this control.

The NSYS command for this control also traced OSRuntime, including OS
backtraces, whereas the formal gap profile traced only CUDA/NVTX. Both arms
share this configuration, but direct comparison to the formal trace has
that additional boundary. The later one-graph replay-timer control uses
the formal settings and verifies them from raw capture metadata.
