# Collection during the first five graph replays: result

Collecting the first five full graph replays did not reproduce the formal
HBM trace's 66.528 us layer-window idle or 416 ns median gap. Even the
inside arm's first-ever collected replay has 20.608 us idle and a 96 ns
median gap. Every later replay in this control retains a 96 ns median gap.
This result bounds the tested warmup treatment; it does not establish the
cause of the formal trace difference.

Both arms use current CUB production HBM, H=65,536/A=1/token111090, three
checkpoint layers, `return_hidden=False`, GPU1/CPU8–15 and the fixed NSYS
CUPTI provider. Both construct with `FullExtendGraphCapture` under
collection, contain zero internal event nodes and receive five restored
prefix graph warmups. Only collection during those five warmups differs.
Each receives one additional warmup in the final collection interval.
The harness asserts zero full graph replays after construction, the expected
counter before each first warmup, and exactly five afterward. Source review
confirms preparation uses three eager direct-computation warmups and no
full graph replay. Both arms use ordinary CUDA events outside capture
around replay submission; complete synchronized forward wall is separate.

The frozen harness SHA256 is
`5308371ec070c0a74681b7be0f49fb53068f88b82e4f18ac877a08ec1ab659ec`.
Independent check `q1_replay_boundary_check_20261008_01` passed bitwise
eager/graph and cross-arm prefix/logit checks plus two changed-token replays.
The immutable receipt and evidence are under
`/tmp/cxldsagr-checks/q1-replay-boundary/` with that ID.

The fresh 50-pair AB/BA benchmark `q1_replay_boundary_bench_20261008_01`
retains all samples. Outside/inside replay medians are
1.592448/1.587440 ms; paired inside-minus-outside median is -4.448 us,
with 34/50 faster inside pairs and AB/BA medians -3.744/-5.824 us.
Wall medians are 1.971980/1.959866 ms, with paired median -10.359 us.
The explicit CUPTI provider is also loaded without an NSYS process, so
these are controlled process-condition comparisons. They are not normal
production latency measurements or a production optimization result.

NSYS `q1_replay_boundary_profile_20261008_01` has four intervals:
outside construction; inside construction; inside first-five replays;
final two matched warmups and four formal AB/BA replays. Both construction
intervals have one begin/end capture, 319 capture-info calls, 181 node-list
calls, 8,958 node-type calls, and zero graph launches. The replay intervals
have exactly five and six launches with no capture APIs. No collection-off
warmup appears in the trace.

| Phase / arm | L0–L2 span us | Busy us | Idle us | Median gap ns |
| --- | ---: | ---: | ---: | ---: |
| Inside first replay | 1022.846 | 1002.238 | 20.608 | 96 |
| Inside second replay | 1010.111 | 994.014 | 16.097 | 96 |
| Inside third replay | 1010.910 | 992.894 | 18.016 | 96 |
| Inside fourth replay | 1014.590 | 995.869 | 18.721 | 96 |
| Inside fifth replay | 1008.319 | 990.527 | 17.792 | 96 |
| Matched warmup / outside | 1023.838 | 1004.094 | 19.744 | 96 |
| Matched warmup / inside | 1017.662 | 998.014 | 19.648 | 96 |
| AB / outside | 1011.967 | 992.542 | 19.425 | 96 |
| AB / inside | 1011.646 | 991.838 | 19.808 | 96 |
| BA / inside | 1007.550 | 991.676 | 15.874 | 96 |
| BA / outside | 1016.286 | 997.598 | 18.688 | 96 |

Audit `q1_replay_boundary_audit_20261008_01` revalidates the receipt,
archived sources/request, accepted profile mode, exact interval labels and
AB/BA order. It rereads the current formal raw scope/launch/GPU rows and
checks their evidence hashes. All 11 replays contain exactly 197 GPU
activities: 181 kernels, 15 copies and one memset, with 192 layer-owned
activities. Both templates have zero internal event nodes. Same-process
clone lineage has 394 edges, with executable IDs 38/77. Kernel graph IDs
are checked directly; NSYS memory tables omit graphId, so memory nodes use
the same process, unique launch correlation and exact native-node membership.
All 17 kernel launch fields, copy bytes and full native owner sequences
match the current formal graph.

Every layer window intersects exactly 193 same-process GPU activities,
including the shared final norm crossing the tail by 288–576 ns. No activity
from another graph intersects a layer window. The clipped union of all
same-process GPU activities equals the selected 192-node union in every
case. Node/metadata equivalence does not prove complete graph-edge equality.
The analysis snapshot contains its exact helper source hashes; a later
event-analyzer hardening patch does not alter this archived analysis.

Under NSYS, outside/inside replay medians are 1.739104/1.733904 ms, with
opposite-sign AB/BA paired effects. These include profiler overhead and do
not replace the independent benchmark. Both arms also share a process, so
the inside arm's collection can affect process-wide CUPTI state. This
control therefore does not rule out every profiler lifecycle effect.

The next authorized control will use one graph and vary only ordinary
CUDA event records outside replay. Formal gap profiling records no such
timing events, whereas all controls so far do. No production source or
published result was changed by this control.

The NSYS command for this control also traced OSRuntime, including OS
backtraces, whereas the formal gap profile traced only CUDA/NVTX. Both arms
share this configuration, but direct comparison to the formal trace has
that additional boundary. The later one-graph replay-timer control uses
the formal settings and verifies them from raw capture metadata.
