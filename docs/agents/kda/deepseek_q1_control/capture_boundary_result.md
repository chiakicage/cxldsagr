# Profiler state during graph construction: result

Changing collection state during `prepare_extend_graph` did not reproduce
the previously observed 68.832 us L0–L2 idle. Both arms remain near 16 us
idle, with 96 ns median gaps. This negative control does not identify the
remaining cause; the next isolated variable is the graph inspector/capture
scope used by the formal measurement pipeline.

The task used current production before CUB top-k integration, GPU1/CPU8–15,
H=65,536, A=1, token 111090, HBM cache, three checkpoint layers and
`return_hidden=False`. Both independent model/graph instances contain the
same two external graph-body event nodes. Five graph warmups run with
collection stopped, followed by one replay warmup per arm inside the final
collection interval. The `during` arm brackets the complete
`prepare_extend_graph` call, including its three eager setup warmups, with
profiler start/stop; `before` constructs with collection stopped. Neither
uses `FullExtendGraphCapture` or operator instrumentation.

Independent check `q1_capture_boundary_check_20261008_01` passed bitwise
eager/graph comparisons in each arm, cross-arm prefix and logits comparisons,
and two changed-token graph replays. Its source/input/native-bound receipt
and outputs are in `/tmp/cxldsagr-checks/q1-capture-boundary/` under that ID.

Clean run `q1_capture_boundary_bench_20261008_01` retained 50 balanced AB/BA
pairs without NSYS. Full-graph event medians were 1.478208 ms (`before`) and
1.476672 ms (`during`); paired during-minus-before median was -1.984 us,
with 33/50 faster during samples. The AB and BA paired medians were -0.640 us
and -3.488 us. Synchronized-forward wall medians were 1.832532/1.829328 ms;
the order-specific wall medians have opposite signs, so the small wall
difference is not a stable optimization result.

NSYS `q1_capture_boundary_profile_20261008_01` records two collection
intervals. The first contains exactly one begin/end capture pair and the
`construct=during` NVTX scope. The second has six graph launches: two matched
warmups and four formal AB/BA replays. It contains no construction API.
Both raw reports and `capture_coverage.json` are retained separately from
clean timing.

| Order / arm | L0–L2 span us | Busy us | Idle us | Gaps | Median gap ns | Maximum gap ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AB / before | 1014.462 | 998.527 | 15.935 | 140 | 96 | 576 |
| AB / during | 1021.662 | 1005.148 | 16.514 | 146 | 96 | 736 |
| BA / during | 1018.878 | 1002.238 | 16.640 | 143 | 96 | 672 |
| BA / before | 1021.054 | 1004.317 | 16.737 | 144 | 96 | 544 |

Audit `q1_capture_boundary_audit_20261008_02` verifies one graph launch per
formal scope and 203 complete-replay GPU activities. The complete saved
198-node L0–L2 signature matches uniquely in every replay, including all
kernel names and 17 launch fields, or D2D byte counts. It contains 183 kernels
and 15 copies. Matching this signature establishes scope and launch identity,
not graph-edge equivalence.

A shared final-norm kernel crosses 288–448 ns into the window tail, so 199
raw activities intersect the time interval although 198 belong to the
selected layer signature. The audit retains the crossing node and clips all
intersecting activities to the same window. Their union exactly matches the
selected-node busy union: the crossing final norm fills no idle gap. The
table therefore does not drop GPU activity or sum overlapping durations.
The initial analyzer assertion excluding every boundary-crossing activity was
corrected; the failed analysis directory was deleted, with no GPU rerun or
change to the accepted execution.

NSYS full-graph event medians are 1.757024/1.756112 ms and include profiler
effects; they must not replace the clean 1.478208/1.476672 ms measurements.
Instance creation order remains before-then-during and was not reversed.
Results apply to this H200/SM90 workload and NSYS 2025.6.3 configuration.
Published reports were not modified by this control.

The NSYS command for this control also traced OSRuntime, including OS
backtraces, whereas the formal gap profile traced only CUDA/NVTX. Both arms
share this configuration, but direct comparison to the formal trace has
that additional boundary. The later one-graph replay-timer control uses
the formal settings and verifies them from raw capture metadata.
