# Zero versus two external event nodes: result

Removing the two graph-internal external event nodes did not reproduce the
current formal HBM trace's 66.528 us layer-window idle. Both arms remain near
16–20 us idle with 96 ns median gaps. The independent no-NSYS paired timings
also establish no replay-time gain from event insertion. This is a negative
control, not a production optimization.

Both arms use the current CUB production source on GPU1/CPU8–15 with
H=65,536, A=1, token 111090, three checkpoint layers, HBM cache and
`return_hidden=False`. Both use `FullExtendGraphCapture`, construct under
collection, receive five restored-prefix graph warmups outside collection,
then one in-trace replay warmup. The event arm adds two
`torch.cuda.Event(external=True)` records around the captured graph body.
Both arms are timed by ordinary CUDA events outside capture, around the
same `extend_graph_replay_q_1` scope. Wall time includes the synchronized
complete forward. The fixed NSYS CUPTI library is explicitly loaded in the
check and no-NSYS benchmark too; these controlled process timings are not
normal-model latency measurements.

Independent check `q1_event_boundary_check_20261008_01` passed bitwise
eager/graph and cross-arm prefix/logit comparisons, plus two changed-token
replays. Its immutable receipt and outputs are under
`/tmp/cxldsagr-checks/q1-event-boundary/` with that ID. The native templates
contain 181 kernels, 15 copies and one memset in each arm; only the event
arm has two type-7 event nodes. Each has 192 layer-owned GPU nodes.

The 50-pair AB/BA run `q1_event_boundary_bench_20261008_01` reports:

| Metric | No events | Two events | Paired events-minus-no-events median |
| --- | ---: | ---: | ---: |
| Outside-event replay median, ms | 1.593360 | 1.593008 | -0.000032 |
| Synchronized forward wall median, ms | 1.985183 | 1.978702 | -0.002020 |

Events are faster in 25/50 replay pairs and 27/50 wall pairs. Replay AB/BA
paired medians are -0.576/+0.320 us; wall AB/BA medians are
+5.463/-5.622 us. All outliers remain in the result, including a 2.383360 ms
no-event replay and whole-forward wall values up to 4.331631 ms. The paired
timings do not support a material event-insertion gain.

NSYS `q1_event_boundary_profile_20261008_01` contains three collection
intervals: no-event construction, event construction, then two warmups and
four formal AB/BA replays. Both constructions have one begin/end capture and
319 capture-info calls. The final interval has six graph launches and no
capture APIs.

| Order / arm | L0–L2 span us | Busy us | Idle us | Gaps | Median gap ns | Maximum gap ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AB / no events | 1018.559 | 998.975 | 19.584 | 141 | 96 | 736 |
| AB / two events | 1022.943 | 1006.462 | 16.481 | 141 | 96 | 832 |
| BA / two events | 1012.959 | 993.695 | 19.264 | 139 | 96 | 736 |
| BA / no events | 1011.006 | 991.038 | 19.968 | 141 | 96 | 800 |

Audit `q1_event_boundary_audit_20261008_02` binds every replay to its own
native capture/ownership ledger, same-process clone lineage (396 recorded
clone edges), and executable graph ID (38/77). All 197 GPU activity
signatures match across arms, including all 17 kernel launch fields and D2D
byte counts. The no-event signature also exactly matches the current formal
CUB graph, independently extracted in
`q1_event_formal_reference_20261008_01` from
`deepseek_h64k_a1_cub_20261008_01_h65536_a1_profile`.

Native ownership selects 192 L0–L2 activities. The shared final norm overlaps
288–320 ns into the layer-window tail. Each window intersects exactly 193
same-process GPU activities, with no activity from another graph launch;
clipping this complete set preserves the selected-node busy/idle union. GPU
membership and launch-metadata equality do not prove complete DAG-edge equality. Outside
event replay medians under NSYS are 1.781072/1.768608 ms and include profiler
effects; they do not replace the independent benchmark.

The updated analyzer verifies the accepted profile mode and run ID, receipt
bytes and identity, archived source/input hashes, formal-reference input
hashes, and exact AB/BA scope order. Independent CPU review
`/tmp/cxldsagr-checks/q1_event_boundary_independent_audit_20261008_01.json`
also rehashed 1,374 receipt artifacts, the 1,371 source snapshots in each
phase and 5,551 concrete runtime files; reread seven complete tensor
comparisons; and recomputed all 100 clean and four profile samples. It
matched the formal reference against its raw SQLite rows and independently
recomputed all-process interval unions. Changed-token evidence compares the
two graph arms; it does not include separate eager changed-token outputs.
The analysis-only replacement preserves every node and timing statistic
from the prior audit, as verified in
`/tmp/cxldsagr-checks/q1_event_boundary_analyzer_update_audit_20261008_02.json`.
Raw check, benchmark and profile evidence is unchanged.

The first-ever graph replay remains an unresolved lifecycle difference:
these controls run five graph warmups outside collection, while the formal
path's first graph replay is its in-trace warmup. Graph construction's eager
warmups do not exercise graph replay/upload. The next control will vary
collection during those first five graph replays, with zero internal event
nodes in both arms. No production source or published measurement was
changed by this control.

The NSYS command for this control also traced OSRuntime, including OS
backtraces, whereas the formal gap profile traced only CUDA/NVTX. Both arms
share this configuration, but direct comparison to the formal trace has
that additional boundary. The later one-graph replay-timer control uses
the formal settings and verifies them from raw capture metadata.
