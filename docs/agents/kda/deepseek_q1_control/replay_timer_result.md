# Ordinary CUDA events around one graph replay: result

The same graph retains a 96 ns median recorded gap with or without ordinary
CUDA timing events around replay. Plain formal calls have 16.577/16.864 us
layer-window idle; timed calls have 16.002/15.937 us. The relevant raw NSYS
settings match the current formal profile, including CUDA/NVTX capture with
no OSRuntime tracing. The formal 66.528 us idle / 416 ns median gap remains
unexplained by this control.

This run uses one current CUB production HBM model, prefix snapshot and
native full graph on GPU1/CPU8–15, with H=65,536/A=1/token111090, three
checkpoint layers and `return_hidden=False`. `FullExtendGraphCapture`
constructs the zero-internal-event graph under collection. Five shared
plain warmups run with collection stopped. Dynamic plain/timed callbacks
then use the same graph/cache/storage, adding zero/two ordinary event
records only around `extend_graph_replay_q_1`. Both restore and synchronize
the same prefix before each call. The clean benchmark measures synchronized
complete forward wall time; it never queries a CUDA elapsed-time metric.

The frozen harness SHA256 is
`e0b999a3f5a0be4df42f7845b7f1ebb52ec68945ac28a1abccdfb34ac3e5e2ec`.
Independent check `q1_replay_timer_check_20261008_01` passed bitwise eager
comparisons for both scopes on the original input and two changed tokens,
and retained the same native graph/cache throughout. Its immutable receipt
and outputs are under `/tmp/cxldsagr-checks/q1-replay-timer/` with that ID.

The fresh 50-pair AB/BA run `q1_replay_timer_bench_20261008_01` reports
plain/timed wall medians of 1.887489/1.901581 ms. Paired timed-minus-plain
median is +12.421 us, with AB/BA medians +15.123/+10.446 us; timed is faster
in 10/50 pairs. All samples are retained. The two event records add observed
wall overhead in this run. Explicit CUPTI loading remains part of both
arms' process conditions; no normal-production latency claim follows.

NSYS `q1_replay_timer_profile_20261008_01` uses
`--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node
--capture-range=cudaProfilerApi --capture-range-end=repeat`.
It saves construction separately from two matched warmups and four formal
AB/BA replays. Audit `q1_replay_timer_audit_20261008_01` reads relevant
`META_DATA_CAPTURE` fields directly and requires exact equality with formal:
only Cuda/NvtxEvents, zero sampling rate, node graph tracing, no CPU/GPU
context-switch trace, and matching CUDA collection settings. Environment
contents are excluded from the metadata report. Earlier controls also
enabled OSRuntime tracing; that difference is removed in this run.

| Phase / arm | L0–L2 span us | Busy us | Idle us | Median gap ns | Event records |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup / plain | 1009.406 | 993.470 | 15.936 | 96 | 0 |
| Warmup / timed | 1006.078 | 989.853 | 16.225 | 96 | 2 |
| AB / plain | 1013.630 | 997.053 | 16.577 | 96 | 0 |
| AB / timed | 1016.510 | 1000.508 | 16.002 | 96 | 2 |
| BA / timed | 1008.255 | 992.318 | 15.937 | 96 | 2 |
| BA / plain | 1011.615 | 994.751 | 16.864 | 96 | 0 |

Every scope has exactly one CUDA graph launch. Plain has no runtime
`cudaEventRecord`; timed has exactly two, with one completed before launch
and one starting after launch returns. All six scopes share executable
graph ID 38 and exactly the same native 197 GPU node IDs. The sole capture
ledger contains 181 kernels, 15 copies and one memset, no internal events,
and 192 layer-owned GPU nodes. Same-process clone lineage has 197 edges.
All 17 kernel launch fields, copy bytes and native owner sequences match
the current formal graph. The audit also checks the receipt, archived
source/request hashes, exact AB/BA order, and formal raw scope/launch rows.

Each layer window intersects 193 same-process GPU activities, including
the shared final norm crossing its tail by 288–416 ns. No other graph
activity intersects a window. Clipping all such activities yields exactly
the selected 192-node busy/idle union. Signature equality does not establish
complete DAG-edge equality or cross-run executable byte equality.

Under NSYS the two-pair wall medians are 2.185464/2.173841 ms. Their direction
differs from the independent 50-pair result, so they are retained only as
profiler-condition evidence. The data does not establish a GPU scheduling
mechanism or justify adding/removing production graph operations. Root owns
further isolation and any change to the formal measurement interpretation.
