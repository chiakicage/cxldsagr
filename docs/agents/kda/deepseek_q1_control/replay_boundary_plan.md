# First graph replay collection boundary: draft and execution plan

## Question and hypothesis

The first three controls leave NSYS layer-window idle near 16–20 us despite
matching the current formal graph's 197 GPU signatures. The formal graph's
first-ever replay is collected; these controls first replay each graph five
times with collection stopped. Profiling the first replay/upload may affect
later graph-node tracing. This is a hypothesis, not an established runtime
or profiler mechanism.

## Fixed contract

Use current production HBM, H=65,536/A=1/token111090, three checkpoint layers,
`return_hidden=False`, GPU1/CPU8–15 and the existing signed request path.
Both independent model/cache/graph instances use the same fixed NSYS CUPTI
provider and `FullExtendGraphCapture`; both construct while collecting and
contain no graph-internal event nodes. Both have five restored-prefix graph
warmups, then one matched replay warmup in the final collection interval.
Both use ordinary CUDA timing events outside capture around
`extend_graph_replay_q_1`; synchronized wall time covers the complete forward.
The first warmup collection state is the sole planned treatment:

- `first5_outside`: all first five full graph replays run with collection off.
- `first5_inside`: all first five full graph replays run with collection on.

No full graph replay may occur before these assigned warmups. Assert the
full graph replay counter is zero after construction, matches each warmup
index before execution, and reaches five afterward. Construction uses only
the implementation's eager setup warmups. Preparation of smaller compute
graphs is common to both arms and is not a full-extend replay.

## Implementation and acceptance

1. Derive a fresh `q1_replay_boundary.py` harness from the completed event
   control without modifying prior sources. Remove internal events in both
   arms. Emit warmup NVTX labels and per-warmup outside-event/wall samples.
2. Independently check eager/graph and cross-arm outputs bitwise, prefix
   identity and changed-token replay. Both native templates must contain
   exactly 197 GPU nodes, 192 layer-owned GPU nodes and zero event nodes.
   Freeze the harness before checking and bind source/native/provider/input
   identity in an immutable correctness receipt.
3. In a fresh no-NSYS process, require that receipt and retain 50 balanced
   AB/BA pairs. CUPTI remains explicitly loaded; no subscriber is added by
   this harness. Preserve every sample and distinguish process-condition
   timings from normal production latency.
4. Run separate NSYS capture with four saved intervals: outside-arm
   construction; inside-arm construction; inside-arm first five replays;
   final two matched warmups plus four formal AB/BA replays. No implicit full
   replay belongs to either construction interval. Collection-off warmups
   must be absent from all trace intervals.
5. Audit construction/replay API coverage, native per-arm capture ownership,
   same-process clone lineage and exact executable graph IDs. Match all
   197 GPU nodes and all 17 kernel launch fields/copy bytes to each other
   and to the current formal reference. Analyze the five collected first
   replays, both matched warmups and formal AB/BA replays separately.
6. Compute each 192-node layer window using native owners. Clip all GPU
   activities intersecting that window and require the full union to equal
   the selected-node busy/idle union, preserving boundary-crossing final
   norm activities. Do not infer complete graph-edge equality or true
   hardware scheduling from NSYS timestamp gaps alone.

Use fresh check, benchmark, profile and analysis run IDs. Root owns any
formal measurement correction or publication. No production edits or
additional experiment follow this control without a new task.
