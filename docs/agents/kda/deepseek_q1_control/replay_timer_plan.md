# Ordinary replay timing events: draft and execution plan

## Question

The four completed controls still record two ordinary CUDA timing events
around every full graph replay. Formal gap profiling uses NVTX and records
no CUDA events at that boundary. Isolate this remaining instrumentation
difference on the same captured graph and model. No mechanism is assumed.

## Fixed contract and treatment

Current CUB production HBM, H=65,536/A=1/token111090, three checkpoint
layers, `return_hidden=False`, GPU1/CPU8–15 and the existing signed request.
Use exactly one model, prefix snapshot and `FullExtendGraphCapture` graph,
with zero internal event nodes. Construct under collection, assert zero
replays afterward, and perform five restored-prefix warmups with collection
stopped. Apply the same per-arm matched trace warmup count before balanced
formal AB/BA calls.

- `plain`: scope yields without recording CUDA events.
- `timed`: same scope records two ordinary CUDA events immediately around
  `extend_graph_replay_q_1`, outside capture.

The scope is chosen per replay of the same graph; graph/native/storage
identities must not change. Both arms restore and synchronize the same
prefix before each call. Time synchronized complete forward wall only in
the clean paired benchmark. Recording GPU timer events in the plain arm
would defeat this treatment, so no GPU-event metric is compared.

## Execution and acceptance

1. Derive and freeze a new `q1_replay_timer.py` without changing previous
   harnesses. Independent check compares both dynamic scopes with eager and
   each other, including changed-token inputs and prefix identity. Verify
   exactly 197 GPU nodes, 192 layer-owned nodes and zero internal events.
   Bind sources/native/provider/input/graph geometry in a fresh receipt.
2. Require that receipt in a fresh process for 50 balanced AB/BA pairs of
   complete synchronized wall timing. Preserve all samples. The fixed
   CUPTI provider is explicitly loaded in all phases.
3. Save two NSYS intervals: construction; two matched per-arm warmups and
   four formal AB/BA calls. Each treatment call has exactly one graph
   launch; assert zero/two cudaEventRecord API calls for plain/timed.
   All first five stopped warmups use plain scope, common to both arms.
   Use the exact formal CLI trace settings: `--trace=cuda,nvtx --sample=none
   --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi
   --capture-range-end=repeat`. Recheck relevant raw `META_DATA_CAPTURE`
   settings against formal, excluding environment contents. Prior controls
   used `cuda,nvtx,osrt`; this run intentionally removes that configuration
   difference while keeping both new arms identical.
4. Resolve the single native ownership ledger and same-process clone
   lineage. All replays must use the identical executable graph ID and
   native GPU node IDs; all 197 signatures/17 kernel fields/copy bytes and
   owner labels must match the current formal graph.
5. Compute each 192-node layer window, verify the clipped union of all
   same-process intersecting GPU activity, retain boundary-crossing final
   norm, and compare record gaps separately from clean wall timing.

Use fresh check, benchmark, profile and audit IDs. Root owns further
measurement corrections and publication. Do not modify production or
infer a hardware scheduling mechanism from the profile alone.
