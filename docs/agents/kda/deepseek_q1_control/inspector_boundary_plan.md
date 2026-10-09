# Formal graph inspector on/off control

The collection-state control did not reproduce the formal trace's large
idle. Start a separate source-bound experiment after root's CUB production
integration; do not edit the completed `q1_capture_boundary.py` or its runs.

Use `q1_inspector_boundary.py` on GPU1/CPU8–15, current HBM production,
H=65,536/A=1/token111090, three layers and `return_hidden=False`. Both arms
construct with collection active, then perform five restored-prefix graph
warmups with collection stopped. The final collection interval contains one
warmup per arm and balanced AB/BA formal replays. Two external graph-body
timing events remain identical in both arms. No matrix operator instrumentation
is installed. Independent model, cache and graph instances remain in the
same creation order, plain then inspect.

The sole treatment is the actual formal `FullExtendGraphCapture` scope:
`plain` uses the timing context only; `inspect` wraps the same context in
the existing scope and calls its existing `finalize` after graph allocation.
This includes repeated CUDA capture-info/node enumeration, CUPTI node IDs,
type reads and stage-ownership bookkeeping. It is not a claim about one API
alone. Labels are metadata; the treatment must not alter outputs or workload.

`GraphInspector` requires the NSYS CUPTI provider. The independent check and
clean benchmark explicitly load the same existing
`/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/libcupti.so.13.2` in both
arms, without adding a CUPTI subscriber. Under NSYS this is the already-loaded
provider. Bind its path/hash plus observed inspector provenance, all model
sources, native artifacts, input bytes and checkpoint identity. Provider
failure propagates; there is no alternative backend.

1. Run fresh independent check: eager/graph bitwise in each arm, cross-arm
   prefix/logits bitwise and two changed-token replays. Save complete evidence
   and immutable receipt in system temporary storage.
2. Require the receipt in a fresh process without NSYS; retain 50 balanced
   AB/BA pairs of full-graph CUDA events and synchronized model wall time.
3. Run NSYS with CUDA-Profiler-API capture ranges and repeat mode. Three
   intervals contain plain construction, inspected construction and matched
   replay warmups/formal AB/BA. Verify expected construction APIs and read
   the actual formal inspector node ledger.
4. Match complete observed GPU nodes across arms using the inspected native
   ownership ledger, graph lineage and each replay's unique launch. CUB
   removes six GPU nodes from the previous full graph, so do not blindly
   reuse the old 198-node layer template. Compare native kernel names/launch
   metadata and D2D bytes, then compute clipped union busy/idle. Preserve any
   shared final-norm overlap with the layer-window tail.
5. The default PyTorch graph releases its raw graph after instantiation.
   Reading edges would require extra capture-time observation or changing
   graph retention. Do neither in the plain arm in this first control;
   report that node/launch equality is not an edge-equivalence proof.
6. Report the measured condition only, without attributing a result to a
   specific inspector API or generalizing to other tools/hardware. Root owns
   any measurement-pipeline correction and formal report replacement.
