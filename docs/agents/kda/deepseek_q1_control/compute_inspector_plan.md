# Initial compute-graph inspection control

The complete unchanged formal pipeline reproduces the long HBM gap:
67.680 us idle versus the earlier 66.528 us, with identical 197 GPU-node
signatures, native ownership and 416 ns median gaps. The five smaller
controls do not. Keep the complete pipeline and change one preparation step.

Replace only `profile_layers.prepare_graphs` with a wrapper that opens the
same profiler interval, then calls the original helper in `check` mode.
This allocates the same compute graphs without `CaptureGraphOperators` and
its `InstrumentOperators` package. Preserve every all-four-method warmup,
cache switch, prefix, full-extend construction and measured replay. Keep
the later `FullExtendGraphCapture` inspection unchanged. This controls the
whole initial inspection package; it does not isolate an individual API.

Use GPU0/CPU0-7 and the exact reproduction's process/compiler environment,
including the literal three-prefix PATH. The original independent check
already uses this compute preparation path. Reuse its receipt only after
the unchanged numerical execution/native identity matches. Archive the
diagnostic wrapper as an additional measurement source and label its
changed preparation boundary explicitly. No production source is edited.

Run ID: `q1_compute_inspector_profile_20261008_01`. NSYS remains
`--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node`
with CUDA-profiler ranges and repeat. Retain all 13 ranges; the first must
contain 12 begin/end captures and 6 graph launches while compute-graph
node-inspection calls disappear. Compare the original HBM warmup and
formal windows by full native node membership and all-process clipped
interval union. Recheck all four saved outputs, source/native identity and
raw relevant NSYS settings.

No compute-operator ledger is fabricated. Without that ledger the generic
prefill attribution and gap gate cannot be asserted; do not call the
ordinary `launch_gap` report or publish this as the formal result matrix.
This is an invasive diagnostic, with no clean-latency or optimization claim.
