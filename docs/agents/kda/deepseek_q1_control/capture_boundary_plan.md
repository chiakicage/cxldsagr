# Profiler state during graph construction: controlled diagnosis

This is a measurement-control experiment, not a GPU implementation change.
The formal current HBM trace has 198 L0–L2 activities and 68.832 us idle;
private traces with matching relative nodes and 17 launch metadata fields
have about 17 us idle. Formal graph construction occurs under
`cudaProfilerStart`; private construction occurred before it. This timing
variable must be isolated before attributing the idle difference to code.

Keep current production on GPU1/CPU8–15: three checkpoint layers, H=65,536,
A=1, token 111090, HBM cache, ordinary append, `return_hidden=False`, complete
extend graph. Use independent model/cache/graph instances for arms `before`
and `during`. No `FullExtendGraphCapture`, matrix instrumentation or graph
node inspection during execution is installed. Two identical external CUDA
event nodes time the graph body in both arms.

For `before`, construct the graph while profiler collection is stopped. For
`during`, bracket the same `prepare_extend_graph` call with profiler start
and stop. Both arms then receive five restored-prefix graph warmups with
collection stopped. The measured phase starts collection for both arms,
performs exactly one restored-prefix replay warmup per arm, then measures
the same calls. Prefix restoration and synchronization occur outside the
wall timer; elapsed CUDA event values are read after synchronized forward.
The graph's own three direct-computation setup warmups remain unchanged.

1. Freeze `q1_capture_boundary.py`, input bytes, current production source,
   checkpoint filesystem identity and runtime/native identity. Check eager
   and graph logits bitwise, both arms bitwise, changed tokens and restored
   prefix semantics. Save source, input/output evidence and immutable receipt
   in a system temporary check directory.
2. In a fresh process without NSYS, require the matching receipt, then run
   50 balanced AB/BA pairs. Save all graph-event and synchronized-forward
   wall samples. Runtime profiler API calls remain in both check and bench,
   but no profiler is attached.
3. Collect a separate NSYS run, tracing graph nodes. Use CUDA Profiler API
   capture-range control so construction is actually collected in only the
   `during` arm. Save all capture intervals; one final interval contains the
   matched replay warmup and formal AB/BA scopes. Do not attach NCU.
4. In SQLite, bind every formal scope to its unique graph launch and all
   GPU activities by correlation/process. Verify the same 198 L0–L2 activities
   against the existing independently inspected node template, checking all
   launch fields. Compute interval-union busy/idle and gap distributions;
   never count summed overlapping kernel times as GPU busy.
5. Report clean timing separately from NSYS timing. Any observed difference
   establishes a measurement effect only. Leave published reports unchanged
   until root decides whether the formal measurement pipeline needs correction.

Keep each check, benchmark and profile in distinct run IDs. Do not mutate
production or relax identity if another agent changes it; stop that run and
coordinate a new frozen source version. Keep cleanup exceptions together with
the original exception and retain normal model drain/close behavior.
