# Reduced HBM replay with the formal environment: result

The short gaps persist after matching GPU0/CPU0–7, the literal formal PATH,
default cache selection, precision and participating recorded runtime artifacts.
The reduced plain calls have 17.889/18.146 us layer-window idle and 96 ns median
gaps; the current FREE formal replay has 66.848 us idle and 416 ns median gaps.
The earlier GPU placement/cache/binary differences do not explain this remaining
discrepancy within the recorded identity coverage. The cause and any production
optimization remain unresolved.

| Phase / arm | L0–L2 span us | Busy us | Idle us | Median gap ns |
| --- | ---: | ---: | ---: | ---: |
| Warmup / plain | 1017.602 | 997.986 | 19.616 | 96 |
| Warmup / timed | 1011.490 | 993.538 | 17.952 | 96 |
| AB / plain | 1013.410 | 995.521 | 17.889 | 96 |
| AB / timed | 1016.226 | 998.978 | 17.248 | 96 |
| BA / timed | 1007.267 | 989.858 | 17.409 | 96 |
| BA / plain | 1016.547 | 998.401 | 18.146 | 96 |
| FREE formal / measured | 1056.580 | 989.732 | 66.848 | 416 |

The unchanged reduced timer uses one HBM model/cache/full graph, real L0–L2,
H65536/A1/token111090 and default last-token logits. Construction is collected;
five plain warmups run with collection stopped. Every invocation restores and
synchronizes the same prefix before timing. Plain/timed callbacks add zero/two
ordinary CUDA event records outside capture. The explicit CUPTI provider and
formal CUDA/NVTX/node/no-sampling profile settings are preserved.

The check `q1_replay_matched_check_20261008_01` passed both scopes against eager
for tokens 111090, 111091 and 111092. Independent CPU rereading confirms six
bitwise output comparisons and the receipt/source/runtime archives; its record
is `/tmp/cxldsagr-checks/q1_replay_matched_check_20261008_01_independent.json`.
The wrapper's independent CPU review also passed 34 acceptance/rejection cases.

Clean `q1_replay_matched_bench_20261008_01` retains all 50 balanced AB/BA pairs.
Plain/timed synchronized complete-forward wall medians are 1.896749/1.910402 ms;
the paired timed-minus-plain median is +15.110 us, with AB/BA medians
+15.390/+14.830 us. Timed wins 7/50 pairs. This measures the extra event records
in the reduced process, not a production optimization or the profile gap.

Profile `q1_replay_matched_profile_20261008_01` and analysis
`q1_replay_matched_audit_20261008_01` verify exactly one graph launch in each
of the six scopes. All share the same executable and 197 native GPU nodes,
including 192 layer-owned nodes; names, 17 launch fields, copy sizes and owners
match FREE formal. The capture contains 181 kernels, 15 copies and one memset,
with no internal event/empty nodes. The 197 clone-lineage mappings are not a
dependency-edge DAG. All-process SQLite rereading finds 193 intersecting rows
per window on device 0 and no foreign process or graph activity. One final norm
starts 320–480 ns before the layer-window ends and finishes after it. Clipping
that norm preserves the 192-node busy/idle union.

All recorded native-JIT, CuTe, linear-quantization, attention-combine and page64
entries match formal exactly. Five unused offload libraries and one decode-hint
specialization remain unobserved rather than being loaded for comparison.
DeepGEMM's per-launch JIT binary identity is not exposed by the formal collector;
CuTe MLIR is identified by in-memory hashes without retained files. The match
does not prove equality outside these recorded boundaries or across lifecycle
histories. The next control adds the formal multi-method preparation history;
its result is pending.

The accepted analysis and its source/reference files remain unchanged. Copies
of the independent wrapper review, saved-output audit and separate integer
interval-union audit are indexed in
`experiments/deepseek_v32_mfu/output/data/q1_replay_matched_audit_20261008_01/independent_review/index.json`.
No experiment README or published performance result is replaced by this control.
