# DeepSeek quantization: executable plan

1. Freeze source hashes for the current indexer formula, activation adapter
   and installed DeepGEMM helper. Extract actual activation widths from
   `official_01/operator_calls.json`. Record hardware/dependency versions.
2. Implement `q0` fused D128 indexer quantization with lazy Triton import and
   an independent eager reference. Preserve output layout/dtypes and both
   scale formats; use reference fallbacks outside the validated fast path.
3. Run CPU contract/import checks and SM90 exact GPU comparisons. Cover
   random values, zeros/tiny values, exponent/FP8 ties, signed zero,
   extreme BF16 values, supported tensor ranks, strides and empty rows.
   Compare bytes/scales directly; fix failures before timing.
4. Benchmark current complete eager API against the fused complete API on
   Q/K workloads. Exclude compilation, preallocate only benchmark inputs,
   include output allocation and launches, alternate implementations, and
   report event and synchronized wall timings. Use temporary run IDs.
5. Assess current compiled official DeepGEMM activation quantization on
   measured widths. If an actual bottleneck remains, coordinate any linear
   adapter modification with parent and open a separate candidate.
6. Read ncu-report-skill before collecting NCU. Inspect launch count,
   occupancy/registers and memory throughput for the admitted candidate;
   separate profiler duration from unprofiled timing.
7. Record candidate/checkpoint evidence, run relevant regression checks,
   and hand exact integration instructions to parent. Parent performs
   isolated complete-model correctness/performance reruns and publication.

Status: q2 accepted for indexer integration after 30 exact tests, paired
complete-API/graph comparisons against eager and compiled official D128,
and NCU full/source profiling. No activation adapter change is included.
See `checkpoint.md` and `investigation_log.md`; whole-model rerun is parent-owned.
