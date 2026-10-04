# C5 draft and evidence

C3 HBM cold history CSV at
experiments/deepseek_v32_motivation/output/data/motivation_c3_profile_20261004_01/analysis/independent_nonmatrix.csv
attributes185.431568 ms/1920 launches to official FlashInfer selection + index
finalization + stable value sort, and22.862042 ms/5120 launches to wrapper work.
The wrapper adds eight launches per call: int32->int64, four isfinite operations,
scalar fill, where, and int64->int32. These are profiled GPU times, not predicted
formal E2E deltas; the launch gaps may add host cost.

Installed flashinfer.topk.get_topk_module().radix_topk already allocates/returns
int32 and accepts sorted_output, deterministic, tie_break, row_states_buffer,
output_values, dsa_graph_safe. The public top_k wrapper immediately widens its
result to int64. Calling the same official registered operation directly retains
all selection and ordering kernels while avoiding the round trip. Reuse the
same named1MiB official row-states cache buffer initially, even though SMALL
currently forces FilteredTopK; do not combine allocation-policy changes.

The plain API has no fused finite-value sentinel option. Ragged/page-table APIs
support lengths and int32 output but omit sorted values and return sequential
IDs for length<=K; they are unsuitable as replacements here. dsa_graph_safe=True
forces VEC_SIZE1; SMALL already forces FilteredTopK. Do not assume toggling that
flag accelerates or fixes capture without measuring the exact installed path.

Candidate1: direct official int32 wrapper plus existing torch finite/where
operations. Candidate2: same wrapper plus one in-place kernel which reads FP32
value bits and writes only nonfinite positions of the int32 index tensor to-1.
No index read or value write is needed. The bit predicate is
(value_bits &0x7fffffff)>=0x7f800000, preserving signed-zero and NaN/inf masking.
A lazy Triton helper fits the existing standalone quantization pattern.

Risk: preserving exactly the public wrapper's sorted/deterministic/SMALL flags;
private installed API compatibility; same official scratch lifetime; value-bit
preservation; graph capture of the registered op and mask; no silent removal of
needed contiguous copies. Frozen uv/installed backend identity must be recorded.
The official185ms sort/selection cost remains after this scoped optimization.
