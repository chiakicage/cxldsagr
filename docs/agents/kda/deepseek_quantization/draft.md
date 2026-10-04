# DeepSeek quantization: design draft

The accepted official run attributes 72 CUDA kernels / 0.488799 ms to six
indexer quantization calls during offload extend1024. These are annotated
exclusive kernel sums, not complete API time. The current eager formula
materializes several FP32 intermediates around a D128 row reduction.

Candidate `q0`: one Triton program handles multiple independent D128 rows.
Each row loads the BF16 values once, computes FP32 abs-max, scales and rounds,
and stores contiguous FP8 data and FP32 scales. Fuse only arithmetic already
inside `quantize_index`; keep Q/K and downstream weighting unchanged.
Start with 16 rows per program and four warps, and use exact CUDA libdevice
math / rounded arithmetic where required by the reference. Layout checks
and fallback choices remain in the Python API; imports stay lazy.

The main correctness risks are scalar division lowering, libdevice versus
approximate log2 near exponent boundaries, signed zero/NaN behavior, and
FP8 cast ties. UE8M0 activation quantization uses an upstream bit-based ceil
instead of the indexer's log2/ceil formula, so the two baselines must be
validated independently. Never assume mathematical equivalence implies
identical stored scales.

The likely first-order gain is avoiding repeated launches and intermediate
traffic. Row grouping can subsequently trade CTA count against register
pressure, but tuning begins only after the initial candidate is correct.
The existing compiled DeepGEMM activation helper may already fuse to one
kernel; measure it before changing its implementation.

Sources inspected: current indexer formula in `echo_model.py`; current
compiled helper in `linear/fp8.py`; installed upstream
`deep_gemm/utils/math.py`; kernel-wiki as a Hopper implementation reference.
No external performance claim is used to admit the candidate.
