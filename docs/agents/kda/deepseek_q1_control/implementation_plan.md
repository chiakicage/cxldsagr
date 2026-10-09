# Draft and executable plan

2026-10-08 baseline inspection found repeated `deep_gemm::transpose_fp32` after
activation quantization. Official `get_mn_major_tma_aligned_tensor` returns the
input unchanged when strides are `(1, get_tma_aligned_size(rows, 4))`.
Official `per_token_cast_to_fp8` has no output-layout option.

Ranked candidates: (1) write existing quantizer scales in official layout for Q1;
(2) reuse dynamic aranges/bounds in the graph only if the first candidate is
insufficient; (3) broader projection fusion requires a separate semantic audit.

Risks: Q1 has a 16-byte TMA stride although only one FP32 value is meaningful;
padding must never become a logical row. Public callers rely on contiguous
scales. Graph replay must write current data, not retain warmed scales. Native
identity must include the actual changed Triton specialization. No expanded
precision tolerance is allowed.

Execution:
1. Add explicit private quantizer output-layout choice and stride-aware stores.
2. Dispatch only dense Q1 linear to this layout; leave public defaults intact.
3. GPU correctness on GPU2/CPU16–23, compare default path + compiled official
   helper, then run existing quantizer/linear tests.
4. Save a fresh independent receipt and clean complete-API bench with source and
   native identities in the MFU experiment output. No checks inside timed loops.
5. Inspect profile for absence of the redundant transpose. Report evidence to
   parent, who owns full-model validation and publication.

Commands will use `.venv/bin/python -m pytest
operators/deepseek_v32/linear/tests -q` and
`python -m experiments.deepseek_v32_mfu.src.q1_control` with distinct check/bench
run IDs. Warm official helper before identity capture because its compiler
initialization may populate `TRITON_PTXAS_BLACKWELL_PATH`.
