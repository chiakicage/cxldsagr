# Consumer-layout harness revision 02

The numerical candidate remains `consumer_layout_01`. Root's first GPU check
stopped in the first empty-row fixture: `contiguous()` preserved a zero final
stride, so reinterpreting FP32 as bytes raised before outputs were saved. No
numerical mismatch was reported. Frozen revision 01 and its run logs are intact.

Revision 02 is prepared at
`/tmp/deepseek_linear_scale_layout_candidate_20261006_02/`. All eight baseline
and numerical candidate source files are byte-identical to revision 01. The
comparison first checks shape/dtype and handles empty payloads directly;
nonempty byte comparison and serialization use explicit contiguous-format
clones. This also handles nonempty singleton dimensions whose strides survive
`contiguous()`. No tolerance, quantization arithmetic, scale-store logic or GEMM
call changed.

CPU validation covers 150 ABI/serialization layouts, 300 guarded launches with
mocked writers, five extra zero-stride layouts and a scalar case. It verifies
empty-shape/dtype rejection, nonempty bit-flip rejection, NaN payloads, signed
zero and untouched padding. The existing source/dispatch/address contracts
were also rerun. CUDA remained uninitialized; these checks do not replace root's
new GPU check. Evidence, source equivalence and the new freeze are in that
private workspace. `ROOT_RUN.md` uses the standard environment, observer/NUMA
wrapper and SSD check output `deepseek_linear_scale_layout_check_20261006_02`.
