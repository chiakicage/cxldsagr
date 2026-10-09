# Complete formal pipeline reproduction

The original long HBM node gap reproduces without production changes.
Run `deepseek_h64k_a1_cub_gap_reproduce_20261008_01` used the unchanged
formal shell and Python pipeline on GPU0/CPU0-7. The H65536/A1/token111090
request, three checkpoint layers, all-four-method preparation, cache
lifecycle and profiler configuration match the authoritative CUB cohort.

| Run / replay | L0-L2 span us | Busy us | Idle us | Median gap ns | CPU graph launch us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original / warmup | 1074.146 | 1007.074 | 67.072 | 416 | 514.264 |
| Original / measured | 1067.459 | 1000.931 | 66.528 | 416 | 359.344 |
| Reproduction / warmup | 1073.538 | 1005.186 | 68.352 | 416 | 516.343 |
| Reproduction / measured | 1066.817 | 999.137 | 67.680 | 416 | 355.141 |

`q1_formal_reproduction_audit_20261008_01/windows.json` binds both raw
SQLite pairs and native templates. All four replays have the same 197 GPU
nodes, including 192 layer-owned nodes, identical ordered names, all 17
kernel launch fields, copy byte counts and ownership. The union of every
process's activity intersecting each layer interval, clipped at both ends,
equals the selected-node busy union. The crossing shared final norm is
retained. The larger full-forward GPU envelope in `gap_audit.json` includes
embedding, head and setup intervals and is not the table's layer window.

Independent CPU audit
`/tmp/cxldsagr-checks/formal_reproduction_independent_audit_20261008_03.json`
verifies the check, benchmark, original profile and reproduction chains:
1,370 execution sources, two measurement sources, request/checkpoint binding,
6,434 concrete runtime/source files and 12,647 total evidence files. All
eight reproduction output tensors are bitwise equal to both the independent
check and original profile. The check's 36 saved tensors and 31 comparisons
also reread bitwise. All 13 capture files have matching GPU tables, active
GPU0 and relevant profiler settings.

The launcher has a different run ID, shell depth and earlier export of
`TRITON_PTXAS_BLACKWELL_PATH`; the recorded in-process execution/compiler
environments and native identities are exactly equal. Four in-memory CuTe
MLIR hashes are equal but have no retained files for independent rehashing.
These limits are preserved in the independent audit. Its JSON is also copied
into the analysis run directory. The reproduction launcher is
`/tmp/cxldsagr_formal_gap_reproduce_20261008_01.py`; the reusable CPU extractor
is `experiments/deepseek_v32_mfu/src/analyze_formal_reproduction.py`.

The five smaller controls remain negative: construction collection state,
full-extend inspector, internal events, first-five-replay collection, and
ordinary outside timing events each leave roughly 16-20 us idle. The first
four also trace OSRuntime; the last has the exact formal CUDA/NVTX settings.
Thus the long gap is repeatable in the complete pipeline, and its cause is
still unknown. The next control retains that pipeline and omits only initial
compute-graph operator inspection, as specified in `compute_inspector_plan.md`.

This repeatability profile does not replace the clean benchmark or published
matrix and establishes no end-to-end gain or hardware mechanism.
