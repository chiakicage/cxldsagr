# Exact Q1 hint: production acceptance and formal refresh

The bounded production mean now dispatches from
`operators/deepseek_v32/indexer/prefetch_hint.py` to `q1_hint_exact.py` and
`csrc/q1_hint_exact.cu`. CUDA arithmetic is byte-identical to the accepted
private source after its opening comment. It keeps the exact reduction tree,
finite mask/count, RN division and separate EMA. Unsupported inputs retain
normal dispatch; direct unsupported native entry rejects them. Official ECHO,
threshold policy, exact selection, cache preparation and FlashMLA are unchanged.

The production DSO is `cxldsagr_q1_hint_exact_mean_sm90.so`, SHA256
`ac892236d5533f441e0b7c37fd01d5e817038334f787aa3d74922b969cc4fe9f`.
Its immutable key is
`cxldsagr_q1_hint_exact_mean_sm90_0e07858049a4278ead3e86cb6a75ec3810cac2439da176994c62ed0b9da0dfee`.
The integrated check and the subsequent formal cohort use this exact artifact.
Declared project/CUDA/compiler/TVM closure is verified. Undeclared system C/C++
headers remain outside the build-time evidence; this is not a hermetic build claim.

## Fresh acceptance

`q1_hint_integrated_check_20261008_01`, under
`/tmp/cxldsagr-checks/q1-hint-integrated/`, uses the installed Torch expression
as its baseline and the unpatched production dispatcher as its candidate.
Its check-only driver is `experiments/deepseek_v32_echo_official/src/q1_hint_integrated.py`,
SHA256 `8cac0131a73305107fd140775b6565d883e3318e8d6e786edfd5a11565562f3d`.

Independent review verified 1,480 receipt artifacts, 1,383 current/archive
sources, seven native archives and 54 hint assemblies. All 18 fixtures and 78
component comparisons passed, with all 16 offsets, owner/value bytes, changed
graphs, unsupported inputs and ordered nondefault-stream consumers. Real
L0–L2 checks cover 16 diagnostic/consumer records, 14 record comparisons and
six clean graph records. All 48 stage proofs were independently recomputed,
including full saved scores and actual Q1→A2 hint consumption.

Reviews under `/tmp/cxldsagr-checks/`:

- `q1_hint_integrated_check_20261008_01_independent.json`, SHA256
  `431192fd84391e122ba0dafb859c5b87008bbf4a83ed8d026e14de7588d08cc5`.
- `q1_hint_integrated_prefetch_independent.json`, SHA256
  `1b1be79ed99bd5aa4c8b51ddcf8694be915a61a9e0fd863fe27d71e14e27da28`.

Operator regression passed 70 checks, provenance/audit regression 63, and
timeline classification regression 77. No GPU skips count as acceptance.

## Formal measurement

The current accepted cohort is `deepseek_h64k_a1_isolated_20261008_01`, which
replaces the earlier HINT mixed-method preparation report. Each method has its
own check, clean benchmark, minimal node profile and operator profile process.
Only its selected method is warmed; offload checks use saved HBM outputs on CPU.
The model/kernel implementation, H65536/A1/token111090, GPU0/CPU0–7, cold restore
and one warmup/three prefill/five step samples remain the declared boundaries.

| Method | Prefill median ms | Step median ms | L0–L2 profile window ms |
| --- | ---: | ---: | ---: |
| HBM | 636.421778 | 1.834721 | 1.013090 |
| ECHO | 639.730964 | 2.588716 | 1.426563 |
| Serial sparse | 640.379290 | 2.395402 | 1.268482 |
| Dense prefetch | 648.942526 | 5.532310 | 4.408651 |

The private 100-pair hint result (2.6486815→2.6053275 ms, 87 wins) retains its
own paired boundary. Cross-batch formal medians do not establish the hint's
speedup. The isolated HBM window has 17.953 us idle; method preparation was a
material part of the earlier gap observation, without identifying its hardware
mechanism. This measurement correction is not a kernel change.

Complete attention API sums over three layers/nine kernels are
44.224/43.232/42.944/44.352 us for HBM/ECHO/serial/dense, including query repeat
and combine. The actual graph reservation remains separately audited; removing
requested transient hint tensors does not imply additional session capacity.

The consolidated accepted report is
`experiments/deepseek_v32_mfu/report/h64k_a1/`. Its `summary.json`,
`window_rows.json`, `input_hashes.json` and `publication_manifest.json` retain
per-method execution, receipt, clean benchmark and original trace identities.
All 13 standard and 44 graph checks passed. The minimal and operator profiles
each have eight saved outputs compared bitwise with the checked outputs;
actual ECHO transitions, native files and raw graph/window ownership were reread.
The replaced HINT formal reports and their obsolete run payloads are removed
at publication; valid component and controlled comparison evidence is retained.

Retained A128+ and capacity results were separately checked in
[the scope audit](q1_hint_retained_scope.md). All five complete capacity plans
and their 1,222 scalar values remain identical. Official SGLang results retain
their different token, GPU, framework and natural residency; cross-implementation
numerical acceptance remains false.
