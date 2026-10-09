# Exact Q1 mean fusion: component gate passed

The private candidate preserves the accepted finite-mean bits and reduces
complete mean-plus-unchanged-EMA graph latency on all three real-layer inputs.
The subsequent [private model gate](q1_hint_model_result.md) passed and the
bounded production relocation has fresh correctness acceptance. Formal
publication remains in progress. This page retains the original component
source/native identities. Official ECHO, strict prediction, top-k, bounded
prepare and FlashMLA attention remain unchanged.

## Candidate and acceptance

The candidate `experiments/deepseek_v32_echo_official/src/q1_hint_exact.cu`
fuses finite mask/count, the fixed Torch sum tree and RN mean publication.
The Python adapter enables it only for SM90 FP32 inference with aligned,
unit-inner-stride `[1,65537]` scores and matching contiguous offset `[16]`.
Unsupported inputs call the existing mean path. Decode EMA remains separate.

Frozen source SHA256:

- `q1_hint_exact.cu`: `bdbc6dc0c191dc2797a2d9b40c10f1e51cc4cee9bd61e1c440acd301fcfc65c0`.
- `q1_hint_exact.py`: `d7b5b356db83f66816bd100c30381fe230851729b1e51d0f2f035c098a7f319a`.
- `q1_hint_candidate.py`: `69da6f89b01df141e580d7d266a46935babfc504f9f4ed3f5670e5811780251f`.
- Loaded candidate DSO: `cb9645437cc0fabb929918a5ba79f54f96470c7b2965e548f584ee208decdd62`.

Independent check `q1_hint_candidate_check_20261008_01` under
`/tmp/cxldsagr-checks/q1-hint-candidate/` passed 78 comparisons on 18 fixtures.
They cover the three real layers, nonfinite inputs, signed zero, subnormals,
overflow/cancellation, vector/tree/tail boundaries and unsupported dispatch.
The oracle is the installed Torch masked FP32 sum and integer count, not a
copy of the candidate tree. All 16 offsets, input owner storage and top-k
value bytes match. Checks include changed captured inputs, an ordered
nondefault-stream producer/consumer and native dispatch counts. The counter
records 13 eager calls plus 13 captures; it does not count GPU graph replays.

The separate CPU review rehashed 62 receipt artifacts, ten current/archived
source files and 48 Triton assembly files, regenerated all fixtures, and
reread all saved bit views. It is saved as
`/tmp/cxldsagr-checks/q1_hint_candidate_check_20261008_01_independent.json`,
SHA256 `e2542c7cf7edd2c08f8873dace2f48c728e4eadf01ff8e869bd9e1587f8f266d`.
The supplementary native audit rehashed the declared 2,182 candidate/CUDA
source and header files, 123 TVM-FFI files, six compiler binaries and 17 loaded
libraries, and matched the immutable DSO across check and bench. Its record is
`/tmp/cxldsagr-checks/q1_hint_candidate_native_runtime_independent.json`, SHA256
`524bea341621cf66cbc13817d741ce714f89544b3c48c5fcbe796d6a261b2ea0`.
The actual Ninja dependency list also contains 325 system C/C++ headers that
were not bound into the original build identity. Their current hashes cannot
prove build-time bytes; this is an exact-DSO acceptance, not a hermetic rebuild
claim. These component tests do not prove the later Q>1 consumer or complete model.

## Independent complete-API timing

Run `q1_hint_candidate_bench_20261008_01` contains 1,200 samples: three real
layers, graph/eager execution, 100 balanced AB/BA pairs and two arms. Both
arms restore positive-zero offsets and synchronize before each timed call.
Events enclose the whole mean-plus-EMA API or one graph replay. All samples
are retained; native identity/libraries match check and all three measured
Triton specializations match entries in its eight accepted variants.

| Layer | Baseline graph median, us | Candidate graph median, us | Paired candidate-minus-baseline median, us | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| L0 | 16.304 | 9.536 | -6.720 | 100 / 100 |
| L1 | 16.128 | 9.664 | -6.496 | 100 / 100 |
| L2 | 16.208 | 9.600 | -6.544 | 100 / 100 |

AB/BA paired medians are -6.736/-6.720, -6.448/-6.496 and -6.560/-6.512 us.
Eager medians are 75.232/42.624, 74.048/42.448 and 74.112/42.016 us
(baseline/candidate), also with 100/100 wins per layer. Eager CUDA-event spans
include host submission gaps; they are not isolated kernel execution times.
This is one independent benchmark process, not 100 independent processes.

The independent CPU benchmark audit recomputes the exact sample order,
summaries, paired deltas and both order strata. Its saved record is
`/tmp/cxldsagr-checks/q1_hint_candidate_bench_20261008_01_independent.json`,
SHA256 `04944f10bcbd4dfe1684aabc870f683690201084c0c5620a7d91834991fe156f`.
Raw source/native/input identity and samples remain in the benchmark run.

The subsequent model gate used independent ECHO models, matching cold
restores and 100 balanced full-forward pairs, plus actual Q1-to-Q2 consumption.
Its separate evidence is linked above. No full-model speedup or formal report
replacement follows from these component numbers alone.
