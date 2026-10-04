# DeepSeek quantization: investigation log

## q0: fused D128 finite-input candidate

Run `deepseek_quantization_q0_20261003`, physical GPU 2 / H200 / SM90.
The initial candidate passed 28 finite-value/reference tests and reduced
complete index Q1024/K1024 API latency versus the eager indexer formula.
Its source snapshot and measurements are under the corresponding `/tmp`
directory. These are operator development measurements, not experiment
publication or whole-model results.

`deepseek_quantization_q0_profile_20261003` collected full+PM sampling and
source profiles. Q `[1024,64,128]` had one kernel, 36 registers/thread,
128 threads/CTA, 4096 CTAs, 9.344 microseconds under NCU replay. Achieved
occupancy was 63.45%; SM throughput 53.66%, DRAM read throughput 37.49%,
L2 throughput 61.46%. This is not a compute-bound matrix operation.
Full/source profiler durations differ from complete API measurements.
The report flags excess sectors / partial waves; no tuning claim is made.

q0 is not eligible for promotion: additional NaN/Inf checks found that
Triton's default min/max discarded NaNs during reduction and clamping.
Finite-value correctness did not establish the entire reference contract.

## q1: explicit NaN propagation

Parent candidate q0; preserve its row grouping, launch geometry and finite
arithmetic. Replace the reduction combiner and clamps with explicit
`tl.PropagateNan.ALL`, matching PyTorch's canonical NaN output/scales.
All 65,536 BF16 encodings (including payload/sign encodings), mixed Inf/NaN
rows and all prior cases now pass byte/scale equality: 30 tests passed.
Run `deepseek_quantization_q1_20261003`; measurement is still in progress.

Admission is expanded to compare the official compiled DeepGEMM helper on
the same D128 Q/K matrices, including wrapper/output-layout adaptation.
Separate single-call API, batched API and CUDA graph kernel-service timing.
The prior linear-agent 5.95 microsecond activation measurement is graph
service time, not single-call wall or Python-inclusive event time.

The official activation `[1024,7168]` full NCU profile has one kernel,
30 registers/thread, 64 threads/CTA and 7168 CTAs; replay duration is
7.520 microseconds. Its first profiling attempt failed before collection
because a temporary harness named `profile.py` shadowed Python's standard
library `profile`; renaming the harness resolved the failure. No failed
attempt is cited as measurement evidence.

q1 versus official D128, provisional measurements before final JSON:
Q1024 complete single API 48.016 vs 97.584 microseconds and batched API
24.047 vs 66.117; graph kernel service 7.244 vs 6.226. K1024 graph service
3.564 vs 1.425. q1 reduces host overhead but does not beat the official
kernel. The official helper also clamps UE8M0 infinity to exponent 254,
unlike the original indexer; it is not an unconditional semantic substitute.

## q2 accepted bounded arithmetic specialization

Parent q1. The fast path only accepts BF16, so its pre-rounded scale can
take only the finite BF16 absolute-maximum values plus the `1e-4` floor.
For that discrete set, test bit-ceil UE8M0 against the original log2/ceil
using all 65,536 BF16 encodings; retain canonical NaN and infinity behavior.
For UE8M0 only, compute one reciprocal per row then multiply values; scales
are exact powers of two. Keep `scale_fmt=None` elementwise `div_rn` intact.
This removes expensive per-element correctly-rounded division from the
common path. Require the entire exact suite before paired full-API and
graph measurements. Revert if exact tests fail or the full call regresses.

The candidate passed all 30 tests and outperformed compiled official D128
on Q1024/K1024 in both complete API and graph kernel service. Source,
measurements and the integration boundary are frozen in `checkpoint.md`.
The activation assessment demonstrates some potential but is not integrated;
its official non-finite semantics differ from the indexer contract.
