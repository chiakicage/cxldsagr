# Reduce compute graph boundary copies

Status: proposal only, prepared while the isolated C4 full trace runs.

## Contract and evidence

The C3 full profile conserves every activity and finds about 55 ms of explicit
graph-input copies per cold request. The largest input is absorbed MLA output
`[Q,128,512]` copied to each finish graph. The pinned official FlashMLA sparse
prefill API allocates its result internally and exposes no output argument.
Do not edit its upstream sources or substitute a different attention backend.

The candidate must preserve the ten independent checkpoint block copies,
full candidate hidden/logits, exact selection, FP32 residual/norm semantics,
owned D2H sources and all stream/failure/capacity rules of C3. Use Q=1024 and
Q=128 on H200/SM90, then the complete H65536/A128/P65536/NH16777216 workload.
Profile-derived sums are prioritization evidence, not forecast wall savings.

## Draft

First measure the direct final-layout output BMM proposed separately by the
graph_profile agent. Its output is `[Q,128,128]`; a transposed `out=` view can
remove the current head-major result repack without changing arithmetic.

Then probe moving only that expansion BMM before the finish graph. The eager
BMM writes directly into a backend-owned static `[Q,128,128]` tensor; the graph
contains the existing output linear, residual norm and MLP. This eliminates
the absorbed-output input copy, but moves one matrix API from graph replay to
ordinary submission. Smaller candidate batches may regress. Exact BMM output
and complete timed callback costs decide promotion, independently for each Q.

The saved residual is also already a stable projection-graph output. A separate
variant could let finish capture read it directly rather than copy it to a
second static tensor. This is legal only with separate private graph pools,
stable captured output identity and enforced sequential execution. Warmup
must not leave finish bound to a temporary saved residual, and capture must
not overwrite still-live projection outputs. Treat this as a separate candidate.

## Executable probe plan

1. Prepare a /tmp-only subclass or adapter using real checkpoint layer weights.
   Keep the ordinary and candidate callbacks alive for identical inputs. No
   production graph changes before the probe passes.
2. Test both residual branches, all six projection outputs, absorbed-output
   expansion and complete hidden/residual output for Q128/Q1024. Include
   changed inputs, long positions, cross-stream ordering and independent
   outputs retained across a later invocation. Require bitwise equality.
3. Compare synchronized complete input-copy/BMM/finish costs after warmup,
   including every ordinary launch and copy. Also record CUDA event timing
   and one trace of actual kernels/copies; do not time replay alone.
4. If a variant improves, update graph input planning, actual-storage audits,
   policy revision, API instrumentation and graph lineage checks together.
   Charge a borrowed saved tensor through its actual existing owner once.
   With no incoming residual it aliases static hidden input; with a residual
   it belongs to the projection private pool. Never count it as a fresh static
   allocation. Keep poison/owner retention
   when graph completion cannot be established.
5. Repeat full real-checkpoint four-scheme validation, then freeze a new source
   and measure the quiet full trajectory. Publish only after corresponding
   actual replay-node attribution and numerical/source audits pass.

The current C4 run, older published reports and valid raw output remain until
replacement acceptance. This proposal has no measured performance result.
