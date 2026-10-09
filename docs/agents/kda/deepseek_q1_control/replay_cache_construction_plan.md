# Cache construction before the matched reduced timer

Status: CPU/static preparation only; no GPU result. The full four-method
prelude reproduces the long-gap regime, while repeated HBM warmups and direct
pinned-allocation lifetimes have not reproduced it in their first processes.
The new factor separates the cache-construction package from model/offload
forward history. It does not isolate an individual allocation or hardware
mechanism. Root decides whether to run it after the full-prelude statistics
profile and its audits are accepted.

Add only `q1_replay_cache_construction.py`. After the unchanged one compute-bank
preparation, call the existing `profile_layers.select_cache_method` in order
`[hbm, echo, serial_sparse, dense_prefetch, hbm]`. Do not run a prelude prefix,
forward, snapshot/restore or extend-graph preparation. Require zero valid
length and no full graphs after every selection, the same compute bank, one
generation advance per selection and five in total. The final HBM selection
must leave no shared pools, sessions or prefetch helpers.

Record the actual selection order and completion in `contract.cache_construction`
and `runtime.cache_construction`, with kind
`deepseek-q1-replay-cache-construction-v1`. Bind the new source while keeping
the matched wrapper, timer and all earlier drivers unchanged. Retain the
matched formal source/environment/runtime gates and honest participating
runtime subsets; constructing offload caches need not execute or load all
offload kernels. Do not load providers to manufacture a full runtime match.

The public constructors retain their own synchronization, native loading,
GPU-buffer allocation and stream creation. The driver adds no flush, host-stat
query/reset, stream or provider. Normal graph-entry behavior, measured HBM
prefix/capture, five warmups, checks and clean/profile timer boundaries remain
unchanged. No host-stat wrapper is included in this factor.

CPU review must verify the exact selection trace, absence of forward/snapshot/
graph helper calls, one-shot execution, all intermediate/final state gates,
stable source/runtime identity and original exceptions without retries.
After source freeze and independent review, any root-run check, clean benchmark
and profile need new IDs and the existing acceptance/raw-window audits. Keep
single-process factor evidence distinct from confirmed treatment replication.

The frozen driver SHA256 is
`45fadcc4d9e49bfb4d88b83c7b990be3092014cf7896114ec20c9ea120b45707`.
Author CPU mocks passed 19 cases; independent source/CPU review passed five.
Ruff, formatting and the real CLI help passed with CUDA uninitialized.
The review files are `/tmp/cxldsagr-checks/q1_replay_cache_construction_cpu_review.json`
and `/tmp/cxldsagr-checks/q1_replay_cache_construction_independent.json`.
These checks do not validate real allocations, native loading, GPU cleanup or timing.

The separate analyzer is `experiments.deepseek_v32_mfu.src.analyze_replay_cache_construction`,
frozen SHA256 `a9c52c4fcc49c36e6f2e3752923bf2e2262498f36ba45cf9301a352924d1994e`.
It reuses the matched raw audit and HBM reference helper, comparing accepted
A1/B1/C1 signatures and owners while allowing exact participating runtime
subsets. It adds `--matched-audit-dir`, `--four-method-audit-dir` and
`--hbm-prelude-audit-dir` to the matched analyzer arguments. Actual reference,
constructor check/bench metadata and four strict rejection gates passed in
`/tmp/cxldsagr-checks/q1_replay_cache_construction_analyzer_cpu_review.json`.
This CPU review does not replace root's raw-profile acceptance.
