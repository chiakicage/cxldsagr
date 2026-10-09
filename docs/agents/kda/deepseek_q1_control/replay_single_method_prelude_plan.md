# One selected method's warmup before the matched timer

Status: new driver and CPU/static review only. Cache construction alone retained
96 ns gaps; the complete four-method prelude retained 416 ns. The next factor
executes one method's complete warmup before returning to HBM. Root will select
`dense_prefetch` first, then `serial_sparse` and `echo` as needed. Differences
separate method-specific initialization/execution packages; they do not isolate
DMA, allocation, stream history or a hardware fence mechanism.

`q1_replay_single_method_prelude.py` accepts `--prelude-method` with choices
`hbm`, `echo`, `serial_sparse`, `dense_prefetch` and default `hbm`. Only that
driver flag is consumed; `--formal-profile` and every original timer argument
continue to the unchanged matched CLI. No earlier driver or analyzer changes.

After the same one compute-bank preparation, select the chosen method and run
the exact public formal sequence: H65536 prefix, snapshot, cold restore, Q1
extend-graph preparation, token111090 forward, delete snapshot. Then select
fresh HBM. Require committed prelude length65537 before the final selection,
the same bank, exactly two generation advances, zero final length and no shared
pools, sessions, helpers or full graphs. Keep normal synchronization and
graph-entry flush behavior; add no stats, flushes, streams or profiler scopes.

Use kind `deepseek-q1-replay-single-method-prelude-v1`. The actual chosen method
and warmup count enter both `contract.single_method_prelude` and
`runtime.single_method_prelude`, so receipts cannot cross methods. Preserve all
matched formal source/environment checks and exact participating runtime
subsets. Each method needs its own check before separate clean bench/profile
runs. Single-process results remain factor evidence, not confirmed treatment
replication. Root owns GPU execution and raw acceptance after source freeze.

Frozen driver SHA256:
`6368e0a8470a39c3a42a2eb9553988a36afa152740bbb446b8429641862b2fb3`.
Seven bounded author CPU cases cover all four methods, CLI preservation,
distinct method identities and three failure gates. Independent source-only
review found no blocker. Ruff, formatting and the selected-method CLI help
passed with CUDA uninitialized. Existing review records are
`/tmp/cxldsagr-checks/q1_replay_single_method_prelude_cpu_review.json` and
`/tmp/cxldsagr-checks/q1_replay_single_method_prelude_independent_source_review.json`.
These checks do not validate real GPU numerical, cleanup or timing behavior.

The distinct analyzer is
`experiments.deepseek_v32_mfu.src.analyze_replay_single_method_prelude`, SHA256
`e5fb3e79e724acc14b7e237791d36d9cc0b1c1e472ecb4d4918aa74bf134be8c`.
Required arguments are `--prelude-method`, `--profile-dir`, `--bench-dir` and
`--output-dir`. It retains the matched analyzer's optional `--formal-profile`.
Optional `--matched-audit-dir`, `--four-method-audit-dir`, `--hbm-prelude-audit-dir`
and `--construction-audit-dir` default to the accepted A1/B1/C1/D1 audits.
It validates exact driver-only source changes, method-bound contract/completion,
receipt identity and participating formal runtime subsets, then delegates the
existing raw activity-union, signature and ownership audits.

Thirteen bounded CPU cases passed using accepted references and synthetic
receipts whose contract/completion dictionaries come from the frozen driver
AST. No actual single-method run was available during this review. Independent
source review found no blocker; Ruff, formatting and CLI help passed. Evidence:
`/tmp/cxldsagr-checks/q1_replay_single_method_analyzer_cpu_review.json` and
`/tmp/cxldsagr-checks/q1_replay_single_method_prelude_analyzer_source_review.json`.
