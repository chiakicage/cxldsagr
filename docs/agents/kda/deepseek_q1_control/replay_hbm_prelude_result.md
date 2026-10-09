# Four HBM warmups: first factor control

Repeating the HBM warmup four times did not reproduce the long-gap regime in
this accepted control. All six profile scopes have 96 ns median recorded gaps;
the two measured plain windows have 17.536/20.417 us idle. The four-method
ABBA control remains 416 ns and 67.361–68.064 us plain idle. This factor adds
one independent benchmark process and one independent profile process. The
50 within-process plain/timed pairs and six profile scopes are not independent
replications of the HBM×4 treatment.

| Scope | Span us | Busy us | Idle us | Median recorded gap ns |
| --- | ---: | ---: | ---: | ---: |
| Warmup plain | 1011.778 | 992.866 | 18.912 | 96 |
| Warmup timed | 1015.907 | 995.683 | 20.224 | 96 |
| Pair 0 plain | 1006.051 | 988.515 | 17.536 | 96 |
| Pair 0 timed | 1013.090 | 995.457 | 17.633 | 96 |
| Pair 1 timed | 1018.306 | 999.744 | 18.562 | 96 |
| Pair 1 plain | 1017.507 | 997.090 | 20.417 | 96 |

The separate clean benchmark retains 50 balanced plain/timed AB/BA pairs.
Complete synchronized forward wall medians are 1.8992515/1.904469 ms; the
paired timed-minus-plain median is +9.5415 us, with 15/50 timed wins. These
pairs measure extra external-event overhead within this process, not an
optimization benefit from the HBM warmups. Clean wall and invasive profile
windows remain separate measurement boundaries.

The driver repeats `hbm` four times after one compute-bank preparation. It
uses the same public prefix, cold-restore and extend-graph helpers as the
four-method prelude. Completion requires five cache-generation advances,
the unchanged compute bank, empty shared pools/sessions/helpers, closed
prelude full graphs and a fresh empty HBM cache before the unchanged reduced
timer. Its formal reference still requires the original four-method order;
that reference contract is distinct from the actual HBM×4 loop. The driver is
`experiments/deepseek_v32_mfu/src/q1_replay_hbm_prelude.py`, SHA256
`0611fe22ee8243c66f4c8cc4310e701bdc1bf23804397e64b81dab7bdafe036f`.

The check is `q1_replay_hbm_prelude_check_20261008_01`, using the distinct
`deepseek-q1-replay-hbm-prelude-v1` receipt. Independent CPU rereading verified
six saved bitwise comparisons for tokens 111090, 111091 and 111092 and rehashed
1,432 receipt artifacts. The check, clean benchmark and profile have identical
execution identities. Observed native/CuTe/Triton entries match the formal
runtime records exactly; five unused offload DSOs and one decode-hint
specialization legitimately remain unobserved. This matches the HBM-only
omission set and differs from the four-method process runtime set. The formal
collector still lacks a DeepGEMM per-launch JIT binary ledger, and CuTe MLIR
hashes have no retained files.

Runs are `q1_replay_hbm_prelude_{bench,profile,audit}_20261008_01`. The analyzer
binds accepted A1 matched and B1 four-method audits, including their distinct
check receipts, and delegates the unchanged raw timer audit. Its source is
`experiments/deepseek_v32_mfu/src/analyze_replay_hbm_prelude.py`, SHA256
`153a12c0020708edfce8b1463148c9a30cd3bd8869c0e5a1ef4733d9ea2b7ae0`.
The independent analyzer review passed 28 actual-evidence and negative-mutation
cases without provider imports or GPU execution. It checks contracts,
completion, sources, archives, receipt kinds, exact participating runtime
subsets and rejection of altered or duplicated native identities.

An independent SQLite review rereads all 197 correlated GPU nodes in each
scope and compares every saved raw field. The 192 layer-owned nodes define
the layer window. Every-process/device queries find 193 intersecting rows per
window on device 0: the same layer nodes plus a crossing final norm. Clipping
that norm preserves the integer interval union. All recorded gaps and busy/
idle values match exactly, with no foreign process or graph activity. Native
node signatures and layer owners agree with the A1 and B1 references; this
does not establish equality of dependency edges.

Supplemental driver, check, analyzer and raw-window reviews are indexed under
`experiments/deepseek_v32_mfu/output/data/q1_replay_hbm_prelude_audit_20261008_01/independent_review/index.json`.
The index preserves hashes of every original audit file; those files remain
unchanged. No new production implementation or formal performance result is
published by this factor. It narrows the next diagnostic step to differences
introduced by the offload preparation/execution package, without excluding
all allocation, stream, graph-history or loading mechanisms. Resource
preparation without offload forwards is the next factor; it has no result yet.
