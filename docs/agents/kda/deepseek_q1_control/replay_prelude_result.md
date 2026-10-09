# Formal prelude control: ABBA confirmation

The four-method preparation package repeatedly reproduces the long-gap regime
in the accepted A1/B1/B2/A2 sequence. Both prelude processes have 416 ns median
recorded gaps in all six profile scopes; both matched HBM-only processes have
96 ns. Plain layer-window idle is 67.361–68.064 us with the prelude and
17.312–18.146 us without it. This confirms the package-level trigger in these
runs, with two independent processes per treatment per metric. It does not
identify a hardware mechanism or establish a stable latency estimate.

| Process / treatment | Clean plain ms | Clean timed ms | Plain profile idle us, two replays | Median gap ns, all six scopes |
| --- | ---: | ---: | --- | ---: |
| A1 / matched HBM-only | 1.896749 | 1.910402 | 17.889 / 18.146 | 96 |
| B1 / four-method prelude | 1.948968 | 1.9577155 | 68.064 / 67.361 | 416 |
| B2 / four-method prelude | 1.980612 | 1.9838795 | 67.936 / 67.392 | 416 |
| A2 / matched HBM-only | 1.890577 | 1.903652 | 17.633 / 17.312 | 96 |

Root ran fresh benchmark processes and fresh profile processes in this order
for each metric. Saved result modification times corroborate completion order;
they are not exact process start times. The clean measurements retain 50
within-process plain/timed AB/BA pairs per process, 400 samples in total.
Those pairs do not provide additional independent prelude replicates. Both B
plain medians exceed both A medians, but the two B medians also differ by
31.644 us. Profile windows and clean complete-forward wall timings remain
separate measurements. FREE formal retains 66.848 us idle and 416 ns gaps.

The prelude reuses the formal public helpers for one H65536/A1 warmup in each
of HBM, ECHO, serial sparse and dense prefetch. It then creates a fresh empty
HBM cache before the original reduced prefix/capture/replay flow. Completion
gates require one unchanged compute bank, five cache-generation advances, and
released offload pools, sessions, helpers and prelude full graphs. The package
changes allocation, registration, stream, graph and library-loading history
together; none is isolated by this experiment.

The accepted check `q1_replay_prelude_check_20261008_01` covers tokens 111090,
111091 and 111092. Independent saved-output rereading confirms six bitwise
eager/scope comparisons, the distinct receipt and completion gates. All formal
runtime entries are observed, including the five offload libraries and one
decode-hint specialization absent from A1. Each matches the formal record.
DeepGEMM per-launch JIT binary identity remains unavailable in the collector;
CuTe MLIR has in-memory hashes without retained files.

The A1/A2 runs are `q1_replay_matched_{bench,profile,audit}_20261008_{01,02}`;
B1/B2 use `q1_replay_prelude_{bench,profile,audit}_20261008_{01,02}`. Each
treatment reuses its separately accepted three-token check; bench/profile
identities match, and the two repetitions of each treatment have identical
execution identities. Both prelude runs observe all formal runtime entries;
both matched runs retain the same five-library/one-hint omissions.

All 24 profile scopes preserve the same 197 native GPU nodes and 192 layer
owners, including names, 17 launch fields and copy sizes. Each has one graph
launch with zero/two ordinary event-record calls according to its scope.
The 197 clone-lineage mappings do not describe dependency edges. Independent
raw rereading reconstructs every native node and recomputes layer statistics
using integer intervals. Every-process/device queries find 192 or 193 rows
per window on device 0, no foreign process or graph activity, and unchanged
busy unions when any crossing final norm is clipped at the layer boundary.

The B1 audit's `addendum/index.json` records the driver mock/source reviews,
independent six-comparison check, analyzer review and separate raw integer-union
audit. The compact confirmation bundle is
`experiments/deepseek_v32_mfu/output/data/q1_replay_prelude_abba_20261008_01/`.
Its result binds all four audits, both correctness receipts, raw SQLite inputs
and source hashes; `clean_samples.csv` retains every sample, and
`artifact_manifest.json` binds the selected files. Existing accepted analysis
files retain their original hashes.

The first HBM×4 factor is now accepted as
`q1_replay_hbm_prelude_audit_20261008_01`. With the same compute bank and five
cache transitions, all six profile scopes retain 96 ns median gaps; the two
plain windows have 17.536/20.417 us idle. Clean plain/timed wall medians are
1.8992515/1.904469 ms. The three-token check passed six saved comparisons.
This adds one benchmark process and one profile process, so HBM-only repetition
did not reproduce the long-gap regime in this run; it does not identify a
lower-level cause. See [HBM warmup control](replay_hbm_prelude_result.md).
The experiment READMEs now state the confirmed ABBA warmup boundary. Published
numbers and report assets remain those of the formal four-method pipeline.
