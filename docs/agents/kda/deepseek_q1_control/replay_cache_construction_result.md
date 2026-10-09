# Cache construction only: first result

Creating and releasing the four cache methods without prelude forwards did
not reproduce the long-gap regime in this control. All six profile scopes
have 96 ns median recorded gaps. The two measured plain windows have
18.850/19.168 us idle, versus 67.361–68.064 us in the confirmed four-method
ABBA prelude. This adds one independent benchmark process and one profile
process; 50 within-process pairs and six scopes are not treatment replications.

| Scope | Span us | Busy us | Idle us | Median recorded gap ns |
| --- | ---: | ---: | ---: | ---: |
| Warmup plain | 1013.186 | 995.809 | 17.377 | 96 |
| Warmup timed | 1017.859 | 999.139 | 18.720 | 96 |
| Pair 0 plain | 1014.659 | 995.809 | 18.850 | 96 |
| Pair 0 timed | 1018.691 | 998.691 | 20.000 | 96 |
| Pair 1 timed | 1020.962 | 1002.178 | 18.784 | 96 |
| Pair 1 plain | 1007.939 | 988.771 | 19.168 | 96 |

The separate clean benchmark retains 50 balanced plain/timed pairs. Complete
synchronized forward wall medians are 1.905263/1.9112395 ms. The paired
timed-minus-plain median is +13.908 us, with 11/50 timed wins. These compare
external-event overhead within the same process; they do not measure an
optimization from cache construction. Clean wall and intrusive profile
windows remain separate measurement boundaries.

After one unchanged compute-bank preparation, the driver calls the existing
cache selector for `hbm`, `echo`, `serial_sparse`, `dense_prefetch`, then `hbm`.
Every selection must leave zero valid length, the same compute bank and no
full graphs. The final HBM state must have five generation advances and no
shared pools, sessions or prefetch helpers. There is no prelude prefix,
forward, snapshot/restore or extend-graph preparation. Public constructors
retain their own native loading, allocations, streams and synchronization;
the driver adds no flush or statistics query.

Construction loads three libraries missing from the HBM-only control:
`echo_indexer`, `recall_dispatch` and `kv_transfer`. The recorded participating
runtime entries match the formal identities exactly. `official_echo_decode`,
`official_prefetch` and the decode-hint specialization remain unobserved.
This is an initialization-package control, not an isolated allocation test.
Its negative result does not rule out all allocation, registration, stream,
graph-history or offload-execution mechanisms.

The driver is `experiments/deepseek_v32_mfu/src/q1_replay_cache_construction.py`,
SHA256 `45fadcc4d9e49bfb4d88b83c7b990be3092014cf7896114ec20c9ea120b45707`.
The independent check `q1_replay_cache_construction_check_20261008_01` uses kind
`deepseek-q1-replay-cache-construction-v1`. Root's CPU reread verified six saved
bitwise output comparisons for tokens 111090/111091/111092 and rehashed 1,435
receipt artifacts. Check, benchmark and profile share one execution identity.

Runs are `q1_replay_cache_construction_{bench,profile,audit}_20261008_01` under
`experiments/deepseek_v32_mfu/output/data/`. The analyzer binds accepted A1/B1/C1
references and delegates the unchanged matched/raw timer audit. Its source is
`experiments/deepseek_v32_mfu/src/analyze_replay_cache_construction.py`, SHA256
`a9c52c4fcc49c36e6f2e3752923bf2e2262498f36ba45cf9301a352924d1994e`.
Root's independent SQLite reread verified all six scopes' 197 correlated nodes,
192 layer owners and exact all-process/device clipped interval unions. These
signatures do not establish equality of graph dependency edges.

Ten existing author, independent source/CPU, actual-check and raw-activity
review files are copied into the audit's `independent_review/` and indexed
by `index.json`, SHA256
`59815aa7b77a1b796532f9083caffde91172c5d9264b6e949c7d88507ef7ba82`.
All 17 original audit files remain byte-for-byte unchanged. Archival reran no
tests or GPU work. The analyzer's independent review was source-only; actual
metadata and raw-window checks are separate records.

The next factor runs one selected method's complete formal warmup before
returning to a fresh HBM cache. It can separate method-specific initialization
and execution packages, while preserving the same numerical and measurement
gates. No production performance value changes from this result.
