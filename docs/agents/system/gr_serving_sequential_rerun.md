# Sequential 16-user 64K rerun

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

2026-10-02. Latest user instruction: first run a smaller case with HBM cache
4 GiB, CPU DRAM cache 64 GiB, 16 users, and 32 requests; no heat distribution,
visit each user in order and repeat the traversal twice.

## Contract

- Keep history 65536 and candidate 128 from the current requested 64K case.
  Access IDs are exactly `list(range(16)) * 2`, visit indices are sixteen zeros
  followed by sixteen ones; there are 16 first visits and 16 revisits per scheme.
  No curve file or stochastic user selection participates in this schedule.
- All eight existing cases run in one process on physical GPU 0: DeepSeek HBM,
  ECHO, serial sparse, dense prefetch; then NOSA HBM, serial sparse, dense prefetch,
  overlap. Each case starts empty after independent warmup. Prefix chunk 1024,
  DeepSeek sparse slots 32768. Exact all-candidate BF16 hidden comparisons remain.
- Keep model, output, memory-accounting and timer boundaries unchanged. This is
  a controlled cyclic workload, not a replay of industrial traffic. Cache misses
  on the second pass remain revisits and include reconstruction latency.
- Expected delivery: 8 cases, 256 measured requests, 192 non-HBM complete-hidden
  comparisons, independently audited identities/budgets/LRU/numerics, six native
  NOSA profile samples on requests 16/17/18, and the accepted report.

## Run 01 command (failed source identity)

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
  bash experiments/gr_serving/scripts/run.sh \
  gr_serving_h200_20261002_sequential_u16_t32_h64k_01 \
  --users 16 --requests 32 --sampling sequential \
  --history-tokens 65536 --candidate-tokens 128 --seed 42 \
  --hbm-budget-gib 4 --dram-budget-gib 64 --deepseek-slots 32768
```

The seed controls synthetic content, not the sequential access order. No
`--access-trace` or heat argument is needed. Sequential metadata must explicitly
show absent heat dataset/field/hash and a deterministic traversal.

## Planning expectations, not measured results

Using current session reservations and the two complete user traversals, expected
capacities are DeepSeek 5/9/9/17 and NOSA 1/22/16/22 in the above scheme order.
DeepSeek's first three schemes and NOSA HBM therefore have 16 revisit misses;
DeepSeek dense prefetch and all NOSA offload schemes retain all 16 histories.
Actual allocations and exact LRU outcomes must still pass the independent audit.
Old 64K cold/hit means suggest about 22.4 minutes of timed request execution;
input generation, loading, comparison I/O, audit and profile are additional.

## Current state and next actions

**Run 01 failed its final source-identity guard and is not a valid experiment.**
All 256 request rows and 256 numerical records were written, but the final guard
detected changes to `models/deepseek_v32/echo_infer.py` and
`models/deepseek_v32/tests/test_echo_infer.py` while measuring. Root identified
these as changes made by an external task. The launcher exited 1 before report
generation and accepted publication. Completed numerical records do not repair
the source-identity failure; no run-01 latency, hit/miss or numerical conclusion
may be published or used as a comparison.

The preceding industrial 1024/4096 run was stopped before measurements, exec
session 56177 exit 143, and PID 503849 confirmed absent. Its external diagnostics
are not experiment results. Source freeze was released to add a true heat-free
sequential workload and an independent order audit. Terminal run-01 identity:

- Run ID `gr_serving_h200_20261002_sequential_u16_t32_h64k_01`.
- Exec session `42234` exited 1; launcher/process-group PID `514472` and
  measurement PID `514478` identify the failed attempt, not a live run to resume.
- Initially recorded aggregate source SHA256:
  `75909d12fb212c7a80277bae5e52acc340136fd88a9848da7bc51cb8e5750e58`.
- Staging root:
  `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving/gr-serving-gr_serving_h200_20261002_sequential_u16_t32_h64k_01.aGAeuj`.
- `log/measure.stderr.log` ends with `RuntimeError: implementation changed while
  measuring` and the two changed paths above. Both JSONL diagnostic files contain
  256 rows. `data/metadata.json` still says `status="running"` because the guard
  raised before the terminal metadata update; that stale field is not evidence
  the process remains active. No accepted run-01 directory exists under
  `experiments/gr_serving/output/data/`.
- Workload checks: 39 CPU tests; audit checks: 86; native profile checks: 62.
  Both real tokenizer workloads passed every input/order identity audit at
  64K+128 across all 32 requests. Access SHA256:
  `c3511ba7f3fb306def87447c0b7d7c545e253f5f113fb98288987612befedda9`.
  These are preparation checks, not performance measurements.

Root will create an isolated worktree and rerun the complete same 16-user /
32-request case as run 02, with a fresh run ID, source snapshot and independent
audit. The exact worktree path, command, run ID and new source identity must be
recorded when that run actually starts; this document does not claim it has
launched. Do not weaken the guard, re-sign run 01 after the fact, or reuse its
measurement rows to avoid the complete rerun. Its diagnostics remain outside
`experiments/` and are not an experimental delivery.

On successful run-02 completion, run the full CPU audit and then the same-GPU
native NOSA profile with `--num-users 16`. Publish only verified results,
preserve all valid slow controls and below-threshold overlap samples, and replace
old 64K artifacts only with the accepted new report publication. Preserve old
4K/16K results until their own replacement is authorized and accepted. The
task/review rewrite in `gr_serving_sequential_publication_draft.md` remains an
unpublished draft; its run/source/status fields must be updated from the actual
accepted replacement, not promoted using run 01.
