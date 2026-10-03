# Industrial 10M trace rerun

**Superseded execution, 2026-10-02:** The user asked to first run a smaller 64K
case: HBM 4 GiB, DRAM 64 GiB, 16 users, 32 requests, sequential IDs 0..15 twice,
without heat sampling. The industrial process below was terminated during
DeepSeek input generation before model loading or measurement; exec session
56177 exited 143 and all process-group members were confirmed absent. It produced
no accepted experiment output. Its diagnostics remain only in the external
staging path below. Do not restart it while the smaller case is being evaluated.
The current execution contract is `gr_serving_sequential_rerun.md`.

2026-10-02. Active user objective: rerun the current experiments using the existing
industrial 10M heat construction with a 1024-user population and 4096 requests.
Latest steering: run the 64K history case first; do not start 4K/16K yet. Cache
budgets are now 64 GiB HBM and 1024 GiB CPU DRAM. The earlier 4/16 GiB proposal
and its 56-hour estimate are superseded.

## Frozen workload and experiment scope

- `GR/generated/industrial_10m_pv_share_t4096_seed42/requests_1024.csv` is the
  selected existing trace. It uses `industrial_10M`, `pv_share`, population and
  sampling seed 42, independent weighted draws, and no per-user revisit cap.
- Raw request CSV SHA256:
  `f2a507dd546c33f5b9d9774a50824945e44f3dfce5709052864b20e7a5888452`.
  Workload canonical access SHA256:
  `31d8a1e01b97f53f3d6db1a098085a28050e92d240db8ed9e2ac7ae680c98f97`.
- Actual coverage: 751 visited users, 273 unvisited, 507 returning users,
  751 first visits and 3345 revisits; maximum visits 129. These are observed
  outputs, not quotas. This is synthetic heat, not an industrial request log.
- Prefix 65536, candidate 128, chunk 1024, DeepSeek sparse slots 32768.
  Both models retain their existing scopes and all four schemes: DeepSeek
  ten-copy dense checkpoint workload then full NOSA. Each case starts empty,
  runs two separate warmups, and measures the complete 4096-request trace.
  Total 8 cases / 32768 measurements / 24576 non-HBM all-hidden comparisons.
- Single physical GPU 0, as in the prior runs. Every request has an exact BF16
  all-candidate-hidden comparison. DeepSeek also executes its last-token LM head;
  NOSA only returns hidden. Model weights/ordinary activations are outside cache
  quotas; reservations and actual cache samples remain independently audited.

## Planning estimate, not new performance evidence

Old 64K cold/hit means combined with an exact LRU simulation of the new trace
give about 30.8 hours of request execution, excluding generation, model loading,
numerical serialization, independent audit and native profiling. The old
short-trace means have weak statistical support; live progress must supersede
this scheduling estimate.

| Model | Scheme | Session capacity | Predicted revisit misses / 3345 | Estimated hours |
|---|---|---:|---:|---:|
| DeepSeek | HBM | 80 | 2168 | 7.32 |
| DeepSeek | ECHO | 145 | 1544 | 7.36 |
| DeepSeek | Serial sparse | 145 | 1544 | 6.19 |
| DeepSeek | Dense prefetch | 273 | 785 | 3.86 |
| NOSA | HBM | 30 | 2803 | 2.37 |
| NOSA | Serial sparse | 366 | 454 | 1.21 |
| NOSA | Dense prefetch | 271 | 793 | 1.34 |
| NOSA | Overlap | 366 | 454 | 1.14 |

The host reports about 2 TiB RAM with 1.7 TiB available before launch and unlimited
memlock. GPU 0 has approximately 140 GiB usable physical memory. These are
preflight observations, not proof of run-time peak memory or accepted allocations.

## Commands and acceptance

Launched latency run ID:
`gr_serving_h200_20261002_industrial10m_u1024_t4096_h64k_01`.

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
  bash experiments/gr_serving/scripts/run.sh \
  gr_serving_h200_20261002_industrial10m_u1024_t4096_h64k_01 \
  --users 1024 --requests 4096 --history-tokens 65536 --candidate-tokens 128 \
  --heat-dataset industrial_10M --heat-field pv_share --seed 42 \
  --access-trace GR/generated/industrial_10m_pv_share_t4096_seed42/requests_1024.csv \
  --hbm-budget-gib 64 --dram-budget-gib 1024 --deepseek-slots 32768
```

The launcher keeps incomplete/failed diagnostics outside experiments and only
copies successful run outputs into the experiment. It is not independently
accepted until the CPU auditor passes with `--expected-trace-directory
GR/generated/industrial_10m_pv_share_t4096_seed42`. Then run a new native profile
with the first three saved NOSA revisits, `--num-users 1024`, and verify all six
serial/overlap samples. Keep all valid samples even below the 90% overlap gate.

Publish the accepted 64K report and provenance, inspect rendered figures, then
remove the replaced old 64K latency/profile outputs and report assets in the same
publication update. Preserve old 4K/16K results until their replacements are
accepted. No old performance number represents the new trace or budgets.

## Superseded execution record

The latency launcher was stopped while building the complete DeepSeek workload.
Tool exec session: `56177`; launcher PID `503840`; measurement PID `503849`.
Frozen aggregate source SHA256:
`6c0e1d9457d52ff325687378eb9bf1b267a7d780febf86f54f04e6e39d9e0e62`.
Staging root (outside experiments):
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving/gr-serving-gr_serving_h200_20261002_industrial10m_u1024_t4096_h64k_01.nSuqaN`.
Its `log/measure.stdout.log`, `log/measure.stderr.log`, `data/metadata.json`, and
`data/measurements.jsonl` are the live evidence. Poll the existing exec session or
confirm the process identity before deciding it stopped; do not restart on an
observation timeout. No new performance result is accepted yet.

Required CPU checks passed: workload 32, measurement 7, independent audit 71,
native profile 61. Both real tokenizers passed 64K+128 initial requests; complete
4096-access schedules matched the original CSV for both model formats and all
three configured history lengths. These are preparation checks, not experiments.

The former source freeze was released after process termination. These source
identities describe the stopped attempt only. The parent objective remains active;
the latest requested first delivery is the smaller sequential case.
