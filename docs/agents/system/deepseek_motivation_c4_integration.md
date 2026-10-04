# C4 accepted run and CPU audit

Status: full run and post-run CPU checks accepted; not published. This record
does not validate later C5 output-layout or C6 cache/graph-boundary changes.

## Run and source identity

- Formal run: `motivation_c4_20261004_u16_r2_02`.
- Frozen source: `/tmp/deepseek-motivation-c4-frozen-48yhnrcs`.
- Source manifest SHA-256:
  `88796ef27ddbbd5b0b6e969552b9d88d6ed9feb581cc8520bd9e28c9aa2dfa75`.
- FLOPs analysis: `motivation_c4_flops_20261004_02`.
- Comparison reference: `motivation_c3_20261004_u16_r2_01`.

The workload uses H=65,536, A=128, P=65,536, NH=16,777,216, 16 users and
two complete rounds on H200. It runs ten independent dense checkpoint block
copies sourced from layers 0–2, with embedding, final norm and last-token
LM head. It remains a checkpoint workload surrogate, not a trained ten-layer
model or validation of the full 61-layer model.

C4 includes packed CPU token input, exact single-query storage accounting,
guarded cleanup after completed transient discard, and the resident indexer's
causal-tail mask restriction. The latter masks only columns at or beyond
`query_start`, avoiding an unnecessary full-history boolean mask. This is a
combined candidate; the run does not isolate each optimization's contribution.

Execution-source changes against C3 are confined to these four files among
the model/cache/serving/operator implementation files:

| File | C4 SHA-256 |
|---|---|
| `serving/persistent.py` | `e07b33d5192372b64042f650c308baf99a9e944d3ba7ed54abc597162fad63f7` |
| `cache/sparse_token_pool.py` | `202ba4352118055f2ad3c2e33236487d7f6edd04eb49e5c2cdf2e751deeb2974` |
| `models/deepseek_v32/serving_backend.py` | `23d63e3fd08b63a9c7afb4fab554deb6f426ffecac22b37939c10cf198d2f67a` |
| `operators/deepseek_v32/indexer/echo.py` | `30492794f10c4bd34d730c0bb926c79827095feb59e7cad35765ff0bced4c219` |

Data/log run directories were moved to the original repository without
overwriting any destination. All 1,570 files were hashed before and after the
cross-device move; the run-local `relocation_manifest.json` records every
identity, including the intact source snapshot.

## Acceptance and analysis

The frozen-source report command independently rechecked all 128 measured
requests, all 12 warmup requests, saved source/workload/tensor identities,
session behavior, graph replay counts and timing-stage conservation. All 96
offload request outputs match their HBM references exactly, including every
candidate hidden element and last-token vocabulary logit.

`analysis/cross_version_c3_hbm.json` additionally compares all 32 complete
request records and every saved HBM payload field between C3 and C4. Inputs,
hidden states and logits are byte-exact. The executable CPU check is preserved
at `analysis/cross_version_check.py`.

CPU entrypoints ran under `CUDA_VISIBLE_DEVICES=''` from the C4 frozen tree:

```bash
python -m experiments.deepseek_v32_motivation.src.report \
  --run-dir C4_RUN --output-dir C4_RUN/analysis/reaudit_report
python -m experiments.deepseek_v32_motivation.src.flops \
  --run-dir C4_RUN --output-dir C4_FLOPS \
  --peak-report ORIGINAL_REPO/experiments/deepseek_v32_echo_prefill/report/layers3/summary.json
```

Here `C4_RUN` is
`experiments/deepseek_v32_motivation/output/data/motivation_c4_20261004_u16_r2_02`
under the original repository; `C4_FLOPS` is the corresponding
`output/data/motivation_c4_flops_20261004_02`. These analyses introduced no
GPU measurement and did not update `report/` or publication README results.

## Latency and MFU

Times below are arithmetic means of all 16 measured requests in each group,
rounded to three decimals. Prefix entries displayed as zero are below
0.001 ms; those hits execute no history model work. HBM revisits rebuild their
history under the fixed HBM quota.

| Scheme / visit | E2E ms | Admission ms | History ms | Candidate ms | Cleanup ms | E2E MFU % |
|---|---:|---:|---:|---:|---:|---:|
| HBM / first | 2415.863 | 4.486 | 2399.780 | 11.355 | 0.242 | 40.044 |
| HBM / revisit | 2422.401 | 4.404 | 2406.282 | 11.468 | 0.247 | 39.936 |
| ECHO / first | 2532.248 | 5.453 | 2511.787 | 13.563 | 1.445 | 38.203 |
| ECHO / revisit | 39.909 | 4.115 | 0.000 | 34.126 | 1.668 | 5.629 |
| Serial sparse / first | 2453.739 | 5.540 | 2434.554 | 12.184 | 1.461 | 39.426 |
| Serial sparse / revisit | 33.398 | 4.224 | 0.000 | 27.528 | 1.646 | 6.726 |
| Dense prefetch / first | 2470.470 | 5.585 | 2451.028 | 12.365 | 1.492 | 39.159 |
| Dense prefetch / revisit | 40.330 | 4.310 | 0.000 | 34.310 | 1.710 | 5.570 |

MFU is `100 * sum(ideal compute ms) / sum(measured wall ms)`, not an average
of per-request percentages or speedup ratios. Precision-specific useful work
uses the recorded no-TF32 policy and dense H200 reference peaks FP8=1979,
BF16=989.5 and FP32=67 TFLOP/s. History plus candidate ideal time is
967.406637 ms per miss request; a history-hit candidate has 2.246394 ms of
ideal work. These are reference-peak effective MFUs, not hardware counters.
No C4 operator-MFU profile has been accepted; C3's operator timings must not
be presented as C4 measurements.

Against C3, first-visit E2E means fall by 110.687/98.915/117.131/104.548 ms
for HBM/ECHO/serial/dense. ECHO and serial revisit admission falls by
3.094/3.114 ms and cleanup by 1.365/1.410 ms; their candidate means change by
+0.956/+0.017 ms. Their total revisit means improve by 3.502/4.507 ms.
These paired stage summaries support the CPU-overhead reduction, while the
complete run includes the separate indexer-mask change.

The first-visit ECHO/serial/dense gaps above HBM remain
116.386/37.876/54.607 ms. The active latency/MFU optimization goal is not
complete.

## Outliers and comparison limits

All samples remain in every mean and MFU denominator. C3 dense request 16
took 674.951 ms, including 663.633 ms in candidate execution. Its other 15
revisits were roughly 44.6–45.7 ms. That one observation strongly affects the
C3 dense revisit mean of 84.533 ms. C4's mean is 40.330 ms; this 44.202 ms
mean difference must not be attributed wholly to C4's CPU changes. Dense
revisit medians are 45.064 ms in C3 and 39.893 ms in C4. The cause of the C3
slow request is not established by these stage measurements.

The slowest C4 revisit in each offload scheme is:

| Scheme | Request | E2E ms | Group median ms | Admission ms | Candidate ms | Cleanup ms |
|---|---:|---:|---:|---:|---:|---:|
| ECHO | 29 | 41.380 | 39.601 | 4.172 | 35.614 | 1.594 |
| Serial sparse | 25 | 36.255 | 33.170 | 5.500 | 29.083 | 1.671 |
| Dense prefetch | 16 | 44.538 | 39.893 | 4.085 | 38.343 | 2.109 |

`analysis/c3_c4_stage_comparison.csv` preserves all mean stage differences.
`analysis/c4_slowest_requests.json` records the three slowest requests per
scheme/visit and their median absolute deviations. These are descriptive
records, not exclusion criteria or causal classifications.

## GPU process observations

Root confirmed monitor session 88224 exited 0 before its JSONL was attached.
The run contains an exact copy at `gpu_observations.jsonl`, SHA-256
`478a7815756f2e1e0e64df323195dc55d22335fbf2442233d1fa2175d48b911d`.
`gpu_observations_boundary.json` records the precise sampling interval.

There are 21 discrete process-list samples from
2026-10-04 03:20:34.992828 to 03:24:17.388442 Asia/Shanghai
(2026-10-03 19:20:34.992828 to 19:24:17.388442 UTC). Sampling intervals range
from 10.097 to 30.495 seconds. The first sample is 235.238 seconds after the
metadata run start; the final sample is 47.802 seconds after measured-run
completion. Twenty samples show only expected PID 1727805; the final sample
shows no compute process. No unexpected PID is observed within these samples.

The monitor began near the end of HBM / ECHO warmup, after run launch. This
does not establish continuous initial-phase or between-sample isolation.
Root separately observed only the main PID in earlier launch and intermediate
`nvidia-smi` readbacks; those readbacks are not part of this JSONL. The limited
observation coverage is preserved rather than generalized to the whole run.
