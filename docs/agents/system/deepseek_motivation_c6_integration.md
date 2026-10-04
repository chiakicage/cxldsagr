# C6 accepted run and independent audit

Status: complete four-scheme trace and CPU audits accepted; publication pending.
The latency/MFU optimization goal remains active. This run includes C5 top-k,
the direct value-expansion output layout, graph policy v2 and C6 native sparse
recall metadata. It does not include the working-tree C7 dense metadata change.

## Run and implementation identity

- Run: `motivation_c6_20261004_u16_r2_01`; exec 52572 exited 0.
- Frozen root: `/tmp/deepseek-motivation-c6-frozen-5vbx7859`.
- Executed-source manifest SHA-256:
  `bd3e91265fed36f182d291b1bea3a659a2214c686e8236ed42da663c07775bff`.
- Freeze map SHA-256:
  `c7694caab06ea8ecd33a644fc26cd29ce2d5c6b288157cf31b3052118270b5b6`.
- FLOPs analysis: `motivation_c6_flops_20261004_01`, generated from frozen C6.
- Comparison: `motivation_c4_20261004_u16_r2_02`.

H=65536, A=128, C=1024, P=65536, NH=16777216; sixteen users visit twice.
The workload remains ten independent checkpoint dense-block copies with
embedding, final norm, all candidate hidden states and final-token logits.
HBM revisits rebuild history under the fixed token quota; offload revisits
retain history. This is the checkpoint workload surrogate, not a trained
ten-layer model or complete 61-layer verification.

The original repository now contains the run's `output/data/` and `output/log/`
directories. All 1573 relocated files matched their before/after SHA-256 hashes;
the data directory records `relocation_manifest.json`. No accepted report was
replaced or deleted.

## Acceptance

Frozen-source `src.report` rechecked all 128 measured outputs, source and input
identities, warmup records, session behavior, graph replay deltas and timing
stage sums. The separate `analysis/independent_formal_audit/audit_source.py`
imports no production check/report code. Its independent audit confirms:

- All 96 offload candidate hidden/logit payloads are byte-exact against HBM;
  all 128 payloads are finite and cover the complete required output.
- All 32 C4/C6 request records and complete HBM payload fields are byte-exact.
- Session LRU, host-page/token admission, transient candidate counters, memory
  fields and all 104960 graph replay deltas agree with the workload contract.
- All twelve saved warmup trajectory rows agree. Warmup tensors were not saved
  for a second independent numerical comparison.
- All eight report groups and 24 formal MFU stages independently reproduce
  from raw rows and precision-specific useful-work arithmetic.

Evidence is in the accepted run's `analysis/reaudit_report/` and
`analysis/independent_formal_audit/`, including stage comparisons, all output
hashes and descriptive slowest-request records. No samples were excluded.

## Formal latency and MFU

All entries are means of sixteen requests. MFU is total ideal compute time
divided by total measured time, using only the relevant stage rows. The
overlapping prefix/extend/E2E rows are never added together.

| Scheme / visit | E2E ms | Admission ms | History ms | Candidate ms | Cleanup ms | E2E MFU % |
|---|---:|---:|---:|---:|---:|---:|
| HBM / first | 2321.445 | 4.962 | 2305.139 | 11.067 | 0.278 | 41.673 |
| HBM / revisit | 2328.641 | 4.813 | 2312.507 | 11.064 | 0.257 | 41.544 |
| ECHO / first | 2442.999 | 5.491 | 2422.515 | 13.580 | 1.414 | 39.599 |
| ECHO / revisit | 27.568 | 4.149 | 0.000 | 21.741 | 1.678 | 8.148 |
| Serial sparse / first | 2361.138 | 5.637 | 2342.154 | 11.898 | 1.449 | 40.972 |
| Serial sparse / revisit | 20.732 | 4.175 | 0.000 | 14.913 | 1.643 | 10.835 |
| Dense prefetch / first | 2376.950 | 5.649 | 2357.703 | 12.161 | 1.436 | 40.699 |
| Dense prefetch / revisit | 40.262 | 4.223 | 0.000 | 34.335 | 1.703 | 5.579 |

Against C4, first-visit means improve by 94.417/89.249/92.601/93.520 ms in
HBM/ECHO/serial/dense order. ECHO and serial revisit means improve by
12.341/12.665 ms, mostly within candidate execution. Dense revisit changes
by only -0.069 ms; its median is 40.091 ms versus C4's 39.893 ms. These data
do not show a meaningful dense revisit improvement.

The current first-visit gaps above contemporaneous HBM are 121.554 ms
(5.236%) for ECHO, 39.693 ms (1.710%) for serial and 55.505 ms (2.391%) for
dense. C6 is a combined candidate, so the formal trace does not isolate each
component's contribution. C3 matrix-API timings cannot serve as C6 API MFU.
The matching dense-only C6 diagnostic completed and passed independent audits;
see [profile checkpoint](deepseek_motivation_graph_profile_checkpoint.md). Final
four-scheme API attribution is still required before report replacement.

## GPU observation boundary

All delegated CUDA work was held until the formal process exited. The attached
`gpu_observations.jsonl` contains 67 discrete NVML samples, SHA-256
`d6e9e1bd71d2e298f3c687076f8a917141aa5038a6d1011b568fc2cc7d46376d`.
Sampling starts 92.245 seconds after metadata start and ends 47.537 seconds
after measured-run completion, with intervals from 5.097 to 27.425 seconds.
No unexpected PID was observed. This does not establish continuous coverage
of the initial phase or intervals between samples. Exact times are retained
in `gpu_observations_boundary.json`.
