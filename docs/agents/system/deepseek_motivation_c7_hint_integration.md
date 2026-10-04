# C7a and exact ECHO hint accepted run

Status: full four-scheme trace, output audit and saved-data arithmetic accepted;
publication pending. The MFU optimization goal remains active. C7 includes the
C6 implementation plus dense-prefetch metadata and exact ECHO hint updates.
The capped gather, mixed-type norm and MLP packing prototypes are not included.

## Run and source identity

- Run: `motivation_c7_hint_20261004_u16_r2_01`; exec 25953 / PID1810918 exited 0.
- Frozen root: `/tmp/deepseek-motivation-c7_hint-frozen-a44l_2uk`.
- Executed-source SHA-256:
  `cc141544c664dfe0e99119d7889b1fa6d1f2728062995ab564b279dca7285652`.
- Freeze-map SHA-256:
  `d08a95eced7c8923945699384f84785ff29d48d3ba63ab8eb5ee9d152736fba3`.
- FLOPs analysis: `motivation_c7_hint_flops_20261004_01`, from frozen C7.
- Reference: `motivation_c6_20261004_u16_r2_01`.

H=65536, A=128, C=1024, P=65536, NH=16777216; sixteen users visit twice.
The workload remains ten independent checkpoint dense-block copies, embedding,
final norm, all candidate hidden states and final-token logits. It is a
checkpoint workload surrogate, not a trained ten-layer model. GPU0 is reported
as NVIDIA H200 / SM90 by PyTorch and M403 by nvidia-smi.

The freeze copied 474 files; the CPU execution-snapshot precheck verified 1259
files including new dependencies. All 1577 relocated data/log files matched
their source hashes. They reside in the experiment's original
`output/data/<run_id>/` and `output/log/<run_id>/` trees. Existing published
reports and their backing outputs remain until final replacement acceptance.

## Acceptance

Frozen-source report re-audit exec 35981 and independent audit exec 16671 exited
0. The independent audit imports no production check/report code. It verifies:

- All 128 saved payloads are finite and complete. All 96 offload/HBM comparisons
  of candidate hidden states and final logits are byte-exact.
- All 32 C6/C7 request records and complete HBM payload fields are byte-exact.
- LRU admission, host-page/token capacities, transient candidate counters,
  stage-time sums and all 104960 graph replay deltas agree with the contract.
- Warmup trajectory/counter records agree; warmup tensors were not saved for
  a second independent numerical comparison.

The saved-data arithmetic audit independently reproduces all eight report
groups and 24 MFU stages from raw request rows and recorded precision-specific
useful FLOPs/reference peaks. It does not measure hardware instructions or
matrix API time. Evidence is under the run's `analysis/reaudit_report/` and
`analysis/independent_formal_audit/`, including `arithmetic_source.py` and
`report_arithmetic_crosscheck.json`. All measured requests are retained.

Combined validation before the trace passed 176 GPU checks with no skips or
failures. Global CPU regression passed 2764 tests, 947 explicit optional or
hardware skips and 58 subtests. See
[combined validation](deepseek_motivation_c7_hint_validation.md).

## Formal latency and MFU

Each row contains sixteen requests. MFU is total ideal compute time divided by
total measured stage time. Prefix, extend and E2E rows overlap and are never
added together.

| Scheme / visit | E2E ms | Admission ms | History ms | Candidate ms | Cleanup ms | E2E MFU % |
|---|---:|---:|---:|---:|---:|---:|
| HBM / first | 2320.410 | 4.759 | 2304.404 | 10.994 | 0.253 | 41.691 |
| HBM / revisit | 2328.411 | 4.791 | 2312.372 | 10.993 | 0.255 | 41.548 |
| ECHO / first | 2423.584 | 5.493 | 2403.383 | 13.283 | 1.426 | 39.916 |
| ECHO / revisit | 26.867 | 4.140 | 0.000 | 21.038 | 1.689 | 8.361 |
| Serial sparse / first | 2358.644 | 5.615 | 2339.745 | 11.832 | 1.452 | 41.015 |
| Serial sparse / revisit | 20.684 | 4.165 | 0.000 | 14.785 | 1.733 | 10.861 |
| Dense prefetch / first | 2375.511 | 5.722 | 2356.117 | 12.165 | 1.507 | 40.724 |
| Dense prefetch / revisit | 34.537 | 4.220 | 0.001 | 28.453 | 1.863 | 6.504 |

Relative to C6, ECHO first visits improve by 19.415 ms and dense revisits by
5.725 ms. HBM and serial means change little. The gaps above contemporaneous
HBM first visits are 103.174 ms (4.446%) for ECHO, 38.234 ms (1.648%) for serial
and 55.100 ms (2.375%) for dense. This combined trace does not isolate the
individual component contributions.

C7 has no matching matrix API profile yet. C6 dense cold-request API MFU
(54.14%) and revisit-candidate API MFU (30.74%) describe C6 only. Capped gather,
exact norm I/O and shared MLP quantization remain separate candidates. Final
four-scheme attribution and affected-experiment reruns precede publication.

## GPU observation boundary

All delegated CUDA work was held until formal completion. Monitor exec 69405
exited 0; attached `gpu_observations.jsonl` SHA-256 is
`455578864e6e317f8332a4c186f059863c1d34a0dd78f833455590966f8d7780`.
Its 100 samples include 83 with only expected PID1810918 and none with an
unexpected PID. Sampling begins 37.044 seconds before metadata start and ends
88.432 seconds after measured completion; intervals are 5.096–31.446 seconds.
This is discrete observation, not continuous proof between samples. Exact
timestamps and scope are in `gpu_observations_boundary.json`.
