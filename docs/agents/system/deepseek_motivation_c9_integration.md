# C9 Triton formal trajectory

Run `motivation_c9_triton_20261004_u16_r2_01` completed successfully.
It combines C7 with exact typed norm, packed/shared MLP computation and the
handwritten T1 quantizer. This trace does not isolate individual contributions.

Frozen root: `/tmp/deepseek-motivation-c9_triton-frozen-0496ye32`.
Executed-source SHA-256:
`b762e7502ba5c30e5c0106510e1ee2f8636a4179ceff1f15fbcc86e7a93a328e`.
The 1,590 data and two log files were relocated with matching hashes to
`experiments/deepseek_v32_motivation/output/` under the run ID above.
FLOPs analysis uses `motivation_c9_triton_flops_20261004_01`.

The workload remains H65536, A128, chunk1024, P65536, NH16777216,
sixteen users/two rounds, ten independent checkpoint dense-block copies,
all candidate hidden and last-token logits. Compute graphs are enabled;
production does not use `torch.compile`. Both PTXAS environment settings are
fixed as described in the [validation record](deepseek_motivation_c9_validation.md).
Hardware is PyTorch H200/SM90; nvidia-smi reports M403 for the same device.

| Scheme | First mean ms | Revisit mean ms | First E2E MFU % | Revisit E2E MFU % |
|---|---:|---:|---:|---:|
| hbm | 2203.859 | 2194.362 | 43.8960 | 44.0860 |
| echo | 2306.008 | 26.873 | 41.9516 | 8.3594 |
| serial_sparse | 2245.959 | 20.555 | 43.0732 | 10.9289 |
| dense_prefetch | 2250.105 | 34.022 | 42.9938 | 6.6027 |

Compared with the contemporaneous HBM control, first-visit gaps are 4.64%
for ECHO, 1.91% for serial sparse and 2.10% for dense prefetch. Compared with
C7, first-visit means decrease by 4.78%–5.28%; revisit changes are small.
The full objective remains incomplete. Matching C9 matrix API profiling has
passed its independent audits, with cold-request values 53.38%–56.30%.
C6/C7 operator numbers must not be presented as C9 measurements.

## Independent acceptance

All 128 saved payloads passed shape/dtype/finiteness and identity checks.
All 96 offload/HBM comparisons and all 32 C9/C7 HBM comparisons are byte-exact.
The twelve warmup records, measured LRU/capacity/transaction state, stage sums,
transient candidate writes and zero candidate D2H checks pass. Formal warmup
outputs are not saved; their audit concerns recorded lifecycle/counters.

Every request executes both captured compute islands with zero eager fallback:
104,960 measured replays, comprising 41,600 HBM and 21,120 per offload scheme.
Eight actual Triton specializations were observed and bound to declared/frozen
source and compiler identity. The independent runtime audit rehashed 190
source/dependency/artifact paths and checked retained CUBIN/loaded handles.
All 256 request memory snapshots satisfy allocated <= reserved <= device-used;
graph-private reserved storage is 5,771,362,304 bytes per scheme, within plan.
This trace does not fill NH or establish full-NH physical capacity.

The independent arithmetic check reproduces all eight report groups and
24 MFU stages. MFU aggregates precision-specific ideal compute time divided by
summed E2E wall time; overlapping prefix/extend/E2E stages are not summed.
Evidence is in the run's `analysis/independent_formal_audit/`:

| Record | SHA-256 |
|---|---|
| `audit.json` | `c3f4415174ebb6d125ec90c6a6b55374e6b18832bff82219c1aae48c2e873e91` |
| `runtime_graph_audit.json` | `230ab85cf3860f0fcf606891916d34d319e0af1143294c7f7fa7de8df88bf34d` |
| `report_arithmetic_crosscheck.json` | `553cfd1b64210a718356798ed6081ba2ff2b0efb803d5164c0914df4d8075060` |

The attached GPU monitor contains 102 discrete observations spanning run
startup through completion, with no unexpected GPU PID. These samples do not
continuously observe the intervals between them. Formal execution and monitor
sessions have exited. The new profile has a separate monitor and run ID.

Published experiment reports remain unchanged until the matching profile and
replacement report pass acceptance; old figures do not describe C9.
