# C9 profile: bottlenecks, interval boundaries and evidence

Status: formal trajectory and matching four-scheme profile independently
accepted on 2026-10-04. This is an internal evidence note; the coordinated
motivation and official-report replacement remains separate.

Cold history top-k costs 180–190 ms in the sampled captures. ECHO adds
61.079 ms of history `resident_selection_kernel`, while activation
quantization costs 36.784–37.424 ms per complete cold request. Captured graph
copy kernels add 48.353–50.039 ms per cold request. These observations support
examining selection, metadata and graph auxiliary work alongside matrix APIs;
they do not support attributing the entire remaining MFU gap to SFA or
activation quantization, or treating any kernel-duration sum as an assured
end-to-end saving.

## Workload and acceptance

The workload has H=65,536, A=128, chunk=1,024, P=65,536,
NH=16,777,216, sixteen users and two rounds. It uses ten independent dense
blocks copied from checkpoint source layers 0–2, with embedding, final norm,
all candidate hidden outputs and the last-token LM head. Each copied block
receives its corresponding source block's hidden/residual inputs. This is the
7,827,793,408-parameter checkpoint workload surrogate, not a trained ten-layer
DeepSeek model or a full 61-layer validation. Runtime reports H200 through
PyTorch and M403 through nvidia-smi; execution is on SM90.

All run paths below are relative to
`experiments/deepseek_v32_motivation/output/data/`:

| Role | Run ID |
|---|---|
| Formal | `motivation_c9_triton_20261004_u16_r2_01` |
| FLOPs | `motivation_c9_triton_flops_20261004_01` |
| Profile | `motivation_c9_triton_profile_20261004_01` |
| Earlier numerical reference | `motivation_c7_hint_20261004_u16_r2_01` |

Formal execution source SHA-256 is
`b762e7502ba5c30e5c0106510e1ee2f8636a4179ceff1f15fbcc86e7a93a328e`;
profile source SHA-256 is
`13e87328311d3b7ce2df2cde230f1e612f61c5c2502ebe39dad811d3b19f2996`.
The profile manifest additionally contains instrumentation and analysis.
Independent rehashing checked 1,271 profile snapshots and confirmed equality
of all 1,236 execution-source files with formal. Both used the same eight
actual Triton specializations and 190 verified runtime source/artifact files.
Frozen source, pinned compiler environment and commands are recorded in the
[execution plan](deepseek_motivation_c9_execution_plan.md).

Formal acceptance covers all 128 output payloads, 96 byte-exact offload/HBM
pairs, 32 byte-exact C9/C7 HBM pairs, 104,960 island replays, zero eager
fallbacks, request stage sums, LRU/capacity and candidate lifecycle. It also
checks graph bounds and all 256 request memory snapshots. Each scheme has
5,771,362,304 bytes of graph-private reserved storage. Allocated, reserved and
device usage remain distinct; sixteen resident histories do not fill NH.

The profile saves eighty byte-exact payloads: twelve warmup, four cold, sixty
unprofiled prepare and four revisit. Its twelve SQLite captures include four
graph setups and eight requests. Independent raw-SQL attribution reconciles
45,933 matrix calls/primary kernels, 127,816 matrix-owned GPU activities,
274,122 total GPU activities, 6,560 replays, 186,880 graph GPU activities and
4,520 clone edges. Every replay has complete layer/query/part coverage and no
missing, duplicate or unknown nodes. Arithmetic checks cover 21 aggregate
groups, 195 operator-stage groups and all 369 kernel-inventory rows.

## Formal latency and sampled GPU intervals

Formal values below are means of sixteen requests per scheme/phase, in ms.
HBM misses and rebuilds H on every measured visit; offload revisits retain H.
Candidate-only extend and whole-request latency therefore describe different
amounts of work.

| Formal metric | HBM | ECHO | Serial sparse | Dense prefetch |
|---|---:|---:|---:|---:|
| First E2E | 2203.859 | 2306.008 | 2245.959 | 2250.105 |
| Revisit E2E | 2194.362 | 26.873 | 20.555 | 34.022 |
| Revisit extend | 10.674 | 21.087 | 14.613 | 28.134 |
| Revisit admission | 4.599 | 4.147 | 4.295 | 4.209 |
| Revisit cleanup | 0.250 | 1.639 | 1.645 | 1.678 |

The following intervals come from one captured cold and one captured revisit
per scheme. GPU span is earliest activity start to latest activity end;
busy is the union of all GPU activities; gap is span minus busy. Unowned
means outside the attributed matrix API boundary, including graph auxiliary
work, selection, metadata and transfers. Values are ms.

| Scheme / phase | GPU span | Busy union | Gap | Unowned union |
|---|---:|---:|---:|---:|
| HBM cold | 2198.793 | 2151.618 | 47.176 | 339.423 |
| HBM revisit | 2209.951 | 2168.251 | 41.700 | 340.323 |
| ECHO cold | 2482.124 | 2120.905 | 361.220 | 402.597 |
| ECHO revisit | 36.114 | 16.741 | 19.374 | 3.841 |
| Serial cold | 2260.036 | 2192.595 | 67.441 | 402.654 |
| Serial revisit | 26.735 | 11.839 | 14.896 | 4.459 |
| Dense cold | 2259.011 | 2190.464 | 68.547 | 403.715 |
| Dense revisit | 35.398 | 24.855 | 10.542 | 17.541 |

These sampled gaps identify exposed intervals; they do not attribute a cause
or measure removable CPU time. In particular, ECHO cold has a 361.220 ms gap
in its instrumented sample, while formal E2E is a separate sixteen-request
mean. Subtracting profile API time from formal wall time cannot isolate CPU
overhead. Category intervals can overlap or contain each other.

Matrix API active time is the sum of per-call owned GPU unions, including
quantization, scale transforms, preparation and repair. It is not the global
matrix union. MFU sums useful precision-specific ideal times before division,
using reference peaks FP8 1,979, BF16 989.5 and FP32 67 TFLOP/s, with no
matmul TF32. API MFU is neither SM occupancy nor hardware-instruction MFU.

| Scheme | Cold API active ms / MFU | Cold formal E2E MFU | Revisit API active ms / MFU | Revisit formal E2E MFU |
|---|---:|---:|---:|---:|
| HBM | 1812.394 / 53.38% | 43.90% | 1828.116 / 52.92% | 44.09% |
| ECHO | 1718.323 / 56.30% | 41.95% | 12.899 / 17.41% | 8.36% |
| Serial sparse | 1789.957 / 54.05% | 43.07% | 7.380 / 30.44% | 10.93% |
| Dense prefetch | 1786.764 / 54.14% | 42.99% | 7.315 / 30.71% | 6.60% |

## Cold selection, quantization and copies

The top-k rows sum the three actual FlashInfer kernel families
`FilteredTopKUnifiedKernel`, `StableSortTopKByValueKernel` and
`FinalizeTopKIndicesKernel` in the history segment only. For HBM their separate
costs are 108.164, 46.115 and 35.852 ms. The activation-quantization and copy
rows cover the complete cold request, including its candidate. All entries
are GPU duration sums in ms, not additional terms to add to the interval table.

| Work | HBM | ECHO | Serial sparse | Dense prefetch |
|---|---:|---:|---:|---:|
| History top-k families | 190.132 | 180.543 | 188.065 | 187.785 |
| History resident selection | — | 61.079 | 64.355 | 64.303 |
| Activation quantization | 37.424 | 36.784 | 37.065 | 37.063 |
| Graph-input MEMCPY | 7.619 | 7.541 | 7.607 | 7.607 |
| Unowned graph `direct_copy_kernel` | 50.039 | 48.353 | 49.627 | 49.576 |

History resident selection has 640 kernels per offload cold capture. Each
complete cold request has 5,200 activation-quantization kernels; these are
already charged to their owning matrix APIs. Each offload revisit has eighty
such kernels, costing 0.194 ms. HBM revisit rebuilds history and has 5,200
kernels, costing 37.472 ms. The cold top-k and resident-selection observations
are materially larger than activation quantization. This profile does not
identify a quantization-only mechanism for removing those selection costs.

There are zero CUPTI MEMCPY records inside captured graph nodes, but each cold
request has 3,900 unowned graph nodes whose actual kernel names contain
`direct_copy_kernel`. Each offload revisit has sixty, costing
0.182/0.182/0.181 ms for ECHO/serial/dense. Separate graph-input MEMCPYs cost
0.026/0.027/0.032 ms on those revisits. The copy-kernel audit uses actual
KERNEL records; it does not label arbitrary elementwise kernels as copies.

## Revisit ECHO and serial sparse

ECHO's fused indexer API takes 8.274 ms at 8.402% API MFU in the sampled
revisit; serial's resident indexer takes 2.735 ms at 25.418%. MLA is similar
at 1.216 and 1.218 ms. Separate host-gather work costs 0.354 ms for ECHO,
1.560 ms for serial and 14.720 ms for dense. The fused ECHO API includes
inseparable fetch/reduction work, so its whole kernel window cannot prove
internal transport/compute overlap.

ECHO's cache argsort makes twenty calls and launches 440 GPU activities,
with 1.295 ms GPU duration sum and 2.809 ms CPU exclusive scope time. Serial
makes ten calls/220 activities, with 0.646 ms GPU sum and 1.466 ms CPU
exclusive time; dense has ten calls/220 activities and 0.634 ms GPU sum.
These observations support examining repeated ordering work, subject to exact
cache and validation semantics. They are not an estimate of the end-to-end
benefit of removing a call.

ECHO revisit exact-top-k scope totals 0.678 ms of GPU work. Its three hint
update components total only 0.084 ms: mask count 0.019, Torch reduction 0.051
and publish 0.014 ms. They are too small to account for the main observed
ECHO/serial difference.

Formal offload revisit cleanup averages 1.639–1.678 ms. In the captured
cleanup scopes, `prefix_pool_audit` runs twice and consumes
1.971/2.000/1.946 ms CPU exclusive time for ECHO/serial/dense, with zero GPU
activities in that scope. `backend_truncate` consumes approximately
0.009–0.010 ms, also with zero GPU activities. CPU scopes may include waits;
these sampled times must not be added to GPU intervals or subtracted from
formal cleanup means. At the end of the C9 audit, root selected an isolated
CPU strict-validation prototype for the next cache investigation; this note
does not promote a production implementation change.

## Evidence index and preservation boundary

`F` below means the formal run's `analysis/independent_formal_audit/`;
`P` means the profile run's `analysis/`. Each audit retains its source and
input hashes. Paths in this table refer to ignored run artifacts and are
intentionally plain text.

| Artifact | Independent check | SHA-256 |
|---|---|---|
| `F/audit.json` | Payloads, cross-version equality, request lifecycle and replays | `c3f4415174ebb6d125ec90c6a6b55374e6b18832bff82219c1aae48c2e873e91` |
| `F/runtime_graph_audit.json` | Actual Triton artifacts, graph reservation and memory | `230ab85cf3860f0fcf606891916d34d319e0af1143294c7f7fa7de8df88bf34d` |
| `F/report_arithmetic_crosscheck.json` | Eight formal groups and 24 MFU stages | `553cfd1b64210a718356798ed6081ba2ff2b0efb803d5164c0914df4d8075060` |
| `F/gpu_observation_audit.json` | Formal monitor boundaries and PIDs | `297dc6c497313f28771b2a339d50d06395e967bd4fa478fef0013ac3dc1a7372` |
| `P/independent_numerical_audit.json` | Eighty complete profile payloads | `b223ebc430f6ba6f057e3328703c205117d8eb1dbe0063a8644c5aa8c46484b3` |
| `P/independent_provenance_audit.json` | Profile/formal execution identity and runtime artifacts | `5c5b2b6c3d2b5d813ca56a0d1a5518d691abb7905b0c35c3c0358d625b34e699` |
| `P/operator_mfu/crosscheck.json` | Raw-SQL calls, graph lineage, nodes and GPU intervals | `70ac19a475d745922e1da2b96751e85647c88e42f4e85ba7a759b4b67a10a38d` |
| `P/independent_profile_audit/aggregate_arithmetic.json` | Aggregate and operator-stage arithmetic | `1aa47838f4ff6b8fd51c16ed027a7a122b23279791676097684eb7a8a474f491` |
| `P/independent_profile_audit/bottleneck_summary.json` | Intervals, nonmatrix work, cleanup and monitor | `f35224234b548fcf94d63de83f4c7465a76bd36f65fd137a78181414831b4b11` |
| `P/independent_profile_audit/graph_copy_kernels.json` | Actual unowned graph copy kernels | `12962adba649556c35eb52d4f71bacf3e7b23c16dc45d43519d9dac7ef07f871` |
| `P/independent_profile_audit/kernel_inventory_audit.json` | All 369 kernel groups, including activation quantization | `c0578c36451bc4744f9863162ae2b03ffaefc3bde0de0f26af32211c548ccc5f` |

Detailed raw-derived activity rows are in
`P/independent_gpu_activities.jsonl`; selection family sums are in
`P/independent_nonmatrix.csv`. Primary report tables are under
`P/operator_mfu/` and `P/aggregate_mfu/`; CPU scopes are in
`P/stage_costs.csv`. Cold/revisit captures are 2/3 for HBM, 5/6 for ECHO,
8/9 for serial and 11/12 for dense. Captures 1/4/7/10 provide graph setup
and clone-lineage evidence.

The formal monitor contains 102 observations with only expected PID 1996655
and maximum gap 27.202 s. The profile monitor contains 200 observations with
only expected PID 2005465 and maximum gap 30.027 s; both span recorded run
start/end. These are discrete observations, not continuous exclusion proof.
The single read-only py-spy dump in `unprofiled_prepare_stack.txt` maps
`main (profile.py:756)` to an unprofiled prepare request, outside warmup and
captured requests. Its saved mtime does not measure dump duration. The later
official GPU correctness gate started after the complete profile window.

No old report or backing run was deleted or rebound by this audit. The
existing official comparison pins old motivation report hashes, so its
motivation source run, published report and logs remain held until the
coordinated replacements pass. The reviewed publication plan and cleanup
manifest are in `/tmp/deepseek_motivation_c9_publication_20261004/`.
C9 combines earlier compute-graph work with exact norm/packed shared MLP and
direct Triton quantization; its comparison with C7 does not isolate any one
component. Subsequent implementation changes require their own affected
correctness, formal and profile acceptance before replacing this evidence.
