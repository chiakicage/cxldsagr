# C10 matching profile: accepted analysis and remaining observed costs

The user closed optimization scope at C10. This checkpoint records final
analysis of `motivation_c10_profile_20261004_01`, paired with accepted formal
run `motivation_c10_20261004_u16_r2_01` and FLOPs analysis
`motivation_c10_flops_20261004_01`. T3/C11 was deferred and restored out of
production; see [its integration checkpoint](../kda/deepseek_topk_wrapper/t3_integration_checkpoint.md).

Root reported profile exit 0 and verified relocation before analysis. The
analysis waited for the separate official formal/repeat timing window to end.
The full sequential chain then exited 0 with CUDA hidden and two CPU threads.
All eleven stage stderr files were empty. Production analysis used frozen C10
source under `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0/`.

The chain completed pipeline attribution, FLOPs verification, per-operator
MFU and aggregate MFU, followed by seven independent checks. Accepted evidence:

- All 80 saved candidate hidden/logit payloads match formal HBM bytes: twelve
  warmup, four cold, sixty preparation and four revisit outputs. The numerical
  audit remained on CPU with CUDA uninitialized.
- All 1,273 profile source files were rehashed; 1,238 execution-source files
  match formal C10. Eight observed Triton specializations, three native
  FlashInfer libraries, 380 norm source/dependency records, native validation
  source/compiler/binary identities and graph allocation bounds passed.
- Independent raw SQLite traversal verified 45,933 matrix calls and primary
  kernels, 267,562 GPU activities, 6,560 graph replays, 180,320 graph activities
  and 4,360 clone edges. Every matrix interval and all 195 operator groups
  agree with production analysis. Each of the eight requests has exactly one
  instance-bound native-validation call and one contained raw NVTX interval.
- Independent arithmetic agrees for all 21 aggregate groups and 195 operator
  stage groups. All 369 matrix-kernel inventory rows agree with raw activities.
  Bottleneck interval unions and actual graph copy-kernel rows were retained.
- Graph_validate separately accepted 162 all-device query samples containing
  158 process entries, all for the expected profile PID and device. Four
  unavailable command strings occurred after completion. The maximum sample
  gap was 28.836934 seconds; discrete sampling does not prove continuous isolation.

Profile source identity is
`c7c2f40915ce41897bd9eabd1dde1efb0dfcfcc528afaea039ef9ac03e57623c`;
formal source identity is
`11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f`.
All schemes recorded 5,603,590,144 bytes of graph-private reserved memory in
this profile. This is an observed allocator reservation, distinct from tensor
storage, static allocated bytes, device-used memory and the chosen plan bound.

The table compares complete-request useful work against precision-specific
dense peaks. Formal MFU divides ideal compute time by the mean uninstrumented
request time across sixteen first visits or sixteen revisits. API MFU divides
the same work by summed per-call owned GPU-activity unions in one corresponding
profile capture. HBM revisits rebuild evicted history; offload revisits retain
history. The two timing populations and denominators are kept separate.

| Scheme | Visit | Formal mean ms | Formal MFU % | Profile matrix-API MFU % |
|---|---|---:|---:|---:|
| HBM | first | 2176.886 | 44.440 | 53.383 |
| HBM | revisit | 2170.244 | 44.576 | 52.720 |
| ECHO | first | 2283.433 | 42.366 | 56.182 |
| ECHO | revisit | 24.947 | 9.005 | 17.488 |
| Serial sparse | first | 2217.737 | 43.621 | 53.892 |
| Serial sparse | revisit | 18.456 | 12.172 | 30.391 |
| Dense prefetch | first | 2233.838 | 43.307 | 53.902 |
| Dense prefetch | revisit | 31.889 | 7.044 | 30.676 |

Remaining observed costs are diagnostics, with no new optimization plan:

- First-visit exact-top-k scope GPU unions were 189.943–192.798 ms across
  the four captures. Offload pool-operation scope unions were 65.380–70.137 ms.
  These categories have their own scope boundaries and must not be added to
  nested API/category unions.
- On retained-history revisits, ECHO cache argsort occupied 1.299 ms of GPU
  activity, compared with 0.648 ms for serial sparse and 0.634 ms for dense
  prefetch. Separate host-gather kernels occupied 0.356, 1.560 and 14.723 ms,
  respectively. ECHO's additional fused indexer/fetch work is inside its
  matrix-owned API; the separate gather figure is not total ECHO transfer time.
- The three offload revisit captures had GPU span-minus-busy gaps of 20.715,
  21.440 and 12.262 ms, respectively. These instrumented gaps do not identify
  their cause or the cost removable from formal requests. Formal revisit
  admission means were 2.175–2.208 ms and cleanup means were 1.636–1.668 ms;
  these host intervals can include synchronization waits.
- Actual unowned graph copy kernels remain: 2,600 per first-visit capture
  (19.913–20.640 ms summed GPU duration) and forty per offload revisit
  (about 0.101 ms). No captured-graph CUPTI MEMCPY records were observed;
  that absence does not imply the graph performs no copying.

The retained data root is
`experiments/deepseek_v32_motivation/output/data/motivation_c10_profile_20261004_01/`.
Under its `analysis/`, `independent_profile_audit/completion_receipt.json`
binds the production outputs, all eight independent audit results including
the separate monitor, eleven log pairs and copied execution sources. Its
SHA256 is `cc30286b462b21f1ac542155e871cd2f3843131e6a324914b67d9c89b313a829`.
The raw-SQL `operator_mfu/crosscheck.json` SHA256 is
`bc14c30de3b0f42c3f388c476b2c93b70d4a478a1b3335e3c58e0a9903af2758`.

Formal wall latency and instrumented matrix-API activity come from separate
runs. Per-call GPU unions may overlap across calls; their sum is not a full
request GPU union. Neither their difference from formal latency nor a profile
gap is an isolated CPU cost or guaranteed removable latency. Fused ECHO
internal fetch overlap was not measured. Root and Graph_validate own report
publication and replacement; this checkpoint records analysis acceptance.
