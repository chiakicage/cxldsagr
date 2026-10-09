# Accepted component and full-model validation

Candidate `page64.pack_q1_keys` is promoted into `echo._pack_q1_keys`. It only
changes input adaptation; official MQA computation, score masking and exact
top-k remain unchanged. Current source is uncommitted. The precise pre-promotion
wrapper source and candidate bytes are archived in each run's source manifest.

Correctness: `/tmp/cxldsagr-checks/deepseek_q1_packing/packing_check_20261008_03/result.json`.
Ten N values cover byte patterns, partial-page zero padding and 30 changed-input
graph replays. Real L0–L2 scores and exact selections match bitwise, including
24 fixed graph replays and changed current K/scales/endpoints. The first check
failed before native top-k because PATH omitted ninja; the second completed
checks but could not serialize Triton's GPUTarget. The corrected third run is
the accepted receipt; neither prior run is an experiment result.

Clean benchmark: `q1_packing_bench_20261008_01`, GPU 0 / CPUs 0–7, H200 SM90,
N=65,537, 20 warmups, 7 samples, 100 invocations per sample. This resident API
does not mutate cache state. Each method includes fresh current-input packing.

| Layer | Packing copies us | Packing fused us | Full MQA copies us | Full MQA fused us |
| --- | ---: | ---: | ---: | ---: |
| L0 | 20.405 | 4.587 | 28.261 | 12.841 |
| L1 | 20.328 | 4.313 | 28.107 | 12.951 |
| L2 | 20.337 | 4.308 | 28.116 | 12.993 |

NCU: `q1_packing_ncu_20261008_01`, full/PM and source-counter profiles, personally
collected and parsed by the root agent. The baseline report covers only the
first full-page key copy; the fused report covers keys, scales and page padding.
At base clocks/cache flush, grid shrinks from 16,384 to 1,025 CTAs. The first
baseline copy takes 22.592 us with 62.26% SM and 7.75% DRAM-read throughput;
the complete fused pack takes 6.464 us with 4.93% SM and 27.97% DRAM-read
throughput. The baseline is dominated by its generic byte-copy/address work
and load stalls, rather than saturating memory bandwidth. Fused source stalls
map to key/scale loads; the prebuilt ATen copy has no resolvable source mapping.
PM sampling is sparse for the short fused kernel and cannot support a detailed
tail-shape claim. These profiler durations are not clean API latency.

Permanent verification after promotion: 20 GPU/CPU tests covering page64
delegation, byte preservation, real paged dispatch/masking/graph behavior and
decode EMA passed. Full-model matrix `deepseek_h64k_a1_official_20261008_03`
accepted independent check, clean timing, stage profiles and complete node/IO
audits with official ECHO, scale layout, projection scheduling, promotion and
score-layout changes integrated. The source-bound operator profile/report are
`deepseek_h64k_a1_official_mfu_profile_20261008_04` and
`deepseek_h64k_a1_official_mfu_report_20261008_04`. Packing measurements above
retain their own API boundary; they cannot be summed with other component
benefits to infer the measured whole-model improvement.
