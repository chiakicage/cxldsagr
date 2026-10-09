# Q1 finite-mean baseline: measured bottleneck

The unchanged mean-plus-EMA API now has independent acceptance, clean timing
and two completed NCU collections. The official ECHO kernel and decode EMA
were not modified. This result motivates one private finite-mean fusion; it
does not accept that candidate or change the published full-model numbers.

## Identity and clean boundary

Check `q1_hint_baseline_check_20261008_03` covers six real-layer eager/graph
comparisons, all 16 offset bytes and unchanged input bytes. Its receipt is
under `/tmp/cxldsagr-checks/q1-hint/`. The launcher SHA is
`08b57c51de8f2a18c9d7a7c9810559ac3c7472013fb61c6f9f1ef38ccdb51b21`;
the unchanged baseline driver SHA is
`c9553ab4244f8be823ae5a59644eec0b66ea7ebf59b0e48bb7d7e7f7fb16ddbf`.
The separate clean run `q1_hint_baseline_bench_20261008_03` has 20 warmups
and 100 samples per layer/execution: graph medians 16.128/15.904/16.256 us,
eager medians 73.312/72.608/72.864 us. Each sample restores offsets and
synchronizes before starting its events. These measure the whole finite-mean
plus unchanged EMA API; they are not isolated sum or model latency.

The L0 full/source runs are `q1_hint_baseline_full_20261008_03` and
`q1_hint_baseline_source_20261008_03`. Both completed their original strict
receipt gates and end-of-run source/runtime checks. NCU 2026.1.1 selects one
L0 sum with NVTX, uses kernel replay, flushes caches, and controls base clocks.
Reports and logs remain in this experiment's normal output categories.
The six unavailable CTC metrics in the full collection are retained as warnings.
The two failed launcher attempts are diagnostic evidence under `/tmp`, not
experiment results; see [identity resolution](q1_hint_ncu_identity_resolution.md).

## NCU evidence and limits

Root opened both reports with `ncu_report`, checked launch geometry and
read dynamic absolute-PC SASS. The CPU export is
`experiments/deepseek_v32_echo_official/output/data/q1_hint_ncu_analysis_20261008_01/`.
It retains scalar/instanced metrics, rules, PM series, source lookup results,
absolute function instructions and the separate relative-lookup diagnostic.

| Dimension | Recorded evidence | Interpretation |
| --- | --- | --- |
| Grid and occupancy | `launch__grid_size=1`, block 512, 32 registers/thread; `sm__warps_active.avg.pct_of_peak_sustained_active=24.73584254%` | Only one block can occupy one SM. The active-cycle occupancy is not device-wide utilization. |
| Load balance | One block; no variable-length per-block work distribution | Cross-SM distribution is limited by grid size, not demonstrated imbalance among multiple active blocks. |
| Issue and stalls | `smsp__issue_active.avg.pct_of_peak_sustained_active=9.98813608%`; long-scoreboard per-issue ratio 31.81851257 | Few issue opportunities; waits for load results dominate this cold-cache kernel. |
| Tensor work | Scalar FP32 reduction with vector loads, additions and shuffles | Tensor cores are not relevant to this exact reduction contract. |
| Timeline | PM series retained; full/source each have only 11 PC samples, 10 at one load-dependent FADD | Sparse samples locate a dependency, but do not establish a detailed tail model or a reliable sampled time percentage. |
| Memory | DRAM read 324,864 B, 0.39289872% of peak; L1 hit 0%, L2 hit 50.84671018%; zero local spills in details | This collection does not saturate device DRAM bandwidth. Cache flushing changes the load path relative to clean repeated calls. |

The NCU sum duration is 17.216 us at the reported 1.50 GHz SM frequency,
with caches flushed on replay. It must not replace the clean complete-API
medians or the prior formal sum activity times (8.288/8.416/8.576 us).
NCU's percentage speedup suggestions are diagnostic heuristics, not measured
speedups or a claim that an exact reduction can use every SM.

## Actual sum path and candidate constraint

The source report has 3,767 executed-counter PCs, 190 nonzero rows and a
summed warp-instruction count of 9,049. Absolute-PC instruction data show
one `LDG.E.128` and four accumulator `FADD`s each executed 512 warp times
(16 warps times 32 loop iterations), the tail scalar load/add executed by one
warp, followed by three four-way-combine additions, shared reduction and five
shuffle steps. No source lines are correlated in the installed wheel.
The inspected installed Torch headers supply index/order detail; actual
numerical equivalence still requires independent GPU comparisons.

NCU's relative-PC lookup returned a different instruction stream for many
addresses: the export retains 3,768 absolute function instructions and 4,016
relative diagnostic instructions. Only absolute dynamic PCs were used to
identify the executed path. The relative stream is not proof of this kernel's
execution.

The first private candidate fuses finite masking/counting and publication
into that single-block sum tree. It preserves per-accumulator input order,
shared offsets 256/128/64/32, shuffle offsets 16/8/4/2/1, exact integer finite
count and round-to-nearest FP32 division. The separate official decode EMA
stays unchanged. This can remove two launches and masked scratch traffic;
additional mask/count instructions may offset the benefit. No alternative
reassociation, threshold policy change or candidate promotion is authorized
by this profile. The next gates are bytewise edge/stream/graph acceptance and
balanced complete-API timing under the [existing plan](q1_hint_plan.md).
