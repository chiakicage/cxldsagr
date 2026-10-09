# Current production and open diagnosis

The current published cohort is `deepseek_h64k_a1_isolated_20261008_01`.
All 16 independent method/role processes and the complete CPU publication
audit passed. The selected report is `experiments/deepseek_v32_mfu/report/h64k_a1/`.
The model/kernel implementation remains the accepted exact-hint implementation;
each process now prepares only its selected method.

HBM L0–L2 span/busy/idle is 1013.090/995.137/17.953 us. The official reference
span is 1014.975 us with 14.274 us idle, under different input/residency/framework
and physical GPU conditions. This is a preparation/measurement boundary finding,
not an equivalent-workload speedup or a proved hardware mechanism.

Clean step medians are HBM 1.834721072, ECHO 2.588715986, serial sparse
2.395401942 and dense prefetch 5.532309995 ms. ECHO L0–L2 span/idle is
1426.563/47.492 us. ECHO still needs measured optimization; the overall goal
remains active. Historical controls below retain their original implementation
and measurement boundaries, and do not replace the current formal cohort.

## Prior component evidence

The Q1 consumer-scale candidate was promoted into `linear/fp8.py` only for BF16
Q1 and eight measured checkpoint matrix shapes. Public quantization stays
contiguous by default. Prepared gate/up uses one unchanged quantization. All
matrix arithmetic and the official DeepGEMM call remain unchanged.

Fresh component check `/tmp/cxldsagr-checks/q1_control_check_20261008_02` passed
48 dtype/stride/special-value cases against both default and compiled official
helper, eight complete linears, and32 changed-input graph replays. Independent
bench `q1_control_bench_20261008_01` and single-replay confirmation
`q1_control_bench_single_20261008_01` improve all eight paired medians. Prior
October6 K16384/N7168 regressions remain relevant: this cell's new single-replay
benefit is only0.032–0.144 us by order stratum, so no broad claim is made from it.

`q1_control_profile_20261008_01` independently verifies32 NVTX/CUDA-correlated
API windows. All baseline calls contain quantization/transpose/GEMM; all layout
calls contain quantization/GEMM. GEMM symbols match each corresponding shape.

Full private HBM check `/tmp/cxldsagr-checks/q1_control_model_check_20261008_02`
passed prefix final logits, complete extend hidden/logits, eager-versus-fullgraph,
cross-instance, and two changed-token replays bitwise. Models, caches and graphs
are independently allocated. All source/input signatures remained unchanged.
`q1_control_model_bench_20261008_01` retains20 balanced pairs. Medians are
2.0193485 ms baseline and2.0046615 ms layout; paired median delta−18.2835 us,
with13/20 negative deltas. One baseline sample has approximately149 ms extra
delay; no sample was discarded. Wall measurements are noisy and support only a
small directionally consistent benefit. Both graphs reserve56,623,104 bytes.

Production integration passed88 focused GPU linear/quantizer tests, including
18 new Q1 bitwise/changed-graph/default-dispatch/prepared-lifetime tests.

The prior production matrix `deepseek_h64k_a1_official_20261008_03` passed independent
correctness, cold clean timing, stage/node profiling and source/native/input
audits. Its HBM/echo/serial/dense step medians are 1.866080/2.725594/2.507448/
5.519253 ms. Operator profile `deepseek_h64k_a1_official_mfu_profile_20261008_04`
and report `deepseek_h64k_a1_official_mfu_report_20261008_04` bind the same check
and benchmark. Eight saved profile outputs are bitwise equal to check outputs.
Full CPU regression passed 4,761 tests and 58 subtests; 1,524 GPU/optional cases
were skipped by the CPU entry point, not counted as GPU acceptance.

The final Q1 graph also includes the separately checked projection branch
schedule and physical score-padding view. Projection private model A/B
`q1_projection_model_bench_20261008_01` improved the paired median by 81.1595 us
(16/20 wins); its node profile did not observe cross-stream compute overlap.
Do not attribute the scheduling gain to proven compute overlap. Final graph
private reserved is 62,914,560 bytes per method, up from 56,623,104 in the older
schedule. The local HBM L0-L2 window is 1.073347 ms versus the official reference
1.014975 ms. Different inputs/frameworks remain unisolated; the optimization
goal is still open, especially preparation/control/top-k and node idle time.

Initial component check hit Dynamo's8-recompile limit; the fixed matrix now
declares128 variants before compiling the independent oracle. Initial private
model check lacked `ninja` in PATH; a fresh ID with the venv PATH completed.
Neither failed attempt is a valid result or published performance comparison.

## QKV rejection and current follow-up

Q-A/KV-A/index-K fusion was rejected after clean full-model measurements.
The final ownership-clean version passed 66 projection cases and 1,584
field comparisons, but its 100-pair full-graph event median regressed from
1.476992 to 1.478768 ms. With matching layer events, L0-L2 regressed from
1.046224 to 1.049264 ms and full graph from 1.483744 to 1.486496 ms. Favorable
invasive profiles do not override those clean measurements. No QKV fusion
patch was applied. Rejected timing/candidate trees, prototype entry points
and the unapplied integration mirror were deleted; the file inventory and
former result hashes are in `cleanup_cub_20261008_01.json`. `q1_qkv.py` remains
only because accepted selection experiments import and fingerprint helpers.
Two old profile pairs remain temporarily as inputs to the active gap diagnosis.

The newer accepted CUB/validation cohort is
`deepseek_h64k_a1_cub_20261008_01`; its complete stage and operator reports
were independently audited and published. HBM/echo/serial/dense synchronized
step medians are 1.868587/2.725153/2.482657/5.523886 ms. The local HBM profile
window is 1.067459 ms, busy 1.000931 ms and idle 0.066528 ms. Controls for
profiler state during construction and for the capture inspector do not
reproduce the long idle. The zero-versus-two internal external-event control
also does not reproduce it: the zero-event arm has 19.584/19.968 us idle,
and both arms have 96 ns median gaps. Its complete 197-node GPU signature
matches current formal production, with 192 native layer-owned nodes.
The evidence is `q1_event_boundary_audit_20261008_02` and
`event_boundary_result.md`; clean paired timing shows no event-insertion gain.

The first-five-replay control and same-graph outside-timer control are also
negative. Their accepted evidence is `q1_replay_boundary_audit_20261008_01`
and `q1_replay_timer_audit_20261008_01`; results are described in
`replay_boundary_result.md` and `replay_timer_result.md`. The latter uses
exactly the formal CUDA/NVTX settings and leaves about 16-17 us idle.

The unchanged complete pipeline has now reproduced the long gap in
`deepseek_h64k_a1_cub_gap_reproduce_20261008_01`: measured HBM L0-L2
span 1066.817 us, busy 999.137 us, idle 67.680 us and median gap 416 ns.
Native 197-node signatures/ownership and all-process clipped interval unions
match the original formal trace. Independent source/runtime/receipt/output
checks passed; full details and launch-environment boundaries are in
`formal_reproduction_result.md`. The initial compute-graph inspection control is also negative: its measured
idle is 67.584 us with 416 ns median gaps. `compute_inspector_result.md`
records native/source/runtime/output verification and confirms that 294
GetNodes and all 2,301 NodeGetType inspection calls were removed. Initial
compute-bank construction was then moved outside collection in another
complete-pipeline control. That also leaves 67.648 us idle and 416 ns median
gaps; see `compute_collection_result.md`. Its source/runtime/output and
empty-first-range audits passed. The cause remains unresolved. Reduced
controls have additional GPU/JIT-cache/binary differences; do not treat
their matching node signatures as fully matched execution environments.
The optimization goal remains open. Projection overlap in the final graph
was observed by the later matching-node audit; the earlier no-overlap statement
above refers only to its component capture.

The retained graph-type inventory also shows no extra empty/event nodes in the
formal HBM capture snapshot. Original and reproduced formal HBM graphs contain
181 kernel, 15 memcpy and one memset node, with zero nodes of types 5, 6 or 7.
Reduced no-event, first-five-replay and outside-timer controls have the same
197-node counts; the event/inspector controls add exactly two type-7 nodes.
This counts every entry in `node_types`, not only `gpu_node_ids`. The collector
enumerates types 0/1/2/5/6/7 using `cudaGraphGetNodes`; it saves the last snapshot
inside the captured scope. No complete edge DAG or independent post-instantiation
enumeration is available. These counts do not explain the idle difference.
See [node-type inventory](graph_node_inventory_result.md) and the bounded CPU
result `/tmp/cxldsagr-checks/bounded_graph_node_inventory_20261008_01.json`.
