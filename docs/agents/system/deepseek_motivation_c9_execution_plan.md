# C9 formal trajectory and matching four-scheme profile

Status: formal trajectory and matching four-scheme profile completed and
independently audited, 2026-10-04. All 128 formal payloads passed; 96
offload/HBM and 32 C9/C7 HBM comparisons were byte-exact. All eighty profile
payloads, source/runtime identities, graph attribution and aggregate
arithmetic also passed. The accepted findings and evidence index are in the
[C9 profile note](deepseek_motivation_c9_profile.md).
Root owns GPU scheduling, source freeze,
artifact relocation and publication. This plan does not authorize this audit
agent to launch CUDA work.

## Frozen source and prerequisites

Use `/tmp/deepseek-motivation-c9_triton-frozen-0496ye32` for every execution and
production analysis command below. Root's 497-file freeze map is
`b65d577b59fa7c57735e079449815af25554df1507eb5c16653340042c6e0de0`.
The CPU snapshot precheck at `/tmp/deepseek-c9-snapshot-precheck-ovhwktx7`
checked 1,268 paths and produced executed-source identity
`b762e7502ba5c30e5c0106510e1ee2f8636a4179ceff1f15fbcc86e7a93a328e`.
Later working-tree changes must not enter this run or its matching profile.

The combined gate is `/tmp/deepseek_c9_combined_validation_20261004_01`.
Its independent audit `independent_audit_graph_profile.json`, SHA-256
`cd06c26ff8b7b3f051575a8f59973fbeb0f6ae8b94dc8fecb18f99c58be7ffca`,
checked 176 passing tests, all 528 setup/call/teardown outcomes, unchanged
before/after identities, 1,598 source/dependency/artifact files and 24 actual
Triton specializations. Four schemes each reported graph-private reserved
storage of 5,771,362,304 bytes. The later rename from
`linear/tests/test_quantization.py` to `test_linear_quantization.py` was
verified byte-preserving and is explicitly recorded in this audit.

The public/small-checkpoint and dense/grouped/packed-consumer gates are
separate evidence owned by their execution agents. Root also reports the
global CPU result: 2,850 passed, 1,022 explicit optional/hardware skips and
58 subtests. Those checks do not supply formal latency or matrix API MFU.

Both CLI help entrypoints were checked from the frozen tree with CUDA hidden.
No production files were edited during this preparation. The packed MLP still
calls separate `gate` and `up` CheckpointLinear instances; the existing
instrumentation forwards `out=` and `quantized=` and retains their separate
matrix-work identities. Shared quantization is charged to the call which
actually performs it. `silu_mul_packed` is already instrumented as nonmatrix
work. Do not reuse old physical node counts: the new Triton quantizer changes
the actual captured node membership even when useful FLOPs are unchanged.

## Environment and formal command

Pin the exact strings below for both formal and profile processes, before any
backend import. The SM90 quantizer uses Triton's bundled PTXAS, resolved SHA-256
`c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.
The Blackwell PTXAS variable is preseeded because FlashInfer otherwise adds it
after the initial identity snapshot; its presence is not a Blackwell execution
claim. Keep the fixed strings across processes, including equivalent path
aliases, because the declared identity records them verbatim.

```bash
cd /tmp/deepseek-motivation-c9_triton-frozen-0496ye32
export PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/bin:$PATH
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export CXLDSAGR_SM90_BACKEND=native
export TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_motivation/scripts/run.sh \
  --run-id motivation_c9_triton_20261004_u16_r2_01 \
  --model-path /preset-models --device cuda:0 \
  --num-users 16 --rounds 2 --history-tokens 65536 --candidate-tokens 128 \
  --chunk-size 1024 --sparse-pool-tokens 65536 --host-arena-tokens 16777216 \
  --compute-graphs --seed 42
```

Root must hold other GPU work for the complete measurement and record a
process monitor spanning startup through completion. Keep monitor intervals,
expected PIDs and start/end timestamps; discrete samples are not continuous
proof of isolation. Record both PyTorch's H200 name and nvidia-smi's M403 name
without retroactively relabeling earlier results.

The script writes under the frozen tree's experiment `output/data/` and
`output/log/`. On failure, its temporary data/log diagnostics stay outside the
experiment delivery tree. A passed run must retain its source snapshot and all
payloads. If root relocates it to the shared repository, verify every copied
file against the original hash before analysis. No command should overwrite an
existing run ID. The matching profile below assumes that verified relocation
has completed.

## Formal acceptance and CPU analysis

Require all four schemes, 32 requests per scheme, and user order
`0..15,0..15`. Each scheme warms requests `0,1,16`, releases warmup state and
starts the measured trajectory from an empty user pool. HBM retains one history
under P and misses on every measured request; offload schemes retain sixteen
histories and hit on all sixteen revisits. NH is not filled by this trajectory.

Check all 128 complete candidate hidden/logit payloads for shape, dtype,
finiteness, token identity and saved file hash. Require all 96 offload/HBM pairs
to be byte-exact. Independently compare all 32 C9 HBM payloads and request
records with accepted C7 HBM; this is a cross-version numerical check, not a
claim that C7 timings describe C9. Formal warmup outputs are not persisted,
so their separate audit covers recorded trajectory/counters only.

Check stage sums, LRU state, host pages and token quotas, retained H, candidate
GPU-transient state, zero candidate D2H writeback and candidate-only transfer
counters. Graph policy remains `deepseek-compute-islands-v2`; every measured
request must have zero eager fallbacks and complete before/after replay deltas
for both islands at every independent layer. The fixed geometry implies
104,960 measured island replays: HBM 41,600 and each offload scheme 21,120.
The H64K correctness gate's intentional A121 fallback must not appear here.

Keep allocated, reserved and device memory separate. Per-request peaks are
cumulative within each scheme; device free is sampled at boundaries. Neither
the graph-reserved figure nor this sixteen-user trajectory proves full-NH
capacity. Do not subtract the old observed allocator gap as a fixed reserve.

The runtime collector already saves actual quantizer metadata under
`flashinfer_runtime_artifacts.linear_quantization_triton`; despite the parent
field's historical name, this is a distinct local Triton artifact entry.
Audit its captured build identity against declared source/compiler identity,
rehash metadata-group PTX/CUBIN files, verify retained CUBIN hashes and loaded
handle flags, and inspect actual no-FTZ/NaN/div.full/SATFINITE PTX. Metadata such
as `enable_reflect_ftz` alone does not establish arithmetic FTZ behavior.

After verified relocation, keep the same frozen working directory and run:

```bash
export C9_BASE=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_motivation/output/data
export C9_DATA=$C9_BASE/motivation_c9_triton_20261004_u16_r2_01
export C9_FLOPS=$C9_BASE/motivation_c9_triton_flops_20261004_01
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.report \
  --run-dir "$C9_DATA" --output-dir "$C9_DATA/analysis/reaudit_report"
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.flops \
  --run-dir "$C9_DATA" --output-dir "$C9_FLOPS" --model-path /preset-models \
  --peak-report /tmp/deepseek-motivation-c9_triton-frozen-0496ye32/experiments/deepseek_v32_echo_prefill/report/layers3/summary.json
```

The FLOPs tool reads checkpoint headers, not weight arrays. It requires the
recorded H200 identity and dense reference peaks FP8 1,979, BF16 989.5 and FP32
67 TFLOP/s. Preserve actual precision evidence and the no-TF32 policy. Compute
aggregate MFU as the sum of precision-specific ideal times divided by the sum
of measured stage times, retaining all sixteen samples in each group. Prefix,
extend and E2E are overlapping views and must not be added together.

The independent formal audits have passed and are saved under
`experiments/deepseek_v32_motivation/output/data/motivation_c9_triton_20261004_u16_r2_01/analysis/independent_formal_audit/`,
with their exact source and input hashes. `audit.json` verifies all 128
payloads, 96 offload/HBM pairs, 32 C9/C7 HBM pairs, 104,960 actual island
replays, LRU admission, transient candidates and stage sums. Its SHA-256 is
`c3f4415174ebb6d125ec90c6a6b55374e6b18832bff82219c1aae48c2e873e91`.

`runtime_graph_audit.json` verifies eight actual Triton specializations,
190 recorded source/dependency/artifact files, graph plan/allocation bounds
and all 256 request memory snapshots. Its SHA-256 is
`230ab85cf3860f0fcf606891916d34d319e0af1143294c7f7fa7de8df88bf34d`.
`report_arithmetic_crosscheck.json` independently reconciles all eight report
groups and 24 formal MFU stages; its SHA-256 is
`553cfd1b64210a718356798ed6081ba2ff2b0efb803d5164c0914df4d8075060`.
The accepted C7 audit sources and results were left unchanged.

`gpu_observation_audit.json` verifies 102 saved process observations spanning
startup through completion, with no unexpected PID. The largest sample gap
is 27.202 seconds, so these remain discrete observations. Its SHA-256 is
`297dc6c497313f28771b2a339d50d06395e967bd4fa478fef0013ac3dc1a7372`.

## Matching four-scheme graph-node profile

After formal acceptance and a new exclusive GPU grant, from the same frozen
root and pinned environment:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_motivation/scripts/profile.sh \
  --run-id motivation_c9_triton_profile_20261004_01 \
  --reference-run "$C9_DATA" --device cuda:0
```

Omit `--scheme`: all four are required. `profile.sh` supplies `--nsys`, CUDA,
NVTX and OS-runtime tracing, `--cuda-graph-trace=node`, CUDA-profiler capture
boundaries and SQLite exports. It stages outputs outside the experiment and
publishes only after process/output/export checks succeed.

Each scheme repeats three warmups, then independently executes requests
0 through 16 so the revisit occurs after all first-round users. Only request
0 and request 16 are profiled, plus graph setup. Expected captures are:

| Scheme | Setup SQLite | Cold SQLite | Revisit SQLite |
|---|---|---|---|
| hbm | capture_1.sqlite | capture_2.sqlite | capture_3.sqlite |
| echo | capture_4.sqlite | capture_5.sqlite | capture_6.sqlite |
| serial_sparse | capture_7.sqlite | capture_8.sqlite | capture_9.sqlite |
| dense_prefetch | capture_10.sqlite | capture_11.sqlite | capture_12.sqlite |

Require 12 Nsight captures: four setup and eight request captures. Require 80
saved byte-exact output payloads: twelve warmups plus 68 trajectory requests,
all independently compared with matching formal C9 HBM. The profiled requests
contain 6,560 island replays: 2,600 HBM and 1,320 per offload scheme. Expected
physical nodes and matrix calls must come from current templates and expanded
work, not prior C3/C6 constants.

`load_reference` rejects changed model/cache/operator/serving/executor/vendor
source. The profile manifest intentionally differs from the formal manifest
because it adds analysis/instrumentation sources. Preserve both identities and
all setup captures: request traces alone cannot establish clone lineage.

After verified profile relocation, run this CPU chain from the frozen root:

```bash
export C9_PROFILE=$C9_BASE/motivation_c9_triton_profile_20261004_01
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.verify_profile_flops \
  --profile-run "$C9_PROFILE" --flops "$C9_FLOPS/flops.json"
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.analyze_pipeline \
  --sqlite "$C9_PROFILE/capture_2.sqlite" --sqlite "$C9_PROFILE/capture_3.sqlite" \
  --sqlite "$C9_PROFILE/capture_5.sqlite" --sqlite "$C9_PROFILE/capture_6.sqlite" \
  --sqlite "$C9_PROFILE/capture_8.sqlite" --sqlite "$C9_PROFILE/capture_9.sqlite" \
  --sqlite "$C9_PROFILE/capture_11.sqlite" --sqlite "$C9_PROFILE/capture_12.sqlite" \
  --output-dir "$C9_PROFILE/analysis"
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.operator_mfu \
  --profile-run "$C9_PROFILE" --flops "$C9_FLOPS/flops.json" \
  --output-dir "$C9_PROFILE/analysis/operator_mfu"
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.aggregate_mfu \
  --operator-dir "$C9_PROFILE/analysis/operator_mfu" --flops-dir "$C9_FLOPS" \
  --output-dir "$C9_PROFILE/analysis/aggregate_mfu"
```

The independent C9 audit adapters are retained at
`/tmp/deepseek_c9_profile_numerical_audit_graph_profile.py` and
`/tmp/deepseek_c9_profile_attribution_audit_graph_profile.py`. They accept
explicit profile/formal/FLOPs paths, cover four schemes and eighty payloads,
and derive physical node and matrix counts from the new captures. Both ran
successfully after capture completion and production analysis; their sources,
input hashes and passing results are saved with the run. The checks require
no missing/duplicate/unknown nodes per replay,
one traced process across setup/replay lineage, actual executable graph IDs,
actual independent layer/weight identities, complete eager-launch correlation
and all memcpy/memset/kernel activities. Node-only copy records are valid only
through their complete process/launch/node-lineage evidence.

Verify useful FLOPs and API counts by scheme, phase, segment, chunk, independent
layer, operator and precision. MLA splitting may increase API count only while
conserving work. Recompute every matrix invocation's primary-kernel time,
owned-helper GPU union, sum and span, and every aggregate. Include quantization,
scale transforms, preparation and repair in the owning API boundary. Keep
unowned graph auxiliary work, eager nonmatrix work and graph input copies
visible without double-counting matrix-owned helpers.

The profile samples one cold and one revisit request per scheme. Its instrumented
wall time does not replace the formal 16-sample means. Matrix API MFU uses
actual owned GPU intervals and reference-peak useful work, not hardware
instructions or SM occupancy. Full-kernel ECHO windows cannot prove internal
prefetch/compute overlap. If transport overlap is reported, independently
intersect actual copy and matrix intervals by segment, chunk and target layer;
zero-copy ratios are undefined. CPU API/GPU interval intersection measures
exposure, not causality or wholly removable CPU time.

## Publication boundary

Keep all currently valid README/report/output evidence until C9 formal output,
source/runtime provenance, matrix attribution and arithmetic audits pass and a
reviewable replacement report exists. Label C9's combined change honestly:
C7 plus exact norm/packed-shared MLP and direct Triton quantization. Formal C9
versus C7 does not isolate the quantizer alone. Do not reuse C6/C7 API MFU as
C9 data. Component comparisons used shared explicit Triton PTXAS and do not
retroactively redefine historical default-Torch-PTXAS timing. Shared-module
effects on other experiments still need their own rerun/publication scope.
