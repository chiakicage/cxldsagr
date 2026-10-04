# C8 common norm and MLP integration

Status: component integration and combined production correctness passed;
serving measurement has not run. The user then requested avoiding
`torch.compile`, so the formal run is deferred while the remaining linear
activation quantizer is replaced through KDA. The user's latest instruction
selects handwritten Triton; the earlier CUDA C++ candidates were not promoted.
See the [current quantizer plan](../kda/deepseek_linear_quantization/triton_implementation_plan.md).
The MFU goal remains active.
C7a plus the exact hint remains the latest accepted formal source; see
[C7 integration](deepseek_motivation_c7_hint_integration.md).

The accepted C8 integration is frozen at
`/tmp/deepseek-motivation-c8_torch_reference-frozen-rb0_2eqx`: 483 copied files,
freeze-map SHA-256
`cae2da8b9a58907ae35f99b5eeda017348b1a0a1ce1e1421fd6ca7703c956bb6`.
It is a correctness/component reference, not a new formal latency result.

Production norm passed 71 focused GPU tests and four live IR/source checks.
Its temporary wrapper failed only while reading the wrong runtime report key
after those checks; corrected independent CPU audits accepted the saved data
without rerunning CUDA. MLP passed 37 focused GPU tests after isolating compiler
state between independent cases, plus nine real-checkpoint layer/Q cases with
exact full-down outputs, streams and changed-input graph replays. The first
heterogeneous test run exhausted Dynamo's recompile limit; no production limit
or algorithm changed for the test rerun.

The combined four-scheme H64K gate passed 176 tests in 92.39 seconds, with all
119 recorded source paths and both native/norm identities unchanged. Actual
graph-private reserved storage was 5,771,362,304 bytes for every scheme,
within the existing reservation. This is not a full-NH capacity or process-peak
measurement. Global CPU regression passed 2778 tests with 1005 explicit skips
and 58 subtests; 100 additional motivation/runtime-report tests passed. The
later compiler-isolation test-only edit passed its focused CPU and GPU suites.

## Selected components

- Norm: reuse the official CuTe FP32 constructor/reduction with local typed
  input/output kernels. Plain and fused adapters passed 3120/2880 matrix cases,
  36/48 lifetime cases and independent source/key audits. Complete-API run
  `benchmark_measure_01` passed 64 before/after check records, 32 comparison
  groups, 3840 wall/event samples and 64 separate traces. Both wall and event
  medians improve in all 32 measured comparisons. Initial production dispatch
  covers the measured BF16 domain; other existing paths retain their behavior.
- MLP: select direct packed gate/up outputs plus one call-local quantization.
  Keep independent original weights, two official GEMMs and unchanged down
  projection. `graph_bench_01` passed 18 layer/shape/boundary cases, 10800
  wall/event samples, 432 immutable-fixture checks and 216 traces. Combined
  full-MLP graph event medians improve roughly 4%–6%; eager small-Q gains are
  larger. Independent audits rechecked 899 source/interface files. These are
  component results, not additive predictions of request speedup.

Norm artifacts are under `/tmp/deepseek_norm_adapter_20261004/`; MLP artifacts
are under `/tmp/deepseek_mlp_packing_probe_20261004/`. Preserve immutable
original source locations before editing live files. The C7_hint frozen tree
provides the baseline where recorded hashes match.

C7b gather is separate. Two 1500-sample sweeps confirm only about 2% benefit
for cap264 under the current gather-first component schedule. Lower caps and
serial/gather-only regressions are retained. The 42-case diagnostic profile completed; CPU interval attribution is pending.
No generic transport or dense-serving integration is selected in this plan.

## Ownership and implementation

`graph_validate` owns `operators/deepseek_v32/norm/`, norm tests and
`models/deepseek_v32/nonmatrix.py`. `cache_c3` owns FP8/CheckpointLinear/MLP
integration and profiler passthrough. Coordinate any packed SiLU helper through
the owner of `nonmatrix.py`; avoid simultaneous edits to that file.

The MLP API must preserve both CheckpointLinear scopes and attribute shared
quantization exactly once. Prepared activation state is explicit and local to
one MLP call. No weight merging, object-level activation cache, hidden padding
copy or borrowed-output behavior is introduced into the default linear API.
Validate explicit output views, caller/input/weight ownership and fallbacks.

Norm retains unrounded FP32 residual arithmetic and separately rounded saved
output. Local JIT identity covers its source and actual vendor/include/DSL
dependencies. Importing CPU/reference paths must not initialize CUDA or load
native extensions. Keep vendor and installed files unchanged.

## Required production gates

1. Review component source deltas and run focused CPU tests. Add meaningful
   GPU checks for exact vendor FP32/BF16 results, real MLP weights, fallback
   dispatch, output/storage ownership, changed inputs and nondefault streams.
2. Validate all four real-checkpoint schemes from independent empty caches at
   H=P65536, C1024, A121 eager/A128 graph, with displaced retained history.
   Compare all candidate hidden/logits and prefix outputs, independent copied
   weights/state, transient counters and borrowed/owned graph lifetimes.
3. Recheck graph planning against actual private-pool reserved bytes and real
   static allocations. Do not assume component memory deltas equal model peak
   or reduce a reservation solely from allocated-byte observations.
4. Preserve matrix API scopes and graph-node attribution. Run the appropriate
   instrumentation checks; shared quantization cannot disappear from totals
   or be counted under both gate and up. Run the global CPU regression after
   the combined source is settled.
5. Freeze the accepted production source, verify new dependency snapshot
   coverage, then run a quiet four-scheme 16-user/two-round formal trajectory.
   Check every saved output against contemporaneous HBM and the accepted C7
   request/input/payload baseline. Reconcile stage times, LRU, capacities,
   graph counts, source identity and precision-specific MFU independently.
6. Obtain matching four-scheme graph/matrix API profiling and review remaining
   nonmatrix/fetch/launch work. Check affected dependent experiments before
   publication; preserve existing reports and backing outputs until replacement
   evidence is accepted and published in the same update.

GPU correctness and timing windows are scheduled separately. No unrelated
CUDA process may run during a performance window. Poll existing sessions to
completion rather than restarting on an observation timeout.
