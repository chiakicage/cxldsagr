# Dense MLP packing complete-API benchmark plan

Status: this extended driver passed CPU review and its separate 72-graph CUDA
correctness screen. The original probe's eager timing also completed under an
exclusive grant. Extended-driver graph timing also completed; run IDs and evidence
are in [checkpoint.md](checkpoint.md). This plan records the measured prototype;
production integration has a separate [plan](integration_plan.md).
The original `probe.py` and its accepted `screen_01.json` remain immutable.
A separate `benchmark.py` imports the screened computation and adds measurement
orchestration; it does not modify production functions or weights.

## Matrix and gates

Use actual checkpoint layers 0, 1 and 2, independently loaded gate/up/down
weights, and Q121/128/1024. Compare `baseline`, `packed_only`, `shared_only` and
`packed_shared` for complete gate/up/SiLU and full MLP. Both GEMMs preserve
their original M/N/K, recipe, kernel family and independent weight/scales.

Before CUDA initialization, require the initial exact result, all 17 fixture
records, the 51 variant checks, unchanged source identity and matching checkpoint
path. Record the new driver and installed DeepGEMM library/header identity too.
Do not bypass the gate if another component changes a hashed production file.
Such a change requires an explicit refreshed correctness screen.

The driver's separate `screen` captures all four modes at both boundaries for
all nine checkpoint-layer/query combinations: 72 distinct graphs. For each,
change the input three times, compare borrowed and owned results to current
production baseline bytes, retain the owned results across later replays, and
check that static inputs and graph outputs own distinct storage. Run the
input predecessor, copy, replay and output consumer on a non-default stream;
join its completion before inspection. Inputs and independent weights/scales
must remain unchanged. Graph private pools must be independent across modes.
Timing requires this driver's exact screen with identical full identity,
the exact 18 subject/Q/boundary keys and all four modes' required pass flags.

## Measurement boundaries

1. **Eager complete API:** actual `stage` or `full_mlp`, from host entry through
   device completion. Include validation, padding checks, allocation and views,
   activation quantization, both GEMMs, SiLU and down where named. No packing,
   quantization or candidate-specific allocation is hoisted outside the call.
2. **Graph replay with borrowed output:** replay a previously captured complete
   helper/MLP and wait for completion. Static input was populated before timing.
   Capture includes all of that helper's kernels, but Python validation, setup,
   allocations and capture cost occur during setup and are reported separately.
   This boundary measures graph computation and does not provide owned output.
3. **Graph API with owned output:** copy a caller input into static input, replay,
   clone the output, and wait for completion. Both device copies and output
   allocation count. This is a controlled ownership boundary, not a claim that
   the serving finish graph performs either copy for each MLP invocation.

Measure host completion and separately CUDA-event elapsed time for each API.
Keep event instrumentation out of host-wall samples. Prewarm every mode and
boundary, then rotate mode order across repeats. Use five warmups and 25
repeats by default; record all samples. No concurrent CUDA work is allowed
during timing. Fixture loading, capture, correctness, memory snapshots and
profiling remain outside latency samples. Before each API's samples and after
its samples/profiles, run the actual timing callables against the baseline on
the original, changed and restored fixture. Check caller/static input
preservation, output identity and owned results surviving later calls. Store
these pre/post checks in the timing record; null marks inapplicable ownership
properties for borrowed outputs or eager inputs. Both checks share the same
immutable input snapshot and baseline output captured before graph setup and
sampling; the post-check must not accept a newly snapshotted mutated input.

For each API report allocated/reserved/device-used snapshots and peak allocated
delta, with live output bytes. For graphs also report static input bytes,
capture setup seconds and allocator segments belonging to their private pool.
Concurrent graph objects are retained for the four controls, so global process
memory is context, not an isolated per-variant footprint. Do not infer actual
HBM budget compliance from allocated alone or add observations as fixed reserves.

Optional profiler traces encompass exactly one API call plus synchronization.
Store kernel names/counts/work sum, memcpy/memset separately, and elapsed kernel
envelope. Work sums and envelopes have different definitions. Inspect packed
paths for absence of cat and unexplained packing/output-copy kernels, while
recognizing that the owned-output graph boundary intentionally adds copies.
Do not label any component result full serving performance or measured MFU.

## Eventual production attribution

Preserve both `CheckpointLinear.__call__` invocations and their individual
shape/FLOP records. A narrow optional `out` plus call-local prepared-activation
carrier can let gate create the official quantized pair inside the gate API,
then let up consume that exact pair inside the up API. The carrier must bind
the original tensor identity/layout and have one-call lifetime; no object cache,
global activation memoization or cross-request reuse is allowed. Down remains
unchanged. Unsupported shapes/backends retain the current path.

The current instrumentation wrapper accepts only `(instance, x)`. If optional
arguments are introduced, every relevant wrapper must forward them, and tests
must show three dense MLP API records, unchanged individual GEMM work counts,
one gate/up quantization charged to gate, and independent down quantization.
Packed allocation/SiLU cost remains inside the surrounding complete MLP scope.
An alternative paired API requires explicit profiler and report changes before
integration; silently bypassing `CheckpointLinear` is unacceptable. The current
temporary prototype does not implement these production scopes, so its timing
cannot establish final instrumented API cost or measured operator MFU.

## Commands

`benchmark.py` current SHA-256 is
`a3705277a6a35508a08da2b9a65c495134571d54273d75f3b27c8d6647c9a790`.
Ruff format/check passed. `benchmark_prepare_04.json` records the driver,
original candidate and installed DeepGEMM Python/interface/binary/header
identities, the full 72-graph matrix, and `cuda_initialized=false`. The initial
independent source review found no correctness-screen blocker and requested
stronger accepted-record checks and actual timing-fixture validation. Those
changes are implemented. Final delta review passed at the current hash;
`independent_benchmark_source_review.json` verifies prepare 04, all 899 source
hashes and the immutable pre/post fixture guard. This is CPU review only.
Earlier prepares belong to earlier revisions and do not prove the current
CUDA path.

All paths are engineering artifacts under
`/tmp/deepseek_mlp_packing_probe_20261004/`. CPU prepare, then independently
review the new driver. Only root's later grant selects the physical CUDA device
and permits its `screen` or `bench` command. Every output filename is new.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  /tmp/deepseek_mlp_packing_probe_20261004/benchmark.py prepare \
  --checkpoint /preset-models \
  --initial-screen /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json \
  --output /tmp/deepseek_mlp_packing_probe_20261004/benchmark_prepare_04.json
```

That prepare command already completed; use a new output path for any rerun.
The later commands, only after the relevant review and root grant, are:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/benchmark.py screen \
  --checkpoint /preset-models \
  --initial-screen /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json \
  --output /tmp/deepseek_mlp_packing_probe_20261004/graph_screen_01.json

PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/benchmark.py bench \
  --checkpoint /preset-models \
  --initial-screen /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json \
  --graph-screen /tmp/deepseek_mlp_packing_probe_20261004/graph_screen_01.json \
  --profile --warmups 5 --repeats 25 \
  --output /tmp/deepseek_mlp_packing_probe_20261004/graph_bench_01.json
```
