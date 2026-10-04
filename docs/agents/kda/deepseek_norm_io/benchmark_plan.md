# Complete norm API benchmark plan

Date: 2026-10-04. Status: source review, the 64-case GPU driver check, complete-API timing and
independent result/trace audits passed. See the checkpoint for exact run IDs. Root grants those windows
separately. The passing full numerical matrices are in [checkpoint.md](checkpoint.md).

## Frozen inputs and scope

Temporary driver: `/tmp/deepseek_norm_adapter_20261004/benchmark.py`, SHA-256
`98daa59211e240cdb8869ab71c8340c353ae90085f431c9acb0039ffab4bd7dc`.
It imports the unchanged `probe.py`, `validation.py`, `local_plain.py` and
`local_fused.py`; no production or installed source is modified.

The driver verifies every source recorded by `plain_validate_02/result.json`
and `fused_validate_01/result.json`, then requires every benchmark subject to
exist in their passing matrices. Default subjects are source-0 plain input
norm D7168, query-A norm D1536, KV-A norm D512, and fused post-attention norm
D7168. Q128/1024, BF16, contiguous/row-strided layout, epsilon `1e-6` yield
16 fixtures. The KV-A row stride is 576. Optional Q1/9/121, FP32, offset and
other validated epsilons require corresponding passing validation keys.
The ordinary baseline requires matching input/residual dtypes; mixed-dtype
fused validation does not become an unsupported baseline timing comparison.

These are component fixtures with checkpoint weights. They do not establish
full-model numerical acceptance, serving latency/MFU or physical capacity.

## Boundaries and checks

- `eager`: actual `models.deepseek_v32.nonmatrix.rms_norm` or
  `residual_rms_norm` versus the validated local adapter. Include all allocation,
  dtype conversion, explicit packing and kernel work.
- `graph_owned`: copy caller inputs into independent static buffers preserving
  full layout, replay the captured complete API, then clone each output into
  fresh caller-owned storage. Include copies, replay and clones in both wall
  and event samples. Captured outputs remain private. Replay is bound to the
  graph's construction stream; complete lifecycle stream tests remain in the
  already completed validation suite.

Correctness checks precede and follow sampling on the actual callables. Preserve
immutable complete backing-storage guards for original/alternate inputs and
weights from before oracle creation or graph capture. Compare original and
changed-input outputs exactly against the ordinary model baseline. Retain the
actual first return through the changed-input call, and require all returned
allocations to be independent of callers, graph static inputs and captured
outputs. Fused normalized and saved outputs must be independently owned.

For graph calls, keep an immutable full static-storage snapshot from before
warmup/capture. Reconstruct each expected owner from that snapshot, update its
logical view exactly as the public copy does, and compare it after replay.
This catches guard corruption left by timed samples as well as checking calls.
Verify private-pool segment containment of every captured output. Baseline and
local graphs have different pools and allocations. Save both check phases and
audit the exact fixture/execution/variant key set. Measurement records additionally
audit every fixture, metric, repetition and variant key plus alternating order.

## Timing and memory

After the driver correctness run passes, a quiet root-assigned window may run
five warmups and 30 repetitions per variant by default. Alternate baseline/local
order each repetition. CPU wall samples cover full API plus current-stream
synchronization without CUDA timing events. Separate CUDA-event samples cover
the complete API enqueue window, including gaps and all enqueued work. Event
elapsed time is not labelled kernel-work time.

Exclude fixture creation, checkpoint loading, compilation, warmup/capture setup,
correctness checks, memory snapshots and profiler passes. Preserve every raw
sample. Optional separate Torch profiler runs retain the full Chrome trace,
its hash, actual kernel names/count and kernel-work sum. The sum is not an
elapsed union or whole-API latency.

Report complete returned owning storage, graph static-input capacity and
private reserved/active segments alongside PyTorch allocated/reserved and
device-used observations. Baseline/local graph objects and oracle/guard tensors
coexist, so these are contextual component observations rather than isolated
variant peaks or serving budget acceptance.

## Commands

Run CPU preparation from the repository root with a new output path:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python \
  /tmp/deepseek_norm_adapter_20261004/benchmark.py prepare \
  --checkpoint /preset-models --output /tmp/norm_benchmark_prepare_new
```

After independent source review and an explicit correctness grant:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES="$NORM_GPU" OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_norm_adapter_20261004/benchmark.py check \
  --checkpoint /preset-models --output /tmp/norm_benchmark_check_new
```

After record audit and a separate quiet timing grant, preserving the same
subject/execution arguments:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES="$NORM_GPU" OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_norm_adapter_20261004/benchmark.py measure \
  --checkpoint /preset-models \
  --check-record /tmp/norm_benchmark_check_new/result.json \
  --output /tmp/norm_benchmark_measure_new --profile
```

Save stdout/stderr separately next to the temporary output directory. A source
change invalidates the passing driver check and requires new preparation,
review and correctness execution before timing. Root decides promotion and
owns subsequent complete-checkpoint integration and serving measurements.
