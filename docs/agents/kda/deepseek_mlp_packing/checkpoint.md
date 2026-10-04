# Dense MLP packing source and prototype checkpoint

Status: source review, the granted correctness screen and the separately
granted eager benchmark completed. Sessions 56769 and 22744 exited 0. Extended
graph correctness session 81934 also exited 0; GPU 3 was explicitly released
after every run. Extended graph timing session 6560 also completed. Root
selected the combined candidate and authorized production integration; those
edits have passed focused CPU checks and await independent source review/GPU
acceptance. Prototype results do not establish integrated serving performance.

The official SM90 TMA output-stride chain is documented in [draft.md](draft.md).
The compiled CUDA path passed the descriptor, guard and byte-exactness checks
below for the prototype's stated domain.

Temporary harness:
`/tmp/deepseek_mlp_packing_probe_20261004/probe.py`, SHA-256
`f32c57290752a0ef5df2e2a3ea7273b6be58c23953e72028342afe0c301aba03`.
CPU `prepare_03.json` records 17 selected source hashes, pinned DeepGEMM commit
and tracked-diff identity, primary layout/alignment rules and
`cuda_initialized=false`. Ruff format/check passed; E402 is deliberately ignored
for temporary-script repository path setup. Earlier CPU prepare records refer
to earlier harness revisions and are not CUDA measurements.

The four variants are the production baseline, direct packed outputs with two
quantizations, shared quantization with separate outputs/cat, and both changes.
All retain separate gate/up official GEMMs and independent weights/scales;
down still calls the production FP8 linear with its own quantization. The
prototype only accepts aligned BF16 SM90 rank-two input with unit inner stride.
It rejects unsupported paths rather than hiding a contiguous output copy.

Completed screening compared both original quantizations, gate/up/SiLU/down
bytes, input and weight immutability, full owner storage and half offsets.
Packed guards verify that gate leaves up untouched and up leaves gate untouched,
plus leading/trailing rows. Actual eager outputs are retained across a later
call and input overwrite; intermediate/output allocations must not alias inputs
or weights. Non-default-stream predecessor/consumer and changed-input graph
replays are also included. The graph check retains owned result clones and
reports graph-private reserved memory; it does not time replay.

The independent reviewer found no blocker in the source/harness review,
including its final eager-lifetime/alias checks and failure JSON. The executed
command was:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/probe.py screen \
  --checkpoint /preset-models \
  --output /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json \
  > /tmp/deepseek_mlp_packing_probe_20261004/screen_01.stdout \
  2> /tmp/deepseek_mlp_packing_probe_20261004/screen_01.stderr
```

`screen_01.json` has status `exact`, SHA-256
`207295ea6a5db9e8dcb8b78a6e51335aece66fb70ff3b495b9c91222ec4845aa`.
The hardware was NVIDIA H200 SM90 with PyTorch 2.12.1+cu130 and CUDA 13.0.
Seventeen fixtures produced 51 exact comparisons: synthetic N64/K128 and
N192/K256 at Q1/7/65/129 with row padding 0 or 16, followed by actual checkpoint
layers 0, 1 and 2 at Q121/128/1024, N18432/K7168. The guards, unchanged inputs
and weights/scales, explicit allocation ownership, eager returned-output
lifetime and non-default-stream checks all passed.

The layer-0 `packed_shared` full-MLP graph checks passed three changed-input
replays each at Q128 and Q1024, with stable graph output identity and owned
clones surviving replay. Reported private reserved bytes were 23,068,672 and
136,314,880 respectively. These are component graph pool observations, not
whole-process HBM peaks or serving capacity bounds. All 17 selected source
hashes remained unchanged through the run and a separate post-exit rehash.
The stderr file is empty and `timings=[]`.

`installed_source_postscreen_audit.json` compares installed and local
DeepGEMM quantizer Python, SM90 1D2D kernel, common TMA copy and PTX TMA bytes;
all four match. Those installed-file hashes were collected after this screen,
not before it. The screen itself records the installed module path and the
selected local/FlashInfer identities; do not retroactively describe the later
comparison as its pre-run identity coverage.

The independent CPU record/source audit also passed:
`independent_screen_record_audit.json`. It rechecked the 17 source hashes,
fixture keys, 51 comparison flags, guarded owner size and graph scope against
the harness. It did not rerun CUDA or compare unsaved raw tensor payloads.

The original probe is retained unchanged. A separate benchmark driver is
CPU-prepared under the same temporary directory, following
[benchmark_plan.md](benchmark_plan.md). It will bind the accepted screen and
extend graph correctness to every mode/boundary/shape before timing. Eager
complete calls, borrowed-output replay, and an API including input copy and
owned output copy will be reported separately. Production attribution and
full-model acceptance remain pending.

## Exclusive eager benchmark

Root granted GPU 3 after the other CUDA users released their devices.
Session 22744, PID 1836919, exited 0. The executed command was:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/probe.py bench \
  --checkpoint /preset-models \
  --screen-result /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json \
  --profile --warmups 5 --repeats 25 \
  --output /tmp/deepseek_mlp_packing_probe_20261004/bench_01.json \
  > /tmp/deepseek_mlp_packing_probe_20261004/bench_01.stdout \
  2> /tmp/deepseek_mlp_packing_probe_20261004/bench_01.stderr
```

`bench_01.json` SHA-256 is
`75c434ee572ce56ccc4dfba40274acd36fe0f09fbcd0883eb98e86d86e4a416d`.
The same H200/PyTorch/CUDA environment measured actual checkpoint layers 0-2
with synthetic BF16 activation fixtures. It does not propagate model hidden
states. Nine layer/Q combinations, two complete boundaries, four modes and
25 samples give 1,800 host-wall completion samples. Five warmups precede each
mode; order rotates across repeats. The complete boundary includes candidate
validation, padding, allocation/views, official quantization, GEMMs, SiLU and
down when named, followed by device synchronization. Inspection and the 72
single-call diagnostic profiles are outside latency samples.

The following values are the median of the three per-layer medians, in ms;
the underlying per-layer samples remain in the run file. These are component
eager costs, not whole-request latency or measured operator MFU.

| Boundary | Q | Baseline | Packed only | Shared only | Packed + shared |
|---|---:|---:|---:|---:|---:|
| Gate/up/SiLU | 121 | 0.323106 | 0.287406 | 0.230779 | 0.230992 |
| Gate/up/SiLU | 128 | 0.323334 | 0.289694 | 0.231741 | 0.232152 |
| Gate/up/SiLU | 1024 | 0.661543 | 0.632477 | 0.658554 | 0.623318 |
| Full MLP | 121 | 0.481374 | 0.443605 | 0.384506 | 0.368513 |
| Full MLP | 128 | 0.481399 | 0.444913 | 0.384974 | 0.367658 |
| Full MLP | 1024 | 0.931724 | 0.903926 | 0.932689 | 0.895219 |

The combined candidate improves all nine full-MLP medians: 1.300-1.306x at
Q121, 1.308-1.312x at Q128 and 1.032-1.042x at Q1024. Sharing alone has no
full-MLP improvement at Q1024 in this run. Larger small-Q host-wall savings
must not be carried over to graph replay before its separate measurement.

`bench_01_analysis.json` checks every trace's kernel roles and complete GEMM
names against its baseline. All 72 match the expected sequence: helper
8/7/7/6 launches and full MLP 11/10/10/9 for the four modes. Packing removes
cat and sharing removes one activation quantizer. The two official gate/up
GEMMs and both official scale-transpose launches remain. No unexpected kernel
or GPU memcpy/memset event appears. Kernel work sums are diagnostic single-call
observations, not elapsed unions or independently repeated performance samples.

Peak allocated deltas for baseline versus the combined candidate are
22,302,720 versus 14,276,096 bytes at Q121, 23,855,104 versus 15,101,952 at
Q128, and 188,743,680 versus 120,815,616 at Q1024. The values agree across
layers and both boundaries. Sharing alone raises the observed peak allocated
delta. These probes did not measure reserved/device-used peaks; they cannot
establish actual HBM budget compliance.

`bench_01_source_before.json` records 898 original and installed DeepGEMM
Python, interface, binary and header identities immediately before timing.
`bench_01_source_after.json` confirms all were unchanged after exit, along with
the original probe's own source checks. Stderr contains profiler start/stop
diagnostics and no failure. This original probe did not time graph replay.

Root's independent audit, `independent_bench_audit.py` and its JSON result,
recomputed all 1,800 samples and 72 medians, checked all 72 trace kernel
counts/names/work sums, and rehashed the 898 recorded files. It passed. This
confirms the reported component record and source identity; it adds no graph
or full-serving performance evidence.

## Extended graph correctness

After the final CPU source review and the previous exclusive timing window
released CUDA, root's conditional correctness-only grant allowed GPU 3.
Session 81934 exited 0 with empty stderr. The command was:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/benchmark.py screen \
  --checkpoint /preset-models \
  --initial-screen /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json \
  --output /tmp/deepseek_mlp_packing_probe_20261004/graph_screen_01.json \
  > /tmp/deepseek_mlp_packing_probe_20261004/graph_screen_01.stdout \
  2> /tmp/deepseek_mlp_packing_probe_20261004/graph_screen_01.stderr
```

The driver SHA-256 is
`a3705277a6a35508a08da2b9a65c495134571d54273d75f3b27c8d6647c9a790`.
`graph_screen_01.json` is exact, with SHA-256
`2b5d1e2c94c7c6dd76c1c8009f603755a77fcfb5a0a4145192ff79fd479862d7`.
All four modes passed at both boundaries for checkpoint layers 0-2 and
Q121/128/1024: 18 groups, 72 independent graphs, 216 changed-input replays
with borrowed output and 216 with owned output. Every group compares eager,
borrowed and owned output bytes to the production baseline. Non-default-stream
predecessor/consumer ordering, unchanged caller/static inputs and weights,
independent graph pools, stable borrowed identity and retained owned outputs
passed. The original 17-fixture guard screen remains the descriptor/half-boundary
evidence; this extended run adds graph coverage.

All 899 recorded source/interface/binary/header hashes stayed unchanged through
the run and the separate `graph_screen_01_postexit_audit.json` recheck.
`timings=[]`. Its capture/setup observations and graph pool bytes are not
replay performance samples or full-model capacity evidence. Root's independent
`independent_graph_screen_audit.json` passed the exact 18 keys, 72 graphs,
216 changed inputs per API, all required flags and all 899 current hashes.
The separately granted timing run follows below.

## Complete graph benchmark and integration decision

Under root's exclusive GPU 3 grant, `benchmark.py bench --profile --warmups 5
--repeats 25` used the original and extended exact screens as gates and wrote
`graph_bench_01.json`; stdout/stderr use matching separate filenames. Session
6560, PID 1854420, exited 0. The result SHA-256 is
`3e6675ea2f04edd8d0b75213056e27d155d0a440ac8939499ac3d7a01dbab254`.
All 18 boundary cases and 216 API/mode entries completed: 5,400 wall samples,
5,400 separate CUDA-event samples, 432 immutable pre/post fixture checks and
216 diagnostic traces. All required numerical/ownership checks passed and all
899 source hashes remained unchanged through the run and post-exit audit.
GPU 3 was released before analysis. Stderr contains profiler diagnostics only.

Root's `independent_graph_bench_audit_root.py` and JSON independently verified
all 10,800 samples, 216 traces, 432 fixture phases and 899 hashes. Eager calls
include all preparation/allocation; borrowed replay excludes graph setup and
uses static input/output; owned graph API includes input copy, replay and output
allocation/clone. Graph setup and memory observations are reported separately.
Each API and wall/event series is sampled separately; do not subtract the
series or interpret their difference as a measured launch/copy cost.

Full-MLP CUDA-event values below are medians of three per-layer medians, in ms:

| Graph API | Q | Baseline | Packed only | Shared only | Packed + shared |
|---|---:|---:|---:|---:|---:|
| Borrowed output | 121 | 0.164352 | 0.159392 | 0.161472 | 0.156448 |
| Borrowed output | 128 | 0.163840 | 0.159296 | 0.161440 | 0.156928 |
| Borrowed output | 1024 | 0.867456 | 0.826368 | 0.859040 | 0.820288 |
| Owned output | 121 | 0.180992 | 0.176512 | 0.178912 | 0.173824 |
| Owned output | 128 | 0.177088 | 0.172320 | 0.174752 | 0.169696 |
| Owned output | 1024 | 0.871296 | 0.831584 | 0.863904 | 0.823648 |

The combined candidate improves every full-MLP graph case, including the host
completion metric. Borrowed event speedups are 1.041-1.051x at Q121,
1.044-1.048x at Q128 and 1.057-1.059x at Q1024. Owned event speedups are
1.035-1.043x, 1.041-1.044x and 1.056-1.058x respectively. Sharing adds a
smaller consistent graph improvement to packing, while retaining the larger
small-Q eager improvement. Root therefore selected `packed_shared` for the
narrow production integration in [integration_plan.md](integration_plan.md).
These component APIs do not establish whole-serving latency or measured MFU.

Before production edits, `baseline_source_mapping.json` verified that the
original FP8, echo model/block, nonmatrix and FlashInfer adapter hashes all
match immutable files under `/tmp/deepseek-motivation-c7_hint-frozen-a44l_2uk`.
The instrumentation file also matches that tree; its identity is labeled a
preintegration snapshot because the component did not use those scopes.
The original absolute paths in timing JSON now refer to edited live files;
use the mapping for original source reconstruction, not current-file hashes.

The first production CPU run passed 48 tests with 26 explicit CUDA skips in
2.58 seconds, covering FP8/model/block tests, prepared ownership and attribution.
Ruff passed on the changed owned files. The skips are unrun GPU cases, not GPU
passes. `independent_production_review.json` then passed 40 CPU metadata,
rejection, dispatch and owner-lifetime checks. It confirmed that packed and
prepared buffers are released before down, with all three matrix scopes intact.

Root took over the production GPU commands. Its first synthetic run
`production_gate_root_01` failed with 23 passes and 14 Dynamo recompile-limit
errors after independent shape/stride/grad fixtures filled the quantizer code
cache. The saved log contains 14 `Dynamo recompile limit exceeded` failures;
it does not show a numerical mismatch. Production source remained unchanged.
This failed engineering run is not correctness acceptance or a performance
result. Root separately reported that the combined four-scheme gate passed;
that evidence belongs to its combined-validation record.

Only test isolation changed: the linear and packed-MLP test modules now reset
the compiler cache at each case's setup/teardown, retaining all compilation
reuse inside each eager/stream/graph case. No production compilation limit or
algorithm changed. Focused CPU checks passed 11 tests with 26 CUDA skips in
1.98 seconds. `independent_recompile_failure_review.json` in
`/tmp/deepseek_mlp_packing_integration_20261004/` confirms that every test-body
AST is unchanged and reset occurs only around each case.

Root's fresh synthetic run `production_gate_root_02.json` passed 37 tests,
with zero failures and zero skips (PID 1886981, exec session 11984, exit 0).
Its SHA-256 is
`da059bc4e3846bbb48d642d9fd3c72eaf142aa3e0eecbcc3ec32819fe6fa95ae`.
The separate actual-checkpoint driver passed all nine source-layer/Q pairs:
layers 0-2 crossed with Q121/128/1024, N18432/K7168. It compared the complete
down-output bytes, retained earlier eager outputs, checked unchanged inputs
and weights, nondefault-stream execution, and 27 changed-input graph replays
with stable borrowed output identity and retained owned clones.

The checkpoint result `checkpoint_root_01.json` has SHA-256
`c52487de5be7fde1cedef2fff98c6fb811d5faff68be430857491c553682d153`
(exec session 31500, exit 0). Its driver `check_checkpoint.py` has SHA-256
`ae5e46f2ce7d215ddeadfc42c7c1b93ca395f8009613dcf6eccde55124013891`.
These files are under `/tmp/deepseek_mlp_packing_integration_20261004/`.
The test uses real checkpoint weights and synthetic activations; it does not
claim complete model propagation or performance.

The independent saved-record audit
`independent_production_records_cache_c3.json` passed the exact 37-test counts,
nine checkpoint keys, all required flags and 27 graph replays, and rehashed
893 source files with no changes. This is a CPU record/source audit, not a
second GPU run or a replay of unsaved tensor outputs. Root separately owns
combined four-scheme correctness and global regression evidence.

Component production correctness is complete. Integrated serving latency and
operator MFU are still unmeasured. The accepted C8 source is frozen at
`/tmp/deepseek-motivation-c8_torch_reference-frozen-rb0_2eqx`; its 483-file
mapping SHA-256 is
`cae2da8b9a58907ae35f99b5eeda017348b1a0a1ce1e1421fd6ca7703c956bb6`.
The subsequent native linear-quantization task has its own exactness and timing
gates; these compiled-quantizer results do not validate that replacement.
