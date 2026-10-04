# Direct MLA rotary destination: production validation

The main MLA projection now supplies the final Q tail to FlashInfer's existing
RoPE kernel. The Q packing allocation and final Q-tail copy are absent from
this source path. The kernel math, FP32 trig table, latent BMM, owned rotated
K, KV concatenation and indexer rotary path remain unchanged. No GPU activity
trace was collected here to count the removed launches.

The production change is in `operators/flashinfer.py` and
`models/deepseek_v32/echo_model.py`. The model enables the destination adapter
only for CUDA main Q with shape `[tokens,128,64]`. The adapter requires the
same-device tensor metadata, an independent Q destination, interleaved D64,
16-byte pointer alignment and vector-aligned strides. Its direct output strides
must also describe non-overlapping token/head rows. Other layouts retain the
ordinary adapter followed by the destination copy. CPU reference and default
`rotary_pair` calls retain their previous paths.

`cache_c3` independently reviewed destination aliasing, disjoint latent/tail
writes, fallback completion before copy suppression, unchanged indexer calls
and owned K lifetime. It found no blocker for this selected model path.

## Accepted checks

CPU checks passed **32 tests** with six existing CUDA cases deliberately
deselected. They cover destination dispatch, alignment fallback, malformed
metadata, alias rejection, overlapping outputs, K ownership and existing
model/reference behavior. Ruff check and formatting passed. The new dispatch
tests are in `operators/tests/test_flashinfer_rotary.py`.

Physical GPU 0 was granted exclusively by root. The production focused gate
ran the complete `models/deepseek_v32/tests/test_echo_rotary.py` module:
**26 passed, no skips**, one dependency warning. This includes 24 GPU cases
and two CPU reference cases. The GPU cases cover the previous YaRN checks,
sixteen aligned BF16/FP16 destination cases at Q1/Q121/Q128/Q1024 and offsets
0/8, and two changed-input graph/stream/owned-key cases. Byte comparisons
preserve signed-zero distinctions. Unaligned offset 3 is outside the vendor
vector contract and is not an accepted GPU fixture.

The complete production projection comparison then passed all six tensors
(`q`, `kv`, `index_q`, `index_k`, `index_weights`, `index_scale`) for real
checkpoint source layers 0–2 at Q121/Q128/Q1024. It used both direct method
calls and full projection-island callbacks, including residual normalization
and saved residuals. There were **27 paired graph-step comparisons**, of which
**18 changed inputs and positions**, for **54 total graph replays** across the
two versions. Positions include 65,536, 131,072 and the last legal start for
each query count. Inputs remain unchanged and owned output copies survive
subsequent calls/replays.

The inputs are controlled checkpoint embeddings with deterministic changes;
source layers do not receive a propagated complete-model hidden trajectory.
This is checkpoint-weight projection acceptance, not full-model or serving
acceptance. The graph outputs are borrowed. Retention checks explicitly clone
the outputs; timing excludes those ownership copies.

## Complete callback timing

All nine correctness cases completed before timing. Every source-layer/Q/mode
group used five warmups and thirty randomized baseline/candidate pairs, for
720 retained samples. Eager timing includes residual normalization, positions,
the complete allocating `project_positions` API and its returned outputs.
Graph timing includes static hidden/residual copies, start-scalar update and
full projection-island replay. Capture and compilation are outside timing.
Wall timing includes the event bracket and completion wait. GPU values are
CUDA event elapsed times, including exposed launch gaps; they are not sums of
kernel durations.

The following values are GPU event medians in microseconds. The last column
is the median saving within randomized pairs, not subtraction of medians.

| Source layer | Queries | Frozen C9 graph | Production graph | Paired saving |
|---|---:|---:|---:|---:|
| 0 | 128 | 174.544 | 166.832 | 7.648 |
| 1 | 128 | 178.736 | 171.072 | 7.680 |
| 2 | 128 | 179.424 | 172.016 | 7.744 |
| 0 | 1024 | 424.512 | 383.888 | 40.512 |
| 1 | 1024 | 441.232 | 400.928 | 40.288 |
| 2 | 1024 | 440.912 | 400.736 | 39.920 |

Separate eager medians decrease in all six groups, but the layer-0/Q1024
paired result is effectively flat: -0.076 microseconds wall and -0.144
microseconds GPU. Do not claim a robust eager improvement for that group.
These component results cannot be subtracted from formal serving latency.

## Reproduction and identity

The first prototype is preserved under
`/tmp/deepseek_rotary_layout_20261004/projection_01`. Its candidate differs
from frozen C9 by exactly two AST nodes: the main RoPE call and the final Q
copy. The production comparison uses the unchanged production method AST and
the unchanged production adapter source. It retains the original frozen
adapter for its baseline and shares frozen projection dependencies and loaded
checkpoint weights. `torch.compile` is forbidden before production imports.

The production gate used exec session 89475. Focused-test PID 2057138 and
projection PID 2057208 ran sequentially, both exited 0, and GPU 0 was released
immediately afterwards. Separate stdout/stderr logs are beside the run
directories. Core commands, with `PYTHONPATH` set to the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
export TRITON_PTXAS_PATH="$PWD/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python -u /tmp/deepseek_rotary_layout_20261004/production_test_gate.py
.venv/bin/python -u /tmp/deepseek_rotary_layout_20261004/production_projection_probe.py run \
  --output /tmp/deepseek_rotary_layout_20261004/production_projection_01 --pairs 30
```

The recorded output paths must be changed for a new run. The focused wrapper
also has its output path explicitly embedded; do not overwrite this evidence.

Before/after source identity matches. Six live Triton quantizer specializations
and their loaded handles remain unchanged between correctness and timing.
The FlashInfer registry and actual process mapping identify
`/root/.cache/flashinfer/0.6.18/90a/cached_ops/rope/rope.so`, SHA-256
`6bd248d922781f40d746dc1dff63cd921fa7c565609ff0db271ab1cd4dce2f9c`.

Production source SHA-256 values:

| Source | SHA-256 |
|---|---|
| `operators/flashinfer.py` | `55c368b9ffa69301b8407de7f9aa69bfce03db3b6bfc4c59646b7d3fe9d36a8d` |
| `models/deepseek_v32/echo_model.py` | `ee042a19953494c45120bc19208ac767bc36b9670519865f304ee80805e71be6` |

Evidence is under `/tmp/deepseek_rotary_layout_20261004/`:

| Artifact | SHA-256 |
|---|---|
| `production_tests_01/result.json` | `0bae76f478e6833e3fdcb51139bbae21a0d00fb23415770a94b91ded52fd4ba8` |
| `production_projection_01/result.json` | `0642d9566d537070b4aa129cd230cc6c12e8d819cc573db87ceab2247a3e68fd` |
| `production_projection_01/screen.json` | `6fa0ed1aa2f853774dffc7a21e63c066c0e1fda856479ebba50f00b4abec6b0d` |
| `production_projection_01/source_before.json` and `source_after.json` | `3df0445e410f8e03dd9b192c2d685a784d0090a1dad4a5566ab30bf39a4992ce` |
| `production_projection_01/timing.json` | `3089f3e50d3c207cf2f5eece062ca038dd708b7b95bf0a9464840ee4235a45ff` |

`production_projection_01/independent_arithmetic.json` verifies all requested
case identities, outcomes, replay counts, 720 raw samples and paired medians.
The probe's historical result field `changed_graph_steps=27` counts all graph
steps, including each initial step; use the explicit 27/18/54 definitions
above. No existing experiment report or backing run was replaced. Combined
serving correctness and new formal/profile acceptance remain root's next gates.
