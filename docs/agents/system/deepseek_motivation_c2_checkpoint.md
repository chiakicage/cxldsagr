# C2: block outputs and absorbed-query layout

Status: ready for integrated source freeze, 2026-10-04. This is engineering
screening evidence for the active motivation optimization, not a new experiment
publication or complete serving acceptance.

## Implementation

- `CheckpointBlock.forward` directly returns existing norm and MLP results when
  the complete input fits one chunk. Serving already sets the block chunk size
  to the current batch length. The multi-chunk path is unchanged. The exact
  FP32 residual/norm arithmetic and separate residual stream remain unchanged;
  no independent replay-input copy is removed.
- CUDA `CheckpointAttention.project` allocates its final contiguous
  `[Q,128,576]` query tensor before absorption. `torch.bmm` writes directly into
  the transposed view of its first 512 columns; the rotated 64-column tail is
  copied afterward. This eliminates the generic rank-three concatenation of a
  head-major BMM output. CPU retains the former path. No custom CUDA kernel,
  vendor implementation, FP32 formula, or checkpoint weight changed.

Source identities at handoff:

| Source | SHA256 |
|---|---|
| `models/deepseek_v32/echo_model.py` | `3feccef6ba014b0b32abf853f9377ed24c615403d1cae8bc05d5e9a20bc36e55` |
| `models/deepseek_v32/echo_block.py` | `81b35bbb5216da12778e322b639fb750dc93f55965ec22634fcfa58d9978310a` |
| `models/deepseek_v32/tests/test_echo_block.py` | `8b900b2e8b09fe5c12e2c2c2a069828e0bdf65cd6c9b47ac6e8a0dc368a26d82` |
| Pre-C2 `echo_model.py` snapshot | `202db3c2ed5365a8d794b9bfc46a681221925d7e034cc03d7758c0a45aa7344d` |

The baseline snapshot is `/tmp/deepseek_c2_echo_model_before.py`. The wider
worktree already contained unrelated changes; these hashes identify this
candidate independently of the Git diff.

## Correctness

`test_echo_block.py` now covers chunks smaller than, equal to, and larger than
the batch, with and without an incoming residual, for dense and MoE fixtures.
All outputs match its independent CPU oracle exactly; hidden/residual inputs
remain unchanged. Model and block tests: **30 passed**. Ruff check and format
check passed for the three changed files.

Real checkpoint `/preset-models` source layers 0–2 used FP8 linears. For each
layer, compare pre-C2 and current complete projection at `(Q,start)` equal to
`(1024,0)`, `(1024,64512)`, and `(128,65536)`. All six projection tensors in
each case are byte-exact: **54 of 54 checks passed**, all outputs contiguous,
input hidden unchanged. These are independent projection calls, not sequential
full-model propagation. Additional absorbed-query comparisons with the actual
layer-0 weight at Q=1,5,127,129,256,512,2048 for two seeds were also byte-exact.

## Measurements

Physical GPU 1, UUID `GPU-a2226185-cb05-a411-80da-f365154128fe`, reports
`NVIDIA M403`, SM90, 143771 MiB. The harness used the repository environment,
BF16 queries/absorbed weight, and explicit IEEE FP32 matmul mode for checkpoint
projection. GPU 1 was assigned exclusively to these probes.

The synthetic inputs have the actual query strides: latent output
`[Q,128,512]` has strides `(512,Q*512,1)` and rotated Q has strides `(8192,64,1)`.
Timing includes allocating the output. CUDA events bracket each complete call;
100 repetitions follow 20 warmups. Values below are median milliseconds.

| Q | Cat only | Two slice copies | BMM + cat | BMM + two copies | BMM directly into final Q |
|---|---:|---:|---:|---:|---:|
| 1024 | 0.324928 | 0.190432 | 0.379920 | 0.245696 | 0.085808 |
| 128 | 0.042928 | 0.028784 | 0.058080 | 0.041024 | 0.031616 |

All candidates were byte-exact. Direct BMM was selected because it avoids
repacking the large absorbed tensor, with no new kernel or arithmetic change.

Complete layer-0 projection has many Python-submitted calls. Eager event
intervals still include those launch gaps and show almost no improvement:

| Q / start | Baseline eager ms | C2 eager ms | Baseline graph replay ms | C2 graph replay ms |
|---|---:|---:|---:|---:|
| 1024 / 64512 | 1.530720 | 1.524464 | 0.737664 | 0.442192 |
| 128 / 65536 | 1.487280 | 1.482240 | 0.199488 | 0.159536 |

Eager uses 10 warmups and 50 repetitions; pure-projection graph replay uses
20 warmups and 100 repetitions. Capture/compilation is outside the graph replay
time. All captured projection outputs match baseline byte-for-byte. The graph
experiment does not include cache operations, session pointers, transactions,
writeback, or serving. Production graph support is not introduced or claimed.
These measurements show why reducing GPU packing alone cannot establish the
requested end-to-end convergence: host submission remains dominant.

## Evidence and commands

Temporary engineering harnesses and JSON contain percentile/wall timings,
correctness, shapes, and source hashes:

- `/tmp/deepseek_c2_pack_probe.py` and `/tmp/deepseek_c2_pack_probe.json`
- `/tmp/deepseek_c2_projection_probe.py` and `/tmp/deepseek_c2_projection_probe.json`
- `/tmp/deepseek_c2_projection_graph_probe.py` and `/tmp/deepseek_c2_projection_graph_probe.json`

Run from the repository root:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/deepseek_c2_pack_probe.py
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/deepseek_c2_projection_probe.py
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/deepseek_c2_projection_graph_probe.py
.venv/bin/python -m pytest models/deepseek_v32/tests/test_echo_model.py models/deepseek_v32/tests/test_echo_block.py -q
```

Keep the published motivation results until integrated all-output validation,
full four-scheme request traces, profiling, MFU recomputation, and affected
shared-consumer reruns are accepted. The root task owns that publication gate.
