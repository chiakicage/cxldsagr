# T3 checkpoint: prototype passed, integration deferred at C10 closure

The user closed optimization scope at C10 on 2026-10-04. T3 integration was
archived and its five production/test/provenance paths were restored to the
accepted frozen C10 bytes. See [the integration checkpoint](t3_integration_checkpoint.md)
for completed component checks and the restoration audit. T3 is deferred;
the prototype measurements below do not describe the current production code.

2026-10-04. T3 uses a fixed threshold of four, a sort-free bounded-neighbor
rank/scatter kernel, fresh per-row int32 flags, and the exact general repair
only on flagged long rows. See [the contract](t3_task.md),
[design](t3_draft.md) and [plan](t3_implementation_plan.md). All temporary
sources and evidence are under `/tmp/deepseek_topk_order_t3_20261004/`.
The temporary gate below passed and root selected T3 for production integration.
See [the separate integration plan](t3_integration_plan.md). This checkpoint
does not report production API or complete-model acceptance. Published
experiment reports remain unchanged by this work.

## Prototype complete API results

GPU1 correctness passed, then root granted an exclusive GPU1 timing window
after an all-device empty process snapshot. Timing used ten warmups and forty
alternating C5/T3 pairs. Every sample was retained. The window ended with both
processes exited 0 and an all-device empty process snapshot; no further GPU
work is authorized by that expired window.

Wall and event medians improved in all six cases. Numbers are milliseconds;
paired delta is the median of `T3 - C5` within each repetition, not a subtraction
of independently computed medians. Negative means faster.

| Q / N | Pattern | C5 wall median | T3 wall median | Paired wall delta | C5 event median | T3 event median | Paired event delta |
|---|---|---:|---:|---:|---:|---:|---:|
| 1,024 / 1,024 | causal | 0.103191 | 0.083539 | -0.019653 | 0.090208 | 0.070496 | -0.019584 |
| 1,024 / 32,768 | causal | 0.332834 | 0.296944 | -0.036184 | 0.319648 | 0.283552 | -0.036256 |
| 1,024 / 65,536 | causal | 0.438551 | 0.400030 | -0.038812 | 0.425184 | 0.386624 | -0.038736 |
| 128 / 65,664 | causal | 0.095360 | 0.092386 | -0.003032 | 0.082448 | 0.079312 | -0.003024 |
| 128 / 65,664 | ties | 0.117671 | 0.111136 | -0.006633 | 0.104560 | 0.098304 | -0.006576 |
| 128 / 65,664 | invalid | 0.149249 | 0.144897 | -0.004250 | 0.136304 | 0.131776 | -0.004208 |

This is a complete component API screen: official selection/value sort,
layout conversion, flags, both kernels, allocations and launch gaps are
included. Prototype identity collection was outside timing. Production must
separately measure its added in-API capture/observe calls. No model-level
latency or MFU gain follows from this table.

Hardware/software were PyTorch-reported NVIDIA H200 (nvidia-smi NVIDIA M403),
SM90, driver570.124.06, PyTorch2.12.1+cu130, CUDA runtime13.0 and Triton3.7.1.
The timing pre/post snapshots recorded GPU1 at 1,980 MHz SM and 3,201 MHz memory
clocks. They are instantaneous observations, not continuous telemetry. Both
PTXAS environment paths were set before imports.

The CPU artifact audit passed: 133 compiler/source dependencies per process,
34 correctness and six timing live specializations, 170/30 Triton artifacts
and five vendor records per process all matched. Both kernel names, actual
module/function handles, retained CUBIN hashes and GPU1 visibility were checked.
Evidence: `gpu1_correctness.json`, `gpu1_screen.json`, separate stdout/stderr
logs, the before/after observation JSONs and `identity_and_timing_audit.json`.

The original C5 bytes used by this prototype are also preserved at
`/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0/operators/deepseek_v32/indexer/selection.py`,
SHA256 `0e68ecb4e16e40e9dda1541693ce3e00e41ccd4a05ea2e708d04dd2545a4dcd8`.
That hash matches the prototype result's baseline source record. Production
integration changes the live repository wrapper, so new comparisons use the
explicit frozen baseline rather than rerunning the old live-import prototype
commands as if the baseline were unchanged.

## CPU and offline evidence

The CPU proof passed 20,605 complete value-bit/index comparisons: 14,444 short
rows, 6,159 long rows and two complete-baseline fallback cases. It covers all
widths 1..2,048, raw/special bits, run lengths 2/3/4/5, all-finite long runs,
warp-alignment boundaries, signed zeros and nonfinite masking, the uint32
packing boundary and maximum real key, and all 3,072 saved real rows with both
baseline and reversed tie IDs. It independently checks true run lengths
against row flags, verifies that short-run destinations form a permutation,
and confirms flagged rows remain untouched before general repair. The final
relation is the independent C5 index-ascending and stable value-descending
two-pass model. Evidence: `cpu_proof.py`, `cpu_proof.json`, and separate logs.

The final CPU ABI gate (`cpu_abi_ready.json`) verifies the official flags and
unchanged 1 MiB scratch ABI, 20 invalid metadata/device rejections, three
complete-baseline pre-selection fallbacks, fresh nonaliasing flag buffers,
fast/fallback launch order and shared flag identity, and Q0 without a flag
allocation or launch. These are CPU/mock/meta checks, not CUDA acceptance.

`audit_cpu.py` passed driver-fixture plumbing with both kernels replaced by
CPU models: 26 synthetic batches and six real batches. GPU fixtures include
length 3/4/5/17 runs at starts 0, 30/31/32, 62/63/64, 126/127/128 with width257,
plus explicit positive/negative-zero groups of length4/5 at starts31/63 with
width129. Both widths leave padding in their next-power-of-two block. Every
fixture checks row flags as well as every value bit and output ID.

Explicit SM90 offline compilation emitted twelve PTX/CUBIN pairs. No CUDA
driver, loaded module or GPU execution was needed. Resource observations:

| BLOCK | Fast registers | Fast dynamic shared B | Fallback registers | Fallback dynamic shared B |
|---:|---:|---:|---:|---:|
| 1 | 18 | 0 | 8 | 0 |
| 32 | 21 | 0 | 16 | 0 |
| 128 | 20 | 16 | 27 | 512 |
| 1,024 | 46 | 16 | 63 | 4,096 |
| 2,048 | 80 | 16 | 124 | 8,192 |

BLOCK>=128 also has 1,024 B static shared memory in both kernels; smaller
blocks have none. All emitted kernels have zero stack/local memory. Fallback
at BLOCK2,048 was checked with index bits16,17,21, all with the same resources.
The fast kernel has no sort or prefix scan in its source. All five emitted
fast variants have zero global loads after the final CTA barrier; their
in-place index stores follow that barrier. These static checks do not prove
GPU race freedom or a latency improvement.

`cpu_evidence_audit.json` binds the current sources, final ABI/proof/fixture
results and all emitted PTX/CUBIN hashes. Ruff and syntax checks passed for
the temporary sources. All CPU checks asserted CUDA remained uninitialized.

Independent read-only review by graph_validate found no ownership or long-row
blocker under the stated ordered-bit/unique-ID producer contract. It confirmed
the distance-four long-run test, unique rank permutation, safe fixed-point
store omission, disjoint finite/nonfinite destinations and same-stream phase
ordering. It requested the warp, threshold, signed-zero and padding fixtures
listed above; compiled order and actual GPU checks remain distinct evidence.

## GPU boundary and executed command

Root granted physical GPU1 for correctness only, permitted to overlap with
C10 correctness on GPU0. `CUDA_VISIBLE_DEVICES=1` maps it to `cuda:0` in the
driver. The preflight snapshot was empty on all GPUs at that instant; this
does not establish an isolated performance window. The physical device UUID
is `GPU-a2226185-cb05-a411-80da-f365154128fe`. Source identities and device/process
observations are in `gpu1_correctness_observation_before.json`.

The following correctness command completed successfully. It checked 41 main
shape/pattern cases, 27 K representatives, all 26 synthetic and six real
fixture batches, unsupported packing, overflow ties, nondefault stream,
retained outputs and four changed-input graph replays. A later invocation of
the same wrapper with `--mode benchmark --warmup 10 --repeats 40` produced
`gpu1_screen.json` in the separate timing window described above.

```bash
PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/bin:$PATH \
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas \
TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
.venv/bin/python /tmp/deepseek_topk_order_t3_20261004/runtime_screen.py \
  --mode correctness --device cuda:0 \
  --output /tmp/deepseek_topk_order_t3_20261004/gpu1_correctness.json \
  > /tmp/deepseek_topk_order_t3_20261004/gpu1_correctness.stdout.log \
  2> /tmp/deepseek_topk_order_t3_20261004/gpu1_correctness.stderr.log
```

The runtime wrapper recorded both kernels' actual live specializations and
compiler/source/artifact identities after work. Correctness on GPU1 and the
component timing gains are not whole-model acceptance.

## Frozen prototype identities

| Source | SHA256 |
|---|---|
| `candidate.py` | `d46f962ac450f7d28a415491ea1a7c4572aa21ddac0fd5cdf59b2b81a156cde5` |
| `screen.py` | `f409aba20233c38985c46ed4a23a3a56b18bd1c402712e9174cb6b277c1e7456` |
| `runtime_screen.py` | `da28422fed655fc27ce764a0b28bc970a69f30d71efdf319175b80ff8a71b32c` |
| `cpu_proof.py` | `e15dee4b9c6eb4d3aa7f92c941b71fa0686c52670ca976b79c920dee627d1e39` |
| `gpu1_correctness.json` | `99505e6cb772328191382b01bf61acf143f3ac65d00bfd1a6e782b5bc632cbb5` |
| `gpu1_screen.json` | `21632217507dca97201f4e9f0bd0e62df0a966e2bd97d9a8d94ebbe92cf94bfe` |
