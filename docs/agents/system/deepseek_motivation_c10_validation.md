# C10 combined integration validation

C10 combines the accepted C9 implementation with opt-in native token validation
and the direct main-Q rotary destination. The combined current-source gate
passed on 2026-10-04: **221 tests, no failures or skips**, including all required
actual-checkpoint cases. One upstream FlashInfer deprecation warning remained.
This is correctness and lifecycle evidence. It establishes neither formal
serving performance nor full end-to-end output equivalence to frozen C9.

The CPU validator contract, exactness screen and separate component timings
are in [the request-validation checkpoint](deepseek_request_validation_checkpoint.md).
Default runners retain the original Python `_validate` method and never
prepare the native helper. This gate explicitly selected native validation.

## Gate scope

Execution directory: `/tmp/deepseek_c10_combined_validation_20261004_01/`.
The driver derives from the C9 combined gate and retains its ten selected test
files, adding `operators/tests/test_flashinfer_rotary.py` and
`models/deepseek_v32/tests/test_echo_rotary.py`. It excludes the new default vs.
opt-in policy tests from constructor injection. Production source is unchanged
by the harness; it sets only `native_token_validation=True` for constructed
runners and verifies the actual native callable and runtime identity.

- The actual-checkpoint graph/eager test uses H65536, chunk1024 and A128 across
  HBM, ECHO, serial sparse and dense prefetch. All prefix/candidate hidden and
  logits match exactly; A121 eager fallback and A128 graph replay both pass.
  Independent user histories, changed candidate sizes, retained history,
  delayed outputs and graph capacity checks remain covered.
- The separate packed-input checkpoint lifecycle case retains its original
  H2304, chunk256 and candidate lengths 16/23. With native validation selected,
  its transfer/admission order, history identity, exact outputs and failure
  ownership checks pass for all four schemes. It is not an H64K request-input
  benchmark.
- Existing token-input, transient-candidate, prefetch, host allocation,
  storage-accounting and serving ownership checks pass. The focused rotary
  suite includes ordinary-adapter parity and changed graph/stream inputs.

The process used assigned GPU0, NVIDIA H200, SM90. OMP/MKL threads were eight.
The checkpoint was `/preset-models`. Before imports, the launch supplied the
complete tool PATH and both fixed PTXAS settings:

```text
PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
DEEPSEEK_GRAPH_HISTORY=65536
DEEPSEEK_GRAPH_CHUNK_SIZE=1024
DEEPSEEK_GRAPH_CANDIDATE=128
```

Production `torch.compile` was forbidden before model imports. No performance
job overlapped. Pytest PID 2059568 and driver PID 2059484 exited successfully;
GPU0 was released afterward. Test duration was 88.46 s and subprocess wall
duration 91.357 s; neither is performance evidence.

## Source and runtime checks

Source closure includes `.cpp`, so the native token helper participates in
both source snapshots. Before execution, CPU preparation collected all 221
tests, checked the original default validator AST and the native wrapper's
single predicate difference, and verified the loaded native binary. Preparation
did not initialize CUDA or execute a checkpoint test.

The two execution source snapshots are identical. All 663 pytest phase reports
passed. Twenty successful runner constructions selected the same native
callable: HBM 8, ECHO 10, serial sparse 1 and dense prefetch 1. Those counts
include small CPU fixtures as well as the checkpoint lifecycle cases. Native
source/build/binary identity stayed unchanged. Twenty-four live Triton linear
quantizer specializations were recorded.

Native token fingerprint under the declared PATH:
`75835a3abd6951acc84adab975a3c760aa7619515096949578b97a0035b66bd6`.
Loaded binary SHA-256:
`ed79d3e717a464379b4f638f608853277b65f602ef05ca5b293e3afa6025cd40`.
The full resolved module path, compiler/header closure and actual loaded
binary identity are retained in `runtime.json` and both source snapshots.

Postrun CPU verification rechecked 1,768 file hashes, runtime consistency,
all test outcomes and raw top-k artifact hashes. It did not initialize CUDA.
Each scheme observed 5,603,590,144 bytes of graph-private reserved storage in
this process. Allocated, reserved and device usage were recorded separately.
This observation is not a new fixed reserve or full-NH capacity acceptance.

## Bounded top-k research samples

A correctness-only wrapper called the unchanged official `exact_topk` and
returned its original tensors. It saved the first three calls with scores
shape `[1024, 65536]`, then disabled further capture. Each file holds CPU FP32
values and INT32 indices of shape `[1024, 2048]`: 16 MiB per pair, 48 MiB total.
No score matrix was copied. Output byte hashes match before and after saving.

Files are `topk_capture_00.pt` through `topk_capture_02.pt` in the gate directory.
`topk_captures.json` records sequential IDs, shape/dtype, query interval
`[64512, 65536)`, aligned chunk index 63 and all hashes. Layer IDs were not
available and remain null. These are deterministic sorted outputs with official
SMALL tie mode. They can establish finite equal-bit tie-run lengths, but cannot
recover unsorted selector emission order. This hook is absent from formal runs.

## Evidence index

All paths below are relative to the execution directory above.

| Artifact | SHA-256 |
| --- | --- |
| `run.py` | `567927e3fba3b0b7e922fcae98060b083775ca59134142d1780f6967e25dccdc` |
| `source_before.json` / `source_after.json` | `ead8eb872fe6f10604d0dc507b9a9123cc073fd63720a306030c033c98451ead` |
| `test_outcomes.json` | `fefb056d55bf86e393a215b4dc9060ea7a18f3edc16aa9b704ff21558f4dcbac` |
| `runtime.json` | `3fdee533e9877319055d441a8d1182fddf68ee9678a1a7413057518dc81e2659` |
| `topk_captures.json` | `bb79f0bde64245e96380a0d14b47603709eeefece831889a8fa53ae31d4852fb` |
| `run_identity.json` | `3f517a9793d109124c6b40405d9c1273d8a77b23f572a73a66e7c2d2b227bcec` |
| `verification.json` | `681862d57bc67ccf821b4758f0a33bc3141bf2a3cd99cb550ad0be9ac685e5d4` |
