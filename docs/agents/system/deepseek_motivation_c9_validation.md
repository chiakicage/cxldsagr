# C9 handwritten Triton integration validation

C9 combines the accepted C8 norm and packed/shared MLP changes with the
handwritten T1 linear activation quantizer. Production does not call
`torch.compile`. Correctness acceptance does not establish end-to-end speedup;
the accepted [formal trajectory](deepseek_motivation_c9_integration.md) has its
own output/source audits, and matching matrix API profiling has also passed
independent numerical, source/runtime, attribution and arithmetic checks.

## Production checks

- Public quantizer suite: 34 passed, no skips. Engineering record:
  `/tmp/deepseek_linear_triton_public_tests_20261004_01/result.json`, SHA-256
  `55105f4dbd7e846cf4e31e8d1bece5c5d668107dae25959bc75700113c7de080`.
- Cross-version checkpoint: H=P2304, NH4608, chunk1024, A128, ten independent
  checkpoint block copies, HBM/eager. All 240 prefix and 80 candidate
  activation quantizations, full prefix/candidate hidden and logits match
  frozen C8 exactly. Each version builds from an independent empty session;
  the candidate forbids `torch.compile`. Twelve actual quantizer specializations
  were observed and source/compiler identity stayed unchanged. Record:
  `/tmp/deepseek_linear_triton_c8_cross_20261004_01/result.json`, SHA-256
  `b15473b02212c1d850fea37c44aa130de3280f858ac812ba51312c31a7965838`.
- Four-scheme H64K integration: 176 passed, no skips, including both required
  actual-checkpoint tests. All prefix/candidate hidden and logits match eager
  exactly; A121 fallback, A128 replay, packed inputs, changed-A history reuse,
  and failure cleanup pass. Production `torch.compile` was forbidden before
  imports. Twenty-four live Triton specializations were recorded.
- Independent H64K audit verified all 528 pytest phase reports, before/after
  identities and 1,598 source/dependency/artifact hashes. Each scheme retains
  5,771,362,304 bytes of graph-private reserved storage within the existing
  plan. Allocated, reserved and device usage are recorded separately. This
  dual eager/graph correctness process is not a full-NH capacity measurement.
  Evidence: `/tmp/deepseek_c9_combined_validation_20261004_01/`;
  independent audit SHA-256
  `cd06c26ff8b7b3f051575a8f59973fbeb0f6ae8b94dc8fecb18f99c58be7ffca`.
- Dense/grouped/packed MLP consumer suite: 37 passed, no skips; 25 observed
  quantizer specializations, unchanged source/build identity. Record:
  `/tmp/deepseek_linear_triton_production_20261004/consumer_02.json`, SHA-256
  `cfd96e48bdc6a59c57d8a9b0374e3d3e1fefb8eb0dc92a1bf0a5a7a4980c11c7`.
- Actual-checkpoint MLP: all nine layer/Q cases (source layers 0–2 and
  Q121/128/1024) passed exact output, input/weight preservation, stream/lifetime
  checks and 27 changed-input graph replays. Six live quantizer specializations
  were recorded; source/compiler identity stayed unchanged. Record:
  `/tmp/deepseek_linear_triton_production_20261004/checkpoint_01.json`, SHA-256
  `272a4917a304deeebfcf877434b52e62d444d28ada2a2dba3c54caafd7e1341b`.
- Global CPU regression: 2,850 passed, 1,022 explicit hardware/optional skips,
  58 subtests passed. Command: `CUDA_VISIBLE_DEVICES='' bash scripts/run_tests.sh cpu`.
  The initial collection attempt found the new linear test basename collided
  with the indexer test. The new file was renamed to
  `operators/deepseek_v32/linear/tests/test_linear_quantization.py` without
  changing its bytes, SHA-256
  `cb8a7b9c7b31e9e661b52f1e7ab9ed23567bba402b6015a029badbb25de8df29`.
  Earlier GPU evidence records the original path; the independent audit verifies
  that relocation. Production code was unchanged.
- Separate experiment CPU suites passed: motivation 63 and official ECHO 35.
  They were collected in separate processes because their pre-existing test
  basenames overlap. A combined collection attempt failed before tests ran;
  no test or implementation changes were needed.

These GPU correctness jobs used separate physical devices and could overlap.
Their elapsed times are not performance evidence.

## Compiler environment

The SM90 PTXAS path is explicitly fixed to
`/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas`,
SHA-256 `c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.
The first consumer attempt passed 35 tests and failed two identity guards when
FlashInfer added `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` after
capture. No numerical mismatch occurred. The passing consumer rerun predeclares
that same auxiliary environment value, with binary SHA-256
`a0212ca1a74ca4877e8d063baa1dd0b075fcd255fb6ab9e8a6bf98a276a2d6e8`.
The production guard is unchanged. The kernel target remains SM90; recording
an auxiliary Blackwell compiler setting does not establish Blackwell execution.
Formal and profile runs will predeclare both settings.

## Frozen candidate

Root: `/tmp/deepseek-motivation-c9_triton-frozen-0496ye32`, 497 files.
Freeze-map SHA-256:
`b65d577b59fa7c57735e079449815af25554df1507eb5c16653340042c6e0de0`.
The CPU source-snapshot precheck verified 1,268 executed/dependency paths;
executed-source SHA-256:
`b762e7502ba5c30e5c0106510e1ee2f8636a4179ceff1f15fbcc86e7a93a328e`.
The selected kernel remains byte-identical to T1, SHA-256
`8b666f5832a47f930134067a33cc31325313f08f89d3ef3d3a53671fc62c9768`.

C9 formal execution and independent audits have since passed; see the linked
integration record for that separate measurement boundary.
The official callback/checkpoint gates and final official comparison have their
own acceptance boundary; they do not follow from these local four-scheme tests.
