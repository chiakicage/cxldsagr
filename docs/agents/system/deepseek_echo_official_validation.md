# Official ECHO validation contract

User objective: validate the official ECHO implementation with the model and
budget from `deepseek_v32_motivation`, in a new experiment named
`deepseek_v32_echo_official`.

## Fixed requirements

- P=65,536 history slots per layer; NH=16,777,216 host tokens. No byte
  subbudget, empirical headroom subtraction, or additional admission W.
- Independent copies of checkpoint layers 0/1/2 in ten dense blocks, preserving
  source hidden/residual replay. 7,827,793,408 parameters, including endpoints.
- H=65,536, A=128, history chunks=1,024, seed=42, 16 users in two sequential
  rounds. Whole candidate batch on a GPU tail, then discard without host writes.
- Three warmup requests (user 0, user 1, user 0) including an actual host miss;
  release warmup sessions/shared allocation before a fresh measured trace.
- Full candidate hidden and last-token logits compared against a fresh HBM
  reference for every request. The complete resident trace is independently
  repeated from empty caches to measure the artifact's own numerical variability.
  Kernel-only correctness is insufficient.
- Pinned official ECHO `bc1b75c1000010d0ac6f032ebaac283255c050b1`: execute
  original fused prefill kernel, allocator and residual recall. Binding/adaptation
  code is separate; no edits to upstream tracked sources.
- Preserve the motivation model's projections (including no Hadamard), FlashMLA
  and ordinary linear backends. Use the original ECHO resident/fused indexer and
  default fused top-k in both control and offload. Disclose these adapters
  and the difference from the upstream TP=8 AWQ SGLang deployment.
- Include actual added metadata allocations and runtime memory in observations.
  Do not equate matching P/NH with matching total bytes. Official eviction
  counters are accumulated on GPU inside forward; unobserved selection/hit
  counters remain unavailable.
- New run IDs, source/build/native identities, request/output hashes and an
  independently rechecked report. Existing motivation reports remain valid and
  are not replaced by this new experiment.

## Current evidence

Initial audit rechecked the motivation `_02` run: 128 saved outputs, 96 exact
offload comparisons, 12 warmup records, and all 1,240 source hashes passed. All
recorded baseline source files matched the worktree before the new adapter files
were added. Workload SHA is
`7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`.

The isolated official fused-kernel smoke passed on physical GPU 1: 16 queries,
4,096 historical tokens, BF16 576-element records. Fused logits were bitwise
equal to official nonfused and installed mainline logits; 4,094 prefetched records
matched host contents and bidirectional mappings. This is operator verification,
not an accepted model or performance result.

## Implementation ownership

- `operators/deepseek_v32/indexer/official.py`: unique native bindings, pinned
  dependencies in user cache, and source/native provenance.
- `models/deepseek_v32/official_cache.py`: original artifact Python/Triton cache
  helpers, storage/lifecycle adapter and explicit memory accounting.
- `models/deepseek_v32/official_serving.py`: existing checkpoint workload with the
  official pipeline, sharing existing model weights and private indexer tensors.
- `experiments/deepseek_v32_echo_official/`: measurement, auditing and report.

## Numerical acceptance fixed before formal offload

The official default top-k uses atomic output-slot allocation. A two-call
H=4,096/A=128 audit returned identical selected sets but different ordering in
every query row. Independent full-checkpoint resident replays also differ, so
bitwise output equality is not a valid requirement for the unchanged artifact.
Cutoff-score ties may additionally change selected token IDs; ordering alone
does not establish the cause of every observed model-output difference.

The full H=65,536 calibration used the formal 16-user workload's first four
requests and two independent resident repeats, with no 64K offload output used
to choose the thresholds. Maximum hidden relative L2 error was 0.002521307,
maximum absolute error 0.1796875; at least 0.999928066 of elements satisfied
`abs(error) <= 1/32 + abs(reference)/64`. Logits had maximum relative L2
0.002260235, maximum absolute error 0.0625 and all elements within that bound.
The temporary calibration metrics were
`/tmp/echo-official-resident-calibration-4f8_ul6s/metrics.json`, SHA-256
`8f3e39e902febe1d6dfd2bd7cdf617de1cf6fd6d1a7bdca0f0adaa1ec26119e7`.
These are correctness calibration observations, not performance results.

Frozen acceptance requires finite correctly shaped outputs, relative L2 at most
0.005 for hidden and 0.01 for logits, and at least 99.9% of elements within the
above pointwise bound. All-element closeness, bitwise equality and maximum error
remain observations. Both the full 32-request resident repeat and all 32 offload
outputs must pass; the report must show both error envelopes.

A separate intrusive full-checkpoint data-path test compares every consumed KV
record against an independent mirror populated from original projection writes,
then compares physical-cache and logical-mirror FlashMLA with the same ordered
indices. This addresses offload data correctness without assuming deterministic
top-k ordering across independent model runs.

Shared cache sources were changing concurrently, so formal execution used
the isolated worktree
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-official/20261004-01`, with copied
source bytes and shared read-only upstream dependencies. The accepted performance
result applies to that snapshot, not subsequent main-worktree compute changes.

## Completed validation and formal run

- Native tests: 3 passed, covering fused/nonfused/mainline logits, BF16 transfer,
  bounded argmin and official top-k. Cache tests: 2 passed.
- Final frozen experiment audit tests: 32 passed. Ruff checks passed.
- Full checkpoint test: `test_official_checkpoint.py`, explicit checkpoint opt-in,
  physical GPU 2, 1 passed in 221.57 seconds. It executed requests 0, 1 and 16:
  1,310 attention calls, 2,650,296,320 selected-record occurrences and
  86,151,004,160 attention-value occurrences. Both selected KV and same-order
  FlashMLA were byte-identical to their independent mirrors. Revisit candidate
  H2D was 78,186,240 B; candidate D2H was zero. Intrusive test times are not
  performance measurements.
- Full-test source manifest SHA-256:
  `5a4ac022652e05880616d84e0bdd79b2369c62ef3ac4dc8d99bf392a636f295a`.
  Actual executed sources match the frozen tree. The only inventory difference
  is an unused offline `models/deepseek_v32/capacity.py` accounting update;
  no runtime module in this test imports it.
- Actual checkpoint top-k audit: request 0, all ten layers' final 1,024-token
  history chunk and 128-token candidate, 20 calls / 11,520 query-layer rows.
  All selected score multisets match exact `torch.topk`, with unique indices.
  Maximum cutoff-bucket occupancy was 5,513, below the artifact's 16,384-entry
  intermediate capacity. This checks these calls, not every possible input.
  Temporary evidence `/tmp/echo-official-build-bc1b75c/checkpoint_topk_audit.json`,
  SHA-256 `2a5c3e4455dc5dd0674960504cde04314e74c0b3c60b69ff1d84878b5b0b4643`.
- Formal run started on physical GPU 3 (visible as cuda:0), run ID
  `deepseek_v32_echo_official_20261004_p65536_nh16777216_u16_r2_01`.
  Unified exec session 83884 exited with code 0; PID 1620428 finished. Initial temporary data directory:
  `/tmp/deepseek-echo-official-deepseek_v32_echo_official_20261004_p65536_nh16777216_u16_r2_01-a56c1j1i`.
  After full acceptance, the harness moved this directory to the frozen
  experiment's `output/data/<run_id>/`. It contains all 64 formal rows, 32 resident
  validation rows, 96 tensor payloads and six warmup observations.

## Accepted result and publication

The frozen report audit passed with source identity
`b98ac0b4384e5b58f6f3c62cf9fe8ae341ddd57c4b5293d6a7d66e202fbcefed`.
It verifies 1,254 project source files, 885 official build source inputs and eight
native artifacts. All 32 offload and 32 independent resident-repeat outputs pass
the numerical policy fixed above. Independent-run output equality is not bitwise.
The full checkpoint mirror test remains separate correctness evidence.

| Metric | HBM-only | Adapted official ECHO |
|---|---:|---:|
| First-visit mean, ms | 2341.777291807375 | 4908.212345933862 |
| Revisit mean, ms | 2339.1749241782236 | 66.32175500999438 |
| Sum of 32 request latencies, s | 74.89523545576958 | 79.5925456151017 |
| CUDA allocated peak, GiB | 11.903460025787354 | 13.765183925628662 |
| CUDA reserved peak, GiB | 18.353515625 | 19.103515625 |
| Maximum sampled device used, GiB | 19.02447509765625 | 20.46978759765625 |

Revisit end-to-end mean improves by 35.270 times, but the complete two-round trace
takes 6.2718411% longer with ECHO. First-visit history construction means are
2302.2214973 ms and 4828.0046701 ms. This result does not show whole-trace speedup,
nor does the revisit ratio establish operator speedup or internal overlap.

The root copied all 2,426 accepted data files and two logs to the main experiment
under the same run ID, comparing SHA-256 for every copied file. Six generated
report files were copied byte-for-byte into the versioned `report/` directory.
Existing motivation results were not replaced. The README identifies the frozen
implementation and distinguishes it from newer shared code in the main tree.
Independent post-run review is maintained in
[the publication audit](deepseek_echo_official_publication_audit.md).

The main-only CLI/config compatibility adjustment after measurement handles
motivation's newly added compute-graph option. It preserves the frozen official
configuration schema and does not modify accepted tensors, timing, source
snapshots or the running implementation. Official compute graphs remain disabled;
the frozen report remains the publication source. The updated main experiment's
35 CPU tests pass, Ruff passes, and its `audit_run` successfully rechecks the
accepted frozen run without rewriting the original six report artifacts.

## Existing local implementation comparison

The user requested inclusion of the current local implementation's performance,
then explicitly instructed us not to rerun it and to reuse existing results.
The comparison therefore cites the latest published complete motivation trace,
`deepseek_v32_motivation_20261004_p65536_nh16777216_u16_r2_02`, rather than claiming
to measure current unvalidated C3 changes. Its source identity is
`006cdcddae8276b39ede88f55dd4e282e543f938127f7a35abdf0e081c336cb9`.

The local ECHO row has first/revisit means 4792.370115563244/70.03876618910 ms,
revisit p95 71.55397797760 ms, and total 77.79854210804 s. Its own HBM control
totals 91.94709866593 s. Both controls are displayed to preserve each run's scope.
Workload manifests and request files are byte-identical across the two runs, and
all 32 input hashes match. Hardware is H200/132 SM in both, but physical GPU UUIDs
differ. Local HBM uses mainline DeepGEMM resident logits, local ECHO uses its own
fused indexer/prefetch, and both use FlashInfer top-k. The official arms use the
original ECHO resident/fused logits and top-k. The old local metadata does not record a full
FP32/TF32 precision policy. These are existing cross-run measurements, not a new
controlled intervention on only the cache pipeline.

The CPU-only `src.compare_existing` produces a supplementary CSV/JSON/Markdown
comparison with both input report hashes and source/run identities. It adds
derived reporting artifacts; it does not perform inference or replace accepted
measurement outputs.

Research handoff: retaining DRAM histories makes revisits much cheaper than HBM
reconstruction at this P, but the official adapted history-build cost exceeds the
savings over this particular two-round trace. Neither the synthetic workload's
representativeness nor task quality follows from this engineering experiment.
