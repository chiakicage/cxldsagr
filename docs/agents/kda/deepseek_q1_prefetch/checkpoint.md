# Official Q1 prefetch checkpoint

Current production and publication include stateless page64/table/staging
preparation fusion. The fixed ECHO official core, scheduler, strict prediction,
cap 64, promotion, exact top-k, hints, recall and cleanup remain unchanged.
Preparation rereads current bytes on every call/replay; no packed history or
new storage is retained. All original official-Q1 dispatch conditions survive,
including full-prepared multi-session/transient leases. The separate bounded
free-slot path retains its narrower original predicate.

The completed cohort is `deepseek_h64k_a1_fused_prepare_20261009_01` on
GPU 0 / CPUs 0–7, H200/M403 SM90, Torch 2.12.1+cu130, Triton 3.7.1. It covers real
checkpoint L0–L2, H=65,536 / A=1 / token 111090, ordinary append, P=NH=65600 and cold
offload. Every method has independent check, benchmark, minimal profile and
operator-profile processes, preparing only that method. All 16 children passed.
One warmup, three complete prefill and five complete step samples were retained.

| Method | Clean prefill median ms | Clean step median ms | L0–L2 NSYS window ms |
| --- | ---: | ---: | ---: |
| hbm | 633.801203 | 1.803457 | 1.014466 |
| echo | 639.018574 | 2.562300 | 1.419011 |
| serial_sparse | 648.025304 | 2.401064 | 1.263747 |
| dense_prefetch | 648.195370 | 5.539999 | 4.415209 |

The complete synchronized wall timer includes required input preparation,
transactions and synchronization/commit. Loading, compilation, graph setup and
prefix restoration are outside it. NSYS windows are separate intrusive runs.
HBM's window and idle are 1.014466 ms/16.737 us; official SGLang is
1.014975 ms/14.274 us with different input, residency, capacity, framework and
physical GPU. Cross-implementation numerical acceptance remains false.

## Preparation and acceptance

The reviewed nine-file patch is SHA256
`6d35542bb9aea59a50ad29b6a9ce5637096d6702996fe9467b038281fe3bf624`.
The fused native body equals the accepted private candidate after symbol rename.
Original CUDA preparation/control APIs and promotion/cleanup functions remain.
The general ECHO module's full SM90a CUBIN matches the prior implementation:
1,218,840 B, SHA256
`0131578b8d4ab4e55e9cf1cd728387cec7eff2d36032835344d7847173410e43`,
all 223 sections including 44 kernel text sections. Full native ELF identities
are distinct and were freshly accepted; no hash normalization was used.

Focused CPU acceptance: 261 passed, 24 optional GPU cases unrun. GPU 1 / H200:
151 passed, zero skipped. Global CPU regression: 5090 passed, 1596 optional
GPU/checkpoint/tokenizer skips, 58 subtests. Ruff and diff checks passed.
Tests are engineering acceptance, not experiment results. Exact commands and
runtime observations are under `/tmp/cxldsagr-checks/q1-fused-prepare-production/`.
The observed fused-entry marker binds the actual immutable native adapter;
loading the old packed-input control alone does not set it.

Formal checks retain 13 standard comparisons and 44 full-graph checks. The report
rereads saved outputs, six actual ECHO transition proofs, source/native identities,
process ownership, graph lineage, traffic and all wall samples. Unsaved raw
scores/KV payloads keep their runtime acceptance boundary. Both minimal and
operator traces independently confirm complete graph 263→257, preparation 9→3,
one fused preparation per layer and 254 identical remaining owner/signature
multisets. This is not complete graph-edge equivalence. L0–L2 own 249 nodes;
the shown window also overlaps one shared final-norm node, giving 250 activities.
Preparation sums are 18.656 us (minimal) and 19.040 us (operator profile).

All 15 clean ECHO layer samples prefetch 64 records, then recall 1983–2041;
H2D ranges 2358144–2424960 B per layer. Attempt counts are 567–619 for L0,
3097 for L1 and 33471 for L2. Actual traffic and eligible transitions are audited
per execution; prediction may contain records outside exact top-k. Serial
recalls 2047 records and dense 65536; all offload layers write 1152 B D2H.
All methods retain 62,914,560 B complete-graph private reserved and 512 B static.
Allocated, reserved and device-used observations remain separate; no capacity
increase or continuous machine-isolation claim follows.

## Evidence and result scope

Selected MFU assets are in `experiments/deepseek_v32_mfu/report/h64k_a1/`;
the durable report run is `deepseek_h64k_a1_fused_prepare_report_20261009_01`.
Official comparison and gap report runs are
`combined_decode_fused_prepare_20261009_01` and
`decode_gap_fused_prepare_20261009_01`, selected at the official experiment's
`report/combined_decode/` and `report/decode_gap/`.
The prior isolated formal results were replaced after this publication. Preserve
`experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_isolated_20261008_01/request.json`:
its SHA256 is
`d776856368e64c931805f2181c41417609d55082866cc71e7468dfb6d57b380f`,
and accepted private evidence plus the new cohort reference that exact input.
Valid private controls and unchanged official/A128/C10/capacity reports remain.

Root's source/configuration audit and the independent review verified 12 MFU
and 12 C10 shapes plus A128 single-point/post-top-k paths do not call the changed
Q1 branch. All planning source hashes and five execution-reservation formula
samples are unchanged. This is not a fresh capacity-fill or request-memory run.
See `fused_prepare_production_review.md`, `fused_prepare_integration_plan.md`
and the scoped audit under the temporary production acceptance directory.

The gain is supported by the private predeclared 500 balanced AB/BA pairs:
medians 2.646831→2.6119945 ms, paired median −34.6505 us, 413/500 faster.
Both orders and five time-block medians improve, while some means and p99 worsen.
Do not subtract formal-cohort medians to infer a candidate speedup, or attribute
the entire wall difference to about 5 us of private preparation kernel time.
The private gate is one process with two fixed allocations and does not prove
independent-run confidence or tail-latency benefit. Its frozen sources, check,
NSYS and individually audited NCU actions retain their own exact boundaries.

## Retained bounded-preparation evidence

The following private gate and production regression records belong to the
earlier FREE implementation and retain their original measurement identities.
They support bounded-preparation integration, not a comparison between the
current isolated cohort and the earlier mixed-method formal batch.

The independent private optimization pair is
`q1_free_prepare_model_bench_20261008_01`: 100 balanced AB/BA pairs, each restoring
the cold prefix outside synchronized complete model.forward timing. Baseline
and candidate medians are 2.7214065 and 2.641556 ms, a 2.934% difference of medians;
paired median delta is −87.8005 us, with 91/100 wins. AB/BA medians are −79.723 /
−91.170 us. All samples remain, including outliers. Private correctness covers
three tokens and independent eager/diagnostic/clean graphs bitwise.

Private profile `q1_free_prepare_model_profile_20261008_01` has 290→269 complete
GPU activities and 260 identical non-prepare signatures. Preparation changes
30→9 activities, 68.928→17.920 us, including 12→0 preparation memsets. Native
layer ownership is not available in this private trace; unique prepare anchors
identify those groups. The full graph-span difference is invasive single-pair
evidence and cannot all be attributed to preparation. The FREE formal preparation
is 9 kernels totaling 18.080 us in the stage profile and 18.496 us in the operator
profile. Full attention API sums across 3 layers/9 kernels are 43.520 / 43.168 /
43.040 / 44.480 us for hbm/echo/serial/dense; Q repeat is included here.

Production validation includes 139 accepted focused cache/indexer/official
cases across the completed runs, an 8-state direct bounded-native check, 170
formal validator CPU tests, and global CPU regression 4,884 passed / 1,548
optional-GPU skipped / 58 subtests passed. Skips are not GPU acceptance.

The original FREE formal publication and its cleanup record are described as
historical evidence in `integration_scope.md`. Valid CUB/FREE controls and their
signed source/request/native dependencies remain separate from replaced
selected report assets.

Independent component reports `q1_official_path_report_20261008_06` and
`q1_attention_report_20261008_04` remain valid with their own original evidence.
Attention complete Graph API is 54.92–54.99→15.37–15.55 us; eager≈55 us, unchanged.
Original tolerances were preserved. Cleanup dependencies and unaffected-model
scope are recorded in `integration_scope.md`; active gap-control evidence and
immutable signed request inputs must remain until those dependencies end.
