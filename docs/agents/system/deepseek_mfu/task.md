# Active DeepSeek extend gap and MFU task

## Required outcome

Optimize `hbm / echo / serial_sparse / dense_prefetch`, using standalone cache
management work first, then model work. The cache experiment is now published;
full-model optimization remains active. Do not mark the goal complete.

Current production is append02 plus resident tail mask01 (V10). Published
diagnostics are `deepseek_gap_v10_a128_*_20261006_01`; current standalone
post-top-k-to-MLA runs are `cache_posttopk_check_20261006_01` and
`cache_posttopk_profile_20261006_01`, with unaffected recall04 retained under
its original identity. Earlier sections below record decisions
and must not be used to select V8/V9 or a rejected candidate as today's baseline.

- Workload: real checkpoint layers 0–2, H=65,536, A=128, history chunk=1,024,
  cold offload residency. This is not the C10 serving workload.
- Gap is the complete window minus union(compute, actual IO). Metadata,
  D2D/layout work, empty gather kernels and idle count unless covered by
  compute/IO. A proven zero-IO fused indexer still computes. Remove IO-only
  time from the denominator; use the conservative upper bound for fused IO.
- Complete extend and every layer must each be strictly below 10%. L0 starts
  at complete forward start. Subsequent layers start at the previous layer's
  compute completion. Include the full synchronization/commit tail globally.
- Complete offload prefill absolute gap must be <=1.2 times same-workload HBM.
- Timeline labels describe each computation. No Host outside CUDA lane.
- Equal-priority slot selection may be unstable. Preserve free-first/FIFO
  priorities, selected-hit protection and exact indexer top-k/tie semantics.
  Validate each call from its own before-state and remeasure changed traffic.
- Publish accepted time/MFU/labeled timelines in `deepseek_v32_mfu`; only then
  rerun motivation with fresh check/bench/profile, memory/IO audits and simulation.

Project/model/experiment rules apply. Preserve valid old formal reports until
accepted replacements. Do not rerun removed official ECHO integration. NOSA
impact evidence is in `docs/agents/acceptance/deepseek_mfu_nosa_impact_20261006.json`.

## Accepted standalone cache change

Production host C++ recall dispatch combines allocate, original generic gather
and publish. Classify, both CPU clock commits, owner/wait/invalidation boundaries
and original CUDA kernels remain. Provider initialization precedes pool storage
allocation; no persistent GPU storage was added. The free-only proof is an
explicit boolean. Implementation/contract details are in cache/model READMEs.

Source freeze `/tmp/production_recall_dispatch_cpu_freeze_20261006_02.json`:
`5a957c74f071a2a1cbb72ceb2b865340e0f95d9fe3450f37ee666095dca72613`.
Production validation: 14 new GPU + 504 affected GPU tests, no skips. Global CPU:
4,171 passed, 1,373 GPU/optional skipped, 58 subtests. Evidence:
`docs/agents/acceptance/cache_recall_dispatch_impact_20261006_01.json` and
`/tmp/production_recall_dispatch_gpu_harness_20261006_01/final_audit_02.json`.

Standalone runs at append02 promotion (manager06 later replaced by manager07):
- `cache_recall_{check,bench}_20261006_04`
- `cache_manager_{check,bench,profile}_20261006_06`

`experiments/cache_manager_performance/report/` contains 53 files. The publication
manifest binds the independent audit, actual native before-states, sources,
README and retirement inventory. Seven superseded recall03/cache05 directories
were removed only after publication. Preserve fixture
`output/data/input_fixture_20261006_02/` and `report/input_archives.json`.
Recall has two actual warmups and 279 measured samples; manager has one warmup
and 84 measured samples. Recall retains one independently reconciled terminal
exit race; no continuous-isolation claim. The original recall impact record above
retains its historical source/test meaning. Current append impact and its recall
rerun resolution are bound in `_06/publication_archive/impact.json` and
`joint_acceptance.json`.

Actual mapped native libraries are recorded from `/proc/self/maps` by
`evaluation/local_native.py`, checked before/after with inode/device/hash
protection. Build-input identity remains separate from actual binary identity.

## Earlier full-model evidence: V8, superseded by V9 below

Runs: `deepseek_gap_v8_a128_{check,bench,minimal}_20261006_01`.
Check under `/tmp/cxldsagr-checks/deepseek_v32_mfu/data/`; benchmark/profile under
experiment `output/data/`. Check13/profile25 comparisons and 8 saved tensors are
bitwise equal. Observers:13/18/36 clean discrete samples, not continuous isolation.

Receipt SHA `a6718bce8f50ca522523bad3389b30ef032594172b7f4bbe0b79806f4015bb2b`.
Execution identity `ab86f7d012746e6d9e42701cfe7c7addf3bd38719f19d5dbe5f1e4ad6eb691a5`.

| Method | Prefill ms | Extend ms | Full gap % | L0 % | L1 % | L2 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HBM | 646.747016 | 3.475854 | 20.543081 | 36.233947 | 10.913332 | 10.719608 |
| ECHO | 649.160940 | 5.497532 | 43.135823 | 63.176803 | 31.363547 | 35.128322 |
| Serial | 651.661905 | 3.973981 | 36.442446 | 52.476803 | 28.092352 | 30.616358 |
| Dense | 648.157918 | 5.836212 | 20.188967 | 38.446681 | 1.580981 | 11.662294 |

Prefill absolute gap/HBM: ECHO1.011697972, serial0.992840685, dense1.016504490;
all pass. All full extends and all L0s fail. Timing and intrusive gap are separate.

Independent audit SHA `743005f6617bc7d833313869a6e19f7c3a290a02a342f447d7afa9ae7b7ee019`;
artifact binding SHA `29cc8a34dea74765bf69b7226ccaa4a46de0a57f9aa18786cbae936c9a1864e3`.
The original audits and nine frozen helpers are copied verbatim to the minimal
run's `independent_audit/`, with a relocation manifest. 4,649 artifacts, 3,852
execution-source records, 59,985 activities, 788 partition windows checked.
Actual ECHO/bridge/common DSOs are unchanged across all runs. A fifth CuTe FP32
RMSNorm entry appears only in post-capture output validation; it occurs zero times
in all nine SQL captures. Do not claim the whole profile runtime object is equal.

MFU README now reports V8 diagnostics and preserves v3 formal operator MFU.
V3 ECHO has the documented scale-stage lifetime issue and cannot support current
implementation claims. V7 lacked actual loaded local DSO identity; no retrofit.

## Corrected L0 chart boundary

`timeline.select_layer` now includes full forward startup for first L0 and starts
later prefill L0 at the preceding chunk's final compute completion. This fixes
the plotting helper only; the original V8 gate already used the correct boundary.
`origin_boundary` records the source. Overlap includes visible embedding compute.

57 launch-gap CPU tests and Ruff pass. Corrected L0/L1 previews and frozen helper
sources are under the V8 minimal run's `timeline_preview_l0_l1_20261006_02/`.
All 16 prefill-last/extend layer windows match original gate fields; all eight L1
summaries retain previous metrics. Root viewed labeled extend L0/L1 PNGs. Original
V8 gap/binding and old preview remain unchanged.

## Direct token-entry candidate: not promoted

`/tmp/direct_token_entry_20261006_01/`, owned by graph_opt. Production model code
is unchanged. Explicit candidate model forward replaces the old prototype's
per-forward global embedding patch. L0 embedding writes directly to static hidden
inside projection graph. Original input validation/H2D/hint snapshots remain;
new ID staging and graph launch follow all begin_step calls. Whole ID source stays
owned until confirmed drain/commit. Every chunk checks embedding identity. Raw L0
requires and consumes its staged pair once; unconsumed stages cannot be replaced.

Adds 9,216 logical ID bytes for Q128/1024. Both graph banks' 12 GiB private limits
are charged to a 64 GiB isolated cache budget, distinct from formal 24 GiB setup.

Freeze `/tmp/direct_token_entry_20261006_01/source_freeze.json` SHA
`286b20df29e51b94a8c8a0ac02494f72c0d14bf1714dc4a0232383ea36e4e88b`;
1,291-source identity `27ce3e9fedf23841c668c73725be680b90594d7966d89d2ae66c8d5657020fd9`.
29 CPU tests and Ruff pass. Owner/budget and harness reviewers passed; review JSONs
are alongside candidate sources. No production promotion yet.

Tiny H2304/C256/A128 GPU check completed:65 comparisons passed,9 clean observer
samples; `/tmp/direct_token_entry_tiny_check_20261006_01/` receipt SHA
`d56b12d0be296b6378503eacc46e4d71d038fa3c5ce79ddde7526615a3e36689`.
Full H65536/C1024/A128 check also passed65 comparisons and16 clean observations;
receipt `1a29c8d24f1ed62179ee2186a9effcd8164ce915401dab2e00b51311e946bd10`,
identity `295d8a9a257f7ab00b680d2ca313ada1a3fb4bc960f4d7c92692f29d68b372ca`.
Root re-read all24 saved candidate/baseline tensor pairs and12 logical-KV hash
pairs. `/tmp/direct_token_entry_full_check_root_audit_20261006_01.json` records
the boundary. Actual static allocation rises156,672B, while logical storage rises
9,216B; both banks retain1,684,013,056B private reserved.

41 AB/BA pairs completed for each method,328 sample rows/656 timing values,
129 clean observations. All samples kept. Extend paired saving median/IQR (us):
HBM15.030[-8.175,40.751], ECHO15.258[3.953,36.414],
serial20.191[-3.586,30.364], dense-5.755[-53.704,5.134]. Not promoted.
`/tmp/direct_token_entry_full_bench_root_audit_20261006_01.json` binds the results.

Separately isolated profile adapter is under
`/tmp/direct_token_entry_profile_20261006_01/`. It uses the checked dual-bank
identity and separate NSYS processes per variant, each with2setup+8phase captures.
Setup-only hooks identify L0 embedding nodes; measured forward has no wrappers.
Root lifecycle review and gap_audit source/interval review passed. Both profile
runs completed, each12bitwise checks and31clean observations. Each has59,985
measured activities,788partition windows,1,560measured and1,560warmup replays.
Full gap upper% baseline→direct: HBM20.369331→19.648503,
ECHO42.243009→42.072496, serial35.295208→36.358604,
dense19.380091→19.853835. All full/L0 gates still fail; both prefill gates pass.
Combined independent comparison:
`/tmp/direct_token_entry_profile_comparison_audit_20261006_01.json`, SHA
`6022e56394e823ea235a4304f637c7bf9a4b04bc752471a30bba412eb2278cf5`.
Decision is `/tmp/direct_token_entry_20261006_01/decision.json`: no promotion.
Production model remains unchanged. GPU3 is free.
The timeline helper now prioritizes explicitly attributed embedding over its
projection scope;57tests/Ruff pass, SHA
`2a53a3ee0ecb2dc603e7fdf644ea549eda137b2e81e1802b696c915a3eed960a`.

## Active Q128 bound MLA/finish candidate

attention_graph implemented `/tmp/bound_mla_finish_full_20261006_01/` from
production baseline, without the rejected token-entry change. It captures
MLA→value expansion→finish after unchanged exact recall, copying only Q128×2048
int32 selected IDs. Existing typed norm and FP8 MLP-down output arguments write
directly into original finish outputs; downstream projection aliases stay stable.
No new kernel arithmetic or raw16MiB attention copy. Q1024/H>P retain original
dispatch. Only additional completion graphs bind cache records. Old bindings
are released after successful drain and before old cache storage release; new
bindings are captured after the next cache generation is allocated.

Static index addition is3MiB across3layers. Existing12GiB chosen graph
private limit covers all pure and completion pools together. Setup occurs after
initial graph preparation/cache rebuild, outside forward, and must be reported
separately. 53 CPU contract tests and Ruff pass. graph_opt found and the author
fixed captured selected-index identity replacement; both binding tensor and bank
mapping are checked before copy/replay. The benchmark control will bind exact
production forward_block and _consume functions at setup, avoiding extra private
dispatch checks in the baseline. graph_opt owns a new independent check/bench
harness and receipt identity. Model-only freeze SHA is
`4b03cdf69254363be59d8033d4d7fe55cf90027a2b301d3b2f1dc2ce90f2525c`;
it binds8 private and19 selected production files, not full measurement identity.
Peer review SHA is
`383befecd46ac925f65c8e59945e1130558949af85f4d8e50c6db2acee24bc79`;
root review is in `root_review.json`.

The independent harness `/tmp/bound_mla_finish_ab_20261006_01/` passed10 CPU tests
and root source review. Freeze SHA
`b9d8a9035ea510c022d398cfb73c6ad57ebe52d97a7a532e03747b0b267b9251`
binds1,302 sources with identity
`13be43c34057d528aadf825fe53330110a6604a9a9d531dc1bb0786ae9e2ab98`.
It covers original/changed suffixes, all hidden/logits/layer selections/logical
KV, cache byte/count metrics and committed lengths. Timing remains separate.

Tiny GPU check `/tmp/bound_mla_finish_tiny_check_20261006_01` failed during warmup;
no receipt was issued. HBM baseline and bound warmups completed, then the first
ECHO baseline prefill reported an asynchronous unspecified launch failure at
`echo_empty_event` after projection. The trace retains original and cleanup
exceptions. Seven observer samples found no foreign process or monitor error;
process exit1, GPU3 returned free. This is not numerical acceptance. Frozen
sources remain unchanged while graph_opt investigates and prepares a separately
identified bounded diagnostic; no automatic retry or full benchmark is assigned.

Separate stage-sync diagnostic
`/tmp/bound_mla_failure_stage_sync_20261006_01` also exited1, with7 clean discrete
observer samples. Its source freeze is
`a77084ef9f468a5f8c0af5070ac8d2888b8541481619fea9ef1585b65ca5d782`.
It locates the first failure at ECHO baseline L0, first chunk/position0: Q256
projection-input sync succeeds; Q256 projection graph exit sync fails before
cache/indexer operations. Root found all pure tensor identities unchanged and
identical L0/Q256 projection/finish pool segment snapshots across the ECHO bind;
all held L0/Q256 storages remain active. Internal graph descriptors and compiled
module lifetimes are not established by that check. Root analysis is in
`root_analysis.json` beside the diagnostic. graph_opt is investigating; this
instrumented failure is not a second acceptance attempt or a timing result.

Unfiltered memcheck diagnostic `/tmp/bound_mla_failure_memcheck_20261006_01`
reproduced that same projection failure, exit99,11 clean observer samples.
The28 reported errors are6 initial invalid-context API probes and subsequent
launch/sticky cleanup errors; no invalid-access/TMA/WGMMA site was identified.
Launcher/tool identity is `/tmp/bound_mla_memcheck_launch_20261006_01/identity.json`.
Expanded independent review verified184 pure/weight tensor identities and all12
pure-pool snapshots across bind, and found strong JIT module owners without an
eviction path; `/tmp/bound_mla_failure_memcheck_probe_20261006_01/proposal.json`.

The HBM-only check `/tmp/bound_mla_hbm_only_check_20261006_01` ended before numerical
checks because the four-method harness requires a loaded ECHO DSO even when only
HBM runs. Both HBM warmups and CUDA cleanup succeeded; no receipt was issued.
This is a diagnostic provenance-scope mismatch, not model correctness evidence.
The fresh HBM-only diagnostic `/tmp/bound_mla_hbm_diagnostic_20261006_01/` imports
those frozen Workload/warmup/check functions unchanged and records only applicable
loaded native images. Root verified its freeze SHA
`8b727723cfc4d6f67f4b4a97ac092c20b6a6367bb1107dc0583c689c4fce5320` and all 1,302
source identities before one fresh run. `/tmp/bound_mla_hbm_numerical_20261006_01`
completed 28 bitwise comparisons and 8 cache-state records across generations 1–4,
with 8 clean discrete observer samples. Root independently compared 12 saved tensor
pairs and 6 full-KV hash pairs, checked committed lengths/metrics and unchanged
before/after execution identities, and bound 1,323 artifacts. Audit SHA
`7b4b5a607b2f063bde279655928553f34b838e6d3ef2d8ad7dbb81c91ed7eea9`.
This establishes HBM tiny-input numerical behavior only; no offload acceptance or
performance receipt. ECHO transition-specific failure remains under investigation.
The boundary probe `/tmp/bound_mla_echo_bind_probe_20261006_01/` passed 17 CPU
tests and root review; freeze SHA
`9b39d389438acb9b65f9df03d4bb6a4f0cd0625544a10ff41d07148b3099f024`.
Run `/tmp/bound_mla_echo_bind_boundary_20261006_01` reproduced the failure, exit1,
12 clean discrete observer samples. Nine transition checkpoints plus the reference
replay passed: release bindings, old-cache release, old-reference destruction,
native provider initialization, shared-cache allocation, three runner constructors,
and entry to completion preparation. The eleventh direct Q256 replay failed
immediately after L0 Q128 completion setup. Its preceding drain succeeded.
All 37 tracked tensor identities/bytes, including six Projected fields and L0
attention weights, matched the reference immediately before the failed replay.
No L1/L2 completion capture or ECHO forward ran; original and cleanup failures
remain grouped. Root independently checked all 42 saved raw byte blobs, all 30
snapshots (reference plus 28 later snapshots identical), and bound 1,390 artifacts;
audit SHA `c339eae8a4fc0a94586e908b70fb74aa2a8ec365df362bb9d882957038aba3ef`.
Extra replays/byte reads perturb execution; PyTorch copy internals were not audited
for temporary GPU allocation. No model acceptance or performance claim.

The next distinct CPU probe `/tmp/bound_mla_echo_bind_probe_20261006_02/` adds one
checkpoint after L0's three eager completion warmups and their actual capture-stream
synchronization, before graph construction/capture. This separates eager completion
execution from capture. No copied model body or unchanged failed-request retry.
The installed torch.cuda.graph.__enter__ also empties CUDA and host allocator
caches before capture; untracked workspaces/graph dependencies remain hypotheses,
not established causes. GPU requires root freeze review and handoff.

## Private CUPTI measurement investigation

`/tmp/deepseek_v8_graph_launch_audit_20261006_01.json` establishes that first
HBM measured projection launch takes81.543us inside gap. Prefill projection
launch medians remain66–68us across64warmups and64measured calls, so this is not
only first-use cost. The trace cannot isolate collection overhead or justify
subtraction. NSYS documentation warns of node-trace perturbation; graph-only
envelopes cannot preserve the required compute/control/IO classification.

gap_audit completed read-only installed CUPTI feasibility:
`/tmp/cupti_activity_feasibility_20261006_01.json`, SHA
`9211f1f1d3060c88c1d29e403fbde5c4f0f1175f1fb2e82967932b2d5ce631ee`.
No API-level blocker found; reduced overhead is unproven. Root authorized a new
private collector implementation and CPU/native compilation only. Initial scope:
Runtime+Driver activities and external correlation, Kernel10/Memcpy6/Memset4
actual timestamps/bytes, full cupti-timestamp forward boundary, setup-only graph
created/cloned lineage, explicit zero-drop/complete-record checks and native
source/library identity. Prepare a mixed compute/control/IO graph fixture; GPU
requires root handoff. Do not integrate the full model, change old gate data,
assume Blackwell HES support on Hopper, or infer per-gap tracing subtraction.

Private fixture01 CPU freeze
`/tmp/cupti_native_probe_20261006_01/freeze.json`, SHA
`97f25d41dccc2305f5fef9b788a6bf860dcd0adb50a785d45b08ef204a4a87ea`,
binds383 source/header/tool/native identities.112 parser tests,4 provenance tests,
native ABI/lifecycle tests and CPU build passed; root independently reviewed
fixture/parser, and attention_graph reviewed native lifecycle. Pinned DT_RPATH
and mapped dev/inode/hash checks establish actual provider identity.

First GPU run `runs/cupti_sm90_minimal_20261006_01` has native exit0,5 output checks,
85 actual GPU activities (75 inside5 full windows plus10 diagnostic D2H),6 buffers
all completed/decoded, drops0 and unchanged native fingerprints. Wrapper exit1:
parser rejected duplicate lineage IDs. Original raw trace SHA
`46cd0ab7d8b7d89c415456db7ce5afe81b9edd7db49598e111bb26934ee36024`.
The2 discrete observer samples saw no owned GPU PID because the fixture was
shorter than the sampling interval; no foreign process/query error was observed.
This is not accepted collector or model evidence.

Observed schema corrections: each cloned node emits CREATE then CLONE; CREATE
does not prove a root. Callback nodeType is0 even for known memcpy/memset, so it
cannot classify nodes. Source-node types must be queried outside callbacks;
Runtime/Driver calls inside callbacks are unsupported. Two internal CREATE-only
nodes have no GPU activity and retain unknown type. Runtime-launched activities
correlate directly to Runtime APIs with runtime_correlation0; direct Driver work
correlates to Driver APIs. _01 remains frozen. Root authorized a distinct
`/tmp/cupti_native_probe_20261006_02` CPU implementation: gap_audit owns native
events/setup type queries, attention_graph owns parser/tests, root reviews.
Revision02 completed CPU acceptance: 137 parser tests, 4 provenance tests,
native ABI/lifecycle checks and all static checks. Root independently reviewed
parser/native deltas, verified all 383 frozen identities and scheduled one fresh
GPU fixture. Freeze SHA is
`29bc89deb8fb9ccb0de2e0f7a836ec51859a2908167b4e1517a09092ae6334f2`.

`/tmp/cupti_native_probe_20261006_02/runs/cupti_sm90_minimal_20261006_02/`
passed native and parser validation: 85 GPU activities, 75 inside 5 full windows,
10 post-window diagnostics, 12 query-backed template types, 26 CREATE events and
18 CLONE edges. Six requested buffers all returned and decoded; drops/faults0.
Raw trace SHA `a8ad80b6fcc6d86c73d864eca3f326539bda5704fc5c9e879cacb85ac0a52628`.
Root independently reconstructed native ancestry/caller domains, all interval
unions/IO-only denominators and per-window actual traffic. Audit SHA
`363d2125217fd5e46265bcd40ce40c891f9a5445f667156730612c1c1a81e399`.
Two clean discrete observer samples did not see the short native GPU PID;
continuous isolation is not established. This proves fixture feasibility only,
not lower overhead or any full-model gate.

Root authorized CPU-only work in `/tmp/cupti_model_probe_20261006_01/`:
gap_audit owns the native bridge and split setup-resource/activity-arming lifecycle;
attention_graph owns the exact-production HBM/dense adapter, setup template
attribution and matched clean/private/NSYS controls. No bound-MLA candidate is used.
A new tiny split-lifecycle fixture must first prove that enabling activities after
graph instantiation preserves every node record. Loading/prefix preparation must
not exhaust bounded no-recycle buffers; complete forwards retain L0 startup,
synchronization, commit and execution-scope tail. Root reviews/freeze before GPU.
No model collection or overhead comparison has run yet.

## Rejected candidates / remaining opportunities

Do not repeat trials without a new implementation hypothesis:
- General FIFO threshold selector: rejected.
- Free-only prefetch: cold neutral, warm about0.8% better, prefill slower.
- Capture-check removals:62 CPU,36 GPU variants and capture contracts passed;
  41-pair dense cold regressed78.263us, ECHO cold saved11.711us. Not promoted.
- Old token-entry monkeypatch prototype: slight full-forward regression.
- Raw MLA into finish graph:16MiB A128 copy; paired regression2.32–2.52%.

Free-only recall fusion feasibility is recorded in
`/tmp/free_recall_fusion_feasibility_20261006_01.md`; stage-error boundaries need a
private API. Old removable GPU-control opportunity is only about23–25us cold
across three layers. Actual gather IO is not removable gap. No GPU assigned.
cache_opt independently checked V8 temporal scope partitions to1ns, then its turn
ended on a provider error. Its completed audit is
`/tmp/deepseek_v8_host_temporal_audit_20261006_01.json`. Recurring L1/L2 temporal
gaps: ECHO indexer70.495/109.344us and expansion61.090/49.959us;
serial recall179.328/182.365us. These are not proven removable times.
The active bound-MLA candidate and private CUPTI collector are described above.
The collector02 fixture is accepted as evidence feasibility; model integration
and native split-lifecycle validation are active as described above. No profiling
subtraction or change to the gap definition is authorized.

## Execution and remaining work

GPU3 H200 SM90, UUID `GPU-80ff95c3-176e-fd8a-728f-9c5577c4a779`, CPU24–31/NUMA0.
Use `/tmp/deepseek_mfu_env.sh` and `/tmp/deepseek_observed_run.py`; one GPU job at a
time. Bind the child only. CPU analysis32–39. Never retry failed requests or kill
unrelated processes. Agent GPU runs require root handoff to avoid collisions.

1. Finish independent cache-only candidates: parallel free-slot append and bounded
   host-write enqueue consolidation. Reject regressions before full-model work;
   promotion requires actual-before-state, subsequent-traffic and lifetime checks.
2. Then continue model/launch optimization until every full/layer extend gate
   passes, using authenticated observer traces without subtracting overhead.
3. Fresh formal operator MFU/time/labeled timelines, audit and replacement publication.
4. Fresh C10 motivation check/bench/profile, IO/memory audits and simulation.
5. Affected regressions and document/result cleanup. Update Supervisor-owned research
   status through Supervisor only if research conclusions change; no new update yet.
6. Remove completed process plans after contracts/design/evidence have permanent homes.

## Continuation: late-armed collector and model check (2026-10-06)

The split-lifecycle fixture passed on GPU3 as
`/tmp/cupti_model_probe_20261006_01/split_fixture/runs/cupti_split_sm90_20261006_01/`.
Native freeze `78f58f89a9b5e2a3ea3bb4b0f80f81725cd68e4955a18d52ac45a1d2609ea903`
was verified before and after. All85 GPU activities,12 typed templates,18 clone
edges and5 window interval/byte totals passed independent root recomputation;
activity arming followed two checked prefix replays with zero pre-arm buffer
requests. Root audit SHA
`864162bb04365d6b3c98e8d6acd9d01498f56d4c28161f7e12276e1292e2b2c6`.
Two clean discrete observer samples missed the short GPU PID; no continuous
isolation or model/profiler-overhead acceptance is implied.

Production adapter freeze `069ab1e211d2b957dbf6ca484b7e4f6dcf4869cd96b657e900ca3cbbb655c67d`
passed39 CPU tests. Its first independent HBM/dense check in
`runs/model_check_01` executed both methods and saved reference.pt, but failed
the final combined execution-identity/CUPTI guard. There is no accepted receipt.
The opaque failure did not persist the after-state; a fresh CPU identity matches
the invocation, so runtime environment/provider drift still needs diagnosis.
No automatic retry or model timing occurred. Observer recorded8 clean samples.

Root is implementing the strict NSYS reader/transfer audit; gap_audit owns the
private raw reader. attention_graph is correcting normalized accounting for
production indexer scope names, fully out-of-root GPU work, shared-tail
conservation, mixed-operation uncertainty, and actual-IO evidence paths.
The isolated completion-stream bound candidate is still CPU-only and unpromoted.

## Isolated completion stream: tiny and full numerics passed

Fresh candidate `/tmp/bound_mla_finish_stream_isolated_20261006_01/` separates
completion capture from surviving pure graphs. Model freeze
`d6751fcec76305760104d5da7c7da4a258413a40c1620720a913fc234438eac3`
and harness freeze `ea685d81aad5b06c42aa63eea015823914efd1e1eb86e2306116115cbbaf0e7a`
passed73 and23 CPU tests respectively. Root verified1309 execution source
identities plus400 native-inspector dependencies and reviewed the stream,
workspace planning and drain/release delta. Forward math and exact selection
remain unchanged from the frozen bound candidate. No production source promoted.

Both `/tmp/bound_mla_finish_stream_isolated_tiny_check_20261006_01`
(H2304/C256/A128) and `/tmp/bound_mla_finish_stream_isolated_full_check_20261006_01`
(H65536/C1024/A128) passed130 bitwise comparisons across four cache methods,
baseline/candidate variants and two suffixes. Root independently re-read48 saved
tensor pairs,48 cross-method tensor comparisons,8 full-KV hash groups and32 cache
state records per run. Root audits are respectively
`6040fc4da1dae2d6f86a417845e2b730fcb4c8d231c908485146ea67e22c0fd2` and
`5811f76ed9f025999ee2d85e179b090420616bd72f712e48109966d1e47ac0a8`.
Observers recorded9 and19 clean discrete samples.

Each run contains66 workspace checkpoints over17 generations. Exact exported
PyTorch workspace-map queries plus allocator snapshots show unchanged pure
(handle,stream,address) owners; completion owns32MiB after setup and no owner
after bound graph release. Both variants explicitly use unified cuBLAS/Lt policy1.
The33MiB planned allocator-block ceiling is distinct from32MiB observed allocation;
containing segment reservations and full allocator/device usage are recorded.
This resolves the observed candidate failure without proving the old failure's
live graph-node pointer lifetime. Full41 AB/BA paired timing is now running in
`/tmp/bound_mla_finish_stream_isolated_full_bench_20261006_01`; no timing/gap result yet.

The collector check's false provider guard is understood: libcupti is a passive
ELF dependency of installed libtorch_cpu.so. Loading the shared object does not
establish active profiling. The failed check remains frozen/unaccepted; a fresh
adapter revision will bind passive provider identity separately from explicit
collector/NSYS activation and persist before/after guard evidence.

## Bound candidate timing outcome

The41-pair benchmark completed with134 clean discrete observer samples.
All328 sample rows,656 timed forwards,680 total forward-path proofs and
workspace/source/native/receipt identities passed independent root audit
`d4c184919dbad4a2ff2e697bbc1e4ba335f10e5bbfc6af9e6f981d8649b416f6`.
Extend medians (baseline -> candidate, ms): HBM3.503706 ->3.530453;
ECHO5.573992 ->5.627584; serial4.101868 ->4.143355; dense5.890531 ->5.906703.
The respective regressions are0.7634%,0.9615%,1.0114%,0.2745%.
Paired median candidate-minus-baseline differences are34.289,55.028,36.700,
17.826us; descriptive10000-resample bootstrap95% intervals are all positive.
Prefill medians differ by at most0.15%, without a demonstrated benefit.
The combined bound MLA/finish candidate is rejected for promotion; no repeated
trial or profile is planned without a new evidence-based implementation hypothesis.
The isolated-stream lifetime fix remains useful diagnostic evidence only.

GPU3 is free after this benchmark. Next: fresh production collector adapter02
check and HBM/dense clean/private/NSYS controls, then use authenticated full/layer
gaps to choose the next optimization. Production remains V8.

## Collector model integration and append candidate continuation

Adapter02 preserved the exact production numerics and timing path, authenticated
mapped CUPTI device/inode, and bound the immutable01 bridge by exact module path.
Its first check stopped before model construction because the generic injection
environment filter included unrelated VSCODE_INJECTION. Adapter03 narrowed the
filter to CUDA/NVTX/NSYS keys. Its independent HBM/dense check passed on GPU3:
`/tmp/cupti_model_probe_20261006_03/runs/model_check_01/receipt.json`, SHA
`221535021661a3e2eef464efe387cafaa80208d06fb151005f85c578020b9fc8`.
Root independently reread six finite tensors and five bitwise comparisons;
execution/provider identities match. Observer recorded9 clean discrete samples.

The03 private HBM worker completed all3 warmups and5 measured default forwards,
with zero native errors and8 clean discrete observer samples. The03 NSYS worker
stopped before graph capture because PyTorch explicitly dlopens its environment
CUPTI even when NSYS has already loaded a library with the same SONAME. A CPU-only
loader diagnostic confirmed that LD_PRELOAD alone does not suppress this explicit
load. No loader policy or Torch source was changed. Adapter04 instead distinguishes
one authenticated active NSYS provider from the authenticated passive Torch
dependency; every N graph query binds the NSYS file explicitly. C/P still accept
only the environment CUPTI file; C never activates an observer. Adapter04 freeze
SHA `669b5aec5c91ce49486827f699da6a6042b2bf83d05ec6182e92b32e9727e1bc`
binds1723 files and passed52 CPU tests. Its N HBM feasibility worker passed with
17 clean discrete observer samples and two complete native captures.

Analyzer02 in `/tmp/cupti_model_analysis_20261006_02/` passed226 CPU tests and
freezes25 files, SHA
`3b279c2f1703c099563a30bf9f55412e96fcf8cd06c5fde9cab81e15fb2d4c5f`.
It accepts source-proven provider path aliases, preserves NSYS OVERHEAD rows as
non-GPU metadata, and recognizes both demanglers' direct-copy cast spelling.
The nine FP32 numeric casts per forward count as compute; three integer causal
bound casts remain control. This resolves all previously unknown model activities
without classifying generic copy/layout as computation. P/N feasibility traces
pass complete graph/API/transfer/interval checks. The P five measured full gaps
are18.4051–18.8519%; L0 still fails. These are feasibility observations, not a
four-method or full-prefill acceptance and no profiler cost is subtracted.

The fixed36-worker schedule is running under root GPU3 ownership in
`/tmp/cupti_model_probe_20261006_04/runs/model_observer_01/`, observer
`/tmp/cupti_model_balanced_observer_20261006_04/`. CPU analysis follows completed
workers once in `/tmp/model_observer_analysis_20261006_04/`, independently
recomputing full/layer/tail interval unions and non-IO denominators. Do not start
another GPU job before the schedule exits. No balanced comparison is accepted yet.

A distinct cache-only ordinary append candidate is now CPU-ready:
`/tmp/free_only_append_candidate_20261006_01/`, freeze SHA
`d9789926becfd7c2e17304ab0cd46ef116b329c1b0cdecb06925703309541273`.
Under the existing sole-session lease with new persistent end<=P, one CTA selects
actual free slots and retains full-P priority validation, then calls unchanged
plain-slot planned_append. It does not change recall or prefetch allocation.
Root reviewed the block-uniform count/barrier/rank logic. All52 CPU tests pass;
54 native GPU cases are unrun. V8 append key/sort activity sums were50.304us ECHO
and33.792us serial, plus12.320/8.192us sort-workspace memset; replacement cost and
whole-forward effects remain unmeasured. No speedup or atomic failure-state claim.

Root materialized `/tmp/free_only_append_runtime_20261006_01/` with copied cache
and indexer directories and symlinked unchanged dependencies. The63 copied files
are recorded in materialization.json. No native build or GPU execution yet.
The append agent is preparing a separate CPU-only full-model A/B harness. Both
arms will use the combined native module and identical production model/graph
functions, selecting only original versus candidate append outside timing.
Production remains unchanged V8 and the overall goal remains incomplete.

## Balanced observer comparison completed; serial append selector rejected

The fixed36-worker schedule completed with108 warmups and180 measured forwards.
Independent CPU audit `/tmp/model_observer_final_audit_20261006_04/audit.json`
(SHA `68884e029fff6c7fe72e49cec4d07e947c6e9a155a212bb87a20a004c24acf13`)
verified all worker/source/provider identities and960 full/layer/tail interval
unions across192 P/N forwards. Root reviewed the audit and separately checked
the wall-time aggregates and startup API evidence. Observer recorded379 samples,
36 owned GPU PIDs, no foreign process, four bounded exit races, and a3.446s
maximum sample gap; this is discrete evidence only.

Median of six worker medians C/P/N (ms): HBM3.130470/3.351017/3.335046;
dense5.842081/5.979406/5.962796. Within-order P/N median increases relative to C
are7.2703%/6.5391% for HBM and2.2590%/1.9848% for dense. No observation cost is
subtracted. Measured full gap medians P/N are18.633%/18.166% for HBM and
18.785%/18.523% for dense. Every full and L0 window fails. HBM L1/L2 pass;
dense L1 passes and L2 fails. All samples, including one dense P outlier, remain.
This covers two methods' extend only, not four-method or prefill acceptance.
Labeled diagnostic timelines remain under the corresponding /tmp analysis
directories and do not replace the formal MFU report.

Ordinary append candidate01 was built only in its disposable runtime. All54
direct-native valid-state GPU cases passed in
`/tmp/free_append_native_tests_20261006_01/`; the observer recorded17 clean samples.
Combined ECHO ELF SHA is
`a12747844c78c09e67558afe44c5963aa6a9c7e500a6915a1edb7786a9125fb2`.
The read-only Ninja audit binds1176 dependencies and actual commands/artifacts,
SHA `9becd2e5efa80e8c88a5f5de6aa274780c696239ab488dcc9031fde6aed63d78`.

Separate target-P check `/tmp/free_append_screen_check_20261006_01/` passed six
baseline/candidate cases. It saves three actual GPU before-states; root reread
all tensors, checked chosen-slot eligibility, and independently reconstructed
all six reported post-state hashes. It permits legal unstable physical ties.
Check observer recorded four clean samples. The paired native-API screen
`/tmp/free_append_screen_bench_20261006_01/` completed all240 fixed measurements,
40 balanced AB/BA pairs for each resident count0/8192/65536, after three warmups
per arm/state. Source/state restore is outside timing; native append and required
synchronization are inside. Cache lease, host D2H writeback and model are excluded.

Baseline/candidate medians (us) were58.715/100.302,60.0735/100.465 and
60.2005/174.656. Paired candidate-minus-baseline medians were41.4885,40.2255
and114.211us; fixed10000-resample descriptive95% intervals were respectively
[41.0265,42.2845], [39.4435,40.373] and[113.968,114.7835]us. The three discrete
benchmark observer samples missed the short GPU process; no foreign process was
observed, without an isolation claim. Candidate01 is rejected; no full-model run
or promotion is planned. Its single-CTA full-P scan was not a useful replacement.

Next cache-only hypothesis is parallel full-P priority validation plus per-warp
free masks in existing scratch, followed by bounded mask compaction and unchanged
planned_append. Candidate02 is CPU-only under append_candidate ownership; GPU3 is
free and remains root-scheduled. The mutable01 full-model harness also has an
unfixed symlink import-path issue; it is not frozen or accepted. Production stays
V8, and formal MFU/motivation replacement remains pending the complete goal.

## Parallel free-mask append candidate and completed model timing

Candidate02 replaces the rejected serial full-P scan with parallel full-P
priority validation and 32-slot masks in existing scratch, followed by bounded
mask compaction and unchanged planned_append. It retains the narrow sole-session,
exclusive-lease, persistent-end<=P dispatch. User-authorized unstable cache ties
do not change exact indexer top-k or its tie policy.

Sources `/tmp/free_only_append_candidate_20261006_02/` freeze to
`599860c4dfc6772c508a3a00e588e176ddeda167a8156fefcaba10e291ab51cc`;
runtime `/tmp/free_only_append_runtime_20261006_02/` materialization SHA is
`f18cc72b896ef869f1a8657349ba9ed6c35750c5dfddc10a3a1d2b6444885ae6`.
All60 CPU tests and54 direct-native GPU cases passed. The native dependency
audit binds1176 entries; actual combined ECHO ELF SHA is
`1b7e576919037b35cd3458f353916b9c9c143f042ebcb14346d5408cd6887850`.
Four fresh-process corruption cases failed at synchronization as required;
no post-failure atomicity or continued CUDA execution is claimed. Another33
existing cache/lifecycle/stream/model-cache tests passed with no skips.

Independent target-P native checks save three actual GPU before-states and
verify six resulting states, selected-slot legality, maps, bitmap and counters.
The fixed40-pair AB/BA screen uses resident counts0/8192/65536. Baseline/candidate
native API+synchronize medians were57.910/40.1435,60.3435/40.5875 and
60.419/53.5565us. Paired median differences were-17.398,-19.675 and-6.905us.
Evidence is `/tmp/free_append_screen_{check,bench}_20261006_02/`; root analysis
SHA is `2da24c1d02ded701b47c6a0226320275c0ecb88f4dc4689d6cb725b5c0ad3135`.
This screen excludes cache lease, D2H writeback and model execution.

Full-model harness03 changes only append binding outside timing and calls the
unchanged production model/attention/graph functions. Its source freeze SHA is
`53ef40851a84f266caed010258c51d6ab8d7988bd9d916a51ea85878d697b1d6`.
Tiny and H65536/C1024/A128 checks each passed388 comparisons covering four
methods, both variants, cold/warm and original/changed suffix. Independent root
audits reread32 cases,224 tensor comparisons and160 forward graph proofs each.
Full check receipt SHA is
`f8e123e97b57e34a9cc7e098864a8b8fe7e243a718ab85b722f213fbe334fa17`.
Harness02 failed before checking due virtual Torch module file provenance;
harness03 fixes that accounting only, with no execution or native change.

`/tmp/free_append_fullmodel_bench_20261006_03/` completed successfully: four
methods,40 alternating AB/BA pairs each,320 rows comprising320 complete prefill
and640 cold/warm extend timings. Observer recorded135 samples, one owned GPU
PID, no foreign processes/exit races/errors and maximum gap2.274096s.
Independent paired timing and identity audit is in progress. Early unpaired
medians suggest reduced ECHO/serial extend time; no promotion, gap, MFU or final
acceptance claim follows from these raw medians. GPU3 is now free under root
scheduling. Production remains V8 until the review decision.

Separate host-write candidate01 remains CPU-only at
`/tmp/cache_host_write_enqueue_20261006_01/`. It batches fragmented D2H enqueue
through one FFI call while retaining actual DMA count and explicit ownership.
Review found a teardown liveness defect: after enqueue poisons the pool,
close drains successfully but release_session rejects poison before cleanup.
No premature GPU storage release is established. A separate02 revision is being
prepared; frozen01 and its runtime remain untouched. No native compilation or
performance claim exists for either host-write version yet.

## Append02 promoted; production remeasurement in progress

Independent full-model A/B audit passed at
`/tmp/free_append_fullmodel_bench_audit_20261006_03/audit.json`, SHA
`eb31eeedadf44143f09dc48a60512318a28d90a0974f2bdf316af056e56b124d`.
It checked 1,392 current/archived execution sources, 9,154 file paths, all 960
timings and 1,048 graph-forward proofs. ECHO cold/warm paired median differences
were -63.625/-62.353 us; serial sparse -60.346/-45.940 us. All four pooled and
order-stratified descriptive bootstrap intervals were below zero. Every sample
was retained. HBM and dense follow unchanged append branches; their small
observed arm shifts remain reported without causal regression attribution or
subtraction from sparse gains. Prefill differences were inconclusive.

Root checked unchanged original hashes and promoted the exact four candidate02
implementation files. New CPU dispatch and native eligibility tests are in the
corresponding module test directories; they do not require stable physical ties.
`/tmp/free_append_promotion_20261006_01/promotion.json` binds the before/after
sources and review decision. Frozen temporary evidence is unchanged; its
pre-promotion current-source validators will intentionally reject the new tree.

Production checks: 27 CPU tests passed (8 CUDA cases skipped in CPU mode), Ruff
passed, then all 87 selected SM90 native/cache/dispatch GPU regressions passed
with no skips. Observer `/tmp/free_append_production_gpu_20261006_01_observer/`
recorded 17 clean discrete samples. Full CPU regression is running. Independent
production cache check `cache_manager_check_20261006_06` is running on GPU3;
bench/profile, complete model gap and formal MFU/motivation replacement remain
pending. The old cache `_05` and MFU V8 results remain explicitly identified.

Production follow-up: global CPU regression completed with 4,203 passed, 1,427
skipped, 58 subtests passed and one warning. GPU validation remains the separate
87-case run above. Cache `_06` check/bench/profile all passed; the independent
audit recomputes 807 GPU activities, 324 layer-counter rows and 69 saved-selection
comparisons. Three actual production native before-states were captured at
resident counts 0/8,192/65,536 in `/tmp/cache_append_production_state_20261006_01/`.
The independent reconstruction checks legal selected slots and 11 post hashes per
state and binds the same ECHO ELF as `_06`. Combined audit SHA is
`7f2c11fd84ee38bac24c2291d59c93d18a2b8c709ab531ce6c4a0e4c298773f0`.
Staged report is `/tmp/cache_append_publication_audit_20261006_06/report/`;
canonical replacement is pending the joint recall audit. Fresh recall `_04`
check/bench passed with 18 checks and 279 measured samples, two actual warmups.
The bench observer contains one bounded exit race; it must remain disclosed.

Full model production V9 check02 passed all 13 comparisons. Check01 failed before
model execution because the invocation omitted `--physical-device 3`; its failed
staging remains only in `/tmp`. No request recovery or backend change occurred.
V9 bench/minimal01 also completed. Independent complete audit
`/tmp/deepseek_gap_v9_independent_audit_20261006_01/audit_complete.json` SHA
`eef92a294a8c135ba2d0aa0a2145da25216d7b7bd436cc3481029daff9354ab8`
verified 59,949 activities, 788 partition windows and 3,855 source records.
Full extend gap upper bounds are HBM 20.7023%, ECHO 39.8249%, serial 34.8273%,
dense 19.1793%. Every L0 fails; only dense L1 passes. Complete prefill absolute
gap ratios are ECHO 1.020770, serial 1.003421, dense 1.025386, all below 1.2.
The independent benchmark medians (prefill/extend ms) are HBM
646.319826/3.501378, ECHO 647.623217/5.449775, serial 649.815754/3.920567,
dense 646.284514/5.840686. No final gap/MFU goal completion is claimed.
Labeled full and L0/L1/L2 diagnostic timelines are being rendered from these
audited data. Observer counts for check/bench/minimal are 13/17/35, all clean
discrete observations.

Host-write02 is frozen at `/tmp/cache_host_write_enqueue_20261006_02/`; 61 CPU
tests and Ruff passed. It repairs poisoned teardown with guarded terminal drain
and device-wide confirmation before CPU-only handle retirement; successful
execution remains unchanged from candidate01. Root reviewed the delta. The
isolated runtime02 materialization SHA is
`f46ccd12b3069e8b630f5cab7ac158f3485c814351c2a4053da399f4e3ef745a`.
Native compile/load passed; actual common-transfer ELF SHA is
`295b553ff3089618aa52ec929077a000d79024f2f75ddc975c4b1b1311fb5188`,
with 373 recorded Ninja dependencies. This is not D2H correctness or performance
acceptance. The GPU check/screen harness is still under CPU review; root found
that the delayed producer must overwrite deliberately wrong bytes to test
dependency ordering, and requested direct poisoned release/close plus unjoined
private-stream completion cases before freezing it. Production host-write code
is unchanged.

## Cache publication and host-write02 decision

Root published production manager `_06` and recall `_04` with 53 report files.
Current publication SHA is
`aaff4c8887ce76e0e6f95738af725208124102d3a14b7079d97ee02834048327`.
The final README SHA is
`e85d1c6f3b7cb9ab0e7a8ad7cb4d95d8e8c8c82c1fa996ab17ff5d2bdaaeee70`.
Root verified 52 staged report hashes, 82 source/archive pairs, 83 recall
artifacts, the five preserved fixture files, all old report hashes and the exact
271-file retirement inventory. Only after replacement publication were the seven
superseded cache05/recall03 performance directories removed. Correctness receipts,
new runs, native actual-before-state evidence and other experiments remain.
The report's archive includes joint acceptance, root review, publication script
and the separately recomputed README/traffic/interval audit.

Frozen host-write02 check passed 12 arm/shape comparisons and eight stream,
validation, ownership and terminal-cleanup checks. The fixed screen completed
480 calls (six shapes, 40 balanced AB/BA pairs, three warmups/arm/shape; two
separate fixture primers). Each observer retained three samples with one owned
process observation, no foreign process/race/error; this is discrete evidence.
Independent CPU byte reconstruction, source/native identity and paired audit:
`/tmp/cache_host_write_screen_audit_20261006_02/audit.json`, SHA
`40c548050eddf0179fe23ab2c76aa5806428fa96d924fb5b0bcc02626d8750a4`.
It binds 579 artifacts, 116 source copies, 20 imported-source records and 373
native dependencies. It verifies saved GPU-output hashes against a separate
byte formula; no raw GPU tensor dump exists for re-reading.

Every shape and both order strata regress. BF16 contiguous 128x576 V8/02
medians are 48.035/64.518 us; paired median +17.058 us. Fragmented counterparts
are 66.5705/73.629 us; paired median +8.200 us. All six pooled paired medians
increase by 5.936–17.058 us. Root independently recomputed all pairs and rejected
promotion in `root_decision.json` beside the audit. No production host-write
change and no repeated trial are planned. A stream-bound native submitter would
be a distinct unimplemented hypothesis, not evidence to reinterpret this result.

A new cache-only CPU prototype is authorized under
`/tmp/prebound_recall_candidate_20261006_01/`: move repeated classify/complete/map
validation into a prepared native call family while preserving each original
stage, current TensorView validation, wait_host and CPU clock/error boundaries.
No cached-success validation, kernel/math change, build or GPU is authorized yet.
The V9 labeled diagnostic figures are complete in the independent audit's
`timelines_v2/`; all four retain an intrusive-profile caveat and unchanged bar
geometry. The L2 dense caption explicitly has no following-layer prefetch.
Current gap gates still fail; formal MFU/motivation replacement remains pending.

## V9 diagnostic publication and independent preservation check

Root published the complete V9 diagnostic report, including full extend,
L0/L1/L2 extend and last-prefill-chunk L0/L1/L2 figures. Final publication SHA is
`859c06a05eb7a09a4c0584d83be114cb6b084c3eeb832a92654ce45974b00fba`;
the separate current README binding SHA is
`c43cf0653ced008e7551a7613edc0e44c328979653f64ab0b1e6437d06b29826`.
The exact 95-file renderer/audit archive is retained in the V9 profile data run's
`publication_archive/`, together with root review and publication/retirement
scripts. Its original manifest SHA remains
`992bd0faaa73fc6d6749776326fc2738e1a2d84bb1b0a30ddeafba13a043c034`.

Only after publication, root retired the reviewed 33 V4–V8 diagnostic
directories: 14,580 files and 1,172,597,685 logical bytes. Root retained a
per-file pre-deletion hash inventory; this is not a reclaimed-filesystem-block
estimate. The independent post-publication audit at
`/tmp/deepseek_v9_final_publication_audit_20261006_01/audit.json`, SHA
`8282ecebc12f42468719035a0be9cbc7e00c8d8f3fa96483352dd3e26bea5912`,
verified 4,090 file hashes, including 3,932 bound current V9 inputs, all selected
V9 assets, all 18 v3 formal report assets and the 95-file exact archive. It
checked both publication bindings, v3 raw SQLite inputs/results and its receipt
payload signature, every retired target's absence, and the exact four remaining
run directories in each output category. The v3 README tail is unchanged.
This is preservation/binding verification, not a repeated numerical audit.

## Early L0 scheduling review, queued after the recall screen

Read-only evidence and source analysis are at
`/tmp/deepseek_early_l0_schedule_review_20261006_01/`; the initial evidence JSON
SHA is `c222646e9db7d1ce7b93c65f247e2430a2bd3f062f9efd5be60647ea5b58bc95`.
The token-H2D-API-return to eager-embedding-launch interval contains
57.399/109.451/87.608/68.928 us of uncovered gap for HBM/ECHO/serial/dense.
It encloses multiple preparations and is not a predicted removable cost.
First-projection API gap remains 82.822/90.694/95.670/0 us. Projection compute
spans are 156–157 us; entire graph GPU spans, including final KV concat/control,
are 159.104–159.904 us. Dense's approximately 1 ms launch-return-to-completion
interval includes its initial DMA dependency and is not projection duration.

The narrow prospective candidate keeps eager embedding and the original graph,
then overlaps pure L0 projection with required hint/cache preparation. Invalid
input, poisoned state and unsupported configuration must fail before resource
use; all checks, append-source reservation, owned-KV lifetime and rollback/drain
requirements remain. Independent allocation/CUDA failures may surface in a
different order under the new schedule; universal old OOM/error precedence is
not required. This clarification supersedes the initial memo's stronger wording.
The [executable private plan](early_l0_plan.md) is ready for one supported
single-GPU, three-layer, single-chunk persistent extend; dense initial-wait
reordering is excluded. It reuses extracted begin guards for a read-only
preflight and retains the normal guarded begin after replay, so all added guard
cost remains measured. No model implementation, native build or GPU work is
authorized before the recall screen decision.

## Prebound validator01 rejected; next private work authorized

The three-extra-FFI validator candidate completed 127 native metadata and 17
actual CUDA checks, then all 720 fixed screen calls. Root's paired/state audit
is `/tmp/prebound_recall_root_audit_20261006_01/paired_audit.json`, SHA
`bfd44de7c0a41bf909641c25f0fb6c169cfac8ab92b642be3a5a5386147fd9e8`.
It verified 54 states/219 tensors, 78 archived source records, 628 native
dependencies and every sample. All six changed cases regress by paired median
14.278–21.622 us, with pooled and both AB/BA descriptive intervals above zero.
Certified-resident controls are near zero and are not subtracted. Check/bench
observers have 13/7 clean discrete samples; prepare/probe observers missed the
short process. Root rejected promotion in `root_decision.json`, SHA
`bd9c8b711b53aa6beb9f4e40189753b4d3002697bee2ccd8af091e5f88ddfe17`.

Production remains V9. Root now authorizes the queued early-L0 private
implementation in parallel with read-only cache-dispatch feasibility work.
No production model change or new model GPU result is implied by that authority.
The separate cache feasibility compares a complete native stage family with
combined classify/map validator-and-launch entries. The allocation workspace
query is already cached by geometry/device; merging fresh validation into that
query cannot remove an FFI crossing on warmed calls without incorrectly caching
validation success. The narrow alternative therefore keeps original Python
allocation guards while combining classify/map validation with existing launches.

## Narrow02 rejected; stream03 remains a separate feasibility review

The narrow02 candidate froze 21 files at
`/tmp/combined_recall_candidate_20261006_02/freeze.json`, SHA
`c52b043ff145d73cc46e694e3575bfee7f3d52917c94f3725bd451462a6bf3ee`.
Its seven-file screen freeze is
`/tmp/combined_recall_screen_20261006_02/freeze.json`, SHA
`8c96cf5eac5c7553dd1b243a21eb3cfc6332fd062a07379c998a246852f860e5`.
The author completed 168 CPU checks and patch validation without applying it;
root then passed 77 native and 12 actual CUDA checks and independently reread
54 saved before/after observations and 219 tensors. This covers saved before
storage and after consumed/live state, without a full post-scratch/storage claim.

Root completed the fixed 720-call screen. Restored-all-hit L0/L1/L2 paired
candidate-minus-baseline medians are +9.6225/+9.0595/+11.371 us; cold-sparse-miss
medians are +6.2545/+7.592/+4.987 us. Both AB and BA medians are positive for
each changed case. Root finalized rejection at
`/tmp/combined_recall_root_audit_20261006_02/root_decision.json`, SHA
`468ccb74bdb431d3cf008345e55ebcae6f436bd45406199c57787632ad0429e4`,
and will not run this candidate in the model. Its independent paired/state audit
is `paired_audit.json` in the same directory, SHA
`02b8241c40d5818c9f81a6ee70311d1ce5aa6cee378482127847268a8aeb3c24`.
The prepare/probe/check/bench
observers have 4/4/13/7 clean discrete samples; the short prepare/probe processes
were not observed while active. Production remains V9, and both rejected
candidate freezes remain unchanged.

Root authorized a distinct CPU-only stream03 feasibility review under
`/tmp/recall_stream_context_review_20261006_01/`. It investigates whether the
authenticated Torch exporter already supplies the invocation stream, allowing
selected redundant outer `use_torch_stream` wrappers to be omitted while retaining
device selection, validation, native kernels and restoration. There is no
stream03 implementation, freeze, build or GPU result. The baseline cProfile
diagnostic is intrusive and supports call-count/source investigation only;
its cumulative timings do not estimate removable gap or speedup. Early-L0 work
continues under its separate plan.

## Early L0 GPU checks and call-count diagnostic

The unchanged private early-L0 candidate froze at
`/tmp/early_l0_candidate_20261006_01/freeze.json`, SHA
`836bfbbf910f6cebdec2dfceafd19587fc92498bb70de551a3cb1c7313b17cd7`.
Its 65 CPU checks and independent source review passed. The first tiny GPU
check stopped in the **baseline ECHO** default-versus-hidden comparison: the
original harness incorrectly required identical physical slot assignments.
No receipt was issued, no failed request was resumed, and neither the candidate
nor that frozen harness was changed. The failed run remains engineering evidence
under `/tmp/early_l0_tiny_check_20261006_01/`, outside experiment publications.

The separate harness02 saves actual before/after physical maps, priorities,
free state, clocks, host KV and live records. Before-state must match physically;
after-state must match by logical resident token, priority and record bytes,
with unique inverse maps and an explicit physical-slot permutation. Slot zero,
resident-slot preservation, exact selections/outputs, counters, traffic and zero
evictions remain checked. Differing resident token sets are not accepted as
permutations. Source review traces the permitted permutation to ECHO's atomic
host claims/ranks and free-recall compaction, not the append02 mask expansion.

Harness02 freeze SHA is
`e426e9364bd12afb1f29fcd005e2f84efb79106a13f9af3cefeaca65e8488da0`;
runtime bundle SHA is
`a2081ed945444eb2caf90e11ced7f1607d1b5b5ecf32a0b8d0718b6e6fbf40c8`.
Root ran `/tmp/early_l0_tiny_check_20261006_02/`: 284 output/selection/KV
comparisons, 306 map-equivalence rows and 54 before/after pairs passed. Some
valid tiny comparisons change 2,298 physical slots while preserving canonical
state. Its observer has 13 discrete samples, one owned PID and no observed
foreign process, race or query error. Independent reread remains in progress.
The full H65536/A128 check is running at
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/early_l0_full_check_20261006_02/`.
That SSD output location avoids exhausting the remaining local `/tmp` space;
the frozen implementation and harness remain under `/tmp`. No paired model
timing or early-L0 production promotion has occurred.

Root also ran one unchanged V9 complete-forward cProfile diagnostic per method
after three cold-extend warmups. Results are at
`/tmp/model_python_cost_20261006_01/output/result.json`, SHA
`af31c3dab5b6e9363180377226624d9c24f60f79267506d348e0e9d2b1068aa5`.
Its observer has eight clean discrete samples and one owned PID. The diagnostic
supports source/call-count hypotheses only; its host instrumentation changes
overlap, and some recursive decorator entries have inconsistent cumulative
accounting. Do not use these durations for gap attribution or speedup claims.
Production source hashes remained unchanged. Stream03 is now authorized for
separate CPU-only implementation and native-probe preparation; no build/GPU
result exists yet.

The full early-L0 check and fixed paired screen subsequently completed. Full
check retained 108 state files (38 GiB on SSD), with 284 output/selection/KV
checks and 306 layer-state comparisons. The separate reader verified all files,
3,348 state tensors, 150 distinct output tensors (168 references), 162 layer
transitions and 303,013 changed-slot pairs. It also authored the harness; root
therefore independently reread all saved output comparisons and the three
methods' cold/original/default before/after states across all nine layers.
Both checks passed. The support audit is
`/tmp/early_l0_root_support_20261006_01/full_audit.json`, SHA
`7858f6f0d48291cea718815d6df4514327b95788b7f7ad28b05e72e907d6181d`;
root's narrower independent evidence is `full_root_spot.json` alongside it.

The benchmark is
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/early_l0_bench_20261006_02/`:
240 arm rows, comprising 720 complete timed calls over 40 balanced pairs for
each method. Root retained every sample and independently checked schedule,
receipt identity, medians, paired differences and all raw observer samples.
ECHO cold/warm paired median regressions are 50.024/48.906 us; serial sparse
regressions are 52.924/57.889 us. All four pooled and AB/BA descriptive bootstrap
intervals are above zero. HBM cold/warm deltas are -5.317/-1.837 us, with all
intervals crossing zero. Prefill changes are inconclusive and not subtracted.
The benchmark observer has 118 clean discrete samples and one owned PID.

Root rejected early-L0 promotion in
`/tmp/early_l0_root_support_20261006_01/root_decision.json`, SHA
`8e6c156d07d8546ba39265108adc10c2173c18b7633063222c42509f1a78a027`.
No new gap/MFU profile was taken. The separately frozen GPU lifetime script is
unexecuted because the performance gate rejected this candidate. Production
remains append02/V9; do not repeat the unchanged candidate.

Memory checks retain raw allocated-before, allocated/reserved peaks and
peak-minus-before separately. Raw HBM peaks differ because these checks have
different retained snapshot/reference state; the difference is not attributable
to the eager embedding. Default-call peak allocation increments are identical
for HBM (37,880,832 B) and serial (38,028,288 B); ECHO is 39,540,224 B in baseline
and 39,540,224–39,687,680 B in candidate. All observed peak reserved values are
14,547,943,424 B. These are diagnostic checks, not a serving hard-budget result.

## Recall stream03 passes standalone screen; model trial pending

Candidate `/tmp/recall_stream_candidate_20261006_03/freeze.json` has SHA
`02c052b67d50444ed72d19d5080a0c889804563f0e4fe1c03ba4400e80141098`;
its screen freeze is `f06dc8ac1a166ac6cf8b2931feceb6da4f708ae8b97b65a6a8199a8efb730be4`.
The candidate omits only eligible outer FFI stream scopes, retaining the CUDA
device scope, original Python guards, complete bridge and every native kernel.
Setup authenticates actual mapped Torch/FFI providers; each call still obtains
the current stream. Supported history is the installed builtin or generic Torch
provider, not arbitrary historical custom capsules. Production remains V9.

Independent source review and 42 CPU checks passed. Root passed all 10 actual
GPU stream/failure/fresh-worker checks, then completed the isolated correctness
and fixed 720-invocation screen. The independent reader reconstructed all 54
before/after observations and 219 unique tensor blobs, including actual maps,
KV, legal free/FIFO choices, clocks and traffic. After-state covers consumed/live
state, without claiming full post-call scratch/storage equality.

Restored-all-hit L0/L1/L2 paired median deltas are -12.9675/-12.1625/-14.4155 us;
cold-sparse-miss deltas are -6.469/-5.139/-5.426 us. All six pooled descriptive
bootstrap intervals are below zero, and both order-stratum medians improve.
Cold L2's AB interval crosses zero; this sensitivity remains disclosed. Every
sample is retained and unchanged certified-resident controls are not subtracted.
Probe/prepare/check/bench observers contain 5/3/13/7 clean discrete samples.
Root reread their raw records; the short prepare PID was not sampled.

Root approved a separate private full-model check and fixed paired trial at
`/tmp/recall_stream_root_audit_20261006_03/root_decision.json`, SHA
`a1e2d58b57c3f44c6ff664ccac25dc14d2997dde00c744df0d631805c09eb03b`.
No production promotion, new model latency, gap or MFU result is implied.
The model harness will use exact production model/graph functions and select
the two metadata-op providers outside timing, preserving full synchronization
and commit boundaries. It remains CPU-only and pending review.

A distinct resident-indexer tail-mask fusion is being prepared privately on
CPU. It keeps official DeepGEMM output and exact top-k unchanged and replaces
the arange/comparison/masked-fill tail with one write-only mask launch. Its
scope is logical invalid cells only, preserving strides, offset, valid cells
and unexposed padding. No implementation is promoted or GPU result available.

The private stream03 model harness completed all 32 tiny-model output scenarios
and 72 before/after observations, then failed its final source inventory gate:
`models/nosa/infer.py` and `models/nosa/request_format.py` were imported by shared
request tooling but absent from the freeze. Before/after runtime identities
matched; no receipt was issued. Frozen harness03 and the failed engineering run
remain unchanged. Harness04 will only correct source inventory/path bindings;
the candidate and production sources remain unchanged. Fresh tiny/full acceptance
is required before timing.

Independent V9 opportunity evidence is at
`/tmp/deepseek_v9_next_opportunity_review_20261006_01/memo.md`, SHA
`f70619ee0a3c82681841249269c451ea3923a543881fd1dc9323255702d2ab00`.
Post-device-barrier duplicate commit synchronizations total 8.340/18.125/14.965/
18.843 us in HBM/ECHO/serial/dense; batching owner drains could improve only the
shared tail, not any layer gate. A separate later quantizer-layout hypothesis
could remove 27 scale-transpose control kernels, totaling about 59.5 us in three
methods; dense's actual DMA overlap leaves only 19.84 us exposed. These are
single-profile opportunity bounds, not measured savings. Quantizer work must
follow KDA and preserve official GEMM calls and scale/FP8 bits. Neither hypothesis
has been implemented or authorized for GPU execution.

## Recall stream03 full-model acceptance and tail-mask screen

Harness04 changes only the omitted source inventory and its own path bindings;
its final freeze SHA is
`0bb5321e4010d247ad01479c307cdef7ca73dee5648708191ff18243c0c31a86`.
The fresh tiny04 run passed, followed by the H65536/C1024/A128 full04 run at
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/recall_stream_model_full_check_20261006_04/`.
The full process exited zero with 176 clean discrete GPU observations. Root's
independent reader verified 224 output comparisons and 64 default-call state
files across all four methods, both residencies and both suffixes (48 layers).
Physical BEFORE states match exactly; legal cold ECHO/serial slot permutations
preserve logical records, priorities, maps, clocks and actual traffic. Root's
report is `/tmp/recall_stream_model_root_audit_20261006_03/full_states.json`.
The harness-author support reader additionally passed all prefill/all-hidden
states: 144 files, 4,644 tensors, 408 scheduled equivalence comparisons and
302,680 changed-slot pairs. Its report is
`/tmp/recall_stream_model_support_20261006_03/full04_audit.json`, SHA
`2565a1dbef1379ea221fc89ef388e3e5281a48666293059abd2b0f68023de2fa`.
Runtime, memory and observer support checks passed, including 6,491 tree-hash
records over 6,043 unique files. The fixed 40-pair full-model benchmark is now
running with other heavy CPU/GPU work stopped. No promotion or new gap/MFU
result is implied by these checks.

While the CPU state audit ran, root executed the separately frozen resident
tail-mask candidate01's GPU check. Freeze SHA is
`33bffc3209fb52009b065922ce4ff64d86e34540ccadd832fa0a7174d7f5e306`.
The check at `/tmp/resident_tail_mask_check_20261006_01/` passed, with five clean
discrete observer samples. The separate saved-evidence reader reconstructed
invalid-tail stores from actual before-storage/strides/offset/endpoints, compared
complete scores and hints, rebuilt exact radix-key top-k with the original
small-index ties, and checked actual MQA/Triton artifacts. Its report is
`/tmp/resident_tail_mask_root_support_20261006_01/audit.json`, SHA
`ee6caa99c11ed5d4cf6fe7b5e78ba9408297be0e067ef3a34edefc71ed336468`.
This is synthetic standalone adapter acceptance; model/cache behavior and
performance remain unmeasured. Its independent launch profile passed with four
clean discrete observer samples. Raw traces retain one unchanged official MQA
kernel and show three tail-mask kernels replaced by one for all three masked
cases; the no-mask control has no added launch. Timing remains pending.

The full native recall-entry feasibility memo is
`/tmp/full_recall_entry_feasibility_20261006_01/memo.md`, SHA
`a8ad2902d5d501be9665b4bfc4621679a68320416c6ed8fb3698a78e536bbc20`.
One eager call could nominally reduce 32 Tensor exports to 21, but would cross
existing CPU clock commits, waits, late allocations and fresh validation.
A faithful staged frame needs at least two Python callbacks unless those
operations migrate native. This is a distinct unmeasured mechanism, without
a ready drop-in candidate or predicted speedup. A separate narrow shared-owner
commit/drain design is now authorized for private CPU-only preparation.

The full stream03 model benchmark subsequently completed: 320 arm rows and 960
complete timed calls, with 156 clean discrete observer samples. Root and the
support reader checked every fixed sample, receipt/runtime identity and all
1,048 forward proofs. Root's pooled candidate-minus-baseline medians (us) are
ECHO cold/warm +1.039/-0.677 and serial cold/warm +2.320/-7.475. All 12 pooled
phase intervals cross zero; several AB/BA strata reverse sign. The standalone
recall benefit therefore did not establish a complete-model benefit. Production
promotion is rejected in
`/tmp/recall_stream_model_root_audit_20261006_03/root_decision.json`, SHA
`c797150d09eb2b82a965fca3bc92f424f411c6f2866269b5e38820a70f81909f`.
Correctness evidence remains valid within its private provider scope; do not
repeat the unchanged candidate or use it as the baseline for later work.

The mask01 fixed standalone screen also completed. All 640 invocations remain;
full-adapter paired median improvements are 7.0325/10.5375/6.5995 us for first
prefill chunk, last prefill chunk and extend. Their pooled and both order-stratum
descriptive intervals are below zero. The no-mask adapter control is inconclusive
and not subtracted. Check/profile/bench observers contain 5/4/4 clean samples.
The independent paired/profile reader is
`/tmp/resident_tail_mask_root_support_20261006_01/paired_profile_audit.json`, SHA
`f3aeecc256bcb6abd976b21e31091931e03f0ab19571d1dfc35fa393696efb19`.
Root authorized a private full-model trial, not production promotion, in
`/tmp/resident_tail_mask_root_support_20261006_01/root_decision.json`, SHA
`5b9c8315fd40716c23c48ae241c71c94f145378853154ddff1dcc652edda9889`.
The model harness is being prepared under
`/tmp/resident_tail_mask_fullmodel_20261006_01/` against unchanged append02/V9.

Two additional cache-only preparations remain private: a shared-owner commit
batch and CPython-PyObject classify/map metadata guards. The latter is distinct
from the rejected TensorView/FFI validators and does not add a Cython dependency;
it must preserve fresh Python getters, short-circuit order and original stage
boundaries. Neither has native/GPU timing evidence. A read-only two-buffer hint
proposal targets only ECHO startup, cannot preserve arbitrary external aliases,
and remains unimplemented. Its V9 three-copy GPU duration is 2.624 us; the
35.253 us enclosing interval is not a predicted saving. The full/layer gap gates,
fresh formal MFU and motivation rerun remain outstanding.

## Mask01 private model trial

The model harness froze at
`/tmp/resident_tail_mask_fullmodel_20261006_01/source_freeze.json`, SHA
`2f2066c39ee251606d72a4c4ffd4e1e5d602bf0888209edd5b01ecfbaf831db8`
(1,342 files). Model/layers/attention changes are exact import aliases. The
private resident adapter imports canonical `_module` and `_PreparedPrefetch`;
every cache, native bridge and fused ECHO kernel remains the current production
implementation. Root reviewed source and independently passed all 67 CPU checks.
Review SHA is `24c9cb72436964e44967575577e660abff2441bc5a9d1314471641a09c48c44a`
at `/tmp/resident_tail_mask_fullmodel_root_20261006_01/source_review.json`.

Tiny GPU check
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/resident_tail_mask_model_tiny_check_20261006_01/`
exited zero and issued a receipt, with 23 clean discrete observer samples. Root's
adapted independent reader passed 224 output comparisons and all 64 default-call
before/after files (48 layers), preserving exact physical before-state and legal
logical after-state. The reader changes only receipt kind, arm names and run paths;
lineage is recorded beside `tiny_states.json`. A separately authored comprehensive
saved-evidence/route/runtime reader is being prepared under
`/tmp/resident_tail_mask_model_audit_20261006_01/`.
The full H65536/C1024/A128 check is now running at the analogous
`resident_tail_mask_model_full_check_20261006_01` SSD path. No full mask-model
timing or production promotion has occurred.

The commit-batch CPU screen measured approximately 6.3 us additional dispatch
cost with canonical cache objects and mocked no-op CUDA synchronization. It
does not establish a GPU speedup. Its contract explicitly excludes user release
callbacks that submit CUDA work: retaining normal write sources past publication
alone cannot certify completion of work enqueued by such callbacks. A small
separate real-CUDA commit-only check/paired screen is being prepared before any
full-model trial. Public commit/drain and production code remain unchanged.

## Mask01 full-model timing and commit-batch decision

The mask01 full check completed successfully. Root independently verified 224
output comparisons and 64 default-call state files across 48 layers. The
separate comprehensive audit verified 144 states, 4,644 tensors, 216 transitions,
408 equivalences and 302,812 legal changed-slot pairs. Actual residency and
append transitions reconcile H2D/D2H counts. Audit SHA is
`0d943a489da9c15c545960685e71f62d2946458777f0149b9c34f83643441e14`
at `/tmp/resident_tail_mask_model_audit_20261006_01/full01_audit.json`.

The fixed benchmark at
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/resident_tail_mask_model_bench_20261006_01/`
completed all 320 rows and 960 calls with 157 clean discrete observer samples.
Root's paired arithmetic audit is
`/tmp/resident_tail_mask_fullmodel_root_20261006_01/paired_audit.json`, SHA
`f9fda316f0165dc291062293cf0a5d94d0a9e634554dc6728d497ee9950bff34`.
All four pooled prefill intervals are below zero. Serial warm improves by a
paired median 39.502 us, with pooled and both order-stratum intervals below zero.
Cold extend intervals cross zero for every method; ECHO cold median is -0.368 us.
The complete runtime/binding/artifact benchmark audit is pending. No production
promotion or new gap/MFU result is implied.

The private batch-commit CUDA check and fixed 80-pair screen completed at
`/tmp/cache_commit_drain_cuda_check_20261006_01/` and the analogous `bench` path.
All 12 stream/event/lifetime/state cases pass. Independent audit verifies 1,203
evidence files, all 26 before/final tensors, per-case hashes, 320 timed calls and
both observer logs. Its SHA is
`71c97488d6a85bf3a1a2f96e9fba9734e0ec87f07c52899da2d013523e8033eb`
at `/tmp/cache_commit_drain_independent_audit_20261006_01/full_audit.json`.
Zero tickets improve by 1.354 us; two normal tickets regress by 2.715 us. Pooled
and both order-stratum descriptive intervals agree with each sign. Root rejects
production promotion and a full-model trial in
`/tmp/cache_commit_drain_root_20261006_01/root_decision.json`, SHA
`7751cf0f5411003589e69ce00da5fcbaf26bb90e06909d696015bcd0f916a1e4`.
Do not repeat the unchanged candidate. Public commit/drain and append02/V9 remain
unchanged; the full/layer gap gates, formal MFU and motivation rerun are pending.

## Current production: V10 resident tail mask

Root accepted incremental integration after the mask01 complete benchmark audit
passed. Audit SHA is
`e97df127f4a409f05b188e0344b6929c8a572f3e33d72bffc857db8529670cf9`;
the integration decision is
`/tmp/resident_tail_mask_fullmodel_root_20261006_01/root_integration_decision.json`,
SHA `f4ecff1db2144df41099661ec5808c2a4188359b6022e332c37fca88639a5b4b`.
Production `echo.py` now has SHA
`de20ebc3b4905b3dddb646167fb770d27a92146e76934be3160ad1b5b149c562`.
Its AST matches the frozen candidate; formatting is the only source difference.
The cache analyzer recognizes the causal mask as compute only in its indexer
scope. Existing GPU storage tests now cover column strides 1 and 2; 21 CPU gap
classification tests and Ruff pass.

The first production public check passed all 51 tests but failed its final
archive step because the runner incorrectly required an unused common-transfer
DSO. No receipt was issued. Fresh production02 explicitly requires the one
actual ECHO DSO and passes all 51 tests, without source changes or skipped cases.
Independent audit at `/tmp/resident_tail_mask_production_audit_20261006_02/audit.json`
has SHA `11e87c91afc9328c6f742e4147a2ab50efd5138100312c25767be963315d83b3`.
It checks 816 scoped source/archive files, raw JUnit/reports and executable maps.
Actual ECHO fingerprint is `c8f43b73bc3d148d`; ELF SHA is
`09165f76c0ef4a75067939afa1c071bb68b9e8bf28e229d27681cbefdb2e0a11`.
Before/after native equality is an assertion in the preserved runner; only the
final raw map is saved. This is public-operator acceptance, separate from private
full-model evidence.

Production V10 check/bench/minimal profile completed with new run IDs
`deepseek_gap_v10_a128_check_20261006_01`,
`deepseek_gap_v10_a128_bench_20261006_01` and
`deepseek_gap_v10_a128_minimal_20261006_01`. Check passed all 13 bitwise comparisons;
the three observer logs contain 12/17/36 clean discrete samples. Independent
V10 source/output/interval audit is in progress. Initial complete-extend gap
upper bounds remain 20.2915/40.4162/34.5568/19.4249 percent in method order
HBM/ECHO/serial/dense. All L0 windows fail and only dense L1 passes; prefill's
absolute-gap ratio gate passes. These pending-audit diagnostics do not complete
the goal or replace formal operator MFU. Cache manager production check07 passed
all 12 cases with nine clean discrete observer samples; matching bench/profile
supplements are in progress. Prior valid reports remain identified with their
original implementation. The V10 runtime collector does not retain the resident
mask's live Triton CUBIN identity; source hashes and raw profile kernel activity
must not be described as that missing evidence.

The PyObject guard candidate01 froze, but root's compile-only attempt failed on
CPython internal C11 atomics included in its C++20 translation unit. No DSO was
produced or loaded. Frozen01 and build logs remain unchanged; candidate02 will
use a separate C shim for internal layout observations and address domain cleanup
error preservation before fresh root compilation. No guard implementation is
promoted. A consumer-layout scale candidate is being planned under the existing
linear-quantization KDA component; production linear/kernel sources are unchanged.

## Guard02 rejected; V10 and manager07 audits completed

Guard02 froze with SHA
`e7f29b128f20d3b0afdc45346e922847bb459ddd4d63a95747d2baee7986c82a`.
Its separate C11/C++ build succeeded; the actual ELF has SHA
`96569c9a703a39b206f9e091ac899295dfb45ca6e10e09d12fc99cc0f354daa7`.
The first CPU parity attempt passed 189 tests and failed one fixture: both
baseline and candidate retain a temporary through the thrown exception's
traceback, contrary to the fixture's immediate-destruction assertion. Frozen
implementation02 was unchanged. Verification02b separately records the corrected
deferred-finalizer assertion and passes 190 CPU plus 24 CUDA-metadata tests.
The first host-screen command failed provider authentication before CUDA because
`dladdr` returned the relative argv0 `python`; the fresh screen02c explicitly
invoked the same interpreter by absolute path. No fallback or failed request
continuation occurred.

Screen02c retains all 40 AB/BA pairs per stage, with 64 calls per arm/batch.
Classify/map regress by paired medians 3.187422/1.653641 us. Both order strata and
all descriptive 95% intervals are positive. Root rejects unchanged02 and any
larger recall/model trial; production guards stay unchanged. The arithmetic,
receipt, source, native identity, raw JUnit and observer audit is
`/tmp/recall_pyobject_guard_root_20261006_02/decision.json`.
Parity/screen have five/three clean discrete observer samples. These tests only
exercise metadata guards with no-op native targets; they do not run CUDA cache
kernels or establish recall/model performance.

V10's independent audit passed at
`/tmp/deepseek_gap_v10_independent_audit_20261006_01/audit_complete.json`, SHA
`8127dd95bf0614180da4ea8dcb2389efb813470ca9268615b49bcf24c25e3e33`.
It confirms the previously reported failed full/layer gates; labeled diagnostic
publication staging is in progress. Manager07 check/bench/profile also completed,
with 9/9/18 clean discrete observer samples. Root audit verifies 168 source
records, 747 activities, 558 artifacts and all 84 timing samples; a separate
reader checks 72 saved selection tensors against check HBM and causal/top-k
invariants. Cache state was checked by the executing verifier but not saved for
that reader, so independent physical-state reconstruction is not claimed.
Evidence is under `/tmp/cache_manager_root_audit_20261006_07/`.

Only `echo.py` differs from recall04's scoped source identity. The retained V9
and V10 ECHO ELF `.nv_fatbin` sections are bitwise equal (1,206,848 bytes, SHA
`0f1a2b6a5f1dba3ca45c519af99088eb5c75ce45df29f9804c93381191612494`).
This supports retaining unaffected native recall/append evidence under its
original ELF identity; it does not provide the missing separate live Triton mask
CUBIN. The comparison is
`/tmp/resident_mask_native_payload_review_20261006_01/comparison.json`.
Consumer-layout scale01 has passed private CPU source/dispatch/address checks;
its root-run GPU harness is being prepared. No production linear change exists.

Root published V10 diagnostics and manager07 after source/data/visual review.
Public manifest SHAs are
`c820df034f01ab5656df7bb523d5a8570939854648953bc0392a516f844630a1`
(V10) and
`61a5a2f8b98ea169b595661037413aee75226d2f3bc8f84c99da0f704c77c633`
(manager07). Replaced V9 diagnostics and manager06 outputs were removed only
after new publication; exact retirement inventories and root receipts are in
the current runs' `publication_archive/` directories. Unaffected recall04,
native append01, the captured input fixture and formal v3 MFU remain. V9's
request bytes are preserved verbatim in V10 bench/request.json; historical argv
is not edited. Seven V10 figures carry computation-purpose labels, include
complete/L0 startup boundaries and retain all gap/IO qualifications. The goal
still fails and formal MFU/motivation remain pending.

A bounded CPU frontend review found no material opportunity in precision or
hint scopes: `_precision_policy` costs 2.156 us and the three hint context
entries/exits together cost 4.025 us in the isolated fixture. All eight V10
saved precision policies match the examined state without collected exceptions.
These are standalone host costs, not predicted removable model gap. Source-bound
memo: `/tmp/deepseek_precision_frontend_review_20261006_01/memo.md`.

Scale-layout numerical candidate01 froze with manifest SHA
`4e1d9b390bf4727fe578c3a6fa82722e0abd2887b4cf7cd25e63903f6e76d9e2`.
Root's first CUDA check failed in the first empty-row comparison: empty FP32
scales retain stride zero after contiguous(), so view(uint8) rejects that empty
layout. No numerical mismatch or receipt was produced. The four observer samples
were clean. Frozen01 remains unchanged; harness02 will compare empty tensors by
shape/dtype and the empty byte string in both comparison and serialization,
preserving the numerical candidate and all nonempty bitwise checks.

## Scale-layout harness03 and full-recall preflight investigation

Harness02 preserved all eight numerical source files and fixed byte handling,
but its fresh check stopped after the empty fixture. Torch Inductor lazily set
`TRITON_PTXAS_PATH` to Torch's CUDA 13.0 assembler after the quantizer had captured
the implicit Triton CUDA 12.8 assembler identity. The unchanged identity check
correctly rejected this change; no numerical mismatch or successful receipt was
produced. Seven discrete observer samples were clean.

CPU reconstruction is preserved under
`/tmp/deepseek_linear_scale_layout_identity_diagnosis_20261006_01/`.
Harness03 explicitly declares the existing effective SM90 Triton assembler
before identity capture, then invokes the installed Inductor setup and requires
no identity change. It preserves the assembler bytes, all eight numerical
sources, and strict identity checks. Freeze SHA is
`c776c424253e16c6de8a8fb2a8350becf7e73dd1537b711c407d0beb404e1de3`
at `/tmp/deepseek_linear_scale_layout_candidate_20261006_03/freeze_manifest.json`.
Root verified all 34 files and started fresh check03 on GPU3, CPUs24–31/NUMA0.
No scale-layout correctness or performance acceptance is claimed yet.

In parallel, a distinct full eager recall entry is under CPU-only planning.
It would preflight fresh metadata and allocate scratch before execution, then
submit the original classify/allocate/gather/publish/map functions through one
native call. A host progress ledger would preserve completed CPU clock events
on synchronous failure without stage callbacks. This explicitly revises the
private preflight failure boundary; it does not remove public guards, cache
leases, original native errors or unsafe-owner retention. Root subsequently
authorized private source implementation and narrow CPU checks following review
of `full_recall_entry_plan.md`; native builds, GPU screens and production changes
remain separate gates. The current compute-only graph rule remains in force.

Scale-layout check03 completed successfully: 213 adversarial fixtures, four
small-row padding cases, eight lifecycle/stream/graph sequences and 33 complete
linear cases passed. Its result SHA is
`b5cb612c9755005b82eae5caaf831502331aae658563131597519bb1ec71db57`,
under `/mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_check_20261006_03/`.
The observer has 213 clean discrete samples, with a maximum 2.280058-second gap.
Independent saved-output/identity review is pending; no timing or promotion yet.

The saved-output review subsequently passed: 1,278 quantizer tensors and 132
linear tensors, 918 byte comparisons and 4,137 bound files were checked. Reader
adaptations account for inherited CUDA_VISIBLE_DEVICES and Torch's UUID without
the nvidia-smi `GPU-` prefix; raw run records remain unchanged. Audit SHA is
`ce6b566dfe6c337696d047939a7b94f6c062c6480583f3b06329af9b74cdf735`
at `/tmp/deepseek_linear_scale_layout_independent_audit_20261006_03/audit.json`.
The separate compiler declaration review also passed.

Bench03 completed all 30,720 calls with 15 clean observer samples. Its result
SHA is `e2bafea409f19484058ebb200a56d84bfab00b2818681025bb2da7e1a5d37c7c`.
Independent arithmetic/runtime audit SHA is
`8ed4093bced01c42e5abf69d5f779cb8d7b9601bfcec4af1f93da6a18beddb57`
at `/tmp/deepseek_linear_scale_layout_bench_audit_20261006_03/audit.json`.
All Q128/Q1024 eager and owned-graph paired medians improve in pooled and both
order strata for both timers. Q1/K16384/N7168 graph calls regress; Q1024/K7168/
N1536 borrowed replay has an order reversal. Unrestricted deployment is excluded.
The KDA contract now explicitly limits prospective model deployment to verified
Q128/Q1024 shapes and retains all other results. Confirmation and independent
profile remain required; no full-model benefit or production change is claimed.

Confirmation03 and independent profile03 review have now passed. The confirmation
contains another unchanged 30,720 calls; its audit SHA is
`62d14ea00139a5f671cc3cf95d7b7a86bd1002d72ff4d18f49ed33ab7f9f70a6`.
The two-window comparison SHA is
`e8e769614479980873f529d8e3be8b220a4f55a8dcd3b7e844decf352110af6f`;
both are under `/tmp/deepseek_linear_scale_layout_bench_audit_20261006_03/`.
Each window retains all Q128/Q1024 eager/owned paired-median improvements in
pooled and both order strata. Q1/K16384/N7168 graph regressions persist;
confirmation owned wall/event deltas are +0.9845/+1.648 us. Q1024 borrowed replay
also retains AB/BA reversals, including K7168/N576 in confirmation. These
excluded/qualified cells remain in the evidence; no universal speedup is claimed.

Independent profile audit is
`/tmp/deepseek_linear_scale_layout_profile_audit_20261006_03/audit.json`, SHA
`6ac5ad180259b923612a5bf0cf54624d05fa233655c87ef85fd2756f235e8da5`.
It checks 192 scopes/96 pairs, 480 kernels, 128 expected owned-graph D2D copies,
160 graph-node lineages and reverse coverage of all 416 scoped submission APIs.
Every baseline has one activation transpose and every candidate has none; no
replacement copy or memset appears. Original GEMM call AST and launch signatures
match. The actual provider and quantizer artifacts are bound; separately observed
per-GEMM live JIT CUBIN identity is not available. All 41 observer samples are
clean, with a maximum 3.439745-second sampling gap. This is component evidence,
not a new complete-model gap or formal MFU result.

The narrowed scale candidate is eligible for a private Q128/Q1024 model trial
after integration checks. Cache-manager candidates remain first in execution
priority: the depth-one full recall entry is finishing terminal-disposal CPU
acceptance; root is preparing independent native ABI/stream probes and its
fixed 54-observation/720-call harness. A separate dense post-clear submission
plan is under source review. Production remains V10 and the goal is incomplete.

Full eager recall candidate01 is frozen at
`/tmp/full_eager_recall_candidate_20261006_01/freeze.json`, SHA
`e7eecee5b33df8c8fce0824d4d525bb39d4ae34d386661b810ca03cc87432a91`.
Root verified all 23 files and six production baseline hashes. Its 43 CPU checks
cover dispatch/lifetime and private irreversible terminal disposal. Root also
ran 54 host-native descriptor/order/progress/error tests on the unchanged C++.
Actual CUDA terminal disposal and original-kernel state/fault acceptance remain
separate gates; nested model leases remain excluded.

Root's independent real Torch transport probe passed 96 offset/suffix/stream/
allocator/fault combinations, ten invalid metadata cases, a same-Tensor
valid/invalid/restored descriptor sequence and real pageable-host rejection at
the original post-allocation boundary. Result SHA is
`24583e8a26f8f89bbdb9f3c970ef5f1c3e5af09dd04e71ce411264a6f18619ca`
under `/mnt/ssd-wlcb/chenkaiqi/codex-checks/full_eager_recall_transport_20261006_01/`.
Independent audit SHA is
`10474cca5f985f551ab4d9d16fd88a99c4155f1a8e00c9e79b866ce62c7f6a92`
at `/tmp/full_eager_recall_transport_audit_20261006_01/audit.json`.
The observer has four clean discrete samples but observed no owned GPU PID;
the brief CUDA interval fell between samples. Runtime reports the assigned GPU
UUID and affinity. The probe used stage stubs, not cache kernels; it binds the
mapped probe path/hash without the stronger live inode/device attestation.

The fixed large-fixture correctness check may run while the smaller original-
kernel adversarial/fault harness is being prepared. This changes scheduling
only: no performance screen is permitted until both correctness gates pass.
The fixed harness additionally saves full before/after storage and CPU proofs,
asserts transaction/lease/poison state, and compares candidate/baseline proof
state while allowing reference-only differences. It retains 54 observations
and the later fixed 720-call AB/BA screen. No current result claims a cache
speedup, complete-model gate, or production promotion.

The dense post-clear plan passed independent source review and its isolated
implementation/CPU checks are authorized under
`/tmp/dense_post_clear_candidate_20261006_01/`. It combines wait/copy/publication
after the original clear/clock/event/lease boundary, with explicit copy-stream
binding after FFI conversion. Original CUDA work and both event records remain;
2-to-1 post-clear FFI calls do not imply a measured benefit.

## Full eager recall standalone gates passed; nested integration active

Candidate01 remains frozen. The fixed large-input check passed 36 arm/reference
checks and saved 54 complete before/after observations. Result SHA is
`62c63c2eb5fa21104c0391c9d8131d6e4e7ebd3c597920910694d084e337500d`
at `/mnt/ssd-wlcb/chenkaiqi/codex-checks/full_eager_recall_fixed_check_20261006_01/result.json`.
Independent audit SHA is
`dc3caa180019a459312d85c8c878a57d47b6411582b424b636d036ed6beeb735`
at `/tmp/full_eager_recall_fixed_audit_20261006_01/audit.json`.
It reconstructs records, host state, maps, free bitmap, priorities, clocks,
physical outputs, traffic, counters and 18 paired CPU-proof states. All 22
observer samples were clean. Actual bridge ELF SHA is
`bb70691046077dbd2ed4c208db9aadb5fba2bc59dce84e7218f0a312d780746e`.

Small-state harness01 failed because its partial-release fixture passed a
noncontiguous int64 view to the correctly guarded production API. It issued no
complete receipt. Frozen harness02 materializes the same IDs contiguously and
adds post-terminal CPU owner evidence; candidate01 remains unchanged.
All 25 success observations and 20 separate original-kernel fault processes pass.
The independent combined receipt is
`/tmp/full_eager_recall_state_independent_review_20261006_02/receipt.json`, SHA
`6d0813f21140086e906b27f3e9d65a0dd93fb653ca78da47b107c1d4a3e159b9`.
The check and fault audits alongside it have SHAs
`a1b24e02298ae743de458908555ef7987fe8a77751eb00a86288e95444b6f7b8`
and `4cfef30865f5cfd2de0c3c62fd5f382ae5cc5e3e247d2b54c86a17a1259fba7b`.
The reader checks own-before FIFO/free-first state, all-session counters,
subsequent accesses, exact signed stage traces, frame/21 ledger, partial CPU/GPU
clocks and post-terminal owner clearing. Check/fault observers contain 4/61
clean samples with owned PIDs; maximum interval is 2.250041 seconds.
Actual fault-forwarder ELF SHA is
`477dfd54e10038242f7f88c0f9d355c93047068ddbaca2a90342daac0a645b6c`.
Exact Cartesian commands are indexed by
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/full_eager_recall_fault_batch_20261006_02/completed.json`.

The unchanged fixed screen completed all 720 calls at
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/full_eager_recall_fixed_bench_20261006_01/`,
using the matching fixed-check receipt. Result SHA is
`112911ce433b1aec473f2a86c7514dcdc1cb3a09771e60abb3d5f568437eb3f5`.
Restored-hit paired wall deltas are -24.567/-22.8135/-24.455 us; cold deltas are
-9.013/-9.214/-8.6355 us for L0/L1/L2. All six changed cells improve in pooled
and both order strata for wall/enqueue, with descriptive intervals below zero.
Certified controls regress by pooled medians +0.770/+0.947/+0.9755 us; retain
these results without subtraction. This is single-window evidence, not
run-to-run assurance or a complete-model gap result.

Root arithmetic SHA is
`4109b1e5ff607347561f7c3fe526f91eb2eb1ddaffc4b7de3afb6be6a256b9ba`
at `/tmp/full_eager_recall_root_screen_audit_20261006_01/arithmetic.json`.
Independent audit SHA is
`fa738844e00c7a1e87c6f07692470107d24f334d4ca9dcbe7aaa93fcc8347df8`
at `/tmp/full_eager_recall_bench_independent_review_20261006_01/audit.json`.
It verifies 1,267 files, receipt/source/native identity, all 720 rows/360 paired
metric equalities, actual miss bytes and exact summaries. All eight observer
samples were clean. Root decision SHA is
`dde020bc4322336966fbc0faf144ebb49fd278fa3aeeff314a142c4f1ef65dcf`
at `/tmp/full_eager_recall_root_screen_audit_20261006_01/decision.json`.

Root authorized the reviewed `nested_recall_lifetime_plan.md` as a separate
private implementation under `/tmp/full_eager_recall_nested_candidate_20261006_01/`.
Its source is now frozen with SHA
`b71eee6a94dd80fecaf577c4ff165a3e0e552ccbc4f6001927a8ad6e9f22b793`.
All 67 CPU checks passed without CUDA initialization; receipt SHA is
`74ce3c6afc23e5d83c23d14fa9e51ab13065bc09662c9c79f6d7d93502e9e353`.
Root source review verified every frozen input and all six production hashes,
and accepted the one-call, one-stream-per-lease private scope for hardware checks.
The review is `/tmp/full_eager_recall_nested_root_review_20261006_01/audit.json`,
SHA `6b9338d5e8ceb6e14f0cd6b3dd796ef01bd86c25560a62c867e2088c31eb67f5`.
Attention_graph is adapting the accepted state/fault and fixed-screen harnesses;
root owns all builds/GPU work. New token/wrapper costs must pass those fresh gates
before model timing. The cold standalone margin is only about 9 us. No unchanged
standalone rerun or production promotion follows from source acceptance.

## Dense post-clear native probes passed; actual-state harness active

The separately frozen dense candidate passed 30 CPU/mock cases and root's
host-only compile/symbol-load check. Actual candidate ELF SHA is
`b9a9f160bc11e61362074a177844615fb07f3744957ebc0f3fdb7ca630d08396`.
Native-probe01 was not executed: review found cleanup paths that could bypass
draining and a probe library name excluded by the live-DSO collector.
Frozen probe02 fixes these harness defects without changing the candidate.
Its freeze SHA is
`c69ded2dc5441b854950ee88d5b06cae6670a00972991e06387d3ca092217563`.

Root ran 56 ABI/stream/fault cases plus seven metadata rejections on GPU3, then
repeated that coverage with devices 3/4 visible and current logical device 1 to
exercise restoration. Both passed. Main/device result SHAs are
`bddb79ececab6e3ef15d612c2804dbf8ce7902ed156c1e1cf95a9a10e824bb4f`
and `9db113e7d131fc475407abace6d7be76e801e77cc73a5515cb4109ca55fa7050`.
Actual probe ELF SHA is
`e4bd1c8a4d103ddb893fe5efc5ebad16a6782c013857bfbd1130616297915ff5`.
The 5/4 observer samples were clean. Independent audit SHA is
`4d90ab75ad2ddddcbac14951ada4e4d61c2d922005c7be2d31ac81a5e4b92ed9`
at `/tmp/dense_post_clear_native_execution_audit_20261006_02/audit.json`.
The reader verifies scenario coverage, source archives and actual native
libraries. Pointer/stream/restoration identities remain source-bound live
assertions; individual expected pointer/stream tuples were not serialized.
The real CUDA event-wait API was called, but copy/publication targets were
native stubs. This establishes no actual DMA/state or performance result.

Actual-state harness01 completed its state comparisons but failed final runtime
identity and issued no acceptance result. Source review found that the original
recall bridge was first loaded by pool construction, after the initial identity
snapshot. Exact runtime differences were not saved in that failed run. Frozen
harness02 explicitly initializes the production provider before the snapshot and
saves both runtime records while preserving strict equality. Its freeze SHA is
`6dfb1dbf93b26e7de598677e98b75952569ce6002576602b47660f0abbbdea36`.
The unchanged fixture/cleanup logic reuses its 15 CPU cases; no new CPU execution
is claimed for the three-call initialization/evidence repair.

All fresh small, captured cold/certified, 11 controlled fault and one actual
asynchronous CUDA-assert gates now passed. Independent acceptance bundle is
`/tmp/dense_post_clear_state_independent_review_20261006_02/acceptance_bundle.json`,
SHA `6a4e430eb00aa06d24e90d4b590bfcc24523d2f67a148e60cd9aa5f42a9a4c19`.
It binds 42 complete own-before layer-state reconstructions and 43 NSYS scopes
with 452,987,408 actual H2D bytes. Captured cold contributes 452,984,832 bytes
across both arms and three layers; certified contributes zero. The controlled
drain and actual CUDA-assert cases retain owners and read no after-state.
The actual assert is after successful dense submission, not inside DMA, and its
three clean observer samples saw no CUDA PID; that isolation limitation remains.
The input receipt's canonical payload digest and file-byte hash were independently
verified as separate fields; their difference was not an artifact mutation.

The fixed standalone screen is frozen at
`/tmp/dense_post_clear_screen_20261006_01/freeze.json`, SHA
`31d197be70c6046180cfac4be8a1a6ce49d325d8543a97877c7419a91ffc1f1d`.
Root source/schedule review SHA is
`48df2a2fa18feaa4a60f48b882dc34a683d29ad22db2973525ec82dc17ec8ba1`.
The screen retains four warmup and 32 balanced measured pairs per cold/certified
state, raw enqueue and complete wall endpoints, and actual per-layer counters.
Root launched `dense_post_clear_fixed_bench_20261006_01` in the engineering
acceptance directory; independent sample/runtime/observer audit is pending.
Production remains V10. No dense speedup, model promotion or gap acceptance is
established by these correctness gates.

## Scale-layout model adapter: CPU preparation only

The private Q128/Q1024 model-binding plan is
`../../kda/deepseek_linear_quantization/consumer_layout_model_trial_plan.md`.
The adapter at `/tmp/deepseek_scale_layout_model_adapter_20261006_01/` borrows the
original 27 checkpoint linears and replaces their calls before graph preparation.
It leaves unsupported shapes on the original implementation and rejects an
incompatible private prepared activation through the original consumer guard.
The frozen numerical component remains unchanged.

All 17 current-source CPU binding/dispatch/rollback tests passed; these use meta
weights and mock CUDA descriptors, not GPU numerical execution. Source/CPU freeze
SHA is `0f62318c662bd0a9c1f4819a5cbbaba2fb9255289fe14892603de573f4748f05`.
Independent source review passed and verified 50 bound files. The receipt is
`/tmp/deepseek_scale_layout_model_adapter_independent_review_20261006_01/source_review.json`,
SHA `efe04286a111d2643c14d6bf0d768ec0181ed35173f166cd1abd7edf1b0cfb3e`.
Cache state/fault and standalone timing gates remain first; no scale-layout
model trial or production change has run.

## Nested recall hardware acceptance and dense screen decision

Nested small-state check completed all 25 observations. Independent own-before
state, token, source, native and observer audit is
`/tmp/full_eager_recall_nested_independent_review_20261006_01/check_audit.json`,
SHA `a91b0807e70a6c19ce02eed8bba1f560fb4650dd74307b010c7698311e71508f`.
The targeted fault batch completed its first 14 cases and stopped at case 15,
`free_body_gather_post_pending`: the fixture's D2H event had completed before
the required pending-state check. The candidate had not been called. Case 16
was not run. The failed fixture is not accepted and the original batch is not
retried. A separate instrumented setup diagnosis is checking where completion
occurs; pending-body and pending-swallowed still require valid fresh fixtures.

The fresh fixed-input correctness check ran independently while the pending
fixture was being diagnosed. All 54 saved observations were generated under
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/full_eager_recall_nested_fixed_check_20261006_01/`.
Result SHA is
`592dcaf1cd68ed38afbaeba65e837b03b749322a57fc86fb152d203687d13047`;
21 discrete observer samples were clean, with maximum interval 2.215412 seconds.
Independent saved-state review is pending. No nested benchmark is authorized
until both this review and all targeted fault gates pass.

Dense post-clear screen01 completed 144 calls and 432 layer records. Result SHA
is `b815c66860b2acb3dd3d3f0ee01a80f2e6ca5c5ca1005fc6f39d16afc109aa6a`
under `/mnt/ssd-wlcb/chenkaiqi/codex-checks/dense_post_clear_fixed_bench_20261006_01/`.
Paired candidate-minus-baseline complete-wall medians are +28.388 us for cold
and +51.569 us for certified state. Cold AB/BA medians are +10.004/+40.939 us;
certified AB/BA are +19.223/+84.155 us. Cold enqueue improves by 21.584 us,
but complete latency regresses in both orders. Root rejects this candidate
for model integration; independent arithmetic/provenance review is pending.
The unchanged candidate will not be rerun. Production remains V10; no new
complete/layer gap, formal MFU or motivation result is claimed.

The dense screen independent audit subsequently passed: all 72 pairs, 432 layer
records, exact bootstrap/order-stratum statistics and 1,259 bound files were
verified. Audit SHA is
`3aa1e48feb28d9330eafe183d55fad0f384ae27c4f3eca667fc558fd10dea0cc`;
rejection receipt SHA is
`129e9ecc0c43195ba9768eab992bdd27d756d11459ad09abbd9691160941299a`,
both under `/tmp/dense_post_clear_screen_independent_review_20261006_01/`.
No dense post-clear work remains for this unchanged candidate.

Nested fixed54 independent reconstruction also passed: audit SHA
`6e54fdd5a199fa24049e3209bc19a47e691f9832eb621b8b650931cb1dfd7f5d`
in `/tmp/full_eager_recall_nested_independent_review_20261006_01/fixed_audit.json`.
The first14 fault audit SHA is
`e8a419cf34ab2cb98ce6399f953f1aafe17862757e9ef7b30132b5c9b50a7ddf`.
Pending fixture diagnosis01 found the delay already consumed before write_host;
diagnosis02 found an incomplete event just after write_host, then completion
during the remaining roughly 302 ms of initialization. Both attempts stopped
before the intended recall call. Harness03 initializes the actual cold path once
with the accepted success helper, saves its two setup observations separately,
then constructs the delayed fixture. It preserves the 500M-cycle delay and
incomplete-event assertion, with no retry. Freeze SHA is
`a924ec50ce17d6a3cf508ce55e41cbb44fc5dfc8d6111b1dee03c6c5fb4c4db9`.

The scale-layout full-model harness is source-ready at
`/tmp/deepseek_scale_layout_fullmodel_20261006_01/`, freeze SHA
`cd3335f47c6ce1a65fb6b6851275b5c4b29c648c3b7d57e05bed5cea84341b32`.
Its 44 CPU tests passed; root source review verified 1,371 files, SHA
`9603c6a199afdfbf5c76c4fc553061ee484e4f88bcf7dff5204ae8e91e9a53ad`
at `/tmp/deepseek_scale_layout_fullmodel_root_review_20261006_01/audit.json`.
Two separately owned canonical models/banks compare only the pre-graph adapter
binding. The check retains all extend hidden/top-k and default last-token logits,
without extra all-prefill hidden or all-token logits diagnostics. Cache-manager
gates retain execution priority; this is not GPU numerical or timing evidence.

## Cache candidate closeout; scale-layout model check started

Both harness03 pending fault gates passed. Their independent audit verifies
the same incomplete D2H event/source before recall and after outer unwind, then
source/event clearing only after terminal drain. Combined nested acceptance is
`/tmp/full_eager_recall_nested_independent_review_20261006_01/receipt.json`, SHA
`23b9fc1c45949e093dc5180389c092e693bb78b1fd049fe590e9b12a056da840`.
It binds 25 small-state observations, 16 targeted fault cases, four separately
saved cold-initialization observations and the fixed 54-observation check.

The fresh wrapper-inclusive 720-call benchmark completed, but its model gate
failed. Paired complete-wall candidate-minus-baseline medians (L0/L1/L2) are
+6.0925/+5.5135/+6.315 us for certified state, -16.303/-18.193/-18.955 us for
restored all-hit state, and +0.5055/-3.2555/-5.486 us for cold state. Cold L0
regresses pooled and AB; cold L1 reverses order (AB +2.7555, BA -5.693 us).
Enqueue improvements do not establish complete-call improvement. Root rejects
this nested version for model integration and will not repeat it unchanged.
Independent full benchmark audit is pending; root arithmetic SHA is
`6377842527a32cea8ff48b219e32ef2c4b77ebe6957e204048616722ef4508e5`
at `/tmp/full_eager_recall_nested_root_screen_audit_20261006_01/arithmetic.json`.
The benchmark observer has nine clean samples, PID 308809, maximum interval
2.207224 seconds. Conditional full-model notes remain unimplemented at
`/tmp/full_eager_recall_nested_fullmodel_plan_20261006_01/memo.md`.

A further read-only V10 review found no distinct high-impact cache-only
candidate beyond these evaluated families. Memo SHA is
`cefe06956913d569f91e88308e331a1327bbbda73452843d1c13dc58d4647418`
at `/tmp/deepseek_remaining_cache_review_20261006_01/memo.md`. This is bounded
to the current measured implementation, not a claim that all cache designs are
exhausted. Root moved to the narrowed scale-layout model path only after these
cache gates. Fresh full-model check is running under
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_scale_layout_fullmodel_check_20261006_01/`.
Production remains V10; complete/layer gap gates remain unmet.

The nested benchmark independent integrity audit subsequently passed and
confirmed rejection for model integration. Final audit SHA is
`7ff2dba61e2794449578f3915b320c5b847ee6154c60c0f11434be682adfc117`
at `/tmp/full_eager_recall_nested_independent_review_20261006_01/bench_audit_final.json`.
It verifies all 720 rows, 360 paired metric equalities, both timers and order
strata, exact source/native/receipt identity and nine clean observer samples.
The numerical acceptance remains valid; the performance rejection does not
invalidate the state/lifetime evidence. No unchanged nested model trial follows.

Scale-layout full-model check01 completed 388 comparisons and saved 32 output
cases plus 144 before/after cache-state files. Result SHA is
`3a24f50b5262263e6116bcd1e2d6f8fec88a44daed6ee3609777850b129ca2f4`;
receipt file SHA is
`e8b0961c94aa8ad7392cbb1d3d42345791085a5fb08bbd1b6b63c1c32f2d4db3`.
The before/after runtime records match and both model owners closed without
execution or cleanup errors. All 181 observer samples were clean, with PID
310129 and maximum interval 2.240374 seconds. Independent saved-byte and runtime
audits are in progress; the 40-pair model benchmark has not started.

The independent scale-layout model acceptance has now passed. Combined receipt
SHA is `35b043a942dc5eb356affd6afb2d8d323ba36da1f03c3f4c5d65fac6470ceaea`
at `/tmp/deepseek_scale_layout_fullmodel_independent_audit_20261006_01/receipt.json`.
It binds 8,794 files and independently checks 32 output cases, 144 states,
216 own-before layer transitions, 408 layer equivalence comparisons and all
388 emitted numerical checks. It also verifies 32 bindings, 160 forwards,
216 setup calls and 192 quantizer entries. Cross-owner slot permutations are
proved against logical values and priorities; actual own-call traffic is retained.

The limits remain explicit: every extend hidden row and top-k plus default
last-token logits are covered, not full-prefill hidden/all-token logits or unused
free-slot payloads. Checkpoint metadata and 163 shard file identities are checked,
not complete weight payload hashes. Raw live Python marshal was matched to its
frozen expected digest but independently reconstructed only after code
normalization; no serialized live code object was saved. Live resident-tail
mask CUBIN/MLIR and run-time NVCC binary identity were not separately retained.

Root launched the fixed 40-pair model benchmark with the accepted matching
check receipt at
`/mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_scale_layout_fullmodel_bench_20261006_01/`.
It covers 320 arm rows (160 pairs) and 960 complete forward timings across four methods
and prefill/cold/warm phases. No model speedup or gap acceptance is claimed yet.

## User pause and timeline delivery (2026-10-06)

The user explicitly stopped optimization and requested a progress report plus
timelines. Optimization is paused; no confirmation run was declared or started,
and no GPU/profile/optimization process remains running. Timeline work uses only
the accepted V10 capture and preserves its original measurement identity.

The first scale-layout model benchmark completed 960 timings. Root provenance
audit passed (8,647 files), SHA
`fc066e7daf13f0a9cd2ca7f6e74bf1249a0c8fb14f4e4792803bafb6d7e7145c`,
at `/tmp/deepseek_scale_layout_fullmodel_root_bench_audit_20261006_01/provenance.json`.
Independent arithmetic/order review also passed, final review SHA
`50db945f79381021443f38fcdc32e9e119a195b8a3fb16cd947608ede523c83a`,
at `/tmp/deepseek_scale_layout_fullmodel_bench_independent_audit_20261006_01/review.json`.
Paired candidate-minus-baseline median microseconds (prefill/cold/warm) are
HBM -2214.059/-54.091/-55.572; ECHO -3454.092/-82.778/-66.157;
serial -2710.877/-51.543/-63.381; dense -3250.454/-17.418/-60.219.
Dense cold AB/cold-first is +2.847 us and BA/warm-first is -42.442 us.
Arm and residency order are confounded; the pooled negative estimate does not
establish a universal dense-cold speedup. Production integration remains deferred.
No new scale-layout gap profile, formal MFU or motivation run exists.

Production remains V10. All four complete extend and L0 gap gates fail; only
dense L1 passes among twelve layer windows. All complete offload prefill gap
ratios pass the 1.2x HBM gate. The current drawing task is a CPU-only presentation
of existing evidence, not an optimization continuation or a new measurement.

The requested timeline delivery is complete at
`experiments/deepseek_v32_mfu/report/gap_v10/with_gap/`.
CPU-only redraw ID `deepseek_gap_v10_timeline_redraw_20261006_02` preserves
profile ID `deepseek_gap_v10_a128_minimal_20261006_01`. Seven SVG/PNG pairs
add an explicit red gap lane and retain computation-purpose labels. Independent
review reproduces all 28 integer-nanosecond gap complements, original windows
and ratio bounds; all 15 inputs and 14 image hashes match. Final render receipt
SHA is `4a73823c9baf8762092eb38a71a7bc8954ba0b84fa7964295c237a467abf10e4`.
Only the new drawing source, this ledger, report supplement and MFU README were
changed for delivery; all 19 originally published V10 assets remain unchanged.
The first drawing draft was superseded and removed. Optimization stays paused.

The user requested both prefill and extend timelines directly. An additional
complete-prefill overview is now published alongside complete extend under
`report/gap_v10/with_gap/complete_prefill.{svg,png}`. It reuses the same accepted
V10 capture, covers all 64 chunks and startup/shared tail, and places all
9,475 computation kernels per method in eight named-purpose lanes. GPU activity
counts 14,213/14,492/14,285/14,405 and complete gap totals exactly match the
accepted audit. Drawing ID is `deepseek_gap_v10_full_prefill_timeline_20261006_01`;
its source and imported redraw helper are bound in publication.json. Eight
figure pairs are available in total. No GPU measurement or optimization resumed.

The user requested all computation in one row with less fragmentation. The
primary display is now `report/gap_v10/with_gap/prefill_extend.{svg,png}`:
four method rows by two phase columns, one Compute lane plus IO and Gap per
panel, four coarse purpose colors and fused-IO hatching. All eight complete
windows and original kernel gaps remain unchanged. Drawing ID is
`deepseek_gap_v10_compact_timeline_20261006_01`. No GPU work or optimization.

The user requested separate prefill and extend figures while retaining one
Compute lane. Primary figures are now `with_gap/prefill.{svg,png}` and
`with_gap/extend.{svg,png}`, produced by compact_timeline --layout separate.
Drawing ID `deepseek_gap_v10_split_timeline_20261006_01` retains exact equality
of all eight metric dictionaries with the combined version. No GPU work.

The user requested distinct IO colors and only one labeled prefill chunk.
Current primary split figures use orange H2D and purple D2H within upper/lower
bands of the same IO lane, preserving simultaneous dense reads/writes. Each
prefill method marks only the final chunk64 bracket; the entire phase is shown.
Drawing ID is `deepseek_gap_v10_io_colors_20261006_02`; all eight metric
dictionaries exactly match the prior combined/split views. Both figures and
publication hashes were updated; optimization remains paused.


## Final README timeline scope: last prefill chunk and three-layer extend

The user confirmed L0 first compute kernel start through L2 last compute kernel
end for both timeline and displayed gap. Prefill now shows only chunk 64/64
(1,024 tokens), extend the 128-token batch. The experiment README embeds only
the two final figures; prior timeline variants are no longer displayed there.
Compute purposes use color within one lane, with names only in the legend.
H2D and D2H use separate orange and purple labeled lanes.

Fused compute+IO is entirely productive, excluded from gap, and retained in the
gap-ratio denominator. Only standalone IO outside computation is subtracted.
The eight windows were independently checked with an integer-nanosecond event
sweep; the final deterministic ratios were checked against those intervals.
Extend ratios for hbm/echo/serial_sparse/dense_prefetch are
10.104131422/15.670954489/25.445241008/4.587226794 percent.

CPU-only drawing ID: deepseek_gap_v10_three_layer_window_20261006_03.
Published artifacts remain under report/gap_v10/with_gap, with windows.csv,
split_receipt.json and window_audit.json binding the final views. Original V10
profiles, complete-phase benchmark times and production implementation remain
unchanged. No new GPU work, candidate promotion or optimization was started.
