# C5 output layout and C6 boundary probe

Status: component GPU screening accepted on 2026-10-04. The assigned GPU 0
processes have exited 0 and the window was released to root. These results do
not replace the formal C4 run or establish integrated C6 serving correctness.

## Source and validation boundary

C5 changed only the CUDA destination layout of
`CheckpointAttention.output`: contiguous `[Q,heads,v_dim]` storage receives
the vendor BMM through its transposed `out=` view. Flattening for the existing
`o_proj` aliases that storage. CPU dispatch and validation were unchanged.

The measured model source SHA-256 is
`a50b73b283b79acf6628d708564af1d9fd66f4b0269920f42ef5a05f78c99e27`;
its before-change snapshot is `/tmp/deepseek_output_layout/echo_model_before.py`,
SHA-256 `cb4bfcd40f7675ccb76097a2c7eaa36f13cfe27eeb497ac709866849f7a8a0a7`.
Root subsequently factored this code for graph policy v2. That newer source
is not covered by this source identity or these component results.

CPU model/block/instrumentation and layout tests passed: 44 cases. The eight
new layout cases retained CPU tensor dispatch while selecting the CUDA branch;
they checked exact values, storage ownership, input preservation and contiguous
flattening at Q=1/5/128/1024 with contiguous/strided inputs. They were not
reported as CUDA checks.

The subsequent GPU probes used the assigned quiet GPU 0, NVIDIA H200 / SM90,
the repository environment and explicit IEEE FP32 matmul policy. Actual
checkpoint source layers 0–2 retained their FP8 linear weights. Primitive
fixtures and graph adapters were under `/tmp`; no third-party source changed.

## C5 complete output API

Harness: `/tmp/deepseek_output_layout/gpu_probe.py`.
Result: `/tmp/deepseek_output_layout/gpu_20261004_01/result.json`.

All 18 random BF16 edge cases (layers 0–2, Q=1/5/127/128/129/1024) matched
expanded values and complete output bytes exactly. All six additional
Q=128/1024 cases used official FlashMLA outputs from real checkpoint
projections over a local all-causal Q-token chunk at long RoPE positions.
Expanded values, complete output, captured output and input preservation were
exact. Long RoPE positions do not imply a retained 64K history in this fixture.

Ten warmups precede 50 interleaved samples per variant. The table gives ranges
of median synchronized wall times across source layers 0–2, in milliseconds:

| Q | Scope | Before | C5 direct destination |
|---|---|---:|---:|
| 128 | Expansion API, including allocation/packing | 0.0615–0.0738 | 0.0439–0.0506 |
| 128 | Complete eager expansion plus `o_proj` API | 0.2510–0.2716 | 0.2397–0.2578 |
| 128 | Captured complete output API | 0.0905–0.0974 | 0.0794–0.0849 |
| 1024 | Expansion API, including allocation/packing | 0.1329–0.1356 | 0.0929–0.0955 |
| 1024 | Complete eager expansion plus `o_proj` API | 0.4364–0.4429 | 0.4198–0.4266 |
| 1024 | Captured complete output API | 0.3286–0.3339 | 0.2809–0.2860 |

Captured output API timing includes submission and synchronized completion of
that API. It does not include the production finish callback's input copies,
post-attention norm or MLP. Those costs are included in the separate C6
callback measurement below. JSON preserves every wall/event sample and p90.

The two diagnostic traces and `trace_audit.json` correlate every GPU activity
through actual Kineto external/correlation IDs. Q=128 and Q=1024 each retain
the same named vendor BMM entry point while eliminating one BF16 packing
kernel. In the diagnostic expansion traces that removed kernel takes
6.144/42.112 microseconds respectively. Complete output goes from five to
four kernels, with the same small BMM workspace memset. No replacement output
packing kernel was observed. These trace durations are separate from the
uninstrumented timing samples.

## C6 complete finish callback

Harness: `/tmp/deepseek_graph_boundary_probe/probe.py`.
Result: `/tmp/deepseek_graph_boundary_probe/gpu_20261004_01/result.json`.
Twelve complete-callback traces and their `trace_audit.json` are alongside it.

The temporary adapter compares five variants: C4 copied attention, C5 direct
output with copied attention, C5 plus stable saved-residual reuse, eager
expansion into a smaller static input, and eager expansion plus saved reuse.
The captured finish after eager expansion contains the unchanged output
linear, residual RMS norm and MLP.

All twelve source-layer/query/residual-branch configurations passed two
changed input/position cases. Checks cover all six projection tensors,
captured saved state, expanded values, output projection, normalized hidden,
final hidden and residual bytes. Explicit producer/consumer event ordering,
independent finish outputs and cloned owned outputs retained across later
callbacks also passed. The official FlashMLA fixture is local and causal;
this remains a component test rather than full cache/serving propagation.

Intermediate retention is used only in validation captures. Those captures
are released before separate faithful timing captures, which retain only the
normal final output pair. Warmup projection temporaries never become the
borrowed saved tensor: projection capture assigns it once, then finish warmup
and capture read that stable object. Projection and finish use separate private
graph pools. When the incoming residual is absent, saved aliases projection's
static hidden input; otherwise it belongs to projection private storage. Its
actual owner is charged once.

Every timing sample includes all callback input copies, eager BMM when
selected, graph replay, timing events and synchronized completion. Projection,
FlashMLA input construction, capture, compilation and validation clones are
outside timing. The probe's common shape/storage assertions remain inside the
callback. No replay-only timing is substituted for the complete callback.
Ten warmups precede 50 interleaved samples per variant.

Ranges of median wall times across three layers and both residual branches:

| Variant | Q=128 ms | Q=1024 ms |
|---|---:|---:|
| C4 copied attention | 0.2643–0.2855 | 1.3397–1.3513 |
| C5 direct output, copied attention | 0.2579–0.2785 | 1.2939–1.3047 |
| C5 plus stable saved | 0.2535–0.2722 | 1.2821–1.2933 |
| Eager expansion, copied saved | 0.2572–0.2771 | 1.2434–1.2490 |
| Eager expansion plus stable saved | 0.2541–0.2697 | 1.2309–1.2390 |

Against the paired C5 callback, eager expansion alone lowers median wall time
by 0.646–2.216 microseconds at Q=128 and 50.438–56.815 microseconds at Q=1024.
With stable saved reuse, the corresponding reductions are 3.849–9.849 and
62.935–67.696 microseconds. This screen observes no Q=128 regression, but the
small eager-expansion-only margin still requires integrated validation.

Static finish input capacity falls from 17.75 to 4 MiB at Q=128 and from
142 to 32 MiB at Q=1024 when both changes are used. Projection storage is
reported separately. Actual private/reserved graph-pool bytes are recorded
in each result; static input capacity is not the process memory peak.

At Q=1024, the trace audit finds 21 kernels plus two GPU copies for C4;
C5 removes one kernel. Eager expansion plus stable saved has 20 kernels and
zero GPU copies in the callback, with one BMM outside the graph and the
remaining 19 kernels inside it. The unchanged BMM workspace memset remains.
Thus the trace verifies the intended physical work removal; it does not
predict a formal end-to-end speedup by multiplying per-call differences.

Root owns graph policy v2 integration, resource accounting, source freeze,
full numerical checks and replacement formal/profile runs. Current valid
reports and raw runs remain until that publication gate is complete.
