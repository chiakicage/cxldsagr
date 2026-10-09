# Published and private HBM profile boundary audit

Status: read-only diagnosis on 2026-10-08. Production and frozen candidate
sources were not changed. The Q-A/KV-A/index-K fusion is rejected by the clean
paired model timing; these cross-run profile observations do not reverse that
decision. Capture-time profiler causality remains pending the independent
`review_a1` control.

## Matched work and timing

The authoritative source is `local_hbm` in
`experiments/deepseek_v32_echo_official/report/decode_gap/kernel_details.csv`,
joined by graph node ID and start timestamp to
`experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_official_20261008_03_h65536_a1_profile/capture_4.sqlite`.
Its first input norm through the last L2 MLP-down operation contains 198 GPU
activities: 183 kernels and 15 device-to-device copies. Replay node IDs are
214748364801 through 214748364998. The normal full graph starts with embedding,
so these are GPU activity ordinals 1 through 198, inclusive.

Both private baseline profiles contain exactly the same 198 relative graph
node positions, kernel names, activity types, and 17 checked launch attributes:
launch type, cache config, register count, grid/block dimensions, static/dynamic
and executed shared memory, shared-memory limit config, cluster dimensions and
cluster scheduling policy. This establishes matched work and launch geometry;
the old files do not contain graph edges and cannot establish identical DAGs.

| Run | Matched span, us | Busy union, us | Idle, us | Kernel duration sum, us |
| --- | ---: | ---: | ---: | ---: |
| Published HBM | 1073.347 | 1004.515 | 68.832 | 1076.643 |
| `q1_qkv_model_profile_20261008_02`, baseline | 1018.242 | 1000.898 | 17.344 | 1079.298 |
| `q1_qkv_clean_model_profile_20261008_01`, baseline | 1024.194 | 1007.169 | 17.025 | 1085.441 |

The previous private inferred windows were 1020.930 and 1026.690 us. They
included one additional endpoint fused RMSNorm kernel. Removing that node
shortens them by 2.688 and 2.496 us; idle is unchanged. The endpoint norm starts
slightly before the preceding MLP kernel's reported end, so selecting only by
timestamp would retain part of an out-of-scope node. The audit selects the
matched node set instead.

The published 68.832 us idle consists of 150 short gaps, median 416 ns and
maximum 1088 ns. Private mock/clean baselines have 144/146 gaps, both with
median 96 ns and maxima 768/608 ns. This is a distributed difference, not a
single missing synchronization or long stall. Kernel busy time does not show
a comparable difference. Same-implementation cross-run timing is insufficient
to identify the cause or claim an optimization gain.

## Explanations checked

The published full graph already overlaps Q-A and KV-A. At L0, relative to the
first norm, KV-A runs at 8.256–19.072 us on stream 7 and Q-A at 9.280–22.016 us
on stream 168: 9.792 us overlap. L1/L2 also overlap. Earlier documentation about
no observed overlap refers to a separate component profile.

The 1368 common source hashes, request hash, GPU UUID, CPU affinity 0–7, and
PyTorch 2.12.1+cu130 match. All 12 common recorded JIT artifact hashes match.
The published all-method process additionally loads five local ECHO/offload
libraries; the private HBM-only process does not. Native/quantization build
metadata also differs in the `DG_JIT_WITH_LINEINFO` environment setting.

The published HBM template has exactly 203 nodes: 187 kernels, 15 copies and
one memset. It contains no explicit event record/wait or empty nodes. The
other three published method templates also contain only GPU node types.
`GraphInspector.snapshot` calls capture-info, graph enumeration, node-type and
CUPTI-ID APIs. It adds no nodes or dependencies in source. Its saved templates
contain no edge ledger, so implicit runtime effects have not been excluded.

`profile_layers.profile_extend_graph` enters `gap_profile.profiler_capture`
before creating `FullExtendGraphCapture`, preparing the graph, and finalizing
the template. The private harnesses prepare their graph before
`cudaProfilerStart`, perform five pre-trace replay warmups, and then perform
three in-trace warmups. The published measured phase has one in-trace warmup.
All graph preparations perform the production allocator's three direct eager
warmups. Both use NSYS 2025.6.3.541, export schema 3.24.8 and node tracing.
The private command additionally traces OSRuntime. These conditions were not
controlled independently in the old runs.

`return_hidden` is false in the published run and true in the private harness.
`model._execute_gpu_body` branches on it only after L0–L2; Q1 selects one row
either way. It does not explain a difference in transformer work. The clean
private harness has two external timing-event nodes; the mock harness does
not. Both private baselines show the small gaps, so these events alone do not
explain the observation.

## Controlled next experiment and evidence

Keep production HBM, request/weights, return_hidden=False, CPU/GPU allocation,
profiler flags, the three eager capture warmups and one in-trace replay warmup
fixed. First vary only whether CUDA graph construction occurs before or after
ProfilerStart, without `FullExtendGraphCapture`. Compare clean CUDA-event
timing and node-trace timing separately. Then, if needed, compare hooks on/off
under the same profiler state. Export graph topology and match the complete
198-node work signature before attributing gaps. This control is assigned to
`review_a1`; it has not completed at this checkpoint.

Temporary independent audit records:

- `/tmp/cxldsagr-checks/q1_qkv_profile_boundary_audit.json`: matched interval
  arithmetic and checked launch-attribute differences, which are empty.
- `/tmp/cxldsagr-checks/q1_qkv_profile_boundary_reference.json`: all 198 formal
  node signatures, layer/stage ownership, copy sizes and launch attributes.

The reference is for matching this unchanged production implementation.
Adding external graph events shifts raw node IDs; match the full signature
after separating non-GPU nodes. Do not infer layer ownership from one repeated
kernel name. Query NSYS metadata by explicit field whitelist: the capture
metadata also stores unrelated process environment variables.
