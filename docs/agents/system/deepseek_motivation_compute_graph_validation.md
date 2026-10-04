# C3 compute graph validation checkpoint

Date: 2026-10-04. The current C3 implementation passes the correctness and
ownership checks below. This document records engineering validation; it is
not a latency/MFU result or an experiment publication. The full serving
performance and graph-node attribution gates remain separate.

## Contract and implementation boundary

Follow the [compute graph plan](deepseek_motivation_compute_graph_plan.md):
capture only projection/input norm and output/post norm/dense MLP, retain eager
cache operations and exact selection, use distinct graph pairs for the ten
checkpoint block copies, and charge full graph-private segment capacity.

The validation agent owns changes in
[`compute_graphs.py`](../../../models/deepseek_v32/compute_graphs.py) and
[`test_compute_graphs.py`](../../../models/deepseek_v32/tests/test_compute_graphs.py).
The existing implementation was first checked at H=2304, C=256, A=16/23 across
all four schemes. It passed before the following lifecycle hardening changes:

- Record matmul/TF32/reduced-reduction/autocast policy at capture and reject a
  changed policy at the next execution entry. Expose the captured policy in
  resource diagnostics. A policy mismatch fails before replay and does not
  itself indicate an asynchronous CUDA failure.
- Mark graph storage failed when a dependency on the previous execution
  stream cannot be established, or when close cannot drain graph consumers.
  Retain graph/input/output references on those failures. The parent task
  separately propagates graph failure into backend poisoning/owner retention.

No arithmetic, graph shape, cache semantics, or captured operator was changed
by these hardening edits. Production source was frozen before the final
validation command and the independent performance screen.

## Reproduction and source identity

Repository HEAD was `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`; the worktree
contained changes, so the following SHA256 values identify the directly
involved sources rather than treating HEAD alone as the implementation:

| File | SHA256 |
|---|---|
| `models/deepseek_v32/compute_graphs.py` | `9f609674b458476a096f9941ad3cbfae118a762cbfcd44a6cbef0445fb00eb77` |
| `models/deepseek_v32/tests/test_compute_graphs.py` | `e8f2fa9842ffb4935411a7570cb5d27c1c7027fd76e3aa9beab5c73d5d43e346` |
| `models/deepseek_v32/serving_backend.py` | `09425848a3dc043dbdc5c264bde4d4e488a9e31483f55ec517646653dcd3b55c` |
| `models/deepseek_v32/echo_attention.py` | `5eb38ed3c6a1eee29e95ba610254e484771247caadb3e713d1f2f274e06a9d5e` |
| `models/deepseek_v32/echo_model.py` | `cb4bfcd40f7675ccb76097a2c7eaa36f13cfe27eeb497ac709866849f7a8a0a7` |

This table is not a hash of every transitive dependency or of checkpoint
contents. The independent experiment source snapshot supplies the broader
performance-run provenance.

Hardware: physical GPU 1, NVIDIA M403, SM90, driver 570.124.06; PyTorch reports
150,121,545,728 bytes total device memory. Dependencies: PyTorch 2.12.1+cu130,
CUDA runtime 13.0, FlashInfer 0.6.18, DeepGEMM 2.8.1+057ca59, and FlashMLA
1.0.0+ba89a34. Checkpoint: `/preset-models`. The backend uses its normal FP8
checkpoint linears, BF16 hidden/main KV, FP32 normalization and head weights,
and ten independent copies sourced from checkpoint blocks 0–2.

Final command, from the repository root:

```bash
PATH="$PWD/.venv/bin:$PATH" \
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
DEEPSEEK_GRAPH_CHECKPOINT=/preset-models \
DEEPSEEK_GRAPH_HISTORY=65536 \
DEEPSEEK_GRAPH_CHUNK_SIZE=1024 \
DEEPSEEK_GRAPH_CANDIDATE=128 \
.venv/bin/python -m pytest models/deepseek_v32/tests/test_compute_graphs.py -s -q
```

Result: **9 passed**, 15 dependency deprecation warnings, 83.19 seconds.
The test prints to the terminal and creates no persistent test-result
directory. Ruff check passed for both owned Python files. The first attempt
to run the earlier small real-checkpoint test omitted `.venv/bin` from PATH,
so FlashInfer could not find `ninja`; the corrected command resolved that
environment failure.

## Numerical and lifetime evidence

The final checkpoint test uses H=P=65,536, C=1,024, NH=131,072, and candidate
lengths 121 and 128. It creates independent empty eager and graph caches for
each scheme. The token IDs are the deterministic synthetic IDs in the test,
not the sixteen-user performance trace. Candidate length 128 is captured;
121 intentionally exercises the reported eager fallback without recapture or
history reconstruction. The bank has 40 graphs: two islands for each of ten
independent layers at Q=128 and Q=1024.

For `hbm`, `echo`, `serial_sparse`, and `dense_prefetch`, the test verifies:

- Every 64K prefill hidden value and every candidate hidden/logit value is
  exactly equal between graph and eager execution (`rtol=atol=0`). Candidate
  results also match the HBM-only eager oracle across schemes.
- The second user's different full history occupies the same finite offload
  pools before the original user's revisit. Candidate outputs still match.
  This exercises replay at all 64 history chunk starts and candidate start
  65,536, with both incoming-residual branches.
- The source-copy output relation holds for layers 3–9, and distinct source
  copies have different `wq_a` weight pointers. Graph object identities remain
  unchanged across users and candidate sizes.
- All retained FP8 index keys, index scales, and logical main-KV host records
  remain exactly equal after candidates. Candidate D2H bytes are zero and the
  session remains at H. Shared measured bytes fit the declared resource plan.
- Owned returned prefill tensors and retained diagnostic-hook hidden/residual
  references remain unchanged after later replays and requests.
- A=128 executes exactly ten projection replays; A=121 executes ten explicit
  eager fallbacks. Graph-private and static allocated capacity do not grow
  after replay, eviction/revisit, or candidate-size changes.

The small checkpoint lifetime test separately delays D2H on another CUDA
stream while the same graph is replayed. Both copied host KV tensors remain
exact, and their source pointers differ from each other and the borrowed
graph output. Transient candidates borrow the graph KV output because they
do not persist it to host. Replay on a different execution stream remains
exact after the resource's event dependency.

Failure tests verify changed TF32/BF16 reduction policy is rejected before
replay, a failed prior-stream wait prohibits reuse while retaining storage,
and a failed close retains storage. A real GPU graph allocation with a
one-byte private-pool limit fails capacity acceptance after a partial capture;
the partial bank remains owned until synchronized close clears it. A synthetic
allocator snapshot proves that inactive private-pool segments are included,
not only live output allocations.

## Memory evidence

All values below are bytes. The private-pool column includes inactive blocks
in every graph-owned allocator segment. Static inputs are counted separately
using their actual allocator block sizes. Each scheme's values remained
unchanged at the post-request audits.

| Scheme | Private reserved | Static allocated | Setup seconds |
|---|---:|---:|---:|
| HBM-only | 7,260,340,224 | 1,943,457,792 | 3.853 |
| ECHO | 7,260,340,224 | 1,944,419,328 | 0.371 |
| Serial sparse | 7,260,340,224 | 1,944,140,800 | 0.391 |
| Dense prefetch | 7,260,340,224 | 1,943,729,664 | 0.367 |

The observed private capacity is about 6.76 GiB, plus about 1.81 GiB of static
inputs. The plan's 12 GiB private limit remains an explicitly chosen upper
bound. It is not an observed fixed overhead or a claim that 12 GiB was
preallocated. Setup times include warmup/capture and are not request latency;
the first scheme additionally warms previously unused operator variants.

At graph allocation completion, the test records all three physical accounting
views and checks `allocated <= reserved <= device_used <= device_total`:

| Scheme | PyTorch allocated | PyTorch reserved | Device used |
|---|---:|---:|---:|
| HBM-only | 24,860,961,792 | 29,909,581,824 | 30,705,516,544 |
| ECHO | 27,814,863,360 | 33,460,060,160 | 34,266,480,640 |
| Serial sparse | 27,739,214,336 | 33,090,961,408 | 33,899,479,040 |
| Dense prefetch | 27,738,476,544 | 33,239,859,200 | 34,048,376,832 |

These are validation-process snapshots with two loaded backends, retained
diagnostic tensors and ordinary allocator caches. They are not an isolated
single-backend peak, a full-NH capacity result, or a replacement for the
performance run's memory audit. No allocator difference is reclassified as
mandatory preallocated storage.

## Remaining boundaries

The tests exercise the backend directly, including two simultaneously live
HBM test sessions. They do not validate runner LRU admission, the formal
NH=16,777,216 capacity, or the complete sixteen-user/two-round trajectory.
Those belong to the separate full experiment.

Captured weights, operator objects and configuration must remain immutable
during a bank's lifetime. The runner checks attention-object identity and the
bank checks precision policy at execution entry; it does not hash mutable
weight contents on each request. Weight replacement or concurrent mutation of
global precision policy during an active execution is outside this contract.

GPU tests cover delayed successful D2H and cross-stream replay. Failed CUDA
wait/close paths use fault injection; no destructive hardware fault or
asynchronous illegal-access failure was induced. The parent backend's poison
and admission-owner tests cover the integration response to a failed graph
resource. Whole-request/cache capture and concurrent replay remain unsupported.

No graph-enabled latency convergence, matrix-node timing attribution, or MFU
conclusion follows from these numerical and ownership checks. The performance
and profiling agents must complete those gates using fresh run evidence.
