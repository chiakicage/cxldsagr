# DeepSeek GR serving backend

## Task and numerical scope

The user requested a single-GPU, no-MoE approximation to 8B using real DeepSeek
V3.2 layers 0, 1, 2 and copied corresponding layer inputs. The implementation is
`models/deepseek_v32/serving_backend.py`; existing complete 61-layer model and
kernel implementations are unchanged.

The checkpoint at `/preset-models` contains 597,442,816 parameters per source
dense block. Ten independently loaded physical blocks use source sequence
`[0,1,2,0,1,2,0,1,2,0]`. Together with checkpoint embedding, norm and LM head,
this is **7,827,793,408 parameters**, including a 5,974,428,160-parameter dense
backbone. Observed physical GPU weight storage is 9,864,625,792 bytes because
some absorbed MLA projections and endpoints are BF16 and norms are FP32.

Each request first evaluates source blocks 0, 1, 2. Every subsequent copy
receives a clone of the corresponding source block's hidden input **and** its
separate residual input. Copies own separate weights and KV/indexer states;
they do not consume the previous repeated block's output. The backend returns
all candidate final-normalized hidden states and also executes the last token
LM head. This is a checkpoint-backed workload surrogate, not a trained 8B
model or a correctness result for the complete 61-layer DeepSeek checkpoint.

## Serving contract and policies

`DeepSeekServingBackend` implements `estimate_session_bytes`, `create_session`,
`prefill`, `extend`, `truncate`, `session_bytes`, `session_metrics`,
`release_session`, `synchronize`, and `describe`. User cache sessions are retained
by the shared serving LRU manager. Each session owns fixed-capacity main KV,
indexer K/scales and metadata. All policies preserve exact logical top-2048,
causality, the same checkpoint projections and the existing sparse MLA kernel.

- `hbm`: all main records resident; no CPU backing.
- `echo`: existing fused indexer/prefetch, then exact recall into retained finite
  HBM slots. An exact selected union larger than the pool recursively splits
  query consumption without dropping any selected token.
- `serial_sparse`: same finite pool and exact recall, with fused prefetch
  disabled. Indexer finishes before missing selected records are gathered.
- `dense_prefetch`: every layer keeps pinned CPU main KV. Two per-session full
  layer HBM stages alternate; a dedicated copy stream fetches the next layer's
  entire historical KV while the current block executes. Events prevent stage
  overwrite before its prior consumer completes. The new suffix copies directly
  from its projected GPU records into staging and host backing.

Admission estimates include all session KV records, logical/physical maps,
ages, indexer keys/scales, offset, ECHO counters and both dense stages. Actual
accounting deduplicates aliased storage. Weights, transient activations,
selection scores and GEMM workspace are execution memory outside cache budgets;
they must not be confused with process peak HBM. Truncation synchronizes first,
removes the candidate and keeps the user's prefix and reusable sparse residency.

## Validation

CPU scheduling/budget tests: `models/deepseek_v32/tests/test_serving_backend.py`.
Nine tests passed on 2026-10-02: copied input/residual semantics and storage,
independent caches, revisit equivalence after truncation, transaction rollback,
alternating users on shared model weights, foreign/released sessions, cache
reservations and parameter accounting.

Real checkpoint validation on GPU 0 (SM90, NVIDIA M403) passed with ten physical
copies, a 2,304-token prefix (longer than top-2048), candidate lengths 16 and 23,
and a truncated revisit. Each policy built its prefix from an independent empty
session. All candidate hidden states were bitwise equal to HBM, and every
repeated physical block's hidden/residual output was bitwise equal to its source
block on both prefix and candidate calls. Actual cache allocations stayed below
the pre-admission reservations throughout. This is correctness validation only,
not an experiment or a serving performance result.

Reproduce the checkpoint validation with:

```bash
source .venv/bin/activate
DEEPSEEK_SERVING_CHECKPOINT=/preset-models \
  python -m pytest models/deepseek_v32/tests/test_serving_checkpoint.py -q
```

Activating the environment is required because native JIT looks up `ninja` on
`PATH`; invoking `.venv/bin/python` alone did not find it in this shell. The first
attempt failed before GPU indexer execution, then a fresh activated invocation
passed. No failed run is retained as an experiment result.

Formal GR user/popularity sweeps, timing, fairness, report publication and
Supervisor state updates remain owned by the root implementation agent.
