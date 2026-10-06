# Resident history-build hint lifetime

Status: CPU source proposal, 2026-10-04. No implementation, new test run or
CUDA work. Root requested this bounded review while C9's formal trajectory
and matching profile were pending. Priority depends on that profile. This
proposal applies to `DeepSeekServingBackend`'s local ECHO path.

## Finding and scope

An explicitly uninterrupted, backend-owned empty-session history build with
H<=P can compute each layer's hint only on its final history chunk. Every
earlier hint is overwritten before any supported consumer can read it. The
final update must remain inside the history execution interval and complete
before the prefix snapshot or execution lease is released. Local all-hit
status alone is insufficient to authorize omission.

For C9's H=65,536, C=1,024 and ten independent layers, the source-derived
history count would change from 640 updates to ten. These are call counts,
not measured savings. All indexer scores, exact top-k, append/writeback,
protection events, recall and MLA remain in their original order. No history
traffic or required final hint work moves into setup or outside request timing.

## Producer, consumer and caller proof

| Source | Relevant behavior |
| --- | --- |
| `operators/deepseek_v32/indexer/prefetch_hint.py:38` | Overwrites only `offset[0]` with the finite mean of the current last min(4,Q) score rows. It does not accumulate the previous hint. |
| `models/deepseek_v32/echo_attention.py:271` | Passes the current offset into `prepare_prefetch` before logits; updates it after exact top-k and before deleting scores. |
| `cache/sparse_token_cache.py:651` | A certified resident history returns `prefetch=None`, preserving the empty FIFO event. The offset tensor is only attached to a nonresident prefetch request. |
| `operators/deepseek_v32/indexer/echo.py:174` | Resident official logits receive no offset. The fused branch forwards offset storage to the ECHO kernel. |
| `operators/deepseek_v32/indexer/csrc/echo_logits.cuh:476` | Reads the previous offset for histogram/prefetch thresholding; it does not update the hint. |
| `models/deepseek_v32/serving_backend.py:868` and `:938` | The backend execution lease spans all history chunks, transaction completion, final LM head and prefix-offset snapshot. Reentrant backend execution is rejected. |
| `models/deepseek_v32/serving_backend.py:1014` and `:1027` | Starts one cache step for the full history, then visits each independent layer for every chunk. |
| `models/deepseek_v32/serving_backend.py:953` and `:1116` | Saves committed prefix offsets and restores pre-candidate offsets after successful transient candidate discard. |

The residency argument depends on the existing cache proof, not H<=P alone.
At the first chunk the written history is empty. `append_slots` starts a
stable victim order for this new session; `certify_append` records that every
written history token is resident. While the same backend retains ownership,
later appends with end<=P use unused entries of that order. Resident exact
selection protects only previously consumed entries and preserves the append
plan. Other layers use distinct pool state. Hence each following chunk again
has a certified resident history and uses logits without a hint consumer.
See `cache/sparse_token_pool.py:353`, `:478`, `:499` and
`cache/sparse_token_cache.py:144`, `:526`.

By induction, removing an unobserved intermediate hint update cannot alter
the next chunk's scores, selections, records or hidden state. The final
chunk therefore produces the same score tail. Running the unchanged helper
there produces the same offset bytes, which the prefix snapshot retains.
After another session evicts this history, its revisit receives that same
final hint in the fused prefetch path.

## Smallest prospective change

Keep the default runner behavior, including updates after ordinary all-hit
calls. Only the serving backend may establish a temporary history-build
context after checking an empty session, ECHO/shared-pool ownership, complete
target H<=P, one full pending history step and its exclusive execution lease.
The context must cover both eager and compute-graph callback paths; indexer,
exact top-k and hint work are outside the existing compute islands.

Within that context, each interior layer call may omit its update only while
the uninterrupted append/residency proof still holds and prefetch is absent.
Every final layer call runs the existing update immediately while its own
scores remain live. This avoids retaining score storage and requires no new
kernel. Clear the context in `finally`, including construction, projection,
append, synchronization and commit failures. Preserve the existing offset
backups and rollback rules.

The optimized context cannot permit an arbitrary callback or direct cache
operation to observe an intermediate offset or invalidate ownership. The
first candidate should use the ordinary path whenever capture/diagnostic
hooks can make such observations. Profile scopes that only label work can
remain active. An unexpected loss of residency/ownership after omission must
fail and roll back before fused prefetch can read a stale offset; silently
resuming normal updates cannot reconstruct the omitted preceding hint.
Reject or exclude unsupported interleaving before entering the context.

Do not apply this policy to a populated-session extend, H>P, a standalone
`EchoAttentionRunner` or a generic externally managed cache step. Do not extend it to transient candidate hint
updates in the same change. The caller knows the final history boundary;
the low-level runner cannot infer that boundary from `prefetch is None`.

## Why retaining inputs is a broader alternative

A lazy general solution could save the most recent tail and materialize its
hint before a later miss or snapshot. It is unnecessary for the bounded build
and changes memory lifetime:

- Keeping `scores[-4:]` as a view retains the entire Q-by-N score allocation.
  At Q1024/N65536 that is at least 256 MiB per layer, or 2.5 GiB across ten
  layers, before physical padding and allocator overhead.
- Copying an owned `[min(4,Q),N]` tail reduces retained logical storage to
  1 MiB per layer at N65536, but adds one allocation/copy per chunk and needs
  explicit stream, snapshot and rollback ownership. Its copy cost belongs
  in the history API timing. Logical FP32 bits and the original helper's
  reduction shape/order must remain exact.
- Retaining Q/K/weights instead and recomputing scores is not a simple exact
  substitute. Projection graph outputs are overwritten on replay, indexer
  views can change, and a four-query recomputation can select a different
  kernel/reduction schedule. It would need owned inputs, extra accounting
  and an independent numerical gate.

These alternatives must not be used as an implicit fallback for a final-only
implementation that failed to establish its caller-level proof.

## Validation and measurement boundaries

Reuse `test_echo_cache_order.py`, the cold-append and residency-invalidation
tests in `cache/tests/test_sparse_token_pool.py`, the hint exactness/stream
suite and existing serving transaction/compute-graph tests. Add only focused
cases needed to check this new caller behavior:

1. Compare all intermediate selections, complete final hidden/logits, final
   offsets and prefix snapshots with the frozen per-chunk-update baseline.
   Include one chunk, multiple chunks, H=P and a final Q<4 tail. Check that
   the final update uses the final chunk of each independent layer.
2. Build another session to invalidate/evict the first, then revisit the
   first. Assert byte-identical hint input at the fused consumer, exact
   selections/output, correct cache mappings and actual transfer accounting.
   This catches a final hint that was never published or snapshotted.
3. Cover unsupported interleaving/hooks through the unchanged eager-update
   path, and injected proof loss before a prospective fused consumer through
   failure/rollback. Check failures before and after final hint production;
   no temporary context or pending score owner may survive.
4. Check nondefault-stream order and changed-input compute-graph replays.
   Existing candidate discard/truncate must preserve the same prefix offset.
   Same-version eager/graph agreement alone is insufficient; compare against
   the frozen baseline across the complete build and subsequent revisit.

Only after exactness, profile the new update count and measure complete
history/first-visit latency, including final hints and prefix snapshots, with
the same writeback and synchronization boundaries. Use fresh paired timings
and a separate confirmation window. Root then reruns affected four-scheme
serving/MFU measurements before replacing any accepted results. The current
C9 profile can prioritize this candidate; it cannot validate an unimplemented
variant.
