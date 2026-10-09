# Independent bounded-free preparation review

The private H64K full-model gate can proceed. The saved component evidence
is internally consistent, the baseline is the current production DSO, and
the algorithm preserves the first 64 entries of stable FIFO order under its
declared free-capacity proof. Production integration still needs the dispatch
and token changes below. This review is CPU/read-only and adds no GPU run.

## Evidence independently checked

Audit script `/tmp/q1_free_prepare_independent_review_20261008_01.py` and
result `/tmp/cxldsagr-checks/q1_free_prepare_independent_review_20261008_01.json`
verify the immutable receipt
`q1_free_prepare_check_20261008_02`, the benchmark
`q1_free_prepare_bench_20261008_01`, the NSYS candidate call, and cold/scattered
full/source NCU phases. Each of seven phases contains the same 2,735 source
file hashes, input bytes, candidate DSO and production baseline DSO. The
receipt's 14 top-level artifacts are valid; archived source files were
checked separately against the identity manifest.

The baseline DSO SHA256
`3a207197a196d5b3db02f602b156c7bbc577ab3fc47c6912b299735f02f86693`
also matches all three ECHO native records in the current CUB formal profile.
The harness calls production `cache_ops.prepare_prefetch` with the original
query-count-one and host-capacity arguments. It does not redirect baseline
dispatch through the private extension, which also contains the unchanged
ECHO translation unit. Both actually mapped DSOs are recorded.

The CPU review reconstructs all eight states from the signed input and seed,
then performs 154 saved-tensor comparisons. Complete baseline slots match
CPU stable argsort; candidate slots match the first 64 entries and have
MISSING in every unused entry. Every saved journal element, including its
extra final element, is reset; counters/statistics and metadata `[1, 0]`
match. Saved graph results for five changed valid residency states agree
with the corresponding eager component outputs. The original GPU checks
also verified unchanged priority/bitmap/reverse inputs and nondefault
streams; those post-call input tensors were not separately saved for this
CPU reread.

All seven invalid-input child logs terminate with CUDA failures: bad low/high
ages for both variants, plus insufficient free slots and selected bitmap or
reverse-owner mismatch for the candidate. The 31 token/alias rejection checks
are recorded in the signed check. Their execution is not rerun by this review.
The CPU audit recomputes every median, AB/BA ordering and win count in all
800 paired samples: candidate wins 800/800. Cold medians are
27.392 versus 8.800 us; scattered-free medians are 29.856 versus 17.840 us.
NSYS contains exactly the ordered `prepare_masks`, `select_slots`, and
`reset_publish` kernels for its complete candidate call.

## Algorithm and proof

For valid state, every usable priority is either -1 (free) or in
`[0, timestamp]`. Stable ascending argsort therefore puts all free slots first
in physical slot order. The reused ballot/prefix selector emits exactly this
ascending free prefix, checking the selected bitmap and reverse owners.
The first kernel scans all priorities even when the free prefix is found
early. The final kernel reads each selected journal entry and clears that
same entry in one thread, fills all unused slot outputs with MISSING, clears
the complete journal/counter/statistics, and writes Q1 metadata through dead
mask scratch. Thread zero can write metadata while other CTAs still reset
the journal; the following consumer on the same stream waits for the whole
kernel to finish. Stream ordering separates all three phases, so publication
writes do not race the earlier selector.

The intended dispatch proof is sound only with a sole session, an exclusive
operation lease, valid ordinary history state and `P - H >= 64`. Resident
live records satisfy `L <= H`, so at least 64 slots are free. The current
model calls prepare before main-KV append, after making the indexer suffix
visible; the pending query token has not increased main-KV residency at that
point. The component's H=0/P=64 case tests kernel geometry only and is not an
eligible official paged model call.

## Required integration details

1. Require ordinary, nontransient Q1 with `N >= 32768`, initialized history
   equal to query start, sole session, exclusive operation ownership and
   `P - H >= 64`. The normal general preparation path remains selected outside
   these support conditions; a kernel failure must propagate without retry.
2. Preserve the explicit `prepared_limit=64` proof through the actual
   `_PreparedPrefetch` contract. The official consumer must declare its
   required 64-slot bound. Query count one is not a consumption bound:
   the present `consume(prefetch, 1)` argument means rows, and a generic
   one-query consumer may still request up to 8,192 slots. Generic consumers
   must reject a preparation whose bound is smaller than their actual cap.
3. Integrate one-use invalidation, same stream/storage checks and scratch
   retention with the existing prefetch lifecycle. The private adapter has a
   keyword-only bounded `consume` and no `invalidate` method; it cannot be
   returned unchanged through today's production API. Existing pending-lease
   rules must prevent a second prepare from overwriting a live proof.
4. Fix a small declared-range issue when promoting the native code:
   `free_prepare.cu` computes `(slots + 255) / 256` in signed int. Its wrapper
   allows P near INT32_MAX, where that addition overflows. Compute this bound
   from `slots64` instead. This does not affect any measured shape or block
   the H64K private gate. Preserve the frozen candidate artifact and its
   receipt; apply the correction in the new integration source identity.

No additional algorithmic or saved-baseline mismatch blocks the private
full-model gate. Complete-model numerical, cache-transition, capacity and
formal performance acceptance remain required after integration; component
speedup and this read-only audit do not replace them.
