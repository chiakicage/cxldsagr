# C6 draft

The checked miss path materializes unique IDs, filters padding and history,
translates fragmented pages, filters misses, builds a protected slot mask, ranks
available victims, scatters maps, and remaps the original selection. Many dynamic
shape operations synchronize with the CPU. The root C4 investigation identifies
this path as a remaining revisit launch-gap target; C6 has no measured gain yet.

Use the existing shared logical-capacity bitmap for the exact union. A native
classify kernel scans logical order, protects selected hits at timestamp t, and
writes one miss flag per history position. It counts selected/resident records and
misses. One GPU-to-CPU miss count preserves the checked distinction between zero
misses (retain append/residency proof state) and real recall (invalidate exactly
once). Device assertions reject nonnegative IDs outside written length.

For M > 0, stable argsort of priority[1:] supplies the same first M victims as the
old protected-filter sort. All occupied priorities are below t before protection;
selected hits become newest, and H <= P guarantees enough unprotected victims.
Free priorities remain -1; sentinel slot 0 is excluded. A prefix sum of flags
preserves ascending logical miss order even when page IDs are nonmonotonic.

Reuse existing scratch only under the exclusive layer operation lease, with no
pending prefetch: free_slots[int32,P] holds flags; miss_scratch[int64,P] holds the
prefix; allocation_log[int64,P+1] holds compact global miss IDs. The allocated
int64 argsort tensor becomes chosen IDs in-place after compaction. The existing
generic gather_host_records copies the opaque model-provided width/dtype. Tombstone victim maps before the copy so a failed
publication cannot leave another session pointing at overwritten records. Publish
new maps only after that copy launch on the same stream, then map
the original selection and advance the allocation event. Candidate IDs never
enter host translation, FIFO victims, or host writes.

Risks: scratch lifetime; accidentally sorting global IDs; empty/all-hit event
semantics; equal-priority victim ties; clock rollover; candidate padding; event
ordering of pending D2H writes and H2D reads; metrics double counting; stale map
publication. Differential tests compare full records/maps/priorities/free flags,
Python proof state, clocks, counters, and outputs against checked execution.

Independent CPU ordering audit was supplied by root and rerun before kernels:
/tmp/deepseek_c6_allocator_audit/audit.py, SHA256
6e06fe1669764a7ab666199c504c4d1a065d90c261e3036da9b318fb8bb20176.
All 20,000 randomized states passed, P 1..513, with partial free slots, ties,
fragmented/nonmonotonic host pages, padding, hit/miss mixtures and candidate IDs.
This supports the ordering reduction only; it is not CUDA correctness evidence.
