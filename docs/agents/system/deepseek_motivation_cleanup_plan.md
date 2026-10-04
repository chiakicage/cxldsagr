# Completed transient candidate cleanup

C3 request rows show about3 ms cleanup per offload revisit. The runner must keep
both complete allocation audits. The model currently invokes cache.truncate(H)
and restores ten offset tensors even after extend_candidate already discarded
every transient cache step, restored those offsets, and synchronized successfully.

Candidate: return from DeepSeekServingBackend.truncate only when truncating to
the same committed fixed prefix immediately after a successful transient
candidate and all layers still have no active step, no transient suffix or
pending prefetch, and written/indexer-visible endpoints equal committed history.
Keep ordinary persistent append/truncate and shorter-prefix truncation unchanged.
The public backend checks poison, owner and active execution first. No cached
memory footprint or skipped runner allocation audit is introduced.

Validation: unit cases cover the completed-transient no-op, dirty layer state,
ordinary persistent cleanup and shorter truncation. Then run actual checkpoint
four-scheme graph/eager validation and the complete experiment after the current
C3 profiler finishes. Candidate remains unmeasured before those GPU checks.
