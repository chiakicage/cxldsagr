# T3 design reasoning and risks

T2 preserved exact output but put a full bitonic sort in every row program.
Its 2,048-wide specialization used 124 registers even when the dynamic branch
skipped sorting. Its complete API regressed 32K/64K history and the causal tail.
These observations motivate a separate fast kernel; they do not prove which
resource or branch caused those regressions.

The sampled real selected vectors had maximum finite run length three, but
that is neither a workload invariant nor a reason to truncate larger runs.
T3 fixes the short threshold at four and flags every longer row for general
repair. Because values are already sorted by exact ordered bits, equal bits
form contiguous runs. A run exceeds four iff a pair four positions apart has
equal finite bits. This detects long runs at every alignment without reading
IDs, using numerical comparisons, or relying on host scalar values.

For an unflagged row, all members of the current item's finite run lie within
three positions on either side. The number of matching predecessors is the
item's offset from the run start. The number of matching neighbors with a
smaller original logical ID is its desired rank. IDs are unique because the
official selector returns distinct input positions. The destination formula
therefore assigns a unique position inside the same run to each item and
orders IDs ascending. Value bits do not move. An unchanged destination can
skip its store because no other item targets that position.

One CTA owns the whole row, and a barrier separates all input-ID reads/rank
calculations from all scattered writes. This is necessary for in-place repair
across warps. A flagged row must skip all first-kernel ID writes, including
nonfinite masking, so the second kernel receives the original row and masks
it itself. Distinct rows have distinct IDs/flags. The stream dependency between
launches makes flags and output writes visible without an extra host barrier.

The flag vector is fresh per call, so there is no cross-request scratch cache
or output aliasing. During graph capture it belongs to that graph allocation
pool and follows the same stream's execution. Both kernels launch for every
nonempty call even when all flags are zero; the second kernel's branch can
avoid work but not its launch cost or static resource requirements. The first
kernel's six bounded neighbor comparisons also have their own register and
memory cost. Offline compile both kernels before scheduling a GPU screen.

The second kernel reuses T2's exact run-start/index packing and complete C5
fallback boundary. This keeps long-run semantics independent of the threshold.
Retain signed-zero distinction, arbitrary finalizer NaN payloads, every width
1..2,048, padding cases, and the maximum representable packed key. The first
candidate has one threshold, no autotuning, no query-tile split, no new uint64
fallback, and no production modification.
