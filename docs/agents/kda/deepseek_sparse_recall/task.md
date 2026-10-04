# C6 sparse recall metadata

Reduce private exact-top-k recall overhead for a shared SparseTokenCache with
history H <= pool slots P, including histories displaced by another session.
Keep exact logical union, SMALL/top-k selection, stable FIFO event semantics,
record bytes, physical candidate tails, session ownership, page tables, metrics,
map_generation, append/resident proof invalidations, and rollback/drain behavior.

The existing all-resident C3 path remains preferred. Public arbitrary selections,
unsupported tensor metadata, H > P, and clock rollover between the two events use
the checked path. Every nonnegative private ID must still be checked on-device
against written length. No approximate selection or reduced working set is allowed.

GPU target: Hopper SM90, root-controlled quiet GPU 2 windows. Correctness must
precede timing. C5 wrapper validation runs before any C6 CUDA compilation.
Engineering artifacts stay in /tmp; full-model experiment acceptance/publication
belongs to root. No persistent allocation may be added without capacity accounting.
