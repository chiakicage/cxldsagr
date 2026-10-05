# T1 draft: combine exact ordering after official selection

C9 history top-k families cost 180.543–190.132 ms. HBM has 108.164 ms selector,
35.852 ms index finalizer and 46.115 ms stable value sort; ECHO has
102.469/34.409/43.665 ms. T1 targets the latter two and the existing mask.
These sums do not predict an API or full-request gain. Evidence:
C9 profile note（Git `934485b:docs/agents/system/deepseek_motivation_c9_profile.md`）.

Installed `TopKDispatch` controls index finalization with `deterministic` and
stable value sorting with `sorted_output`. Its selector computes
`deterministic_selection = deterministic || tie_break != None`, and SMALL
forces FilteredTopK. `False, False, SMALL` therefore retains the same selection
specialization/set, with arbitrary emission order. No vendor edit is required.

For FP32 bits b and original logical index i:

```text
ordered = (~b if sign_bit(b) else b ^ 0x80000000) modulo 2**32
descending_key = ~ordered modulo 2**32
packed = uint64(descending_key) << 32 | uint32(i)
```

Sort packed keys ascending, one Triton program per row, padded to the next power
of two with uint64 max. Invert the high bits to reconstruct the exact FP32 value;
mask the unpacked index to -1 only for exponent-all-ones values. This combines
baseline ascending index sort followed by stable descending ordered-value sort.
The selected logical IDs are unique. Width one still applies the finite mask.
In-place row reads feed the full sorting dependency before stores; rows do not
alias. The first candidate uses one fixed launch policy, without autotuning.

Risk: 2048-wide uint64 bitonic sorting may cost enough registers/shared memory
and comparisons to lose against two CUB radix passes. Offline resource inspection
cannot establish performance. Replacing the selector itself has a much larger
proof surface and is deferred.

Resident selection's 61.079–64.355 ms/history is mapping plus exact distinct
union/counters/priority/clocks, not redundant sorting. The current large-input
kernel already merges CTA-local bitmaps. Naive per-element global atomics can
increase contention, while duplicate stamps lose exact counts. Fusion before
`cache.append` sees incomplete suffix mapping. Keep this implementation while
screening T1; any later resident candidate needs its own metadata differential.

Cache union is set-based, but FlashMLA reads the supplied selected axis in fixed
blocks without sorting. Its finite-precision reduction sequence and the public
selection contract require retaining exact order. Real-number permutation
invariance does not authorize changing that sequence.
