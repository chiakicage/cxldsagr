# Reproduce the complete formal profiling pipeline

Five controlled HBM experiments do not reproduce the formal 66.528 us idle.
The latest uses one graph, zero internal events and exactly the formal NSYS
capture settings; removing ordinary timing events still leaves about 16–17 us
idle. Older private baseline traces also show low idle on the same physical
GPU0, so GPU identity alone does not explain the original observation.

Before changing another isolated variable, rerun the existing full formal
pipeline without modifying its source. Use GPU0/CPUs0–7, the same checkpoint,
request generation, capacities, warmups, NSYS settings and compiler environment
as `deepseek_h64k_a1_cub_20261008_01_h65536_a1_profile`. The existing independent
check and clean benchmark remain valid only if their strict identity checks
pass. Preserve the literal native-fingerprinted PATH, including three venv
prefixes after the existing shell entry point adds its own prefix.

Use new profile run ID `deepseek_h64k_a1_cub_gap_reproduce_20261008_01` through
`scripts/gap_profile.sh`, retaining the ordinary all-four-method preparation,
compute-graph instrumentation and cache lifecycle. Do not replace this with a
new reduced model harness. No production code or published result changes.

After successful collection, verify the full receipt/source/native/input
binding, saved outputs, capture settings, native layer membership and clipped
activity union. Compare the HBM warmup and measured replay with the earlier
formal trace before deciding whether to minimize the formal prelude further.
This is a repeatability investigation, not a new optimization claim or a
replacement for clean timing. Keep existing reports while the discrepancy
remains unresolved. GPU0 is reserved after the private free-prepare profiles
finish; never overlap these measurements on that device.
