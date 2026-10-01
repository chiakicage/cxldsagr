# NOSA resident indexer: implementation plan

## Starting point

Use the accepted BF16-pair implementation described in [checkpoint.md](checkpoint.md).
The original complete run is `kda_main_bf16_pair_v3_development`; L0/L15/L31
complete indexer times are 141.471 / 137.854 / 137.504 µs. None reaches the
87.503 µs target. Historical active-work notes are preserved in
[investigation_log.md](investigation_log.md), not carried forward as open tasks.

## Next implementation steps

1. Reconfirm the exact accepted source/dependency identities before any new
   candidate. Retain incremental model-owned compression, finite validation,
   alias protection, ranked preparation and transaction semantics.
2. Profile the current complete call on real captured and synthetic inputs.
   Separate first-pass QK/normalizer work, exact-bound/pruning work, final
   selection and preparation; do not treat old clock-instrumented phase
   timings as current measurements or hardware lower bounds.
3. Pick a bounded candidate supported by the new profile and the existing
   rejected-direction evidence. Require compiler/resource and synchronization
   checks before GPU work; preserve native register reconfiguration and guard
   actual shared/register occupancy rather than relying on total SM capacity.
4. Validate exact selected IDs/masks, top33 keys, final CIS selection,
   reference numerics, exceptional values, invalid transactions, cache/alias
   behavior, graph/fallback paths and poisoned replays as applicable. Keep
   original tolerances and real/synthetic coverage.
5. Use paired complete-call measurements to admit a candidate. Rebuild main,
   freeze snapshots/hashes, check exact supported attribution graphs, and
   measure complete indexer plus companion attention on all three layers.
   Report kernel sums, event API intervals and wall completion separately.
6. Run the required CPU/GPU regression and investigate the prior unfinished
   resident global CPU pattern-sort check if it still applies. Rerun affected
   synthetic MFU and full-NOSA pattern experiments under new run IDs. Audit
   whether the already-published full-model comparison is affected by the
   new implementation; rerun it if so.
7. Publish only accepted new results, then replace affected old report assets
   and raw runs. Do not label the 40% objective complete until every captured
   layer meets the complete-module target.

The shared [measurement and publication rules](../README.md) apply throughout.
No optimization or remeasurement was performed during this documentation move.
