# Four HBM warmups before the matched reduced timer

The four-method prelude restores approximately 416 ns gaps in its initial
accepted profile, while the matched HBM-only baseline has 96 ns gaps. The next
factor replaces that warmup sequence with four HBM warmups. It separates
repeated cache/full-graph lifecycle from offload-specific history as packages;
it does not isolate allocation or identify a hardware mechanism.

Add only `q1_replay_hbm_prelude.py`; retain the frozen matched wrapper, timer and
four-method driver. After the same one compute-bank preparation, execute
`[hbm, hbm, hbm, hbm]`. Each occurrence selects a fresh cache, forwards the
H65536 prefix, snapshots/restores through the same public helper, prepares the
Q1 logits graph, forwards token111090 and releases the snapshot. Cold-residency
arguments remain unchanged; the helper applies no offload eviction to HBM.
Then select HBM once more. Require five cache-generation advances, the same
compute bank, an empty resident cache, no shared resources and no full graphs.
Keep the usual Torch graph-entry synchronization and allocator-cache flushes.

The original measured prefix, correctness checks, inspected 197-node/192-layer
graph, five plain warmups, 50 clean AB/BA pairs and two profile pairs remain
unchanged. Record actual executed methods in `contract.hbm_prelude` and
`runtime.hbm_prelude`, with receipt kind `deepseek-q1-replay-hbm-prelude-v1`.
Archive the new driver and preserve all formal source/environment/runtime
identity checks. HBM-only native/Triton subsets legitimately omit unused
offload entries; do not require the four-method analyzer's complete formal
runtime set or load unused libraries to manufacture equality.

Freeze after CPU/static review, then root runs separate check, clean bench and
profile processes on GPU0/CPU0-7. Compare native signatures, owners and raw
clipped activity unions against accepted matched/four-method runs. Profile
gaps and clean wall time remain distinct. A result needs fresh-process order
confirmation before a treatment-effect claim. If HBM x4 stays low while the
four-method prelude stays high, an allocation-only factor becomes motivated;
otherwise inspect repeated cache/graph lifecycle first. No allocation-only
implementation or GPU execution is part of this driver-preparation task.

The separate analyzer is `experiments.deepseek_v32_mfu.src.analyze_replay_hbm_prelude`.
It accepts the matched analyzer's profile/bench/formal/output arguments plus
`--matched-audit-dir` and `--four-method-audit-dir`. It reuses unchanged raw
checks, binds both accepted reference audits and their receipts, validates the
actual HBM x4 contract/completion gates, and records legitimate unobserved
formal runtime entries. Each of the six scopes must preserve both reference
signatures and native owners. Analysis of the new GPU profile remains root's
responsibility after the independent analyzer review.
