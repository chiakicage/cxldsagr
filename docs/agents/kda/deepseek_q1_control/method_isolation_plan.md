# Isolate formal cache methods in fresh processes

Status: implemented, all 16 children completed, audited and published as
`deepseek_h64k_a1_isolated_20261008_01`. The plan below records its initial
baseline and design choices; current results are in the component checkpoint
and `experiments/deepseek_v32_mfu/report/h64k_a1/`.

The completed publication covers H=65,536, A=1, GPU0, CPU0–7, the existing
request and HINT implementation/environment, one method warmup, three prefill samples and
five extend samples. Root owns implementation scheduling and GPU execution.
At plan creation, the published cohort was `deepseek_h64k_a1_hint_20261008_01`; the earlier
FREE factor controls retain their own source and runtime identities.

## Evidence and scope

The accepted A1/B1/B2/A2 control found 96 ns median native-node gaps after
HBM-only preparation and 416 ns after the complete four-method prelude, with
matching participating kernels and graph ownership. HBM repeated four times,
cache construction alone and direct pinned allocation alone retained 96 ns.
The first dense-only prelude benchmark/profile processes already reproduce
416 ns; its check and raw activities passed independent rereads. This supports
removing cross-method initialization/execution from a method's formal process.
It does not identify a DMA, fence, allocator or other hardware mechanism, and
does not guarantee an improvement for an offload method after its own warmup.

Keep production model, cache, kernels and graph policy unchanged. A method
process still performs its own complete normal warmup and all normal graph
entry cleanup. Do not add cache flushes, provider preloads, allocator probes,
clock changes or timing events. Fresh process isolation does not mean a fresh
device or a cleared disk JIT cache. Do not parallelize GPU child processes.

## Current constraints found in source

| Location | Assumption requiring attention |
| --- | --- |
| `src/profile_layers.py` | `run` warms all four methods before runtime identity collection. `run_check`, `run_benchmark`, both profile loops and result metadata also use global `METHODS`. |
| `src/run_contract.py` | Schema 3 expects 13 comparisons, four graph check bundles and one identity/receipt for all methods. Non-3 schemas currently take the older v2 branch. `warmups` is not currently identity-bound. |
| `src/backend_provenance.py` | `require_local_native=True` requires ECHO and transfer DSOs even for HBM; the existing collector supports `False` without importing providers. |
| `scripts/run.sh`, `scripts/gap_profile.sh`, `scripts/profile_layers.sh` | Publication gates hard-code schema 3, 13/25 comparisons and eight measured operator captures. Capture export itself already follows `nsys_capture_order`. |
| `src/launch_gap.py` | Requires four prefill and extend captures, then immediately derives the HBM-relative prefill gate from the same directory. |
| `src/compact_timeline.py` | Extraction and receipt bind one profile directory/run ID; final panels must have all four methods. |
| `src/audit_shape_matrix.py`, `src/report_shape_matrix.py` | Each shape has one check/bench/profile directory, one receipt and exactly equal execution identities across those runs. Tensor reread, raw graph audit and timeline audit use that layout. |
| `src/report_full_graph_mfu.py`, `src/report_mfu.py` | Four templates, eight captures, one matching benchmark and one receipt are required. Useful FLOPs equality is a cohort property. |
| `scripts/shape_matrix.sh` | The manifest has one set of child directories per shape; it cannot express independent per-method processes. |

`operator_report.analyze_captures`, native graph attribution, interval union,
per-capture timeline classification and the kernel FLOPs definitions can be
reused. Never concatenate SQLite tables or compare graph/node/PID IDs across
children: those IDs belong to their original capture/process.

## Bounded implementation

1. **Single-method execution and receipt.** Add optional
   `--method {hbm,echo,serial_sparse,dense_prefetch}` to the existing
   `measure`/`profile_layers` CLI. Absence retains the existing four-method
   schema-3 path. Presence selects a one-element immutable method tuple for
   *every* warmup, check, benchmark and profile loop. Use explicit schema 4
   and a new single-method receipt kind; update all schema switches explicitly
   so schema 4 cannot enter v2 logic. Bind selected method, isolation policy,
   prebinding warmup count and ordered preparation contract in execution
   identity. Record process start/PID and completed method sequence separately
   as process provenance, not as cross-process identity fields.

2. **Preserve cross-method numerical acceptance using saved CPU evidence.**
   Run the isolated HBM check first. Offload checks take a check-only
   `--hbm-check-receipt` and load its controls with `map_location="cpu"`.
   Validate its accepted kind, artifacts and an explicit common-workload
   projection: source/dependencies/checkpoint/request/GPU/precision/shape,
   graph/residency policy and budgets. Retain both complete method identities;
   do not require equality of their method or actually loaded runtime entries.
   Run only the selected method's eager/graph/default-output checks, then
   compare saved HBM prefix logits, extend hidden and extend logits. Existing
   tolerances and changed-input/cache checks remain unchanged. ECHO must keep
   its cold transition proof and artifacts. Expected standard comparisons are
   one for HBM and four for each offload method; each has eleven graph checks
   when full-extend graphs are enabled. Profile adds three selected-method
   output comparisons. Encode these sets in shared validators, not counts in
   shell. No CPU comparison becomes a timed operation.

3. **Bind only observed runtime artifacts.** HBM passes
   `require_local_native=False`; offload retains `True`. Keep the existing
   mapped-file integrity checks, selected-method warmup before collection,
   exact check/bench/profile identity equality within that method, and native
   before/after equality. Add method/shape-specific presence checks for the
   adapters actually required by that dispatch; do not accept an empty
   inventory by merely disabling the full four-method gate. Do not load unused
   official decode, prefetch, hint or transfer providers to reproduce the old
   inventory. Record and audit participation, without claiming a per-launch
   DeepGEMM JIT ledger that the collector does not have.

4. **Reuse the existing child entrypoints.** Update their post-exit validators
   to accept valid schema-3 or single-method schema-4 results. Each isolated
   full-graph profile has four captures: compute setup, selected prefill,
   selected extend setup and selected extend replay; two are measured and
   exactly one full-extend template is required. `gap_profile` uses the same
   method tuple and its own unchanged trace-warmup boundary. Preserve NSYS
   flags, affinity, failure propagation, fresh IDs and staging publication.
   Save literal command, runner hash and process identity for each child.

5. **Add one orchestration entrypoint and a real cohort manifest.** Prefer
   `scripts/method_isolation.sh` for this bounded H64K+A1 publication, calling
   existing `run.sh`, `gap_profile.sh` and `profile_layers.sh`; it contains no
   model logic. Prepare one explicit request, run HBM check then the other
   three checks, and run each method's clean benchmark, minimal profile and
   operator profile in separate Python processes. Both profiles bind that
   method's own check and clean benchmark. Child IDs are
   `<cohort>_<method>_{check,bench,profile,operators}`. The manifest records all
   four sets of paths, hashes, exact invocations and execution order. Publish
   it only after all children complete. Check directories stay under `/tmp`;
   failures stay outside experiment outputs. Existing `shape_matrix.sh` stays
   compatible and must not be used to claim method isolation. Generalizing
   the whole H/A matrix is a later, separate publication task.

6. **Join evidence, not synthetic runs.** Add
   `src/method_isolation_report.py` as the CPU cohort entrypoint. Reuse/refactor
   existing audit and rendering helpers to accept a mapping from method to
   child directories. It verifies exact four-method coverage, within-method
   check/bench/two-profile bindings, common workload, saved tensor comparisons,
   ECHO transitions, runtime/source artifacts, per-sample traffic, graph
   ownership, FLOPs and interval conservation. A single-method gap audit
   reports raw prefill/extend quantities; its HBM-relative prefill gate is
   explicitly uncomputed until this cohort join. Render four panels from their
   own captures with a shared scale, and compute each MFU from its own clean
   wall samples. Rows and panel receipts retain the actual child run ID,
   profile directory, SQLite hash and benchmark binding. Never invent a
   four-method `result.json`, share one receipt across methods, or rename/copy
   traces to make them look like one process. Existing schema-3 report readers
   remain strict; a new manifest branch supplies the method mapping.

## Exact edit surface

Required execution files: `src/profile_layers.py`, `src/run_contract.py`,
`src/gap_profile.py`, and `scripts/{run,gap_profile,profile_layers}.sh`.
Add `scripts/method_isolation.sh`. `measure.py` already forwards to the common
driver and needs no separate loop. Collector behavior can be selected through
its existing argument; method-specific validation belongs in the experiment
contract, so no shared `evaluation/local_native.py` change is required.

Required analysis files: `src/launch_gap.py`, `src/compact_timeline.py`,
`src/audit_shape_matrix.py` (extract/reuse per-method tensor/runtime/raw audit
helpers), `src/report_full_graph_mfu.py` and the shared helpers it imports from
`src/report_mfu.py`; add `src/method_isolation_report.py`. Preserve existing
schema-3 APIs/defaults. No changes to `operator_report.py`, `analyze_nsys.py`,
`timeline.py`, classifier semantics or FLOPs are planned unless a concrete
incompatibility is found. `report_shape_matrix.py` and `shape_matrix.sh` stay
unchanged because the new manifest/report handles this one cohort explicitly.

Update focused existing driver/script/gap/full-graph/report tests and add
`tests/test_method_isolation.py` for the new cross-process contract. Include
new driver/contract files and runner identities in source inventories.
After acceptance update the MFU README/report and the official comparison
reader's per-panel provenance in
`experiments/deepseek_v32_echo_official/src/compare_timeline.py`; its current
reader assigns one local profile ID to both panels. Regenerate affected
comparison assets through the existing presentation entrypoints. No official
Engine rerun or change to its numerical acceptance is implied.

## Execution and publication gates

1. CPU checks must reject another method's receipt, old mixed-method receipt,
   changed HBM reference, absent required runtime adapter, extra/missing method,
   duplicate/wrong child binding, mutated trace/artifact and incomplete child.
   Verify fake-driver call histories never execute another method, old APIs
   still pass existing tests, CLI help is side-effect free and shell failure
   status is retained. Recompute cohort saved-output and raw-timing summaries
   independently rather than only re-reading the new report's assertions.
2. Freeze the complete source/runner tree. Run four fresh method checks before
   clean measurements; do not reuse the old formal or factor-control receipts.
   Use all existing numerical/cache/ECHO proof and memory checks. Execute
   clean timing, minimal NSYS and operator NSYS separately with matching own
   receipts. Record all observed process/device rows and actual graph lineage.
3. Audit HBM's raw L0–L2 signature and ownership against the accepted matched
   control, and report span/busy/idle/gaps separately from synchronized wall
   samples. Measure the other methods under their own warmup too. A short gap
   in a profile is not itself evidence of a stable wall-time improvement.
   For a claimed causal improvement, rerun the mixed-method control under the
   final source in independent balanced processes; do not turn one new cohort
   versus an older batch into a speedup estimate.
4. Publish a fresh cohort with check/bench/profile/operator audit and per-method
   provenance before replacing affected H64K+A1 tables, MFU, timelines and
   official comparison assets. Keep original formal results until then. The
   README must describe the changed process/warmup boundary; the old numbers
   cannot become isolated measurements by relabelling. Keep unaffected A128
   and H/A matrix results with their original mixed-process boundary. Preserve
   valid factor controls supporting this diagnosis. Official token, GPU,
   residency and framework differences and failed numerical acceptance remain.

The intended conclusion is that independent methods no longer inherit another
method's same-process warmup state. Any remaining performance difference or
hardware explanation requires its own measured evidence.
