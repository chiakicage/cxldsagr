# Motivation profiling completion record

User request: diagnose why HBM-only first visits are faster, calculate MFU, and inspect compute/fetch pipelines. Scope is diagnostic; no production model/cache/operator implementation was changed.

This file's original sections record the earlier eager diagnosis. The accepted
C3 graph-enabled profile and full independent CPU audit are recorded in
[the graph profile checkpoint](deepseek_motivation_graph_profile_checkpoint.md#full-c3-profile-accepted-and-independently-audited).
Its new results are not yet published and do not validate subsequent admission,
cleanup or layout changes.

## Accepted runs

- Formal latency: `deepseek_v32_motivation_20261004_p65536_nh16777216_u16_r2_02`.
- Nsight Systems: `deepseek_v32_motivation_profile_20261004_01`.
- CPU FLOPs analysis: `motivation_flops_20261004_01`.
- Report: `experiments/deepseek_v32_motivation/report/diagnosis.md`.

The earlier profile process completed and published all eight SQLite/raw captures under the experiment output layout. No process remains to resume for that run. Do not restart or confuse the diagnostic `_01` with the deleted insufficient-warmup formal run. Its completion state does not describe the ongoing optimization goal.

## Verified findings

Formal first-visit E2E means HBM/ECHO/serial/dense = 2872.185 / 4792.370 / 4099.002 / 5285.434 ms. History accounts for 98.8–99.0% of the extra offload wall time. All three captured cold histories have zero historical KV H2D and 720 MiB D2H. Dense history scans 20,643,840 cumulative records and all hit. Pool allocation is outside the timed request. Cache sorting, union deduplication, mapping/scalar synchronization and the slower fused indexer explain implementation costs; do not interpret the difference as unavoidable transfer cost.

Reference-peak effective E2E MFU first visit = 33.68 / 20.19 / 23.60 / 18.30%. History matrix work is 1.502905 PFLOPs, candidate 3.637151 TFLOPs. Mixed precision ideal times 965.160 / 2.246 ms. Formal times remain the denominator: instrumentation significantly inflates wall time. Profile explicitly records no TF32; the formal run did not record the policy, so report its FP32 assumption and TF32 sensitivity.

Dense revisit gather is on non-blocking stream 42, compute on stream 7. Ten gathers total 14.721 ms, zero observed matrix-kernel overlap, only 0.458 ms D2D overlap. Nine lookahead gathers run while 57 prior-layer matrix-related kernels (including helpers) have already been submitted. Readiness events do not make the current layer wait for the next layer's gather; neither default-stream implicit synchronization nor late CPU submission alone explains the result. The 65,536-CTA gather grid suggests GPU scheduling/resource contention, but exact resources remain unmeasured. ECHO fused kernels total 8.022 ms plus residual gather 0.355 ms; their internal overlap was not measured.

All 80 diagnostic output checks passed. All 45,933 matrix calls / 195 operator groups match planned useful FLOPs and dimensions, with 4/3/3 independent source copies. 1,247 source snapshot files verified. All eight traces have zero unattributed GPU activities and conserve GPU count/time and CPU exclusive per-thread unions.

## Sources and evidence

New sources: `src/profile.py`, `scripts/profile.sh`, `src/flops.py`, `src/analyze_pipeline.py`, `src/publish_diagnosis.py`; all under `experiments/deepseek_v32_motivation`. Source identity for post-run analyzer edits is separate from the frozen profile source. `analysis/pipeline.json`, stage/API CSV, numerical/call ledgers, `flops_verification.json`, and `revisit_pipeline_evidence.json` preserve measurements, inputs and attribution details. Pipeline evidence includes its executable analysis source and event joins.

Report artifacts are generated in the profile run's `analysis/publication/` and selected into `report/diagnosis/`. Keep original formal report and output: this task did not replace its execution implementation or performance data. Chinese prose was reviewed using humanizer-zh and the independent pipeline reviewer.

## Research handoff for Supervisor

Relevant status items 2.3–2.5 / 4.1 / T-007: the fixed P/NH result now has a first-visit implementation-cost explanation and a sampled revisit pipeline. It supports neither a generic offload transfer penalty nor a fusion benefit. Dense's software lookahead exists, but actual matrix overlap was zero in the diagnostic capture. Exact scheduling cause, improvements after removing overhead, common-model NOSA comparison and real GR representativeness remain open. Do not revive withdrawn 4 GiB/W/chunk results. This engineering handoff does not rewrite researcher-owned status judgments or initiate a full Supervisor review.

## Operator-MFU supplement

User requested per-operator MFU after the request-level diagnosis. Analysis `motivation_operator_mfu_20261004_01` reuses all eight profile captures; report is `experiments/deepseek_v32_motivation/report/operator_mfu.md`, selected data under `report/operator_mfu/`. New sources: `src/operator_mfu.py`, `src/report_operator_mfu.py`; four meaningful CPU tests pass.

All 45,933 calls map uniquely, one matrix entry point per call, 133,046 matrix-API GPU activities. Eight independent direct-SQL MFU checks match. Main MFU denominator is summed per-call unions of all API GPU activities; separate columns retain primary matrix-kernel and per-call GPU-span denominators. Actual FP32 policy is explicit in this profile. Matrix API boundaries exclude outer recall and top-k, and fused ECHO kernel time includes inseparable prefetch/scalar work.

The prior name classifier included `deep_gemm::transpose_fp32` due its namespace. It is corrected to concrete matrix entry points; split-K reductions are also helpers. Existing diagnostic pipeline assets were regenerated and republished after confirming all eight request/segment activity totals, metadata times and zero-matrix-overlap conclusions are unchanged. Formal performance/MFU data and raw captures are unchanged. No new GPU measurement occurred. Publication and completion audits contain current report/source hashes.
