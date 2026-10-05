# Final NOSA repeated-run review

The final three-version comparison `refactor_final_nosa_three_version_20261005_01`
passed its complete saved-evidence audit for the published run, three P0 runs and
three final runs. All 128 per-request identity, quota, cache-charge and candidate
H2D records match across all six P0/final runs. PyTorch allocated, reserved and
sampled device-used values remain distinct from cache accounting.

The [phase/monitor review](nosa_final_phase_and_monitor_review.json) retains all
per-run phase means, ranges and MAD. The
[matched-request review](nosa_final_matched_request_phase_review.json) additionally
matches each request across P0, the intermediate implementation and final source.
Requests in one process are not independent trial replicates.

The full-trace median rose from 206,538.430 to 206,944.563 ms, a 406.132 ms increase.
P0's three-run range is 223.016 ms; the final range is 409.043 ms. All final totals
exceed the largest P0 total, but the median difference does not exceed both
ranges. The final total is also only 110.172 ms above the intermediate median,
within both those ranges. These aggregate points do not establish a stable causal
regression.

| Observation | Final minus P0, ms | P0 range, ms | Final range, ms | Detail |
| --- | ---: | ---: | ---: | --- |
| Async first-request mean | +10.1513 | 4.2855 | 15.6027 | Ranges overlap; 11/16 matched request medians increase, none has all final samples above all P0 samples |
| Dense revisit mean | +0.1945 | 0.1179 | 0.0981 | Run ranges do not overlap; 14/16 matched request medians increase |
| HBM first candidate `extend_ms` | +0.1107 | 0.0676 | 0.0694 | Run ranges do not overlap; 14/16 matched request medians increase |
| Async revisit mean | -0.0193 | 0.0447 | 0.2105 | Aggregate decrease; 11/16 matched request medians still increase, showing sensitivity to aggregation and outliers |

All eight cleanup groups increase by 0.0299–0.0897 ms. Each group has nonoverlapping
P0/final run ranges, and the median increase exceeds both ranges. All 128 matched
request cleanup medians increase; 119/128 requests have every final cleanup sample
above every P0 sample. This is the clearest repeated phase shift. The earlier
[static inspection](nosa_plan_cpu_profile_01/findings.md) identified extra runtime,
usage-wrapper and mutation bookkeeping; it did not time those components or prove
which causes the observed difference. Fresh audit and synchronization remain.

Compared with the intermediate pre-layout-cache diagnostic, offload revisit
admission improves by 1.100–1.202 ms, consistent in direction with the independently
measured reduction in pure planning work. Its `extend_ms` increases by 1.102–1.475 ms;
these phase movements cannot be counted as an end-to-end saving. Relative to P0,
final revisit admission is +0.007 ms for sync, +0.053 ms for async and +0.127 ms for
dense. First-visit admission and prefix changes have much larger run variation;
none of their positive deltas exceeds both groups' ranges.

All 186 recorded process-monitor samples have zero return codes and match their
own benchmark UTC interval and GPU3 UUID. Each run has 62 samples: 61 matching
compute-process rows and one empty startup sample. Sample spacing is
5.099–5.462 s, including query time. There are no foreign rows in the observations;
short activity between samples, CPU interference, clocks and utilization are not
observed by this query.

The old publication differs in measurement boundary, environment and physical
GPU, so its values remain descriptive historical context. The intermediate
comparison is explicitly retained as an earlier diagnostic, including its missing
process-monitor boundary; neither is relabelled as final evidence.

An [optional four-run interleaved confirmation plan](nosa_interleaved_confirmation_plan.json)
is prepared with fresh IDs and opposite-order adjacent pairs. It has not been
authorized or launched. No arbitrary percentage tolerance is used to close the
remaining dense-revisit, candidate or cleanup observations. This review did not
change source, run GPU work, rerun tests or publish replacement performance assets.
