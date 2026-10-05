# nosa serving version comparison

Comparison run: `refactor_final_nosa_three_version_20261005_01`. All input runs passed their complete saved-evidence audit.

Published values are descriptive historical controls. Only baseline/current share the validated workload, device/CPU/NUMA, precision and timing boundary. Repeat ranges and MAD are empirical detection floors, not permitted regression percentages or confidence intervals. Any stable increase or added synchronization requires investigation, including small increases within the observed range. Three repeats do not establish a tail-latency distribution. Memory samples are not process peaks; cache accounting is separate.

Historical contract differences: measurement_boundary, execution_environment, hardware.

| Role | Run | Source SHA-256 | GPU UUID |
| --- | --- | --- | --- |
| published | nosa_motivation_poolscan_sm90_20261004_01 | 93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79 | a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872 |
| baseline | refactor_p0_nosa_bench_20261005_01 | 2fb94cdfc99a710b6347cd324ab1d6efe1d74c511f3d04a55210c10e7ffd32cb | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| baseline | refactor_p0_nosa_bench_20261005_02 | 2fb94cdfc99a710b6347cd324ab1d6efe1d74c511f3d04a55210c10e7ffd32cb | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| baseline | refactor_p0_nosa_bench_20261005_03 | 2fb94cdfc99a710b6347cd324ab1d6efe1d74c511f3d04a55210c10e7ffd32cb | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| current | refactor_final_nosa_bench_20261005_01 | 4665ccce226af23f631e619f748369cea28d4a09b99c1947cbb6b030e58f120c | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| current | refactor_final_nosa_bench_20261005_02 | 4665ccce226af23f631e619f748369cea28d4a09b99c1947cbb6b030e58f120c | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| current | refactor_final_nosa_bench_20261005_03 | 4665ccce226af23f631e619f748369cea28d4a09b99c1947cbb6b030e58f120c | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |

Memory below is the maximum across methods, requests and each role's runs, in GiB. Post-request samples and CUDA allocator peak counters have separate columns; neither is a continuously observed device-used peak.

| Role | Sampled allocated | Sampled reserved | Sampled device used | Allocator peak allocated | Allocator peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| published | 19.950 | 25.330 | 26.148 | 19.990 | 25.330 |
| baseline | 19.950 | 25.330 | 26.148 | 19.990 | 25.330 |
| current | 19.950 | 25.330 | 26.148 | 19.990 | 25.330 |

Values below are means within each run; baseline/current show the median of independent runs.
All values are milliseconds. Positive change means slower execution.

| Method | Visit | Metric | Published | Baseline | Current | Change | Baseline range | Assessment |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| async_sparse | first | latency_ms | 2758.953 | 2744.305 | 2754.456 | +10.151 | 4.285 | investigate_slowdown |
| async_sparse | first | extend_ms | 22.598 | 22.610 | 22.466 | -0.144 | 0.295 | no_observed_slowdown |
| async_sparse | revisit | latency_ms | 31.405 | 31.971 | 31.952 | -0.019 | 0.045 | no_observed_slowdown |
| async_sparse | revisit | extend_ms | 28.431 | 28.984 | 28.830 | -0.155 | 0.030 | no_observed_slowdown |
| dense_prefetch | first | latency_ms | 2685.632 | 2687.375 | 2693.265 | +5.890 | 14.978 | increase_within_observed_repeat_range |
| dense_prefetch | first | extend_ms | 25.096 | 25.188 | 25.327 | +0.139 | 1.014 | increase_within_observed_repeat_range |
| dense_prefetch | revisit | latency_ms | 71.594 | 71.786 | 71.980 | +0.195 | 0.118 | investigate_slowdown |
| dense_prefetch | revisit | extend_ms | 68.620 | 68.746 | 68.745 | -0.001 | 0.124 | no_observed_slowdown |
| hbm | first | latency_ms | 2338.156 | 2341.024 | 2340.204 | -0.820 | 9.554 | no_observed_slowdown |
| hbm | first | extend_ms | 16.694 | 16.643 | 16.753 | +0.111 | 0.068 | investigate_slowdown |
| hbm | revisit | latency_ms | 2321.743 | 2339.101 | 2342.511 | +3.410 | 10.041 | increase_within_observed_repeat_range |
| hbm | revisit | extend_ms | 16.603 | 16.803 | 16.835 | +0.032 | 0.391 | increase_within_observed_repeat_range |
| sync_sparse | first | latency_ms | 2664.894 | 2663.782 | 2669.825 | +6.042 | 13.449 | increase_within_observed_repeat_range |
| sync_sparse | first | extend_ms | 22.385 | 22.335 | 22.568 | +0.233 | 0.554 | increase_within_observed_repeat_range |
| sync_sparse | revisit | latency_ms | 29.321 | 29.121 | 29.568 | +0.447 | 0.583 | increase_within_observed_repeat_range |
| sync_sparse | revisit | extend_ms | 26.324 | 26.117 | 26.450 | +0.333 | 0.406 | increase_within_observed_repeat_range |

[Complete comparison](comparison.json) retains mean/median/p95/sum, allocation charges, sampled memory, complete contracts, evidence hashes and numerical audit results. [Request samples](request_samples.csv) retain all measured request values.
