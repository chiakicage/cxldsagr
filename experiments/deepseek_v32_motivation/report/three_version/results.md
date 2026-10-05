# deepseek_v32 serving version comparison

Comparison run: `refactor_final_deepseek_three_version_20261005_01`. All input runs passed their complete saved-evidence audit.

Published values are descriptive historical controls. Only baseline/current share the validated workload, device/CPU/NUMA, precision and timing boundary. Repeat ranges and MAD are empirical detection floors, not permitted regression percentages or confidence intervals. Any stable increase or added synchronization requires investigation, including small increases within the observed range. Three repeats do not establish a tail-latency distribution. Memory samples are not process peaks; cache accounting is separate.

Historical contract differences: measurement_boundary, execution_environment, hardware.

| Role | Run | Source SHA-256 | GPU UUID |
| --- | --- | --- | --- |
| published | motivation_c10_20261004_u16_r2_01 | 11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f | 1bdee8b4-22ac-536c-208b-bfb4ed38b878 |
| baseline | refactor_p0_deepseek_bench_20261005_01 | d31a0ff6bde97acab24bb2f2c648e9d91ae46ba18f6c074b5f572cf49aaaab7a | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| baseline | refactor_p0_deepseek_bench_20261005_02 | d31a0ff6bde97acab24bb2f2c648e9d91ae46ba18f6c074b5f572cf49aaaab7a | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| baseline | refactor_p0_deepseek_bench_20261005_03 | d31a0ff6bde97acab24bb2f2c648e9d91ae46ba18f6c074b5f572cf49aaaab7a | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| current | refactor_final_deepseek_bench_20261005_01 | 1319a540b7f78da9c363dd09c0f1568b450df3df5e4d3e8b60155a58c348b3dc | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| current | refactor_final_deepseek_bench_20261005_02 | 1319a540b7f78da9c363dd09c0f1568b450df3df5e4d3e8b60155a58c348b3dc | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |
| current | refactor_final_deepseek_bench_20261005_03 | 1319a540b7f78da9c363dd09c0f1568b450df3df5e4d3e8b60155a58c348b3dc | 80ff95c3-176e-fd8a-728f-9c5577c4a779 |

Memory below is the maximum across methods, requests and each role's runs, in GiB. Post-request samples and CUDA allocator peak counters have separate columns; neither is a continuously observed device-used peak.

| Role | Sampled allocated | Sampled reserved | Sampled device used | Allocator peak allocated | Allocator peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| published | 14.593 | 24.305 | 25.675 | 16.359 | 24.305 |
| baseline | 14.593 | 24.305 | 25.675 | 16.359 | 24.305 |
| current | 14.593 | 24.305 | 25.675 | 16.359 | 24.305 |

Values below are means within each run; baseline/current show the median of independent runs.
All values are milliseconds. Positive change means slower execution.

| Method | Visit | Metric | Published | Baseline | Current | Change | Baseline range | Assessment |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense_prefetch | first | latency_ms | 2233.838 | 2239.907 | 2236.624 | -3.283 | 7.674 | no_observed_slowdown |
| dense_prefetch | first | extend_ms | 11.735 | 11.593 | 11.636 | +0.043 | 0.204 | increase_within_observed_repeat_range |
| dense_prefetch | revisit | latency_ms | 31.889 | 32.123 | 32.634 | +0.511 | 0.372 | investigate_slowdown |
| dense_prefetch | revisit | extend_ms | 28.057 | 28.752 | 28.933 | +0.182 | 0.298 | increase_within_observed_repeat_range |
| echo | first | latency_ms | 2283.433 | 2292.119 | 2294.786 | +2.667 | 5.737 | increase_within_observed_repeat_range |
| echo | first | extend_ms | 12.749 | 12.480 | 12.654 | +0.173 | 0.092 | investigate_slowdown |
| echo | revisit | latency_ms | 24.947 | 24.128 | 24.530 | +0.402 | 0.103 | investigate_slowdown |
| echo | revisit | extend_ms | 21.072 | 20.796 | 20.865 | +0.068 | 0.102 | investigate_slowdown |
| hbm | first | latency_ms | 2176.886 | 2193.969 | 2190.912 | -3.057 | 4.233 | no_observed_slowdown |
| hbm | first | extend_ms | 10.596 | 10.601 | 10.718 | +0.118 | 0.016 | investigate_slowdown |
| hbm | revisit | latency_ms | 2170.244 | 2185.788 | 2187.106 | +1.318 | 13.054 | increase_within_observed_repeat_range |
| hbm | revisit | extend_ms | 10.601 | 10.634 | 10.703 | +0.069 | 0.077 | increase_within_observed_repeat_range |
| serial_sparse | first | latency_ms | 2217.737 | 2225.885 | 2225.141 | -0.743 | 3.336 | no_observed_slowdown |
| serial_sparse | first | extend_ms | 11.501 | 11.304 | 11.324 | +0.020 | 0.066 | increase_within_observed_repeat_range |
| serial_sparse | revisit | latency_ms | 18.456 | 17.763 | 18.084 | +0.321 | 0.271 | investigate_slowdown |
| serial_sparse | revisit | extend_ms | 14.624 | 14.415 | 14.439 | +0.023 | 0.219 | increase_within_observed_repeat_range |

[Complete comparison](comparison.json) retains mean/median/p95/sum, allocation charges, sampled memory, complete contracts, evidence hashes and numerical audit results. [Request samples](request_samples.csv) retain all measured request values.

旧完整测量、profile 和 FLOPs 在当前报告验收后清理。保留的比较 JSON/CSV 可重算选定对照并追溯原身份，不支持完整历史源码、native 或数值重审；P0 与当前三次控制运行继续保留。
