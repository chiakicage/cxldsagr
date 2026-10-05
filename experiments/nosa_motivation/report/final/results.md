# NOSA motivation: `refactor_final_nosa_bench_20261005_01`

Complete 32-layer NOSA checkpoint with full NOSA sparse policy and normalized candidate hidden output. No LM head; the DeepSeek motivation experiment also executes its last-token LM head and uses a checkpoint workload substitute. Synthetic cyclic GR input does not establish task quality or representativeness.

Synchronized runner wall latency includes token validation/transfer, admission, eviction, independent sparse history construction on miss, all candidate hidden states, and candidate discard. Model loading, request generation, warmup, counters and reports are outside latency. Bench performs no full-output CPU copies, numerical comparison or output saving between requests. Check publishes no timings.

| Method | Visit | Requests | History hits | Request mean ms | Candidate mean ms | Candidate MFU % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hbm | first | 16 | 0 | 2340.204338 | 16.687044 | 14.242021 |
| hbm | revisit | 16 | 0 | 2342.510886 | 16.846617 | 14.107119 |
| dense_prefetch | first | 16 | 0 | 2691.766039 | 25.343662 | 9.377383 |
| dense_prefetch | revisit | 16 | 16 | 71.903374 | 68.744575 | 3.457105 |
| sync_sparse | first | 16 | 0 | 2671.701466 | 22.600388 | 10.515626 |
| sync_sparse | revisit | 16 | 16 | 29.568324 | 26.449870 | 8.985195 |
| async_sparse | first | 16 | 0 | 2754.456084 | 22.901707 | 10.377272 |
| async_sparse | revisit | 16 | 16 | 31.924646 | 28.815592 | 8.247522 |

Diagnostic source: `refactor_final_nosa_profile_20261005_02`. Numerical reference: `refactor_final_nosa_check_20261005_01`. Every runner wall denominator in api_comparison.json comes from clean bench `refactor_final_nosa_bench_20261005_01`.

Async versus synchronous sparse whole-trace latency gate: **failed**. Every applicable layer/sample page-and-stripe 90% overlap gate: **failed**.

[API comparisons](api_comparison.json) retain complete API spans and activity unions separately. Independent matrix/attention medians form composition estimates, not measured complete requests. Timing validity does not imply a speedup or a passed efficiency gate.

[Memory](memory.csv) separates allocator peaks, after-request device samples, cache charges and cache reservations. Charges include graph private reservations; they are not a tensor-payload sum. These observations do not validate physical capacity at a filled NH quota.

Benchmark source: `4665ccce226af23f631e619f748369cea28d4a09b99c1947cbb6b030e58f120c`. Workload: `d8569499b36e8a4ee9f1367e760b875494a7f1a545245c47e35535c6970a8040`.
