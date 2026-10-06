# Top-k completion to MLA compute start

Analysis ID: `cache_posttopk_profile_20261006_01_transition`. Values come from actual NSYS GPU timestamps.

The sole measured interval starts after the last exact-top-k GPU activity, including invalid-ID masking, and ends at the first actual MLA compute kernel. Indexer, top-k and MLA computation are outside it. No stage-duration subtraction or extra mid-path synchronization is used. The interval retains hint updates, KV append/writeback, exact recall, mapping, GPU control, idle, and MLA preparation/launch delays.

The independent replay executes the production MLA consumer. Complete-model captures, when supplied, are a separate workload. These are intrusive profile measurements, not unprofiled wall-clock benchmarks; profiler overhead remains.

## standalone_replay

Source run: `cache_posttopk_profile_20261006_01`.

Times are µs. `all` first sums the separate layer windows within each sample, then takes the median; it is not a complete request duration. Q1/Q3 use inclusive sample quantiles.

| Method | Phase | Layer | Samples | Median | Q1 | Q3 |
|---|---|---:|---:|---:|---:|---:|
| echo | extend_cold | 0 | 7 | 98.560 | 98.303 | 99.120 |
| echo | extend_cold | 1 | 7 | 77.952 | 76.736 | 78.864 |
| echo | extend_cold | 2 | 7 | 127.231 | 125.167 | 127.327 |
| echo | extend_cold | all | 7 | 301.886 | 299.808 | 304.832 |
| echo | extend_warm | 0 | 7 | 70.688 | 69.887 | 71.456 |
| echo | extend_warm | 1 | 7 | 70.335 | 69.952 | 70.607 |
| echo | extend_warm | 2 | 7 | 69.472 | 69.296 | 70.576 |
| echo | extend_warm | all | 7 | 210.751 | 209.774 | 212.671 |
| serial_sparse | extend_cold | 0 | 7 | 509.374 | 491.726 | 519.311 |
| serial_sparse | extend_cold | 1 | 7 | 244.255 | 242.623 | 248.863 |
| serial_sparse | extend_cold | 2 | 7 | 335.423 | 325.839 | 352.335 |
| serial_sparse | extend_cold | all | 7 | 1087.324 | 1073.804 | 1095.389 |
| serial_sparse | extend_warm | 0 | 7 | 454.111 | 431.198 | 469.791 |
| serial_sparse | extend_warm | 1 | 7 | 248.543 | 242.399 | 254.463 |
| serial_sparse | extend_warm | 2 | 7 | 221.215 | 213.135 | 231.215 |
| serial_sparse | extend_warm | all | 7 | 916.094 | 905.309 | 940.077 |

![Measured post-top-k intervals](transition.svg)

## Accounting

`windows.csv` retains every sample's endpoint timestamps, IO union, exposed GPU control, idle, and signed MLA launch offset. IO + exposed control + idle equals each window. Zero-record gathers remain control; persistent append D2H is IO even for warm history. `stages.csv` retains only hint, append, recall and MLA wrapper stages. CPU scope time is a separate submission view and must not be added to GPU execution intervals.

A negative launch offset means MLA was already submitted when top-k completed on GPU. Idle-before-launch counts only idle ending at a unique main-stream kernel, before its correlated launch API begins. Other-stream endings remain unassigned. This does not attribute host residual time to Python, C++ checks, allocation or scheduling.

Replay inputs use an append-built/restored prefix and zero initial hint. This differs from the full model's cache trajectory; differences between workloads are not wrapper-cost estimates. No GR transient candidate, H>P split query, model speedup or complete-model gap gate is claimed.
