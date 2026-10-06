# Top-k completion to MLA compute start

Analysis ID: `cache_attention_transition_model_20261006_02`. Values come from actual NSYS GPU timestamps.

The sole measured interval starts after the last exact-top-k GPU activity, including invalid-ID masking, and ends at the first actual MLA compute kernel. Indexer, top-k and MLA computation are outside it. No stage-duration subtraction or extra mid-path synchronization is used. The interval retains hint updates, KV append/writeback, exact recall, mapping, GPU control, idle, and MLA preparation/launch delays.

The independent replay executes the production MLA consumer. Complete-model captures, when supplied, are a separate workload. These are intrusive profile measurements, not unprofiled wall-clock benchmarks; profiler overhead remains.

## complete_model

Source run: `deepseek_gap_v10_a128_minimal_20261006_01`.

Times are µs. `all` first sums the separate layer windows within each sample, then takes the median; it is not a complete request duration. Q1/Q3 use inclusive sample quantiles.

| Method | Phase | Layer | Samples | Median | Q1 | Q3 |
|---|---|---:|---:|---:|---:|---:|
| echo | extend_cold | 0 | 1 | 134.367 | 134.367 | 134.367 |
| echo | extend_cold | 1 | 1 | 78.784 | 78.784 | 78.784 |
| echo | extend_cold | 2 | 1 | 106.815 | 106.815 | 106.815 |
| echo | extend_cold | all | 1 | 319.966 | 319.966 | 319.966 |
| serial_sparse | extend_cold | 0 | 1 | 465.695 | 465.695 | 465.695 |
| serial_sparse | extend_cold | 1 | 1 | 222.848 | 222.848 | 222.848 |
| serial_sparse | extend_cold | 2 | 1 | 363.519 | 363.519 | 363.519 |
| serial_sparse | extend_cold | all | 1 | 1052.062 | 1052.062 | 1052.062 |

![Measured post-top-k intervals](transition.svg)

## Accounting

`windows.csv` retains every sample's endpoint timestamps, IO union, exposed GPU control, idle, and signed MLA launch offset. IO + exposed control + idle equals each window. Zero-record gathers remain control; persistent append D2H is IO even for warm history. `stages.csv` retains only hint, append, recall and MLA wrapper stages. CPU scope time is a separate submission view and must not be added to GPU execution intervals.

A negative launch offset means MLA was already submitted when top-k completed on GPU. Idle-before-launch counts only idle ending at a unique main-stream kernel, before its correlated launch API begins. Other-stream endings remain unassigned. This does not attribute host residual time to Python, C++ checks, allocation or scheduling.

Replay inputs use an append-built/restored prefix and zero initial hint. This differs from the full model's cache trajectory; differences between workloads are not wrapper-cost estimates. No GR transient candidate, H>P split query, model speedup or complete-model gap gate is claimed.
