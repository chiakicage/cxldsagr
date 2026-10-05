# Bounded CPU cleanup diagnostic

This diagnostic measures the unmodified generic pool and persistent runner using the existing `SharedBackend` fixture. It is engineering evidence, not a NOSA experiment or correctness receipt. The P0 and current processes each exited 0 with CUDA disabled and uninitialized. No production source or tests were changed.

The only local fixture overrides widen the host quota to 1 or 16 retained sessions and replace candidate compute with appending two fixture tokens and returning a preallocated CPU tensor. Each process uses CPU affinity 0–7, NUMA node 0, and one PyTorch/OpenMP/MKL/OpenBLAS thread. Twenty warmups precede seven batches of 500 calls; cProfile uses separate 300-call passes. P0 ran before current, without interleaving or CPU clock monitoring.

| Retained sessions | Metric | P0 median (µs) | Current median (µs) | Difference (µs) |
|---:|---|---:|---:|---:|
| 1 | one pool audit | 3.141 | 5.955 | +2.813 |
| 1 | mock cleanup | 7.471 | 14.487 | +7.015 |
| 1 | mock extend | 3.573 | 12.473 | +8.900 |
| 1 | mock request wall | 17.335 | 40.513 | +23.179 |
| 16 | one pool audit | 27.685 | 46.946 | +19.261 |
| 16 | mock cleanup | 56.044 | 96.249 | +40.205 |
| 16 | mock extend | 3.691 | 13.444 | +9.754 |
| 16 | mock request wall | 66.365 | 123.849 | +57.484 |

All batch values are retained in `baseline.json` and `current.json`. The current 16-session cleanup has one high batch: its full range is 95.530–171.341 µs, compared with 55.842–56.459 µs for P0. Its median is therefore more representative than a mean of the seven batches, but this is still one sequential process pair. The one-audit ranges are 27.504–28.002 µs for P0 and 46.606–47.272 µs for current.

Both versions execute two full pool audits per mock request. The current audit calls `TokenRuntime.session_usage`, its `_check`, and adapter `session_usage` once per retained session. The adapter creates and validates a fresh `ResourceUsage`; backend session accounting is still called once per session. The shared route additionally creates a `CacheFootprint` and `ResourceUsage`, converts it back to a mapping, and the pool constructs a checked footprint from that mapping.

| Calls per one 16-session audit | P0 | Current |
|---|---:|---:|
| backend session byte observation | 16 | 16 |
| runtime session usage and `_check` (each) | 0 | 16 |
| adapter session usage | 0 | 16 |
| ResourceUsage construction/validation | 0 | 17 |
| CacheFootprint `from_mapping` | 17 | 18 |
| CacheFootprint validation | 33 | 34 |
| ResourceUsage `as_mapping` | 0 | 1 |

In a complete mock request, the current path constructs 35 usage objects (32 session plus 3 shared), executes 33 runtime checks, and retains the same two pool audits, four fixture synchronizations, and one fixture truncate. The third shared observation is outside the two cleanup audits. Current strict integer validation also calls `nonnegative` while P0 uses the earlier footprint checks; this remains part of the measured contract cost.

The two-audit timing difference is +5.627 µs with one session and +38.522 µs with 16 sessions. These are close to the mock cleanup changes of +7.015 and +40.205 µs. Together with the call counts, this identifies fresh runtime/usage contract work as a concrete contributor in this fixture. cProfile absolute timings are inflated by instrumentation and its overlapping cumulative rows must not be summed.

The fixture does not exercise real NOSA storage enumeration, allocator/lifecycle audits, mutation leases, or CUDA synchronization. It therefore cannot attribute all of the complete-serving 30–90 µs cleanup increase to these wrappers. No validation removal or source optimization is proposed from this probe. The paired complete-request measurement determines the serving effect; residual overhead must remain visible in the final report.

Original commands, source hashes, timing summaries and retained-file hashes are in `evidence.json`. The diagnostic script is preserved byte-for-byte as `profile_cleanup.py.txt`; it is not installed as repository code. The `.pstats` files retain the original separate call-path profiles.
