# DeepSeek V3.2 shape matrix

Matrix run: `deepseek_shape_matrix_complete_20261007_01`. Each shape has independent cold correctness, clean timing and NSYS profile runs.

The real checkpoint's L0–L2 propagate hidden/residual in order, with embedding, three dense MLPs, final norm and last-token LM head. History uses 1,024-token chunks; extend uses one complete A-token batch and one full CUDA Graph replay. P=NH=H+A; ordinary persistent append. This scope does not represent the full 61-layer model or C10 GR serving.

Times below are medians of synchronized wall-time samples. Input preparation, transactions and required synchronization/commit are included; weight loading, compilation, graph preparation and prefix restoration are excluded. Cold clears offload main-KV HBM residency while retaining DRAM and resident indexer data. Sample counts and all unrounded values are in [timing.csv](timing.csv) and [timing_samples.csv](timing_samples.csv).

[cache_metrics_samples.csv](cache_metrics_samples.csv) preserves each timed sample's actual per-layer counters and traffic, read after the wall timer stops. ECHO's bounded atomic prefetch can select different eligible records across executions when the cap is saturated, so prefetch/recall counts may differ. Check, clean benchmark and profile counters describe their respective executions and are not substituted for one another.

This matrix combines independently completed shape runs from 4 execution batches with the same model and measurement execution code; the hardware query deadline differs from 20 to 120 seconds. 2 prior preflights stopped before model loading after a `nvidia-smi` hardware query exceeded the 20-second timeout; the cause of the query delay is not established. The [exact input matrix manifest](matrix_manifest.json) preserves detailed batch metadata and run membership.

The source comparison verifies that the archived hardware-probe variants and current source differ only in the single `nvidia-smi` subprocess timeout line. Each shape retains exact check, benchmark and profile execution identities. Original hashes, variant paths and the comparison rule are recorded in [summary.json](summary.json).

## Complete prefill wall time (ms)

| H | A | `hbm` | `echo` | `serial_sparse` | `dense_prefetch` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 128 | 29.976 | 30.650 | 30.143 | 29.639 |
| 4,096 | 256 | 29.386 | 30.288 | 30.315 | 29.880 |
| 4,096 | 512 | 29.314 | 30.928 | 30.336 | 30.287 |
| 4,096 | 1,024 | 29.555 | 30.246 | 30.783 | 29.833 |
| 16,384 | 128 | 133.099 | 134.267 | 133.087 | 132.410 |
| 16,384 | 256 | 132.586 | 132.732 | 132.188 | 133.028 |
| 16,384 | 512 | 132.095 | 133.832 | 133.629 | 133.039 |
| 16,384 | 1,024 | 132.319 | 133.486 | 132.515 | 133.298 |
| 65,536 | 128 | 653.778 | 644.048 | 646.058 | 650.294 |
| 65,536 | 256 | 639.735 | 644.128 | 645.333 | 646.474 |
| 65,536 | 512 | 643.931 | 643.398 | 644.864 | 641.960 |
| 65,536 | 1,024 | 637.331 | 648.827 | 650.458 | 645.898 |

## Complete extend wall time (ms)

| H | A | `hbm` | `echo` | `serial_sparse` | `dense_prefetch` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 128 | 2.566 | 3.352 | 3.099 | 2.755 |
| 4,096 | 256 | 3.308 | 4.155 | 3.962 | 3.571 |
| 4,096 | 512 | 5.147 | 6.028 | 5.673 | 5.270 |
| 4,096 | 1,024 | 8.505 | 9.739 | 9.247 | 8.788 |
| 16,384 | 128 | 2.623 | 3.960 | 3.492 | 3.036 |
| 16,384 | 256 | 3.447 | 4.810 | 4.415 | 3.855 |
| 16,384 | 512 | 5.309 | 7.019 | 6.425 | 5.700 |
| 16,384 | 1,024 | 9.223 | 11.618 | 10.421 | 9.601 |
| 65,536 | 128 | 3.332 | 5.656 | 4.216 | 5.698 |
| 65,536 | 256 | 4.192 | 6.734 | 5.176 | 5.894 |
| 65,536 | 512 | 6.839 | 10.474 | 8.330 | 7.642 |
| 65,536 | 1,024 | 12.155 | 17.981 | 13.231 | 12.339 |

## Timelines

Each link compares all four methods from the shape's separate intrusive NSYS capture. Prefill shows only the final 1,024-token chunk's L0–L2; extend shows the complete A-token batch's L0–L2. Main windows end at the last L2 compute kernel and start at first L0 compute, including earlier L0 history H2D for dense extend. Startup views begin at forward entry and retain embedding. Profile windows are distinct from the complete wall-time boundary above.

Compute phases share one lane. H2D and D2H have separate lanes; red marks GPU idle, and ECHO prepare/finalize/hint have their own lane. Absolute timeline scales are shared across methods within a shape and phase, and may differ across shapes. Exact windows and gap values are in [timeline_windows.csv](timeline_windows.csv).

| H | A | Final prefill chunk | Complete extend layers | Extend with startup |
| ---: | ---: | --- | --- | --- |
| 4,096 | 128 | [SVG](h4096_a128/prefill.svg) | [SVG](h4096_a128/extend.svg) | [SVG](h4096_a128/with_startup/extend_with_startup.svg) |
| 4,096 | 256 | [SVG](h4096_a256/prefill.svg) | [SVG](h4096_a256/extend.svg) | [SVG](h4096_a256/with_startup/extend_with_startup.svg) |
| 4,096 | 512 | [SVG](h4096_a512/prefill.svg) | [SVG](h4096_a512/extend.svg) | [SVG](h4096_a512/with_startup/extend_with_startup.svg) |
| 4,096 | 1,024 | [SVG](h4096_a1024/prefill.svg) | [SVG](h4096_a1024/extend.svg) | [SVG](h4096_a1024/with_startup/extend_with_startup.svg) |
| 16,384 | 128 | [SVG](h16384_a128/prefill.svg) | [SVG](h16384_a128/extend.svg) | [SVG](h16384_a128/with_startup/extend_with_startup.svg) |
| 16,384 | 256 | [SVG](h16384_a256/prefill.svg) | [SVG](h16384_a256/extend.svg) | [SVG](h16384_a256/with_startup/extend_with_startup.svg) |
| 16,384 | 512 | [SVG](h16384_a512/prefill.svg) | [SVG](h16384_a512/extend.svg) | [SVG](h16384_a512/with_startup/extend_with_startup.svg) |
| 16,384 | 1,024 | [SVG](h16384_a1024/prefill.svg) | [SVG](h16384_a1024/extend.svg) | [SVG](h16384_a1024/with_startup/extend_with_startup.svg) |
| 65,536 | 128 | [SVG](h65536_a128/prefill.svg) | [SVG](h65536_a128/extend.svg) | [SVG](h65536_a128/with_startup/extend_with_startup.svg) |
| 65,536 | 256 | [SVG](h65536_a256/prefill.svg) | [SVG](h65536_a256/extend.svg) | [SVG](h65536_a256/with_startup/extend_with_startup.svg) |
| 65,536 | 512 | [SVG](h65536_a512/prefill.svg) | [SVG](h65536_a512/extend.svg) | [SVG](h65536_a512/with_startup/extend_with_startup.svg) |
| 65,536 | 1,024 | [SVG](h65536_a1024/prefill.svg) | [SVG](h65536_a1024/extend.svg) | [SVG](h65536_a1024/with_startup/extend_with_startup.svg) |

Discrete GPU/process observations are preserved in the [observer summary](audit/observer.json). These observations do not establish continuous GPU isolation or CPU/DRAM isolation.

Run IDs, execution identities, hardware, precision, dependency versions and input hashes are recorded in [summary.json](summary.json) and [provenance.json](provenance.json). [publication_manifest.json](publication_manifest.json) binds the selected report files. This matrix adds timing and timelines; it does not estimate new per-operator MFU values.
