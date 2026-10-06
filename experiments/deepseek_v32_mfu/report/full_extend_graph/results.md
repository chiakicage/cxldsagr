# Complete extend CUDA Graph

The real checkpoint's L0–L2 run in one full CUDA Graph per extend, including cache operations and actual IO. Cold and warm correctness passed. This file records the original timeline-source measurement; the experiment README contains the current timing, MFU and selected-window gap results.

H=65,536; A=128; prefill chunk=1,024; P=NH=65,664; ordinary persistent append. One matched warmup per method, 3 clean prefill samples and 5 clean extend samples. Graph preparation and prefix restoration are outside timing; begin/synchronization/commit are included.

| Method | Prefill median ms | Extend median ms | Full-forward diagnostic gap % | Graph GPU nodes |
| --- | ---: | ---: | ---: | ---: |
| hbm | 644.938 | 3.326 | 30.47 | 218 |
| echo | 648.375 | 5.716 | 26.23 | 299 |
| serial_sparse | 640.701 | 4.265 | 34.26 | 242 |
| dense_prefetch | 645.800 | 5.706 | 30.96 | 224 |

Figures are displayed only in the experiment README. Prefill shows only the final chunk (64/64). Main windows end at L2 final compute and start at L0 first compute, extended for dense extend to include its L0 history H2D if earlier. Compute phases use color; IO directions are separate; red denotes GPU idle only. ECHO prepare/finalize/hint are separate annotations.

The full-forward diagnostic includes host startup and commit, and assigns startup to L0. The README uses the selected main window and its corresponding layer intervals, including L0 compute and history H2D for dense extend. Gap is the complement of compute and actual IO; fused ECHO is entirely productive and stays in the denominator.

The graph is valid only for its restored fixed prefix, cache/storage/clock state, query shape and output mode. Outputs are borrowed. The result does not establish arbitrary growing-history replay, C10/NOSA full graphs, full 61-layer performance, or serving capacity limits.

GPU execution, exported native traces and interval accounting are covered by the independent audit files. Any postprocessing repair is recorded in summary.postprocess; a null value means none was recorded. GPU ownership observations and their limits are recorded separately in the audit evidence. Discrete observations do not prove machine or CPU exclusivity.

Run IDs and unrounded memory/latency data: [summary](summary.json). Raw sample rows: [timing](timing_samples.csv). Complete and layer gates: [gate table](extend_gate.csv). Cropped windows: [windows](windows.csv). Input hashes: [provenance](provenance.json). Independent evidence: `audit/`.
