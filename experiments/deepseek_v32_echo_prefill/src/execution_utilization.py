"""Precision-normalized useful matrix work over an uninstrumented execution.

This metric is separate from operator kernel utilization. The numerator is
the sum of per-precision ideal compute times for the implementation's useful
matrix work. The denominator is synchronized end-to-end wall time, including
nonmatrix work, transfers, CPU scheduling and gaps. It is not a single-peak
MFU or a measurement of Tensor Core activity.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def precision_normalized_utilization(calls, wall_samples_ms, *, peaks_tflops, scope):
    """Summarize one verified single-device workload, with no timing inference.

    The caller must verify that the call ledger covers the same execution as
    the uninstrumented samples. In particular, a three-layer ledger cannot be
    paired with a full-model latency. Unknown work is rejected; only explicitly
    annotated nonmatrix calls may omit FLOPs. Padding and redundant arithmetic
    are never substituted for useful work.
    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("An explicit measured workload scope is required")
    if not calls or len({(r["mode"], r["phase"]) for r in calls}) != 1:
        raise ValueError("A ledger must cover exactly one mode and phase")
    samples = list(wall_samples_ms)
    if not samples or any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
        for x in samples
    ):
        raise ValueError("Uninstrumented wall samples must be finite and positive")
    work = defaultdict(int)
    matrix_calls = nonmatrix_calls = 0
    for call in calls:
        flops, precision = call.get("useful_flops"), call.get("precision")
        if flops is None:
            if precision is not None or call.get("formula") != "N/A (no matrix multiply)":
                raise ValueError("Unknown work cannot be treated as nonmatrix work")
            nonmatrix_calls += 1
            continue
        if (
            isinstance(flops, bool)
            or not isinstance(flops, int)
            or flops < 0
            or not isinstance(precision, str)
        ):
            raise ValueError("Matrix work requires nonnegative integer FLOPs and precision")
        precision = precision.upper()
        peak = peaks_tflops.get(precision)
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (float, int))
            or not math.isfinite(peak)
            or peak <= 0
        ):
            raise ValueError(f"A positive finite dense peak is required for {precision}")
        work[precision] += flops
        matrix_calls += 1
    if not matrix_calls or not sum(work.values()):
        raise ValueError("The execution contains no useful matrix work")
    ideal = {precision: flops / peaks_tflops[precision] / 1e9 for precision, flops in work.items()}
    ideal_ms = sum(ideal.values())
    median_wall = statistics.median(samples)
    return {
        "metric": "end_to_end_precision_normalized_useful_compute_utilization",
        "scope": scope,
        "device_count": 1,
        "formula": "100 * sum_precision(useful_flops / (dense_peak_tflops * 1e9)) / wall_ms",
        "work_basis": "useful matrix work of the measured implementation, including absorbed MLA",
        "timing_basis": "uninstrumented synchronized end-to-end wall; one semantic ledger per sample",
        "matrix_call_count": matrix_calls,
        "explicit_nonmatrix_call_count": nonmatrix_calls,
        "useful_flops_by_precision": dict(sorted(work.items())),
        "dense_peaks_tflops": {precision: peaks_tflops[precision] for precision in sorted(work)},
        "ideal_compute_ms_by_precision": dict(sorted(ideal.items())),
        "ideal_compute_ms": ideal_ms,
        "wall_samples_ms": samples,
        "wall_median_ms": median_wall,
        "utilization_samples_percent": [100 * ideal_ms / sample for sample in samples],
        "utilization_at_median_wall_percent": 100 * ideal_ms / median_wall,
        "limitations": (
            "Single-device workload only; no extrapolation to other layers or devices. "
            "Reference peak normalization is not achieved-clock normalization or Tensor Core activity. "
            "No arithmetic mean of operator utilization is used. Nonmatrix work, padding and "
            "transfers add no useful matrix FLOPs; all remain in wall time."
        ),
    }
