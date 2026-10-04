"""Nsight GPU interval and host submission accounting."""

import sqlite3

import pytest

from experiments.nosa_baseline_performance.src.dense.analyze import activity_summary, analyze


def test_nsys_busy_union_excludes_overlap_and_counts_host_late_gap():
    activities = [
        {"start": 0, "end": 10000, "correlationId": 1},
        {"start": 5000, "end": 12000, "correlationId": 2},
        {"start": 20000, "end": 30000, "correlationId": 3},
    ]
    result = activity_summary(activities, {3: {"start": 16000}})
    assert result["gpu_busy_ms"] == 0.022
    assert result["gpu_gap_ms"] == 0.008
    assert result["gap_before_next_cuda_api_ms"] == 0.004
    assert result["gpu_busy_pct"] == pytest.approx(100 * 22 / 30)


def _write_nsys_fixture(path, optional_categories):
    ranges = (
        "GR/light/full_prefill/0",
        "GR/light/extend/0",
        "GR/light/extend/1",
        "GR/light/extend/2",
        "GR/detailed/full_prefill/0",
        "GR/detailed/extend/0",
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES
                (1, 'qkv_gemm'), (2, 'cudaLaunchKernel'),
                (3, 'cudaMemcpyAsync'), (4, 'cudaMemsetAsync');
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, correlationId INTEGER,
                nameId INTEGER, globalTid INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, correlationId INTEGER,
                demangledName INTEGER, streamId INTEGER
            );
            """
        )
        for category in optional_categories:
            connection.execute(
                f"CREATE TABLE CUPTI_ACTIVITY_KIND_{category} "
                "(start INTEGER, end INTEGER, correlationId INTEGER, streamId INTEGER)"
            )
        for index, name in enumerate(ranges):
            start = index * 10000
            correlation = index * 10
            phase = name.split("/")[2]
            connection.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL, 1)",
                (
                    (start, start + 9000, name),
                    (start + 50, start + 8000, f"nosa::{phase}/qkv_proj/0"),
                ),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, 2, 1)",
                (start + 100, start + 200, correlation),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 1, 7)",
                (start + 1000, start + 2000, correlation),
            )
            for offset, category in enumerate(optional_categories, start=1):
                api_name_id = 3 if category == "MEMCPY" else 4
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, 1)",
                    (
                        start + 200 + offset * 100,
                        start + 250 + offset * 100,
                        correlation + offset,
                        api_name_id,
                    ),
                )
                connection.execute(
                    f"INSERT INTO CUPTI_ACTIVITY_KIND_{category} VALUES (?, ?, ?, 7)",
                    (
                        start + 2000 + offset * 1000,
                        start + 2500 + offset * 1000,
                        correlation + offset,
                    ),
                )


@pytest.mark.parametrize("optional_categories", [(), ("MEMCPY",), ("MEMSET",)])
def test_analyze_accepts_missing_optional_activity_tables(tmp_path, optional_categories):
    path = tmp_path / "trace.sqlite"
    _write_nsys_fixture(path, optional_categories)

    results = analyze(path)

    assert len(results) == 6
    for row in results:
        assert row["kernel_count"] == 1
        assert row["kernel_ms"] == 0.001
        assert row["gpu_busy_ms"] == pytest.approx(0.001 + len(optional_categories) * 0.0005)
        assert row["kernels"] == {"qkv_gemm": {"count": 1, "gpu_ms": 0.001}}
        if row["range"].startswith("GR/detailed/"):
            module = row["modules"]["qkv_proj"]
            assert module["kernel_count"] == 1
            assert module["activity_count"] == 1 + len(optional_categories)
            assert module["kernel_names"] == {
                "qkv_gemm": 1,
                **dict.fromkeys(optional_categories, 1),
            }


def test_analyze_requires_kernel_activity_table(tmp_path):
    path = tmp_path / "trace.sqlite"
    _write_nsys_fixture(path, ())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")

    with pytest.raises(sqlite3.OperationalError, match="CUPTI_ACTIVITY_KIND_KERNEL"):
        analyze(path)
