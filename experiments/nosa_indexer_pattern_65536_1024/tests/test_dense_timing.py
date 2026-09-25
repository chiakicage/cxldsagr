"""Reconstruct layer timing through launch correlations, independent of GPU queuing."""

import hashlib
import sqlite3

import pytest

from experiments.nosa_indexer_pattern_65536_1024.src.dense_timing import load_dense_layer_timing


def write_trace(path, *, optional=True):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, 'GR/detailed/extend/0');
            CREATE TABLE NVTX_EVENTS
                (start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
                (correlationId INTEGER, start INTEGER, end INTEGER, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                (start INTEGER, end INTEGER, correlationId INTEGER);
            """
        )
        scopes = [
            (0, 1000, None, 1, 7),
            (1, 999, "nosa::extend/model_misc/shared", None, 7),
            (10, 40, "nosa::extend/embedding/shared", None, 7),
            (100, 400, "nosa::extend/decoder/0", None, 7),
            (110, 140, "nosa::extend/qkv_proj/0", None, 7),
            (150, 170, "nosa::extend/rope_apply/shared", None, 7),
            (180, 210, "nosa::extend/attention_core/shared", None, 7),
            (220, 280, "nosa::extend/gate_up_proj/0", None, 7),
            (420, 800, "nosa::extend/decoder/1", None, 7),
            (430, 470, "nosa::extend/qkv_proj/1", None, 7),
            (480, 500, "nosa::extend/rope_apply/shared", None, 7),
            (510, 540, "nosa::extend/attention_core/shared", None, 7),
            (550, 650, "nosa::extend/gate_up_proj/1", None, 7),
            (830, 900, "nosa::extend/final_norm_add_residual/shared", None, 7),
        ]
        connection.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)", scopes)
        # All GPU starts exceed the entire root CPU range; API launches still locate layers.
        launches = [
            (1, 20, 10000, 40),
            (2, 120, 10050, 100),
            (3, 155, 10160, 20),
            (4, 190, 10200, 1000),
            (5, 230, 11210, 200),
            (7, 450, 11500, 120),
            (8, 490, 11640, 25),
            (9, 520, 11700, 1100),
            (10, 560, 12820, 210),
            (12, 850, 13100, 30),
            (999, 1200, 20000, 700),
        ]
        for correlation, cpu, gpu, duration in launches:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, 7)",
                (correlation, cpu, cpu + 2),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                (gpu, gpu + duration, correlation),
            )
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (998, 120, 125, 9)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (21000, 21800, 998)")
        if optional:
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET "
                "(start INTEGER, end INTEGER, correlationId INTEGER)"
            )
            for correlation, cpu, gpu, duration in [(6, 240, 11220, 10), (11, 570, 12830, 11)]:
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, 7)",
                    (correlation, cpu, cpu + 2),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?)",
                    (gpu, gpu + duration, correlation),
                )


def test_shared_scopes_use_decoder_ancestors_and_activity_sums_keep_memset_overlap(tmp_path):
    path = tmp_path / "dense.sqlite"
    write_trace(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_dense_layer_timing(path, 2)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert result["layers"][0]["modules"] == {
        "qkv_proj": 100 / 1e6,
        "rope_apply": 20 / 1e6,
        "attention_core": 1000 / 1e6,
        "gate_up_proj": 210 / 1e6,
    }
    assert result["layers"][1]["modules"] == {
        "qkv_proj": 120 / 1e6,
        "rope_apply": 25 / 1e6,
        "attention_core": 1100 / 1e6,
        "gate_up_proj": 221 / 1e6,
    }
    assert result["shared_modules"] == {
        "embedding": 40 / 1e6,
        "final_norm_add_residual": 30 / 1e6,
    }
    assert [row["gpu_ms"] for row in result["layers"]] == [1330 / 1e6, 1466 / 1e6]
    assert result["modules"]["attention_core"] == 2100 / 1e6
    assert result["total_gpu_ms"] == 2866 / 1e6
    assert result["gpu_busy_ms"] == 2845 / 1e6
    assert result["gpu_overlap_ms"] == 21 / 1e6
    assert sum(result["modules"].values()) == pytest.approx(result["total_gpu_ms"])
    assert sum(row["gpu_ms"] for row in result["layers"]) + sum(
        result["shared_modules"].values()
    ) == pytest.approx(result["total_gpu_ms"])


def test_missing_optional_tables_and_present_memcpy_are_supported(tmp_path):
    path = tmp_path / "dense.sqlite"
    write_trace(path, optional=False)
    result = load_dense_layer_timing(path, 2)
    assert result["total_gpu_ms"] == result["gpu_busy_ms"] == 2845 / 1e6
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
            "(start INTEGER, end INTEGER, correlationId INTEGER)"
        )
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (15000, 15010, 1)")
    result = load_dense_layer_timing(path, 2)
    assert result["shared_modules"]["embedding"] == 50 / 1e6
    assert result["total_gpu_ms"] == 2855 / 1e6


@pytest.mark.parametrize(
    "sql,error",
    [
        ("DELETE FROM NVTX_EVENTS WHERE textId = 1", "exactly one"),
        ("INSERT INTO NVTX_EVENTS VALUES (0,1000,NULL,1,7)", "exactly one"),
        ("DELETE FROM NVTX_EVENTS WHERE text = 'nosa::extend/decoder/1'", "decoder ranges"),
        (
            (
                "UPDATE NVTX_EVENTS SET text = 'nosa::extend/decoder/0' "
                "WHERE text = 'nosa::extend/decoder/1'"
            ),
            "every layer exactly once",
        ),
        ("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId = 9", "one attention"),
        ("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (30000,30100,9)", "one attention"),
        (
            (
                "DELETE FROM NVTX_EVENTS WHERE text IN "
                "('nosa::extend/model_misc/shared', 'nosa::extend/embedding/shared')"
            ),
            "No module NVTX",
        ),
        (
            (
                "UPDATE NVTX_EVENTS SET text = 'nosa::extend/qkv_proj/1' "
                "WHERE text = 'nosa::extend/qkv_proj/0'"
            ),
            "layer ID disagrees",
        ),
        (
            "UPDATE NVTX_EVENTS SET end = 500 WHERE text = 'nosa::extend/decoder/0'",
            "Overlapping decoder host ranges",
        ),
        (
            (
                "UPDATE NVTX_EVENTS SET end = 1001 "
                "WHERE text = 'nosa::extend/final_norm_add_residual/shared'"
            ),
            "within the detailed extend",
        ),
        (
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4,191,195,7)",
            "Duplicate CUDA correlation",
        ),
    ],
)
def test_ambiguous_or_incomplete_source_trace_fails(tmp_path, sql, error):
    path = tmp_path / "dense.sqlite"
    write_trace(path)
    with sqlite3.connect(path) as connection:
        connection.execute(sql)
    with pytest.raises(ValueError, match=error):
        load_dense_layer_timing(path, 2)


def test_missing_sqlite_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        load_dense_layer_timing(path, 2)
    assert not path.exists()


@pytest.mark.parametrize("num_layers", [0, -1, True, 2.5])
def test_invalid_layer_count_fails_before_opening_database(tmp_path, num_layers):
    with pytest.raises(ValueError, match="positive integer"):
        load_dense_layer_timing(tmp_path / "unused.sqlite", num_layers)
