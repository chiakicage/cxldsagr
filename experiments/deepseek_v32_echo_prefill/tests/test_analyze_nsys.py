"""Synthetic Nsight exports verify attribution and measurement definitions."""

import json
import sqlite3
import subprocess
import sys

import pytest

from experiments.deepseek_v32_echo_prefill.src.analyze_nsys import analyze_sqlite

PID = 12 << 24
TID = PID + 7


def make_capture(path, mode="resident"):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, bytes INTEGER);
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER, label TEXT);
        """
    )
    strings = {
        1: "cudaLaunchKernel",
        2: "cuLaunchKernel",
        3: "cudaMemsetAsync",
        4: "cudaMemcpyAsync",
        5: "cudaEventRecord",
        11: "_fp8_linear",
        12: "_sparse_mla_kernel",
        13: "gather_records",
        14: "unmatched_kernel",
        20: f"echo/{mode}/extend_annotated/layer_0/linear",
    }
    connection.executemany("INSERT INTO StringIds VALUES (?,?)", strings.items())
    prefix = f"echo/{mode}/extend_annotated/"
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
        [
            (0, 1_000_000, prefix + "layer_0", None, TID),
            (100_000, 400_000, prefix + "layer_0/attention_projection", None, TID),
            (200_000, 300_000, None, 20, TID),
            (450_000, 700_000, prefix + "layer_0/sparse_mla", None, TID),
            (800_000, 900_000, prefix + "shared/hidden_transfer", None, TID),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
        [
            (210_000, 260_000, TID, 1, 1),
            (230_000, 250_000, TID, 3, 3),
            (840_000, 850_000, TID, 4, 4),
            (180_000, 185_000, TID, 5, 5),
            (700_000, 710_000, TID, 6, 1),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (?,?,?,?,?)",
        [(220_000, 240_000, TID, 1, 2), (550_000, 560_000, TID, 2, 2)],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        [
            (300_000, 500_000, 0, 1, PID, 1, 11),
            (600_000, 800_000, 0, 1, PID, 2, 12),
            (810_000, 830_000, 0, 2, PID, 6, 13),
            (960_000, 980_000, 1, 2, PID, 999, 14),
        ],
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (450000,550000,0,2,?,?,4096)", (PID, 3)
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (850000,950000,1,2,?,?,1,64)", (PID, 4)
    )
    connection.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1,'HtoD')")
    connection.commit()
    connection.close()
    return path


def test_nested_runtime_driver_attribution_counts_kernel_once_and_excludes_events(tmp_path):
    path = make_capture(tmp_path / "resident.sqlite")
    before = path.read_bytes()
    result = analyze_sqlite(path)
    assert path.read_bytes() == before
    stages = {row["stage"]: row for row in result["stages"]}
    linear = stages["linear"]
    assert linear["kernel_count"] == 1
    assert linear["kernel_ms"] == pytest.approx(0.2)
    assert linear["memset_count"] == 1
    assert linear["api_count"] == 3
    assert linear["api_ms"] == pytest.approx(0.05)  # Runtime contains driver and memset APIs.
    assert linear["api_duration_sum_ms"] == pytest.approx(0.09)
    assert stages["attention_projection"]["kernel_count"] == 0
    assert stages["attention_projection"]["api_count"] == 0
    assert stages["sparse_mla"]["driver_api_count"] == 1
    assert stages["sparse_mla"]["kernel_count"] == 1
    assert stages["layer_unscoped"]["kernel_count"] == 1
    attribution = result["attribution"]
    assert attribution["gpu_activity_count"] == 6
    assert attribution["attributed_gpu_activity_count"] == 5
    assert attribution["unattributed_reasons"] == {"no_correlated_api": 1}
    assert attribution["correlated_api_sources"] == {"DRIVER": 2, "RUNTIME": 3}
    assert attribution["excluded_bookkeeping_api_names"] == {"cudaEventRecord": 1}


def test_multigpu_busy_union_and_gaps_include_unattributed_work(tmp_path):
    result = analyze_sqlite(make_capture(tmp_path / "resident.sqlite"))
    devices = {row["device_id"]: row for row in result["devices"]}
    assert devices[0]["busy_ms"] == pytest.approx(0.47)
    assert devices[0]["span_ms"] == pytest.approx(0.53)
    assert devices[0]["gap_ms"] == pytest.approx(0.06)
    assert devices[0]["activity_duration_sum_ms"] == pytest.approx(0.52)
    assert devices[0]["overlap_ms"] == pytest.approx(0.05)
    assert devices[1]["busy_ms"] == pytest.approx(0.12)
    assert devices[1]["span_ms"] == pytest.approx(0.13)
    assert devices[1]["gap_ms"] == pytest.approx(0.01)
    assert result["capture_gpu_envelope"]["span_ms"] == pytest.approx(0.68)
    assert devices[1]["idle_in_capture_gpu_envelope_ms"] == pytest.approx(0.56)


def test_mapped_host_transfer_bytes_require_full_result_not_memcpy_table(tmp_path):
    path = make_capture(tmp_path / "offload.sqlite", "offload")
    unknown = analyze_sqlite(path)["kv_transfers"]
    assert unknown["host_to_device_bytes"] is None
    assert unknown["observed_memcpy_bytes_by_kind"] == {"HtoD": 64}
    full_result = tmp_path / "result.json"
    full_result.write_text(
        json.dumps(
            {
                "run_id": "fixture",
                "measurements": {
                    "offload": {
                        "cache_per_layer": [
                            {"host_to_device_bytes": 1152, "device_to_host_bytes": 2304},
                            {"host_to_device_bytes": 3456, "device_to_host_bytes": 1152},
                        ]
                    }
                },
            }
        )
    )
    report = analyze_sqlite(path, result_path=full_result)
    assert report["kv_transfers"]["host_to_device_bytes"] == 4608
    assert report["kv_transfers"]["device_to_host_bytes"] == 3456
    assert report["kv_transfers"]["reported_layers"] == 2
    assert report["kv_transfers"]["mapped_host_reads_are_memcpy_records"] is False
    assert any(row["category"] == "mapped_host_kv_gather" for row in report["kernel_categories"])


def test_process_identity_prevents_false_correlation_and_driver_only_is_supported(tmp_path):
    path = make_capture(tmp_path / "resident.sqlite")
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (310000,320000,1,2,?,1,11)", (13 << 24,)
        )
    report = analyze_sqlite(path)
    assert report["attribution"]["attributed_gpu_activity_count"] == 2
    assert report["attribution"]["unattributed_reasons"]["correlation_process_mismatch"] == 1
    stages = {row["stage"]: row for row in report["stages"]}
    assert stages["linear"]["driver_api_count"] == 1
    assert stages["linear"]["runtime_api_count"] == 0


def test_final_synchronization_outside_nvtx_is_retained_as_unscoped_api_time(tmp_path):
    path = make_capture(tmp_path / "resident.sqlite")
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO StringIds VALUES (99,'cudaDeviceSynchronize')")
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1000000,1100000,?,99,99)", (TID,)
        )
    report = analyze_sqlite(path)
    assert report["api_summary"]["outside_echo_scope_thread_union_ms"] == pytest.approx(0.1)
    assert report["api_summary"]["thread_union_ms"] == pytest.approx(0.18)
    assert report["attribution"]["non_event_api_outside_echo_scopes"] == 1
    assert report["api_summary"]["calls_by_name"][0]["name"] == "cudaDeviceSynchronize"


def test_cli_writes_separate_json_and_rejects_overwriting_sqlite(tmp_path):
    path = make_capture(tmp_path / "resident.sqlite")
    output = tmp_path / "analysis" / "resident.json"
    command = [
        sys.executable,
        "-m",
        "experiments.deepseek_v32_echo_prefill.src.analyze_nsys",
        "--sqlite",
        str(path),
    ]
    subprocess.run([*command, "--output", str(output)], check=True, capture_output=True, text=True)
    assert json.loads(output.read_text())["mode"] == "resident"
    before = path.read_bytes()
    failed = subprocess.run(
        [*command, "--output", str(path)], capture_output=True, text=True, check=False
    )
    assert failed.returncode != 0 and "must not overwrite" in failed.stderr
    assert path.read_bytes() == before


def test_mixed_mode_or_missing_extend_scopes_fail_explicitly(tmp_path):
    path = make_capture(tmp_path / "mixed.sqlite")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (0,1000000,?,NULL,?)",
            ("echo/offload/extend_annotated/layer_0", TID),
        )
    with pytest.raises(ValueError, match="exactly one mode"):
        analyze_sqlite(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM NVTX_EVENTS")
    with pytest.raises(ValueError, match="exactly one mode"):
        analyze_sqlite(path)
