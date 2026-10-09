"""Mutate raw NSYS ownership across processes, devices and forward boundaries."""

import hashlib
import sqlite3

import pytest

from experiments.deepseek_v32_mfu.src.method_process_audit import (
    CaptureProcessError,
    audit_capture_process,
)

PID = 314159
GLOBAL_PID = (1 << 48) | (PID << 24)
THREAD = GLOBAL_PID | PID
FOREIGN_PID = PID + 1
FOREIGN_GLOBAL = (1 << 48) | (FOREIGN_PID << 24)
FOREIGN_THREAD = FOREIGN_GLOBAL | FOREIGN_PID


@pytest.fixture
def capture(tmp_path):
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER,
                deviceId INTEGER, correlationId INTEGER, graphId INTEGER, graphNodeId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, globalPid INTEGER,
                deviceId INTEGER, correlationId INTEGER, graphId INTEGER, graphNodeId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER, globalPid INTEGER);
        """)
        connection.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [
                (1, "cudaGraphLaunch_v10000"),
                (2, "echo/hbm/extend_annotated/shared/forward_misc/call_0"),
            ],
        )
        connection.executemany(
            "INSERT INTO PROCESSES VALUES (?,?,?)",
            [
                (GLOBAL_PID, PID, "python"),
                (FOREIGN_GLOBAL, FOREIGN_PID, "also-python"),
            ],
        )
        connection.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0,0,?)", (PID,))
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (100,500,?,NULL,2)", (THREAD,))
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
            [
                (200, 280, GLOBAL_PID, 0, 7, 40, 101),
                (280, 300, GLOBAL_PID, 0, 7, 40, 102),
                (20, 30, FOREIGN_GLOBAL, 1, 19, 9, 103),
            ],
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (110,120,?,0,3,0,0)", (GLOBAL_PID,)
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,NULL)",
            [
                (150, 160, THREAD, 7, 1),
                (10, 11, FOREIGN_THREAD, 19, 1),
            ],
        )
    return path


def audit(path, **overrides):
    return audit_capture_process(
        path,
        **{
            "target_pid": PID,
            "method": "hbm",
            "phase": "extend",
            "expected_device": 0,
            "require_single_graph": True,
            **overrides,
        },
    )


def mutate(path, sql, parameters=()):
    with sqlite3.connect(path) as connection:
        connection.execute(sql, parameters)


def test_ownership_binds_real_process_table_and_preserves_outside_observations(capture):
    before = capture.read_bytes()
    result = audit(capture)
    assert result["passed"] and result["sqlite_sha256"] == hashlib.sha256(before).hexdigest()
    assert capture.read_bytes() == before
    assert result["all_capture_gpu"]["os_pids"] == [PID, FOREIGN_PID]
    assert result["all_capture_gpu"]["device_ids"] == [0, 1]
    assert result["measured_forward_gpu"]["os_pids"] == [PID]
    assert result["measured_forward_gpu"]["serialized_process_ids"] == [GLOBAL_PID]
    assert result["measured_forward_gpu"]["device_ids"] == [0]
    assert result["measured_forward_gpu"]["interval_union_ns"] == 110
    assert result["outside_forward_gpu"]["activity_count"] == 1
    assert len(result["foreign_gpu_outside_forward"]) == 1
    assert len(result["foreign_graph_launches_outside_forward"]) == 1
    assert result["foreign_gpu_in_forward"] == []
    assert result["graph_launch_bindings"][0]["gpu_activity_count"] == 2
    assert result["graph_launch_bindings"][0]["os_pid"] == PID
    assert "No inference" in result["boundary"]


@pytest.mark.parametrize("start,end", [(90, 110), (490, 510), (200, 220)])
def test_foreign_gpu_overlap_is_rejected_including_boundary_crossings(capture, start, end):
    mutate(
        capture,
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=?,end=? WHERE globalPid=?",
        (start, end, FOREIGN_GLOBAL),
    )
    with pytest.raises(CaptureProcessError, match="Foreign PID/device") as error:
        audit(capture)
    assert len(error.value.observations["foreign_gpu_in_forward"]) == 1


@pytest.mark.parametrize("start,end", [(90, 100), (500, 510)])
def test_adjacent_foreign_activity_does_not_overlap(capture, start, end):
    mutate(
        capture,
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=?,end=? WHERE globalPid=?",
        (start, end, FOREIGN_GLOBAL),
    )
    assert audit(capture)["foreign_gpu_in_forward"] == []


@pytest.mark.parametrize(
    "table",
    [
        "CUPTI_ACTIVITY_KIND_MEMCPY2",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    ],
)
def test_alternative_and_additional_gpu_tables_are_all_audited(capture, table):
    with sqlite3.connect(capture) as connection:
        connection.execute(
            f"CREATE TABLE {table} AS SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE 0"
        )
        connection.execute(f"INSERT INTO {table} VALUES (180,190,?,0,7,40,104)", (FOREIGN_GLOBAL,))
    with pytest.raises(CaptureProcessError, match="Foreign PID/device") as error:
        audit(capture)
    assert error.value.observations["foreign_gpu_in_forward"][0]["table"] == table


def test_gpu_device_mismatch_is_rejected_even_for_target_pid(capture):
    mutate(capture, "UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET deviceId=1")
    with pytest.raises(CaptureProcessError, match="Foreign PID/device"):
        audit(capture)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("DELETE FROM PROCESSES WHERE pid=314159", "Missing or ambiguous target PID"),
        ("UPDATE TARGET_INFO_CUDA_DEVICE SET cudaId=1", "CUDA-local device 0"),
        ("DELETE FROM NVTX_EVENTS", "exactly one measured forward"),
        ("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS", "exactly one measured forward"),
        ("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid<>0", "no process-bound graph"),
        (
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=7",
            "no target GraphLaunch correlation",
        ),
        (
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=99 WHERE correlationId=7",
            "no target GraphLaunch correlation",
        ),
    ],
)
def test_missing_or_ambiguous_process_scope_and_graph_bindings_fail(capture, mutation, match):
    mutate(capture, mutation)
    with pytest.raises(ValueError, match=match):
        audit(capture)


def test_scope_process_owner_cannot_be_substituted(capture):
    mutate(capture, "UPDATE NVTX_EVENTS SET globalTid=?", (FOREIGN_THREAD,))
    with pytest.raises(ValueError, match="NVTX owner differs"):
        audit(capture)


def test_graph_launch_must_belong_to_forward_thread(capture):
    mutate(
        capture,
        "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=? WHERE correlationId=7",
        (THREAD + 1,),
    )
    with pytest.raises(CaptureProcessError, match="another PID/thread"):
        audit(capture)


def test_foreign_graph_api_crossing_forward_start_remains_visible(capture):
    mutate(
        capture, "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=90,end=110 WHERE correlationId=19"
    )
    with pytest.raises(CaptureProcessError, match="Foreign graph launch API") as error:
        audit(capture)
    assert len(error.value.observations["foreign_graph_launches_in_forward"]) == 1


def test_api_explicit_process_must_agree_with_thread(capture):
    mutate(
        capture,
        "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalPid=? WHERE correlationId=7",
        (FOREIGN_GLOBAL,),
    )
    with pytest.raises(ValueError, match="process and owning thread disagree"):
        audit(capture)


def test_multiple_compute_island_graphs_allowed_but_not_single_extend(capture):
    mutate(
        capture,
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (320,360,?,0,8,41,105)",
        (GLOBAL_PID,),
    )
    mutate(
        capture, "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (310,315,?,8,1,NULL)", (THREAD,)
    )
    with pytest.raises(CaptureProcessError, match="exactly one process-bound graph"):
        audit(capture)
    mutate(
        capture,
        "UPDATE StringIds SET value='echo/hbm/prefill_annotated/shared/forward_misc/call_0' WHERE id=2",
    )
    result = audit(capture, phase="prefill", require_single_graph=False)
    assert len(result["graph_launch_bindings"]) == 2


def test_runtime_and_driver_duplicate_api_observations_do_not_duplicate_gpu_work(capture):
    mutate(
        capture,
        "CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER AS SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=7",
    )
    result = audit(capture)
    assert len(result["measured_forward_graph_launches"]) == 2
    assert len(result["graph_launch_bindings"]) == 1
    assert result["graph_launch_bindings"][0]["gpu_activity_count"] == 2


def test_gpu_before_scope_with_same_graph_launch_id_cannot_be_reassigned(capture):
    mutate(
        capture,
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=90,end=110,correlationId=19 WHERE graphNodeId=101",
    )
    with pytest.raises(CaptureProcessError, match="no target GraphLaunch correlation"):
        audit(capture)


def test_no_target_gpu_activity_is_not_accepted_as_isolation(capture):
    mutate(capture, "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    mutate(capture, "DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    with pytest.raises(CaptureProcessError, match="no target GPU activity"):
        audit(capture)
