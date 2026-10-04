"""Indexed attribution must retain inclusive containment and stable tie order."""

import random

from experiments.nosa_motivation.src.timeline import _ScopeIndex, analyze_trace


def test_indexed_owner_matches_linear_search_across_nested_tied_and_cross_thread_scopes():
    rng = random.Random(717)
    scopes = [
        {"name": "first", "tid": 1, "ts": 10, "dur": 20},
        {"name": "equal-duration-later", "tid": 1, "ts": 9, "dur": 20},
        {"name": "parent", "tid": 1, "ts": 0, "dur": 100},
        {"name": "zero", "tid": 2, "ts": 3, "dur": 0},
    ]
    scopes.extend(
        {
            "name": str(index),
            "tid": rng.randrange(3),
            "ts": rng.randrange(120),
            "dur": rng.randrange(40),
        }
        for index in range(100)
    )
    indexed = _ScopeIndex(scopes)
    queries = [
        {"tid": 1, "ts": 10, "dur": 19},
        {"tid": 1, "ts": 10, "dur": 20},
        {"tid": 2, "ts": 3, "dur": 0},
        {"tid": None, "ts": 3, "dur": 0},
    ]
    queries.extend(
        {"tid": rng.randrange(4), "ts": rng.randrange(160), "dur": rng.randrange(40)}
        for _ in range(300)
    )
    for cpu in queries:
        enclosing = [
            scope
            for scope in scopes
            if scope.get("tid") == cpu.get("tid")
            and scope["ts"] <= cpu["ts"]
            and cpu["ts"] + cpu["dur"] <= scope["ts"] + scope["dur"]
        ]
        enclosing.sort(key=lambda scope: scope["dur"])
        assert indexed.owner(cpu) is (enclosing[0] if enclosing else None)


def test_gpu_annotation_with_the_request_name_is_not_a_second_cpu_root():
    trace = {
        "traceEvents": [
            {"cat": "user_annotation", "ph": "X", "name": "request", "ts": 0, "dur": 100, "tid": 1},
            {
                "cat": "gpu_user_annotation",
                "ph": "X",
                "name": "request",
                "ts": 10,
                "dur": 60,
                "tid": 7,
            },
            {"cat": "kernel", "ph": "X", "name": "work", "ts": 20, "dur": 30, "tid": 7},
        ]
    }
    result = analyze_trace(trace, "request")
    assert result["instrumented_execute_wall_us"] == 100
    assert result["gpu_activity_union_us"] == 30
