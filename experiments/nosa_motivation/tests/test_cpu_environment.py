"""Contract checks for explicit CPU/NUMA environment comparison."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.nosa_motivation.src.cpu_environment import (
    cpu_environment,
    require_matching_cpu_environment,
)


def recorded():
    return {
        "cpu_environment": {
            "affinity": [0, 1],
            "torch_num_threads": 8,
            "numa_policy": {"available": True, "mode": "bind", "nodes": [0]},
        }
    }


def test_matching_and_legacy_are_distinct():
    assert (
        require_matching_cpu_environment(recorded(), recorded())["status"] == "recorded_and_equal"
    )
    assert require_matching_cpu_environment({}, {})["status"] == "not_recorded_in_legacy_pair"


@pytest.mark.parametrize("actual,reference", [(recorded(), {}), ({}, recorded())])
def test_new_and_unrecorded_cannot_match(actual, reference):
    with pytest.raises(ValueError, match="CPU affinity"):
        require_matching_cpu_environment(actual, reference)


@pytest.mark.parametrize(
    "field,value",
    [
        ("affinity", [2, 3]),
        ("torch_num_threads", 1),
        ("numa_policy", {"available": True, "mode": "default", "nodes": []}),
    ],
)
def test_each_binding_change_is_rejected(field, value):
    actual, reference = recorded(), deepcopy(recorded())
    actual["cpu_environment"][field] = value
    with pytest.raises(ValueError, match="CPU affinity"):
        require_matching_cpu_environment(actual, reference)


def test_current_snapshot_is_explicit_and_does_not_need_cuda():
    torch = SimpleNamespace(get_num_threads=lambda: 8, get_num_interop_threads=lambda: 4)
    actual = cpu_environment(torch)
    assert actual["affinity"] == sorted(set(actual["affinity"]))
    assert actual["torch_num_threads"] == 8
    assert actual["torch_num_interop_threads"] == 4
    assert isinstance(actual["numa_policy"]["available"], bool)


def test_completed_record_is_reopened(monkeypatch):
    from experiments.nosa_motivation.src import cpu_environment as module

    metadata = {"hardware": recorded()}
    monkeypatch.setattr(module, "cpu_environment", lambda torch: recorded()["cpu_environment"])
    module.finish_cpu_environment(metadata, None)
    assert module.validate_cpu_environment_record(metadata)["status"] == "recorded_and_equal"
    metadata["cpu_environment_final"]["affinity"] = [3]
    with pytest.raises(ValueError, match="CPU affinity"):
        module.validate_cpu_environment_record(metadata)


def test_recorded_start_requires_final_evidence():
    from experiments.nosa_motivation.src.cpu_environment import validate_cpu_environment_record

    with pytest.raises(ValueError, match="CPU affinity"):
        validate_cpu_environment_record({"hardware": recorded()})
    assert (
        validate_cpu_environment_record({"hardware": {}})["status"] == "not_recorded_in_legacy_pair"
    )


def test_completion_rejects_changed_policy(monkeypatch):
    from experiments.nosa_motivation.src import cpu_environment as module

    changed = recorded()["cpu_environment"]
    changed["numa_policy"] = {"available": True, "mode": "bind", "nodes": [1]}
    monkeypatch.setattr(module, "cpu_environment", lambda torch: changed)
    metadata = {"hardware": recorded()}
    with pytest.raises(ValueError, match="CPU affinity"):
        module.finish_cpu_environment(metadata, None)
    assert metadata["cpu_environment_final"] == changed
    assert metadata["cpu_environment_audit"]["status"] == "rejected"


def test_legacy_cannot_claim_recorded_audit():
    from experiments.nosa_motivation.src.cpu_environment import validate_cpu_environment_record

    with pytest.raises(ValueError, match="legacy"):
        validate_cpu_environment_record(
            {"hardware": {}, "cpu_environment_audit": {"status": "recorded_and_equal"}}
        )
