"""Independent recall allocation rules and timed completion boundary."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import MISSING
from experiments.cache_manager_performance.src.recall import execution_sources, timed_call
from experiments.cache_manager_performance.src.recall_workload import (
    compare_states,
    validate_allocation,
    validate_selection,
)


def state(priorities, occupants):
    d2h = torch.tensor([MISSING, *occupants])
    h2d = torch.full((12,), MISSING, dtype=torch.int32)
    for slot, occupant in enumerate(d2h):
        if occupant != MISSING:
            h2d[occupant] = slot
    return {
        "device_to_host": d2h,
        "host_to_device": h2d,
        "priority": torch.tensor([0, *priorities]),
    }


def test_equal_priority_slots_may_differ_but_selected_residents_stay_protected():
    before = state([9, 0, 0, 0], [0, MISSING, MISSING, MISSING])
    after = state([10, 0, 10, 0], [0, MISSING, 1, MISSING])
    result = validate_allocation(before, after, torch.tensor([0, 1]))
    assert result == {"misses": 1, "allocation_ties": True, "priority_threshold": -1}
    after["host_to_device"][0] = 4
    with pytest.raises(AssertionError, match="already resident"):
        validate_allocation(before, after, torch.tensor([0, 1]))


@pytest.mark.parametrize("slot", [2, 3])
def test_free_slots_are_selected_before_any_live_priority(slot):
    before = state([9, 0, 1], [0, MISSING, 2])
    after = {key: value.clone() for key, value in before.items()}
    after["host_to_device"][1] = slot
    if slot == 2:
        assert validate_allocation(before, after, torch.tensor([0, 1]))["priority_threshold"] == -1
    else:
        with pytest.raises(AssertionError):
            validate_allocation(before, after, torch.tensor([0, 1]))


@pytest.mark.parametrize("slot", [2, 3, 4])
def test_lowest_unprotected_fifo_priority_is_required(slot):
    before = state([1, 3, 5, 7], [0, 2, 3, 4])
    after = {key: value.clone() for key, value in before.items()}
    after["host_to_device"][1] = slot
    if slot == 2:
        assert validate_allocation(before, after, torch.tensor([0, 1])) == {
            "misses": 1,
            "allocation_ties": False,
            "priority_threshold": 3,
        }
    else:
        with pytest.raises(AssertionError):
            validate_allocation(before, after, torch.tensor([0, 1]))


def test_semantic_comparison_only_relaxes_tied_physical_allocation():
    reference = {
        "allocation": {"allocation_ties": True},
        "physical": torch.tensor([1]),
        "metrics": {"recalled": 1},
        "resident_logical": torch.tensor([True]),
    }
    actual = {**reference, "physical": torch.tensor([3])}
    compare_states(actual, reference)
    with pytest.raises(AssertionError, match="metrics"):
        compare_states({**actual, "metrics": {"recalled": 0}}, reference)
    actual["allocation"] = reference["allocation"] = {"allocation_ties": False}
    with pytest.raises(AssertionError):
        compare_states(actual, reference)


def test_selection_validation_rejects_changed_exact_ids():
    config = SimpleNamespace(append=1, history=2048, capacity=2049)
    ids = torch.arange(2048, dtype=torch.int32)[None]
    validate_selection({"indices": ids}, config)
    duplicate = ids.clone()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection({"indices": duplicate}, config)
    future = ids.clone()
    future[0, 0] = 2049
    with pytest.raises(ValueError, match="noncausal"):
        validate_selection({"indices": future}, config)


def test_timer_reports_submission_and_waits_for_actual_completion(monkeypatch):
    from experiments.cache_manager_performance.src import recall

    events = []
    ticks = iter((1_000_000, 1_100_000, 1_900_000))
    monkeypatch.setattr(recall.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("fence", device)))

    def invoke():
        events.append(("invoke", "cuda:0"))
        return "physical"

    physical, timing = timed_call(SimpleNamespace(invoke=invoke, device="cuda:0"))
    assert physical == "physical"
    assert events == [("invoke", "cuda:0"), ("fence", "cuda:0")]
    assert timing == {"enqueue_ms": 0.1, "wall_ms": 0.9}


def test_execution_manifest_excludes_unexecuted_analysis_helpers():
    paths = execution_sources()
    assert any(name.endswith("recall_workload.py") for name in paths)
    assert not any(name.endswith(("analyze.py", "report.py", "launch_gap.py")) for name in paths)
    assert all(path.is_file() for path in paths.values())
