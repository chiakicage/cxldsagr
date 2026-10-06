"""Report integrity checks reject incomplete or semantically different samples."""

import copy

import pytest

from experiments.cache_manager_performance.src.recall_report import STATES, plot, summarize


def _result():
    result = {"samples": []}
    for state in STATES:
        misses = 4 if state == "cold_sparse_miss" else 0
        for layer in range(3):
            for sample in range(31):
                result["samples"].append(
                    {
                        "state": state,
                        "layer": layer,
                        "variant": "baseline",
                        "sample": sample,
                        "wall_ms": sample + 1.0,
                        "enqueue_ms": (sample + 1.0) / 2,
                        "metrics": {
                            "selection_records": 10,
                            "resident_selection_records": 10 - misses,
                            "max_working_set": 10,
                            "recalled_records": misses,
                            "host_to_device_bytes": misses * 1152,
                            "device_to_host_bytes": 0,
                            "written_records": 0,
                            "host_written_records": 0,
                            "transient_written_records": 0,
                            "evicted_records": 0,
                            "prefetched_records": 0,
                            "prefetch_capacity_failures": 0,
                            "capacity_splits": 0,
                            "record_bytes": 1152,
                            "device_slots": 65664,
                            "host_token_capacity": 65664,
                            "session_host_tokens": 65664,
                            "candidate_slots": 0,
                            "padding_slots": 1,
                        },
                    }
                )
    return result


def test_report_retains_layers_and_inclusive_quartiles():
    rows = summarize(_result(), "Baseline", {layer: (10, 4) for layer in range(3)})
    assert len(rows) == 9
    assert {(row["state"], row["layer"]) for row in rows} == {
        (state, layer) for state in STATES for layer in range(3)
    }
    for row in rows:
        assert row["wall_median_ms"] == 16
        assert row["wall_q1_ms"] == 8.5
        assert row["wall_q3_ms"] == 23.5
        assert row["wall_iqr_ms"] == 15
        assert row["enqueue_iqr_ms"] == 7.5


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "nan", "negative", "enqueue", "variant", "bytes", "eviction"]
)
def test_report_rejects_bad_samples(fault):
    result = _result()
    first = result["samples"][0]
    if fault == "missing":
        result["samples"].pop()
    elif fault == "duplicate":
        result["samples"].append(copy.deepcopy(first))
    elif fault == "nan":
        first["wall_ms"] = float("nan")
    elif fault == "negative":
        first["enqueue_ms"] = -1
    elif fault == "enqueue":
        first["enqueue_ms"] = first["wall_ms"] + 1
    elif fault == "variant":
        first["variant"] = "candidate"
    elif fault == "bytes":
        first["metrics"]["host_to_device_bytes"] = 1152
    else:
        first["metrics"]["evicted_records"] = 1
    with pytest.raises(ValueError):
        summarize(result, "Baseline", {layer: (10, 4) for layer in range(3)})


def test_shared_axes_cover_larger_later_states(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    result = _result()
    for sample in result["samples"]:
        factor = 10 ** STATES.index(sample["state"])
        sample["wall_ms"] *= factor
        sample["enqueue_ms"] *= factor
    rows = summarize(result, "Baseline", {layer: (10, 4) for layer in range(3)})
    limits = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, *args, **kwargs: limits.append([axis.get_ylim() for axis in figure.axes]),
    )
    plot(rows, ["Baseline"], tmp_path)
    for axes in limits:
        for row_index, metric in enumerate(("wall", "enqueue")):
            upper = max(row[f"{metric}_q3_ms"] for row in rows)
            assert all(
                low == 0 and high > upper for low, high in axes[3 * row_index : 3 * row_index + 3]
            )


@pytest.mark.parametrize("warmup", [1, 2, 3, 11])
def test_report_accepts_matching_positive_warmup_metadata(warmup):
    from experiments.cache_manager_performance.src.recall_report import _matched_warmup

    assert _matched_warmup({"warmup": warmup}, {"warmup": warmup}) == warmup


@pytest.mark.parametrize("invalid", [None, True, False, 0, -1, 3.0, "3"])
@pytest.mark.parametrize("side", ["check", "bench"])
def test_report_rejects_invalid_warmup_metadata(invalid, side):
    from experiments.cache_manager_performance.src.recall_report import _matched_warmup

    bench, check = {"warmup": 3}, {"warmup": 3}
    (check if side == "check" else bench)["warmup"] = invalid
    with pytest.raises(ValueError, match="matching positive integer warmups"):
        _matched_warmup(bench, check)


@pytest.mark.parametrize("bench,check", [(2, 3), (3, 2)])
def test_report_rejects_mismatched_warmups(bench, check):
    from experiments.cache_manager_performance.src.recall_report import _matched_warmup

    with pytest.raises(ValueError, match="matching positive integer warmups"):
        _matched_warmup({"warmup": bench}, {"warmup": check})


def test_report_preserves_each_runs_warmup_in_prose_and_provenance(monkeypatch, tmp_path):
    import importlib
    import json

    module = importlib.import_module("experiments.cache_manager_performance.src.recall_report")
    runtime = {
        "gpu_name": "fixture GPU",
        "gpu_uuid": "fixture UUID",
        "cpu_affinity": [0, 1],
        "torch": "fixture",
        "cuda": "fixture",
        "tvm_ffi": "fixture",
    }
    identity = {"config": {}, "captures": [], "runtime": runtime}
    runs = [("Two", "bench2", "check2", "observer2"), ("Three", "bench3", "check3", "observer3")]
    bindings = {
        label: {
            "label": label,
            "run_id": bench,
            "check_run_id": check,
            "identity_sha256": "fixture",
            "identity": identity,
            "warmup": warmup,
        }
        for (label, bench, check, _), warmup in zip(runs, (2, 3), strict=True)
    }
    monkeypatch.setattr(
        module,
        "load_run",
        lambda label, *_: (
            bindings[label],
            summarize(_result(), label, {layer: (10, 4) for layer in range(3)}),
        ),
    )
    monkeypatch.setattr(module, "snapshot_report_helpers", lambda *_: {"files": {}})
    monkeypatch.setattr(module, "plot", lambda *_: "fixture")
    output = tmp_path / "report"
    module.generate(runs, output, tmp_path / "sources")
    text = (output / "results.md").read_text()
    assert "2 warmups per case" in text and "3 warmups per case" in text
    provenance = json.loads((output / "provenance.json").read_text())
    assert [(row["run_id"], row["warmup"]) for row in provenance["runs"]] == [
        ("bench2", 2),
        ("bench3", 3),
    ]
