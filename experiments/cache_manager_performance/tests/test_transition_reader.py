"""Repeated profile windows must bind their own counters and real MLA launch."""

import json

import pytest

from evaluation.validation import identity_digest
from experiments.cache_manager_performance.src import transition

PROCESS = 17 << 24
THREAD = PROCESS | 3


@pytest.fixture
def manager_capture(tmp_path, monkeypatch):
    identity = {"config": {"layers": 3}, "environment": {}, "sources": {}}
    result = {
        "schema": "deepseek-cache-manager-transition-v2",
        "run_id": "synthetic-transition",
        "mode": "profile",
        "passed": True,
        "identity_sha256": identity_digest(identity),
        "validation_receipt": "synthetic-receipt",
        "repeats": 2,
        "samples": [],
    }
    scopes, apis, activities = [], [], []

    def scope(scheme, phase, stage, start, end):
        row = {
            "id": len(scopes),
            "scheme": scheme,
            "phase": phase,
            "stage": stage,
            "label": f"manager/{scheme}/{phase}/{stage}",
            "thread": THREAD,
            "start": start,
            "end": end,
        }
        scopes.append(row)
        return row

    def activity(parent, name, start, end, *, launch=None, kind="kernel"):
        call = {
            "id": len(apis),
            "scope": parent,
            "thread": THREAD,
            "process": PROCESS,
            "start": parent["start"] + 1 if launch is None else launch,
            "name": "cudaLaunchKernel" if kind == "kernel" else "cudaMemcpyAsync",
            "source": "RUNTIME",
            "correlation": len(apis),
            "bookkeeping": False,
        }
        call["end"] = call["start"] + 2
        apis.append(call)
        activities.append(
            {
                "id": len(activities),
                "scope": parent,
                "api": call,
                "process": PROCESS,
                "device_id": 0,
                "stream_id": 1,
                "correlation": call["correlation"],
                "start": start,
                "end": end,
                "kind": kind,
                "name": name,
            }
        )

    for phase in ("extend_cold", "extend_warm"):
        for scheme in transition.SCHEMES:
            for sample_index in range(result["repeats"]):
                base = len(result["samples"]) * 10_000
                sample = {
                    "scheme": scheme,
                    "phase": phase,
                    "sample": sample_index,
                    "metrics": [],
                    "checks": [],
                }
                result["samples"].append(sample)
                scope(scheme, phase, "window", base, base + 4000)
                for layer in range(3):
                    offset = base + layer * 1000
                    metrics = {
                        "recalled_records": sample_index * 100 + layer,
                        "prefetched_records": 2 if scheme == "echo" else 0,
                        "record_bytes": 1152,
                    }
                    metrics["host_to_device_bytes"] = 1152 * (
                        metrics["recalled_records"] + metrics["prefetched_records"]
                    )
                    sample["metrics"].append(metrics)
                    sample["checks"].append(
                        {
                            "layer": layer,
                            "exact_records": True,
                            "attention_finite": True,
                            "attention_matches_resident": True,
                            "metrics": dict(metrics),
                        }
                    )
                    indexer = "indexer_prefetch" if scheme == "echo" else "indexer"
                    for stage, cpu_start, cpu_end, gpu_start, gpu_end, name in (
                        (
                            indexer,
                            10,
                            20,
                            30,
                            40,
                            "sm90_fp8_mqa_logits_fuse_prefetch"
                            if scheme == "echo"
                            else "sm90_fp8_mqa_logits",
                        ),
                        ("exact_topk", 50, 60, 70, 80, "FilteredTopKUnifiedKernel"),
                        ("prefetch_hint", 85, 100, 110, 120, "prefetch_hint_reduce"),
                        ("cache_write", 130, 150, 155, 190, "Device-to-Host"),
                        ("offload_exact_recall", 200, 230, 240, 280, "gather_records<uint32_t>"),
                        ("sparse_mla", 290, 340, 350, 365, "CatArrayBatchedCopy"),
                    ):
                        parent = scope(
                            scheme,
                            phase,
                            f"window/layer_{layer}/{stage}",
                            offset + cpu_start,
                            offset + cpu_end,
                        )
                        activity(
                            parent,
                            name,
                            offset + gpu_start,
                            offset + gpu_end,
                            kind="memcpy" if stage == "cache_write" else "kernel",
                        )
                        if stage == "sparse_mla":
                            activity(
                                parent,
                                "sparse_attn_fwd_kernel<sm90>",
                                offset + 380,
                                offset + 440,
                                launch=offset + 330,
                            )
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    (tmp_path / "trace.sqlite").write_bytes(b"synthetic capture")
    monkeypatch.setattr(
        transition, "capture", lambda path, pattern: (scopes, apis, activities, ["synthetic"])
    )
    monkeypatch.setattr(transition, "execution_sources", lambda *args: {})

    def read():
        (tmp_path / "result.json").write_text(json.dumps(result))
        return transition.manager_transitions(tmp_path)

    return result, scopes, apis, activities, read


def test_repeated_labels_bind_chronological_windows_and_matching_counters(manager_capture):
    result, scopes, _, _, read = manager_capture
    # SQLite row order need not be chronological; equal labels are not IDs.
    scopes.reverse()
    rows, provenance = read()
    assert len(rows) == 24
    assert provenance["repeats"] == 2
    for index, row in enumerate(rows):
        sample_index, layer = divmod(index, 3)
        sample = result["samples"][sample_index]
        assert (row["scheme"], row["phase"], row["sample"], row["layer"]) == (
            sample["scheme"],
            sample["phase"],
            sample["sample"],
            layer,
        )
        assert row["boundaries"]["endpoint_kind"] == "attention_kernel"
        post = row["gpu_topk_to_consumer"]
        base = sample_index * 10_000 + layer * 1000
        assert (post["start_ns"], post["end_ns"]) == (base + 80, base + 380)
        recall = next(
            activity for activity in post["activities"] if "gather_records" in activity["name"]
        )
        assert recall["diagnostic_io_records"] == sample["metrics"][layer]["recalled_records"]
        assert recall["diagnostic_io_evidence"].startswith(f"result.json:samples[{sample_index}]")
        assert "gpu_indexer_to_topk" not in row
        assert not any(key.startswith("indexer_") for key in row["boundaries"])
        assert {stage["stage"] for stage in row["stages"]} == {
            "prefetch_hint",
            "cache_write",
            "offload_exact_recall",
            "sparse_mla",
        }
        assert all(
            "sparse_attn_fwd_kernel" not in activity["name"] for activity in post["activities"]
        )


def test_legacy_no_op_schema_is_rejected(manager_capture):
    result, _, _, _, read = manager_capture
    result["schema"] = "deepseek-cache-manager-v1"
    with pytest.raises(ValueError, match="real-MLA v2 schema"):
        read()


@pytest.mark.parametrize("field", ["attention_finite", "attention_matches_resident"])
def test_real_consumer_requires_checked_attention_output(manager_capture, field):
    result, _, _, _, read = manager_capture
    result["samples"][0]["checks"][0][field] = False
    with pytest.raises(ValueError, match="checked real MLA outputs"):
        read()


def test_real_consumer_requires_actual_attention_kernel(manager_capture):
    _, _, _, activities, read = manager_capture
    activities[:] = [activity for activity in activities if activity["id"] != 6]
    with pytest.raises(ValueError, match="actual attention kernel"):
        read()


@pytest.mark.parametrize(
    "mutation", ["duplicate_index", "reordered_indices", "missing_sample", "extra_scheme"]
)
def test_incomplete_or_ambiguous_sample_identity_is_rejected(manager_capture, mutation):
    result, _, _, _, read = manager_capture
    if mutation == "duplicate_index":
        result["samples"][1]["sample"] = 0
    elif mutation == "reordered_indices":
        result["samples"][:2] = reversed(result["samples"][:2])
    elif mutation == "missing_sample":
        result["samples"].pop()
    else:
        result["samples"][0]["scheme"] = "hbm"
    with pytest.raises(ValueError, match="sample|indices"):
        read()


@pytest.mark.parametrize("mutation", ["missing_window", "overlap", "duplicate_id", "scope_escaped"])
def test_incomplete_or_ambiguous_capture_window_is_rejected(manager_capture, mutation):
    _, scopes, _, _, read = manager_capture
    if mutation == "missing_window":
        scopes.pop(0)
    elif mutation == "overlap":
        scopes[0]["end"] = 20_000
    elif mutation == "duplicate_id":
        scopes[1]["id"] = scopes[0]["id"]
    else:
        scopes[1]["end"] = 5000
    with pytest.raises(ValueError, match="window|unique"):
        read()


@pytest.mark.parametrize("mutation", ["later_window_scope", "unattributed", "entirely_escaped"])
def test_activities_cannot_borrow_equal_labels_from_another_repeat(manager_capture, mutation):
    _, scopes, _, activities, read = manager_capture
    if mutation == "later_window_scope":
        activities[0]["scope"] = next(
            scope
            for scope in scopes
            if scope["label"] == activities[0]["scope"]["label"] and scope["start"] > 10_000
        )
    elif mutation == "unattributed":
        activities[0]["scope"] = None
    else:
        activities[0].update(start=5000, end=5010)
    with pytest.raises(ValueError, match="unattributed or escaped"):
        read()
