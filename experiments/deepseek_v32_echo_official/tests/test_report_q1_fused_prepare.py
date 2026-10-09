"""Publication checks reject changed samples instead of silently restating reports."""

import copy
import json
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_official.src import report_q1_fused_prepare as report


def component_fixture():
    rows, samples = [], []
    for layer in range(3):
        for policy in report.POLICIES:
            case = f"kernel_inputs_layer_{layer}"
            rows.append(
                {
                    "case": case,
                    "policy": policy,
                    "pairs": 100,
                    "warmups": 20,
                    "baseline_us": 10.0,
                    "candidate_us": 9.0,
                    "paired_delta_us": -1.0,
                    "candidate_wins": 100,
                }
            )
            for pair in range(100):
                order = report.ARMS if pair % 2 == 0 else report.ARMS[::-1]
                for arm in order:
                    samples.append(
                        {
                            "case": case,
                            "policy": policy,
                            "pair": pair,
                            "variant": arm,
                            "order": list(order),
                            "gpu_us": 10.0 if arm == "baseline" else 9.0,
                            "attempts": 70,
                            "staged": 64,
                            "h2d_bytes": 73728,
                        }
                    )
    return (
        {"mode": "bench", "rows": rows, "samples": samples},
        {"passed": True, "rows": copy.deepcopy(rows), "samples": 3000},
    )


def model_fixture():
    samples, pairs = [], []
    for pair in range(500):
        order = report.ARMS if pair % 2 == 0 else report.ARMS[::-1]
        order_name = "AB" if pair % 2 == 0 else "BA"
        for arm in order:
            samples.append(
                {
                    "pair": pair,
                    "order": order_name,
                    "arm": arm,
                    "wall_ms": 2.0 if arm == "baseline" else 1.5,
                    "layer_metrics": [
                        {
                            "record_bytes": 1152,
                            "prefetched_records": 64,
                            "recalled_records": 1983,
                            "resident_selection_records": 65,
                            "host_to_device_bytes": 2358144,
                            "device_to_host_bytes": 1152,
                        }
                        for _ in range(3)
                    ],
                }
            )
        pairs.append(
            {
                "pair": pair,
                "order": order_name,
                "block": pair // 100,
                "baseline_ms": 2.0,
                "candidate_ms": 1.5,
                "delta_ms": -0.5,
                "relative_delta_percent": -25.0,
            }
        )
    return (
        {"completed": True, "mode": "bench", "pairs": 500, "result": {"samples": samples}},
        {"passed": True, "clean_timing": {"paired_samples": pairs}},
        {"passed": True, "measurement_integrity_accepted": True},
    )


def test_component_includes_every_group_and_order():
    summary, review = component_fixture()
    rows = report.component_rows(summary, review)
    assert len(rows) == 15
    assert all(row["AB_delta_us"] == row["BA_delta_us"] == -1 for row in rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "traffic", "order", "review"])
def test_component_rejects_incomplete_or_altered_evidence(mutation):
    summary, review = component_fixture()
    if mutation == "missing":
        summary["samples"].pop()
    elif mutation == "duplicate":
        summary["samples"][1] = copy.deepcopy(summary["samples"][0])
    elif mutation == "nan":
        summary["samples"][0]["gpu_us"] = float("nan")
    elif mutation == "traffic":
        summary["samples"][0]["h2d_bytes"] += 1152
    elif mutation == "order":
        summary["samples"][0]["order"].reverse()
    else:
        review["rows"][0]["candidate_us"] = 8
    with pytest.raises(ValueError):
        report.component_rows(summary, review)


def test_model_retains_pair_order_blocks_and_actual_traffic():
    pairs = report.model_pairs(*model_fixture())
    assert len(pairs) == 500
    assert pairs[499]["block"] == 4
    assert pairs[499]["order"] == "BA"
    assert pairs[499]["baseline_h2d_bytes"] == 3 * 2358144


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "traffic", "tail", "rejected"])
def test_model_rejects_missing_pairs_or_changed_tail(mutation):
    bench, analysis, review = model_fixture()
    if mutation == "missing":
        bench["result"]["samples"].pop()
    elif mutation == "duplicate":
        bench["result"]["samples"][-1] = copy.deepcopy(bench["result"]["samples"][0])
    elif mutation == "nan":
        bench["result"]["samples"][-1]["wall_ms"] = float("nan")
    elif mutation == "traffic":
        bench["result"]["samples"][-1]["layer_metrics"][0]["host_to_device_bytes"] += 1152
    elif mutation == "tail":
        bench["result"]["samples"][-1]["wall_ms"] = 20
    else:
        review["measurement_integrity_accepted"] = False
    with pytest.raises(ValueError):
        report.model_pairs(bench, analysis, review)


def test_receipt_cannot_change_after_signature(tmp_path):
    payload = {"identity": {"shape": 1}, "checks": {"passed": True}}
    signature = report.identity_digest(payload)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({**payload, "receipt_sha256": signature}))
    assert report.receipt_payload(path, payload["identity"], signature)["checks"]["passed"]
    payload["checks"]["passed"] = False
    path.write_text(json.dumps({**payload, "receipt_sha256": signature}))
    with pytest.raises(ValueError, match="signature"):
        report.receipt_payload(path, payload["identity"], signature)


def test_no_output_outside_experiment(tmp_path):
    args = SimpleNamespace(
        component_dir=tmp_path,
        model_analysis_dir=tmp_path,
        model_check_review=tmp_path,
        output_dir=tmp_path / "publication",
        report_dir=None,
    )
    with pytest.raises(ValueError, match="output/data"):
        report.publish(args)
