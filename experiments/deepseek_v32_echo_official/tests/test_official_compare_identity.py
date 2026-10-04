from copy import deepcopy

import pytest

from experiments.deepseek_v32_echo_official.src.compare_existing import (
    HARDWARE_FIELDS,
    MODEL_FIELDS,
    compare_identity,
)
from experiments.deepseek_v32_echo_official.src.measure import configuration, parser


@pytest.fixture
def comparable_reports():
    config = configuration(parser().parse_args(["--run-id", "test"]))
    backend = {name: f"same_{name}" for name in MODEL_FIELDS}
    metadata = {
        "config": config,
        "cases": [
            {
                "scheme": scheme,
                "requests": config["requests_per_scheme"],
                "started_empty": True,
                "warmup_request_ids": config["warmup_request_indices"],
                "warmup_requests": config["warmup_requests_per_scheme"],
                "backend": backend,
            }
            for scheme in ("hbm", "echo")
        ],
        "workload_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "checkpoint": {"path": "/test/checkpoint", "identity_boundary": "synthetic fixture"},
        "model_dimensions": {"hidden": 7168, "vocabulary": 129280},
        "hardware": {**{name: f"same_{name}" for name in HARDWARE_FIELDS}, "uuid": "same_gpu"},
        "backend_provenance": {
            "installed": {
                name: {"distribution_version": "same_version"}
                for name in ("deep_gemm", "flash_mla", "flashinfer")
            }
        },
        "precision_policy": "test_policy",
        "precision_settings": {
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "bf16_reduced_precision_reduction": True,
            "fp16_reduced_precision_reduction": True,
            "cudnn_allow_tf32": True,
        },
    }
    return {"metadata": metadata}, {"metadata": deepcopy(metadata)}


@pytest.mark.parametrize("enabled", [False, True])
def test_comparison_records_matching_graph_mode(comparable_reports, enabled):
    for report in comparable_reports:
        report["metadata"]["config"]["enable_compute_graphs"] = enabled
    identity = compare_identity(*comparable_reports)
    assert identity["config"]["enable_compute_graphs"] is enabled
    assert identity["full_precision_policy_equality_verified"]


def test_comparison_treats_legacy_absent_mode_as_eager(comparable_reports):
    left, right = comparable_reports
    left["metadata"]["config"].pop("enable_compute_graphs")
    left["metadata"].pop("precision_settings")
    identity = compare_identity(left, right)
    assert identity["config"]["enable_compute_graphs"] is False
    assert not identity["full_precision_policy_equality_verified"]
    assert not identity["recorded_common_precision_settings_equality_verified"]


@pytest.mark.parametrize("which", [0, 1])
def test_comparison_rejects_graph_eager_mismatch(comparable_reports, which):
    comparable_reports[which]["metadata"]["config"]["enable_compute_graphs"] = True
    with pytest.raises(ValueError, match="enable_compute_graphs"):
        compare_identity(*comparable_reports)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_comparison_rejects_non_boolean_graph_mode(comparable_reports, value):
    comparable_reports[0]["metadata"]["config"]["enable_compute_graphs"] = value
    with pytest.raises(ValueError, match="must be bool"):
        compare_identity(*comparable_reports)


def test_comparison_rejects_conflicting_graph_aliases(comparable_reports):
    comparable_reports[0]["metadata"]["config"]["compute_graphs"] = True
    with pytest.raises(ValueError, match="conflicting"):
        compare_identity(*comparable_reports)


@pytest.mark.parametrize("defect", ["missing", "incomplete", "mismatch", "invalid_type"])
def test_graph_comparison_requires_matching_precision_evidence(comparable_reports, defect):
    for report in comparable_reports:
        report["metadata"]["config"]["enable_compute_graphs"] = True
    metadata = comparable_reports[0]["metadata"]
    if defect == "missing":
        metadata.pop("precision_settings")
    elif defect == "incomplete":
        metadata["precision_settings"].pop("cuda_matmul_allow_tf32")
    elif defect == "invalid_type":
        metadata["precision_settings"]["cuda_matmul_allow_tf32"] = 0
    else:
        metadata["precision_settings"]["cuda_matmul_allow_tf32"] = True
    with pytest.raises(ValueError, match="precision"):
        compare_identity(*comparable_reports)


@pytest.mark.parametrize("value", ["invalid", {}])
def test_comparison_rejects_malformed_present_precision_settings(comparable_reports, value):
    comparable_reports[0]["metadata"]["precision_settings"] = value
    with pytest.raises(ValueError, match="precision"):
        compare_identity(*comparable_reports)


def test_comparison_reports_only_common_precision_equality(comparable_reports):
    left, right = comparable_reports
    left["metadata"]["precision_settings"]["cuda_matmul_fp32_precision"] = "ieee"
    right["metadata"]["precision_policy"] = "equivalent_but_distinct_experiment_label"
    identity = compare_identity(left, right)
    assert identity["recorded_common_precision_settings_equality_verified"]
    assert "cuda_matmul_fp32_precision" not in identity["recorded_common_precision_settings"]
    assert not identity["full_precision_policy_equality_verified"]
