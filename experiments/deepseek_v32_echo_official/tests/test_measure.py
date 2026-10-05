import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_echo_official.src.measure import (
    INDEXER_DISPATCH_POLICY,
    configuration,
    configure_precision,
    digest,
    normalize_compute_graphs,
    numerical_comparison,
    parser,
    provenance_files,
    reconcile_official,
    snapshot_official,
)
from experiments.deepseek_v32_motivation.src import measure as motivation


def test_default_config_matches_motivation_except_scheme_matrix():
    arguments = ["--run-id", "test"]
    expected = motivation.configuration(motivation.parser().parse_args(arguments))
    expected["schemes"] = ["hbm", "echo"]
    expected["indexer_dispatch_policy"] = INDEXER_DISPATCH_POLICY
    assert configuration(parser().parse_args(arguments)) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_saved_official_configuration_roundtrips_graph_mode(enabled):
    arguments = ["--run-id", "test"] + (["--compute-graphs"] if enabled else [])
    saved = configuration(parser().parse_args(arguments))
    assert saved["enable_compute_graphs"] is enabled
    assert configuration(SimpleNamespace(run_id="test", **saved)) == saved


def test_legacy_saved_eager_configuration_keeps_absent_graph_field():
    saved = configuration(parser().parse_args(["--run-id", "test"]))
    saved.pop("enable_compute_graphs")
    assert configuration(SimpleNamespace(run_id="test", **saved)) == saved
    assert normalize_compute_graphs(saved) is False


@pytest.mark.parametrize("enabled", [False, True])
def test_matching_compute_graph_aliases_are_accepted(enabled):
    args = parser().parse_args(["--run-id", "test"])
    args.compute_graphs = args.enable_compute_graphs = enabled
    assert configuration(args)["enable_compute_graphs"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_conflicting_compute_graph_aliases_are_rejected(enabled):
    args = parser().parse_args(["--run-id", "test"])
    args.compute_graphs, args.enable_compute_graphs = enabled, not enabled
    with pytest.raises(ValueError, match="conflicting"):
        configuration(args)


@pytest.mark.parametrize("name", ["compute_graphs", "enable_compute_graphs"])
@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_graph_configuration_rejects_non_boolean_values(name, value):
    args = parser().parse_args(["--run-id", "test"])
    setattr(args, name, value)
    with pytest.raises(ValueError, match="must be bool"):
        configuration(args)


def test_precision_is_explicit_and_records_other_effective_flags():
    matmul = SimpleNamespace(
        allow_tf32=True,
        allow_bf16_reduced_precision_reduction=False,
        allow_fp16_reduced_precision_reduction=True,
    )
    state = {"precision": "high"}
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=matmul), cudnn=SimpleNamespace(allow_tf32=True)
        ),
        set_float32_matmul_precision=lambda value: state.update(precision=value),
        get_float32_matmul_precision=lambda: state["precision"],
    )
    assert configure_precision(fake_torch) == {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "bf16_reduced_precision_reduction": False,
        "fp16_reduced_precision_reduction": True,
        "cudnn_allow_tf32": True,
    }


def test_official_artifacts_require_actual_source_and_native_bytes(tmp_path):
    source, native = tmp_path / "source.py", tmp_path / "native.so"
    source.write_bytes(b"source")
    native.write_bytes(b"native")
    provenance = {
        "source_files": {str(source): digest(source)},
        "native_files": {str(native): digest(native)},
    }
    target = tmp_path / "snapshot"
    target.mkdir()
    snapshot_official(provenance, target)
    native.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed before snapshot"):
        snapshot_official(provenance, target)
    with pytest.raises(ValueError, match="native_files"):
        provenance_files({**provenance, "native_files": {}})


def test_run_script_help_is_available_outside_repository():
    script = Path(__file__).resolve().parents[1] / "scripts/run.sh"
    result = subprocess.run(
        ["bash", str(script), "--help"], cwd="/tmp", capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--sparse-pool-tokens" in result.stdout and "--host-arena-tokens" in result.stdout
    assert "--compute-graphs" in result.stdout and "unsupported" not in result.stdout


def test_only_warmup_can_append_native_artifacts():
    previous = {"source_files": {"source.py": "a" * 64}, "native_files": {"module.so": "b" * 64}}
    current = {**previous, "native_files": {**previous["native_files"], "jit.cubin": "c" * 64}}
    reconcile_official(previous, current, allow_new_native=True)
    with pytest.raises(RuntimeError, match="inventory changed after warmup"):
        reconcile_official(previous, current, allow_new_native=False)
    current["native_files"]["module.so"] = "d" * 64
    with pytest.raises(RuntimeError, match="existing official native artifact changed"):
        reconcile_official(previous, current, allow_new_native=True)


def test_numerical_gate_allows_bounded_outliers_without_claiming_allclose():
    reference = torch.ones(10000, dtype=torch.float32)
    actual = reference.clone()
    actual[:9] += 0.1
    result = numerical_comparison(actual, reference, "hidden")
    assert result["numerical_pass"]
    assert not result["allclose"] and not result["bitwise_equal"]
    assert result["elementwise_outliers"] == 9
    assert result["elementwise_pass_fraction"] == 0.9991
    assert result["metrics_dtype"] == "torch.float64"
    actual[:11] = 1.1
    result = numerical_comparison(actual, reference, "hidden")
    assert result["relative_l2"] < 0.005
    assert result["elementwise_pass_fraction"] < 0.999
    assert not result["numerical_pass"]


def test_global_error_limit_is_independent_of_elementwise_fraction():
    reference = torch.ones(10000, dtype=torch.float32)
    actual = reference + 0.006
    hidden = numerical_comparison(actual, reference, "hidden")
    logits = numerical_comparison(actual, reference, "logits")
    assert hidden["allclose"] and hidden["elementwise_pass_fraction"] == 1
    assert not hidden["numerical_pass"] and logits["numerical_pass"]
    actual[0] = 2
    assert not numerical_comparison(actual, reference, "logits")["numerical_pass"]


@pytest.mark.parametrize("defect", ["shape", "dtype", "nonfinite"])
def test_numerical_gate_requires_matching_finite_tensors(defect):
    reference = torch.ones((2, 3), dtype=torch.float32)
    actual = reference.clone()
    if defect == "shape":
        actual = actual.flatten()
    elif defect == "dtype":
        actual = actual.double()
    else:
        actual[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        numerical_comparison(actual, reference, "hidden")


def test_modes_reject_missing_or_unexpected_receipts_before_configuration(monkeypatch):
    from experiments.deepseek_v32_echo_official.src import measure

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mode reached configuration or CUDA")

    monkeypatch.setattr(measure, "configuration", forbidden)
    with pytest.raises(ValueError, match="bench requires"):
        measure.main(["--run-id", "missing_receipt"])
    with pytest.raises(ValueError, match="check creates"):
        measure.main(
            ["--mode", "check", "--run-id", "extra_receipt", "--validation-receipt", "/none"]
        )
