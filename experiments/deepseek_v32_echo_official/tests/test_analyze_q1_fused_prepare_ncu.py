from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_ncu as audit


class Metric:
    def __init__(self, value, valid, instances=(), correlations=None):
        self.scalar = value
        self.valid = valid
        self.values = instances
        self.correlations = correlations

    def value(self, index=None):
        return self.scalar if index is None else self.values[index]

    def has_value(self, index=None):
        return self.valid

    def unit(self):
        return "sample"

    def description(self):
        return "synthetic validity fixture"

    def num_instances(self):
        return len(self.values)

    def has_correlation_ids(self):
        return self.correlations is not None

    def correlation_ids(self):
        return self.correlations


def test_nonzero_invalid_pc_values_are_retained_with_invalid_flag():
    metric = Metric(9, False, (9,), Metric(0, True, (100,)))
    name = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
    action = SimpleNamespace(
        metric_names=lambda: [name],
        source_info=lambda pc: SimpleNamespace(file_name=lambda: "/source.cu", line=lambda: 7),
        sass_by_pc=lambda pc: "LDG.E R1, [R2]",
    )

    class Action:
        def __getitem__(self, key):
            assert key == name
            return metric

        def __getattr__(self, key):
            return getattr(action, key)

    scalars, instances, pcs = audit.extract(Action())
    assert scalars[name]["has_value"] is False
    assert scalars[name]["value"] == 9
    assert instances[name] == [{"has_value": False, "value": 9, "correlation": 100}]
    assert pcs[0]["counters"][name]["has_value"] is False
    assert pcs[0]["source"] == {"file": "/source.cu", "line": 7}


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_values_keep_their_representation(value):
    result = audit.metric_value(Metric(value, False))
    assert result == {"has_value": False, "value": None, "nonfinite_value": repr(value)}


@pytest.fixture
def packing_records():
    environment = {
        "torch": "torch-version",
        "cuda": "cuda-version",
        "triton": "triton-version",
        "device": "H200",
        "capability": [9, 0],
    }
    kernels = [
        {
            "hash": "specialization",
            "asm_sha256": {"cubin": "binary", "ptx": "ptx"},
            "metadata": {"num_warps": 4},
        }
    ]
    identity = {
        **environment,
        "inputs": {"layer_0.pt": "input", "layer_1.pt": "other"},
        "packing_kernels": kernels,
        "precision": {"tf32": False},
    }
    completed = {
        "mode": "profile",
        "profile": {"method": "fused", "output_shape": [1025, 8448]},
        "compiled_kernels": deepcopy(kernels),
        "identity": {
            **environment,
            "inputs_sha256": {"layer_0.pt": "input"},
            "precision": {"matmul_tf32": False},
        },
    }
    return completed, identity


def test_historical_bridge_accepts_identical_input_and_specialization(packing_records):
    audit.packing_bridge(*packing_records)


@pytest.mark.parametrize(
    "mismatch", ("cubin", "ptx", "metadata", "input", "layer", "precision", "shape")
)
def test_historical_bridge_rejects_changed_identity(packing_records, mismatch):
    completed, identity = packing_records
    if mismatch in ("cubin", "ptx"):
        completed["compiled_kernels"][0]["asm_sha256"][mismatch] = "changed"
    elif mismatch == "metadata":
        completed["compiled_kernels"][0]["metadata"]["num_warps"] = 8
    elif mismatch == "input":
        completed["identity"]["inputs_sha256"]["layer_0.pt"] = "changed"
    elif mismatch == "layer":
        completed["identity"]["inputs_sha256"] = {"layer_1.pt": "other"}
    elif mismatch == "precision":
        completed["identity"]["precision"]["matmul_tf32"] = True
    else:
        completed["profile"]["output_shape"] = [1024, 8448]
    with pytest.raises(ValueError):
        audit.packing_bridge(completed, identity)


def test_sass_rebases_branch_addresses_and_omitted_reuse_annotations():
    assert audit.normalize_sass("@P0 BRA 0x13c0", 0x1000) == "@P0 BRA 0x3c0"
    assert audit.normalize_sass("BSSY B0, 0x13d0", 0x1000) == "BSSY B0, 0x3d0"
    assert audit.normalize_sass("LDG R1, [R2+0x400]", 0x1000) == "LDG R1, [R2+0x400]"
    assert audit.normalize_sass("IMAD R29, R23.reuse, 0x2100, RZ ;") == "IMAD R29, R23, 0x2100, RZ"


def test_python_ast_allows_only_formatting_changes(tmp_path):
    before, after = tmp_path / "before.py", tmp_path / "after.py"
    before.write_text("x = f(1, 2)\n")
    after.write_text("x = f(\n  1,\n  2,\n)\n")
    assert audit.python_ast(before) == audit.python_ast(after)
    after.write_text("x = f(1, 3)\n")
    assert audit.python_ast(before) != audit.python_ast(after)
