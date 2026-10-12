"""Frozen workload matrix and clean observation result contract (CPU only)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

SCHEMA = "deepseek-v32-observation-v2"
S_VALUES = (1, 32, 64, 128, 256, 512, 1024)
H_VALUES = (4096, 16384, 32768, 65536)
QB_VALUES = tuple((1, s) for s in S_VALUES) + tuple((s, 1) for s in S_VALUES[1:])
PEAKS_TFLOPS = {"fp8": 1979.0, "bf16": 989.5, "fp32": 67.0}
PEAK_SOURCE = "https://www.nvidia.com/en-us/data-center/h200/"
PEAK_CONVENTION = "H200 SXM dense advertised peaks; FP8/BF16 sparsity peaks divided by two"
GROUPS = ("common", "moe", "attention")


@dataclass(frozen=True)
class Shape:
    B: int
    Q: int
    H: int

    def __post_init__(self):
        for name in ("B", "Q", "H"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def S(self):
        return self.B * self.Q

    @property
    def key(self):
        return f"b{self.B}_q{self.Q}_h{self.H}"

    def as_dict(self):
        return {**asdict(self), "S": self.S, "key": self.key}


def matrix():
    return tuple(Shape(b, q, h) for q, b in QB_VALUES for h in H_VALUES)


def task_keys(group):
    if group not in GROUPS:
        raise ValueError(f"Unknown workload group: {group}")
    return (
        tuple(shape.key for shape in matrix())
        if group == "attention"
        else tuple(f"s{s}" for s in S_VALUES)
    )


def statistics(samples_us):
    import statistics as st

    if len(samples_us) < 3 or any(not math.isfinite(x) or x <= 0 for x in samples_us):
        raise ValueError("At least three finite positive independent timing samples are required")
    quartiles = st.quantiles(samples_us, n=4, method="inclusive")
    return {
        "time_us": st.median(samples_us),
        "mean_us": st.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "q1_us": quartiles[0],
        "q3_us": quartiles[2],
        "sample_count": len(samples_us),
    }


def mfu(work, time_us):
    flops = work.get("useful_flops")
    precision = work.get("precision")
    if flops is None:
        return {"mfu_pct": None, "peak_tflops": None, "mfu_na_reason": "non-matrix operator"}
    if type(flops) is not int or flops <= 0 or not math.isfinite(time_us) or time_us <= 0:
        raise ValueError("Matrix MFU requires positive FLOPs and time")
    if precision == "mixed":
        breakdown = {
            name: work.get("dimensions", {}).get(f"{name}_flops", 0) for name in PEAKS_TFLOPS
        }
        breakdown = {name: value for name, value in breakdown.items() if value}
        if (
            not breakdown
            or any(type(value) is not int or value <= 0 for value in breakdown.values())
            or sum(breakdown.values()) != flops
        ):
            raise ValueError("Mixed MFU requires a complete positive per-precision FLOPs breakdown")
        return {
            "mfu_pct": sum(value / PEAKS_TFLOPS[name] for name, value in breakdown.items())
            / (time_us * 1e6)
            * 100,
            "peak_tflops": None,
            "precision_peaks_tflops": {name: PEAKS_TFLOPS[name] for name in breakdown},
            "precision_flops": breakdown,
            "mfu_na_reason": None,
        }
    peak = PEAKS_TFLOPS.get(precision)
    if peak is None:
        raise ValueError(f"No explicit dense arithmetic peak for {precision}")
    return {
        "mfu_pct": flops / (time_us * 1e6 * peak) * 100,
        "peak_tflops": peak,
        "mfu_na_reason": None,
    }


def validate_result(result, *, require_complete=True):
    if result.get("schema") != SCHEMA:
        raise ValueError("Unknown observation schema")
    if result.get("offload") is not False:
        raise ValueError("Observation requires device-resident inputs without offload")
    groups = result.get("groups", {})
    if require_complete and set(groups) != set(GROUPS):
        raise ValueError("Common, MoE, and attention groups must all be present")
    for group, entries in groups.items():
        expected = set(task_keys(group))
        keys = [entry["key"] for entry in entries]
        if len(keys) != len(set(keys)) or require_complete and set(keys) != expected:
            raise ValueError(f"Incomplete or duplicate {group} workload coverage")
        inventory = None
        conditional_inventories = {}
        for entry in entries:
            if entry["key"] not in expected or not entry.get("receipt_path"):
                raise ValueError("Unknown workload key or missing numerical acceptance")
            operators = entry["operators"]
            names = [op["name"] for op in operators]
            if not names or len(set(names)) != len(names):
                raise ValueError("Each workload needs a unique nonempty operator inventory")
            unconditional = {
                op["name"]
                for op in operators
                if op.get("metadata", {}).get("supported_condition") not in {"S == 1", "S > 1"}
            }
            if any(
                op.get("metadata", {}).get("supported_condition") == "S == 1" and entry["S"] != 1
                for op in operators
            ):
                raise ValueError(
                    "S1 consumer-layout operator appears outside its support condition"
                )
            if any(
                op.get("metadata", {}).get("supported_condition") == "S > 1" and entry["S"] <= 1
                for op in operators
            ):
                raise ValueError(
                    "Multirow scale-layout operator appears outside its support condition"
                )
            if inventory is None:
                inventory = unconditional
            if group != "attention" and unconditional != inventory:
                raise ValueError(f"Inconsistent {group} operator inventory across S")
            if group != "attention":
                for condition, applicable in (
                    ("S == 1", entry["S"] == 1),
                    ("S > 1", entry["S"] > 1),
                ):
                    if applicable:
                        conditional = {
                            op["name"]
                            for op in operators
                            if op.get("metadata", {}).get("supported_condition") == condition
                        }
                        if (
                            condition in conditional_inventories
                            and conditional != conditional_inventories[condition]
                        ):
                            raise ValueError(
                                f"Inconsistent {group} conditional operator inventory for {condition}"
                            )
                        conditional_inventories[condition] = conditional
            for op in operators:
                summary = statistics(op["samples_us"])
                if not math.isclose(op["time_us"], summary["time_us"], rel_tol=1e-12):
                    raise ValueError("Reported timing differs from retained samples")
                expected_mfu = mfu(op["work"], op["time_us"])
                if op["mfu_pct"] != expected_mfu["mfu_pct"]:
                    raise ValueError("Reported MFU differs from useful FLOPs / measured API time")
    return result
