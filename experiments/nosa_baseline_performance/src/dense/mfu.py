"""Compute module MFU from measured execution lengths and nsys GPU durations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def matrix_flops(config, prefix, query):
    layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    heads = config["num_attention_heads"]
    dim = config.get("head_dim", hidden // heads)
    q_width = heads * dim
    kv_width = config["num_key_value_heads"] * dim
    projection = 2 * query * layers * hidden
    return {
        "qkv_proj": projection * (q_width + 2 * kv_width),
        "o_proj": projection * q_width,
        "gate_up_proj": projection * (2 * intermediate),
        "down_proj": projection * intermediate,
        "attention_core": 4 * layers * q_width * (query * prefix + query * (query + 1) // 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--peak-tflops", type=float, default=989.0)
    args = parser.parse_args()
    metadata = json.loads((args.data_dir / "metadata.json").read_text())
    analysis_path = args.data_dir / "analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else []
    if "numerical_acceptance" in metadata:
        from experiments.nosa_baseline_performance.src.acceptance import verify_acceptance

        verify_acceptance(args.data_dir, metadata, "dense")
    config = metadata["model_config"]
    execution = metadata["execution"]
    shapes = metadata["attention_shapes"]
    assert len(shapes) == config["num_hidden_layers"]
    assert all(row["q"][0] == execution["new_tokens"] for row in shapes)
    assert all(row["k"][0] == row["v"][0] == execution["total_tokens"] for row in shapes)
    result = {"peak_tflops": args.peak_tflops, "execution": execution, "model_config": config}
    for phase in ("full_prefill", "extend"):
        prefix, query = (
            (0, execution["total_tokens"])
            if phase == "full_prefill"
            else (execution["prefix_tokens"], execution["new_tokens"])
        )
        flops = matrix_flops(config, prefix, query)
        duration = metadata["medians"][phase]["wall_ms"]
        result[phase] = {
            "end_to_end": {
                "flops": sum(flops.values()),
                "wall_ms": duration,
                "mfu_pct": sum(flops.values()) / duration / 1e9 / args.peak_tflops * 100,
            }
        }
    for row in analysis:
        if not row["range"].startswith("GR/detailed/"):
            continue
        phase = row["range"].split("/")[2]
        prefix, query = (
            (0, execution["total_tokens"])
            if phase == "full_prefill"
            else (execution["prefix_tokens"], execution["new_tokens"])
        )
        flops = matrix_flops(config, prefix, query)
        modules = {}
        for name, count in flops.items():
            duration = row["modules"][name]["gpu_ms"]
            modules[name] = {
                "flops": count,
                "gpu_ms": duration,
                "mfu_pct": count / duration / 1e9 / args.peak_tflops * 100,
            }
        duration = metadata["medians"][phase]["wall_ms"]
        modules["end_to_end"] = {
            "flops": sum(flops.values()),
            "wall_ms": duration,
            "mfu_pct": sum(flops.values()) / duration / 1e9 / args.peak_tflops * 100,
        }
        result[phase] = modules
    (args.data_dir / "mfu.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({phase: result[phase] for phase in ("full_prefill", "extend")}, indent=2))


if __name__ == "__main__":
    main()
