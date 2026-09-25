"""Check dense correctness and profiler wiring on a tiny CUDA model."""

import argparse
import json
from pathlib import Path

import torch

from experiments.legacy.nosa_gr_forward.src.measure import measure_request
from experiments.legacy.nosa_gr_forward.src.profile import profile_request
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from models.nosa.tests.test_model import (
    initialized_model,
    test_flashinfer_prefill_and_decode_match_dense_attention,
    tiny_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    test_flashinfer_prefill_and_decode_match_dense_attention()
    config = tiny_config(hidden_size=256, head_dim=64, intermediate_size=384)
    model = initialized_model(config, device="cuda", dtype=torch.bfloat16, attention=None)
    tokens = torch.tensor([1, 7, 6, 3, 2, 8, 9, 11, 12], device="cuda")
    measurement = measure_request(model, tokens, prefix_tokens=5, chunk_size=3)
    rows, validation = profile_request(
        model, tokens, 5, 3, args.profile_dir / "tiny_profile.trace.json"
    )
    assert validation["finite"]
    assert validation["full_vs_split_last_hidden_max_abs"] < 0.1
    assert {row["phase"] for row in rows} == {
        "full_prefill",
        "prefix_prefill",
        "candidate_extend",
    }
    assert sum(row["gpu_us"] for row in rows) > 0
    assert any(row["module"] == "rope_apply" for row in rows)
    assert any(row["module"] == "attention_core" for row in rows)
    output = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dense_reference": "passed",
        "profile_validation": validation,
        "profile_gpu_activities": sum(row["gpu_activities"] for row in rows),
        "profile_modules": sorted({row["module"] for row in rows}),
        "measurement": measurement,
        "source_sha256": source_hashes(
            Path(__file__),
            Path("experiments/legacy/nosa_gr_forward/src/measure.py"),
            Path("experiments/legacy/nosa_gr_forward/src/profile.py"),
        ),
    }
    (args.output_dir / "gpu_smoke.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
