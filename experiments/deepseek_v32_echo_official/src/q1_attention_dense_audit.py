"""Quantify rounding differences and the existing per-query split contract."""

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.deepseek_v32_echo_official.src.q1_attention_dense import dense_candidate
from experiments.deepseek_v32_echo_official.src.q1_indexer import DEFAULT_INPUT, digest
from operators.deepseek_v32.attention.device_only.mla import sparse_mla


def compare(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {
        "elements": actual.numel(),
        "different_bf16_bits": (actual.view(torch.int16) != expected.view(torch.int16))
        .sum()
        .item(),
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    candidate = source.with_name("q1_attention_dense.py")
    cache = source.parents[1] / "output/runtime/q1-attention-dense/triton"
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    data = torch.load(DEFAULT_INPUT, map_location="cpu", weights_only=True)
    with torch.inference_mode():
        q, kv, ids = data["q"].cuda(), data["kv"].cuda(), data["indices"].cuda()
        real = compare(
            dense_candidate(q, kv, ids, data["attention_scale"]),
            sparse_mla(q, kv, ids, data["attention_scale"]),
        )
        # Match the existing Q65/H128/S137 test's seeded strided input fixture.
        torch.manual_seed(427)
        q = torch.randn(65, 256, 1152, device="cuda", dtype=torch.bfloat16)[:, ::2, ::2]
        kv = torch.randn(846, 1152, device="cuda", dtype=torch.bfloat16)[::2, ::2]
        ids = torch.randint(1, len(kv), (65, 274), device="cuda", dtype=torch.int32)[:, ::2]
        kv[0] = float("nan")
        ids[0] = -1
        ids[1, ::7] = len(kv) + 10
        ids[2, :3] = torch.tensor([1, 1, 2], device="cuda", dtype=torch.int32)
        whole = sparse_mla(q, kv, ids, 576**-0.5)
        original_tail = sparse_mla(q[64:], kv, ids[64:], 576**-0.5)
        candidate_tail = dense_candidate(q[64:], kv, ids[64:], 576**-0.5)
        original_split = compare(original_tail, whole[64:])
        candidate_split = compare(candidate_tail, whole[64:])
    result = {
        "source_sha256": {str(path): digest(path) for path in (source, candidate)},
        "input_sha256": digest(DEFAULT_INPUT),
        "real_layer0": real,
        "q65_original_tail_vs_whole": original_split,
        "q65_dense_tail_vs_whole": candidate_split,
        "boundary": "Diagnostic only. A Q1 automatic dispatch changes the existing Q65 "
        "test's tail reduction; tolerance acceptance does not preserve its bitwise split contract.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for path in (source, candidate):
        (args.output_dir / path.name).write_bytes(path.read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
