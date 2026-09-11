"""Measure unique MLA token indices for one GR request using checkpoint layer 0."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from GR.heat import HeatPopulation
from GR.input_generator import PREFIX, InputGenerator, TextConfig
from GR.scheduling import ScheduleConfig
from model_run.deepseek_v32_decode import QuantizedLinear, V32Config
from model_run.deepseek_v32_extend import V32ExtendRunner
from model_run.deepseek_v32_extend_kernels import quantize_activation
from model_run.deepseek_v32_ops import FlashInferV32Ops, quantize_index
from model_run.sweep_gr_mla_cache import HISTORIES


def config_from_checkpoint(raw):
    rope = raw["rope_scaling"]
    return V32Config(
        dim=raw["hidden_size"],
        n_heads=raw["num_attention_heads"],
        q_lora_rank=raw["q_lora_rank"],
        kv_lora_rank=raw["kv_lora_rank"],
        qk_nope_head_dim=raw["qk_nope_head_dim"],
        qk_rope_head_dim=raw["qk_rope_head_dim"],
        v_head_dim=raw["v_head_dim"],
        index_n_heads=raw["index_n_heads"],
        index_head_dim=raw["index_head_dim"],
        index_topk=raw["index_topk"],
        norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        rope_factor=rope["factor"],
        original_seq_len=rope["original_max_position_embeddings"],
        max_seq_len=raw["max_position_embeddings"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        mscale=rope["mscale"],
    )


def checkpoint_linear(weights, stem):
    layer = QuantizedLinear.__new__(QuantizedLinear)
    data = weights.get_tensor(stem + ".weight").cuda()
    scales = weights.get_tensor(stem + ".weight_scale_inv").cuda()
    assert torch.equal(scales.log2(), scales.log2().round()), stem
    layer.out_features, layer.in_features = data.shape
    layer.mode, layer.weight, layer.recipe_b = "fp8", (data, scales), None
    layer.activation_quantizer = quantize_activation
    return layer


def rmsnorm(x, weight, eps):
    # Preserve the checkpoint's FP32 norm affine weights; store output as BF16.
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/DeepSeek-V3.2"))
    parser.add_argument("--output", type=Path, default=Path("GR/generated/cache_union_64k_4k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history", type=int, choices=HISTORIES, default=65536)
    parser.add_argument("--new", type=int, choices=(1024, 2048, 4096), default=4096)
    args = parser.parse_args()
    history, new = args.history, args.new
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    instruction_tokens = len(tokenizer.encode(PREFIX, add_special_tokens=False).ids)
    generator = InputGenerator(
        HeatPopulation({0: 1.0}, {"source": "single-user measurement"}),
        tokenizer,
        text_config=TextConfig(
            user_lengths=(history - instruction_tokens,),
            user_probabilities=(1.0,),
            item_lengths=(new + instruction_tokens,),
            item_probabilities=(1.0,),
        ),
        schedule_config=ScheduleConfig(seed=args.seed, sampling="sequential"),
    )
    row = next(generator.iter_generate(1))
    assert row["stable_prefix_tokens"] == history
    assert row["candidate_suffix_tokens"] == new
    assert len(row["input_ids"]) == history + new
    (args.output / "input.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "prompt.txt").write_text(row["prompt"])
    print(
        f"Generated {history} prefix + {new} suffix tokens ({instruction_tokens} instruction tokens).",
        flush=True,
    )
    cfg = config_from_checkpoint(json.loads((args.model / "config.json").read_text()))
    cfg = replace(cfg, max_seq_len=max(cfg.max_seq_len, history + new))
    ops = FlashInferV32Ops(cfg)
    with safe_open(
        args.model / "model-00001-of-000163.safetensors", framework="pt", device="cpu"
    ) as ckpt:
        prefix = "model.layers.0."
        attn = prefix + "self_attn."
        embedding = ckpt.get_tensor("model.embed_tokens.weight")
        input_norm = ckpt.get_tensor(prefix + "input_layernorm.weight").cuda()
        q_norm = ckpt.get_tensor(attn + "q_a_layernorm.weight").cuda()
        ops.index_weight = ckpt.get_tensor(attn + "indexer.k_norm.weight").cuda()
        ops.index_bias = ckpt.get_tensor(attn + "indexer.k_norm.bias").cuda()
        wqa = checkpoint_linear(ckpt, attn + "q_a_proj")
        wqi = checkpoint_linear(ckpt, attn + "indexer.wq_b")
        wki = checkpoint_linear(ckpt, attn + "indexer.wk")
        head_weight = ckpt.get_tensor(attn + "indexer.weights_proj.weight").cuda()
    keys = torch.empty((history + new, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(history + new, device="cuda", dtype=torch.float32)
    # Layer-0 K is token-local: prefix preparation needs no prior attention/MLP.
    # Prefix is prepared in batches; all extend queries are computed together.
    for start in range(0, history + new, 4096):
        stop = min(start + 4096, history + new)
        ids = torch.tensor(row["input_ids"][start:stop], dtype=torch.int64)
        x = rmsnorm(embedding[ids].cuda(), input_norm, cfg.norm_eps)
        key = ops.index_norm(wki(x))
        positions = torch.arange(start, stop, device="cuda", dtype=torch.int64)
        if start == history:
            qr = rmsnorm(wqa(x), q_norm, cfg.norm_eps)
            query = wqi(qr).view(new, 64, 128)
            qp, kp = ops.apply_rope(query[..., :64], key[:, :64], positions, is_neox=True)
            query = torch.cat((qp, query[..., 64:]), -1)
            q8, qs = quantize_index(query)
            weights = (
                torch.nn.functional.linear(x, head_weight).float()
                * 64**-0.5
                * qs[..., 0]
                * 128**-0.5
            ).contiguous()
        else:
            _, kp = ops.apply_rope(key[:, :64], key[:, :64], positions, is_neox=True)
        key = torch.cat((kp, key[:, 64:]), -1)
        k8, ks = quantize_index(key)
        keys[start:stop].copy_(k8)
        scales[start:stop].copy_(ks[:, 0])
    from model_run.deepseek_v32_decode import deep_gemm

    ends = torch.arange(history + 1, history + new + 1, device="cuda", dtype=torch.int32)
    if history > 65536:
        from model_run.gr_index_selection import select_batched

        indices, sampled_logits = select_batched(q8, keys, scales, weights, ends, cfg.index_topk)
        del sampled_logits
    else:
        logits = deep_gemm.fp8_fp4_mqa_logits(
            (q8, None),
            (keys, scales),
            weights,
            torch.zeros_like(ends),
            ends,
            False,
            0,
            torch.float32,
        )
        valid = torch.arange(history + new, device="cuda")[None, :] < ends[:, None]
        assert torch.isfinite(logits[valid]).all().item()
        indices = V32ExtendRunner.select_indices(None, logits, ends, cfg.index_topk)
    assert ((indices >= 0) & (indices < ends[:, None])).all().item()
    # Verify every row contains top-k distinct cache-token indices.
    sorted_rows = indices.sort(-1).values
    assert (sorted_rows[:, 1:] != sorted_rows[:, :-1]).all().item()
    counts = torch.bincount(indices.flatten().long(), minlength=history + new).cpu()
    selected = torch.where(counts > 0)[0]
    hist_count = (counts[:history] > 0).sum().item()
    new_count = (counts[history:] > 0).sum().item()
    torch.save(indices.cpu(), args.output / "selected_indices.pt")
    (args.output / "unique_token_ids.json").write_text(json.dumps(selected.tolist()) + "\n")
    result = {
        "layer": 0,
        "weights": "real checkpoint embedding + input RMSNorm + layer-0 indexer",
        "hadamard": False,
        "seed": args.seed,
        "text_material": "GR synthetic product descriptions",
        "history_tokens": history,
        "new_tokens": new,
        "instruction_tokens": instruction_tokens,
        "topk": cfg.index_topk,
        "total_references": indices.numel(),
        "unique_mla_cache_tokens": selected.numel(),
        "unique_history_tokens": hist_count,
        "unique_new_tokens": new_count,
        "history_coverage_pct": hist_count / history * 100,
        "all_cache_coverage_pct": selected.numel() / (history + new) * 100,
        "unique_pages_64_tokens": selected.div(64, rounding_mode="floor").unique().numel(),
        "unique_history_pages_64_tokens": selected[selected < history]
        .div(64, rounding_mode="floor")
        .unique()
        .numel(),
        "packed_cache_bytes_per_token": 656,
        "unique_token_bytes": selected.numel() * 656,
        "mean_references_per_selected_token": indices.numel() / selected.numel(),
        "max_references_per_token": counts.max().item(),
        "queries_computed_together": min(new, 128) if history > 65536 else new,
        "query_union_tokens": new,
        "rope_cache_tokens": cfg.max_seq_len,
        "prefix_k_preparation_batch": 4096,
        "scope": "one request, first attention layer only; union of selected cache tokens, not measured HBM traffic",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
