"""Run checkpoint dense layers 0/1 and record layer 0/1/2 GR index selections."""

import argparse
import gc
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from model_run.deepseek_v32_decode import attention_scale
from model_run.deepseek_v32_extend import V32ExtendRunner
from model_run.deepseek_v32_ops import FlashInferV32Ops, quantize_index
from model_run.measure_gr_mla_cache_union import checkpoint_linear, config_from_checkpoint, rmsnorm
from model_run.validate_gr_numerics import validate_attention, validate_ffn, validate_index


class CheckpointOps(FlashInferV32Ops):
    def q_norm(self, x):
        return rmsnorm(x, self.q_weight, self.cfg.norm_eps)

    def kv_norm(self, x):
        return rmsnorm(x, self.kv_weight, self.cfg.norm_eps)


class HeadLinear:
    def __init__(self, weight):
        self.weight = weight.contiguous()

    def __call__(self, x):
        return torch.bmm(x, self.weight.transpose(1, 2))


def load_layer(ckpt, cfg, layer):
    runner = V32ExtendRunner.__new__(V32ExtendRunner)
    runner.cfg, runner.mode, runner.post_proj = cfg, "fp8", "deepgemm"
    runner.ops = CheckpointOps(cfg)
    stem = f"model.layers.{layer}."
    attn = stem + "self_attn."
    for name, suffix in {
        "wq_a": "q_a_proj",
        "wq_b": "q_b_proj",
        "wkv_a": "kv_a_proj_with_mqa",
        "index_wqi": "indexer.wq_b",
        "index_wki": "indexer.wk",
    }.items():
        setattr(runner, name, checkpoint_linear(ckpt, attn + suffix))
    ow = ckpt.get_tensor(attn + "o_proj.weight").cuda().float()
    oscale = ckpt.get_tensor(attn + "o_proj.weight_scale_inv").cuda()
    ow = (ow * oscale.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.bfloat16)
    runner.wo = lambda x: torch.nn.functional.linear(x, ow)
    runner.ops.q_weight = ckpt.get_tensor(attn + "q_a_layernorm.weight").cuda()
    runner.ops.kv_weight = ckpt.get_tensor(attn + "kv_a_layernorm.weight").cuda()
    runner.ops.index_weight = ckpt.get_tensor(attn + "indexer.k_norm.weight").cuda()
    runner.ops.index_bias = ckpt.get_tensor(attn + "indexer.k_norm.bias").cuda()
    head_weight = ckpt.get_tensor(attn + "indexer.weights_proj.weight").cuda().float()
    runner.index_weights = lambda x: torch.nn.functional.linear(x.float(), head_weight)
    # Restore checkpoint block scales before splitting per-head K/V. Use BF16
    # grouped products to avoid introducing a second quantization of transposed K.
    w = ckpt.get_tensor(attn + "kv_b_proj.weight").cuda().float()
    scale = ckpt.get_tensor(attn + "kv_b_proj.weight_scale_inv").cuda()
    w = (w * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.bfloat16)
    w = w.view(cfg.n_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)
    runner.wk_b = HeadLinear(w[:, : cfg.qk_nope_head_dim].transpose(1, 2))
    runner.wv_b = HeadLinear(w[:, cfg.qk_nope_head_dim :])
    import flash_mla_sm120 as mla

    runner.sparse_prefill = mla.sparse_mla_prefill_fwd
    norms = [
        ckpt.get_tensor(stem + name + ".weight").cuda()
        for name in ("input_layernorm", "post_attention_layernorm")
    ]
    mlp = (
        [
            checkpoint_linear(ckpt, stem + "mlp." + name)
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if layer < 2
        else None
    )
    return runner, norms, mlp


def save_hits(
    output,
    source,
    indices,
    history,
    new,
    layer,
    peak,
    overlap,
    *,
    index_query_batch=None,
    metadata=None,
):
    output.mkdir(parents=True, exist_ok=True)
    indices = indices.cpu()
    assert indices.shape == (new, 2048)
    assert ((indices >= 0) & (indices <= history + torch.arange(new)[:, None])).all()
    sorted_rows = indices.sort(-1).values
    assert (sorted_rows[:, 1:] > sorted_rows[:, :-1]).all()
    counts = torch.bincount(indices.flatten().long(), minlength=history + new)
    unique = torch.where(counts > 0)[0]
    torch.save(indices, output / "selected_indices.pt")
    (output / "unique_token_ids.json").write_text(json.dumps(unique.tolist()) + "\n")
    shutil.copyfile(source / "input.jsonl", output / "input.jsonl")
    shutil.copyfile(source / "prompt.txt", output / "prompt.txt")
    r = {
        "layer": layer,
        "history_tokens": history,
        "new_tokens": new,
        "topk": 2048,
        "total_references": indices.numel(),
        "unique_mla_cache_tokens": len(unique),
        "unique_history_tokens": int((counts[:history] > 0).sum()),
        "unique_new_tokens": int((counts[history:] > 0).sum()),
        "queries_computed_together": min(
            new, index_query_batch or (128 if history > 65536 else new)
        ),
        "query_union_tokens": new,
        "packed_cache_bytes_per_token": 656,
        "unique_token_bytes": len(unique) * 656,
        "hadamard": False,
        "seed": 42,
        "peak_allocated_mib": peak,
        "layer0_reference_overlap": overlap,
        "source_input_sha256": hashlib.sha256((source / "input.jsonl").read_bytes()).hexdigest(),
        "numerics_revision": 2,
        "scope": "checkpoint layers 0/1 attention + residual + dense SwiGLU; layer 2 indexer; current no-Hadamard MLA approximation",
        "grouped_kv_precision": "checkpoint dequantized BF16 K/V bmm and O linear; other projections MXFP8",
    }
    r.update(metadata or {})
    (output / "result.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--new", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--model", type=Path, default=Path("models/DeepSeek-V3.2"))
    parser.add_argument(
        "--output", type=Path, default=Path("GR/generated/multilayer_hits_validated")
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validations = []
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    h, n = args.history, args.new
    source = Path("GR/generated/cache_union_sweep") / f"h{h}_n{n}"
    row = json.loads((source / "input.jsonl").read_text())
    assert len(row["input_ids"]) == h + n
    cfg_raw = json.loads((args.model / "config.json").read_text())
    assert cfg_raw["first_k_dense_replace"] >= 3
    cfg = config_from_checkpoint(cfg_raw)
    cfg = replace(cfg, max_seq_len=max(cfg.max_seq_len, h + n))
    torch.cuda.reset_peak_memory_stats()
    with safe_open(
        args.model / "model-00001-of-000163.safetensors", framework="pt", device="cpu"
    ) as ckpt:
        embedding = ckpt.get_tensor("model.embed_tokens.weight")
        hidden = embedding[torch.tensor(row["input_ids"])]
        del embedding
        for layer in range(3):
            runner, norms, mlp = load_layer(ckpt, cfg, layer)
            case = SimpleNamespace(
                history_len=0,
                new_tokens=h + n,
                position_ids=torch.arange(h + n, device="cuda"),
                kv_cache=torch.zeros(
                    ((h + n + 63) // 64, 64, 1, 656), dtype=torch.uint8, device="cuda"
                ),
                index_keys=torch.empty((h + n, 128), dtype=torch.float8_e4m3fn, device="cuda"),
                index_scales=torch.empty(h + n, device="cuda"),
            )
            output = (
                torch.empty(hidden.shape, device="cpu", dtype=torch.float32) if layer < 2 else None
            )
            segments = [(s, min(s + args.chunk, h)) for s in range(0, h, args.chunk)] + [(h, h + n)]
            for start, stop in segments:
                residual = hidden[start:stop].cuda()
                case.x = rmsnorm(residual.float(), norms[0], cfg.norm_eps).bfloat16()
                positions = case.position_ids[start:stop]
                proj = runner.project(
                    SimpleNamespace(batch=stop - start, x=case.x, position_ids=positions)
                )
                runner.write_chunk(case, proj, start)
                # Layer 2 has no downstream hidden-state consumer: historical
                # queries are needed only at validation samples. Its K/MLA cache
                # must still cover every historical token for the new queries.
                if layer == 2 and stop <= h and not (args.validate and (start == 0 or stop == h)):
                    del proj, residual
                    if stop % 65536 == 0:
                        print(f"Layer {layer}: {stop}/{h + n} tokens (KV prepared)", flush=True)
                    continue
                if h > 65536:
                    from model_run.gr_index_selection import select_batched

                    q8, qs = quantize_index(proj.idx_q[:, 0])
                    weights = (
                        proj.idx_weights * qs[..., 0] * cfg.index_head_dim**-0.5
                    ).contiguous()
                    ends = (positions + 1).int()
                    indices, logits = select_batched(
                        q8, case.index_keys[:stop], case.index_scales[:stop], weights, ends
                    )
                    del q8, qs, weights
                else:
                    logits, ends = runner.index_logits(case, proj, start, stop)
                    assert torch.isfinite(
                        logits.masked_select(
                            torch.arange(logits.shape[1], device="cuda")[None, :] < ends[:, None]
                        )
                    ).all()
                    indices = runner.select_indices(logits, ends, 2048)
                valid_indices = indices >= 0
                assert torch.equal(valid_indices.sum(1), ends.clamp(max=2048).long())
                assert ((indices < ends[:, None]) | ~valid_indices).all()
                sorted_indices = indices.sort(-1).values
                assert (
                    (sorted_indices[:, 1:] > sorted_indices[:, :-1]) | (sorted_indices[:, 1:] < 0)
                ).all()
                check = args.validate and (start == 0 or stop == h or start == h)
                validation = {"layer": layer, "start": start, "stop": stop}
                if check:
                    validation["index"] = validate_index(
                        runner, case, proj, logits, indices, start, stop, sampled_logits=h > 65536
                    )
                del logits, ends, valid_indices, sorted_indices
                if start == h:
                    overlap = None
                    if layer == 0:
                        ref = torch.load(
                            source / "selected_indices.pt", weights_only=True, map_location="cuda"
                        )
                        sorted_ref = ref.sort(-1).values
                        rank = torch.searchsorted(sorted_ref, indices).clamp(max=2047)
                        overlap = float((sorted_ref.gather(1, rank) == indices).float().mean())
                        assert overlap > 0.99, f"Layer-0 overlap too low: {overlap}"
                        del ref, sorted_ref, rank
                    save_hits(
                        args.output / f"layer{layer}" / source.name,
                        source,
                        indices,
                        h,
                        n,
                        layer,
                        torch.cuda.max_memory_allocated() / 2**20,
                        overlap,
                    )
                if layer < 2 or check:
                    attn = runner.sparse_prefill(
                        proj.q_attn,
                        case.kv_cache,
                        indices,
                        attention_scale(cfg),
                        512,
                        bf16_qk=False,
                    )
                    if isinstance(attn, tuple):
                        attn = attn[0]
                    if check:
                        validation["attention"] = validate_attention(
                            runner, case, proj, indices, attn, start, stop, offload=start == h
                        )
                    if layer < 2:
                        # Match the reference's fused residual add + RMSNorm: normalize
                        # FP32 sum, while the residual stream itself is stored in BF16.
                        summed = (
                            residual.bfloat16().float()
                            + runner.post_wo(runner.post_wv_b(attn)).float()
                        )
                        normalized = rmsnorm(summed, norms[1], cfg.norm_eps).bfloat16()
                        gate, up, down = mlp
                        intermediate = (
                            torch.nn.functional.silu(gate(normalized).float())
                            * up(normalized).float()
                        ).bfloat16()
                        ffn = down(intermediate)
                        output[start:stop] = (summed.bfloat16().float() + ffn.float()).cpu()
                        assert torch.isfinite(output[start:stop]).all()
                        if check:
                            validation["ffn"] = validate_ffn(mlp, normalized, ffn)
                        del summed, normalized, intermediate, ffn
                    del attn
                if check:
                    validations.append(validation)
                del proj, indices, residual
                if stop % 65536 == 0 or stop == h + n:
                    print(f"Layer {layer}: {stop}/{h + n} tokens", flush=True)
            if output is not None:
                hidden = output
            del runner, norms, mlp, case, output
            if layer < 2:
                del gate, up, down
            gc.collect()
            torch.cuda.empty_cache()
    if args.validate:
        path = args.output / (source.name + "_validation.json")
        path.write_text(json.dumps(validations, indent=2) + "\n")
    print("Completed; peak allocated MiB:", torch.cuda.max_memory_allocated() / 2**20, flush=True)


if __name__ == "__main__":
    main()
