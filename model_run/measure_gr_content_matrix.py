"""Measure 3 independent histories x 3 item variants with shared causal prefill."""

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from GR.heat import HeatPopulation
from GR.input_generator import (
    DEFAULT_ITEM_LENGTHS,
    DEFAULT_USER_LENGTHS,
    PREFIX,
    InputGenerator,
    TextConfig,
)
from GR.scheduling import ScheduleConfig
from model_run.deepseek_v32_decode import attention_scale
from model_run.deepseek_v32_ops import quantize_index
from model_run.gr_index_selection import select_batched
from model_run.measure_gr_mla_cache_union import config_from_checkpoint, rmsnorm
from model_run.measure_gr_multilayer_hits import load_layer, save_hits
from model_run.validate_gr_numerics import validate_attention, validate_ffn, validate_index

ROOT = Path("GR/generated/content_matrix")


def case_name(h, u, n, i):
    return f"h{h}_u{u}_n{n}_i{i}"


def generate_group(root, model, h, u, new_lengths):
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    instruction = len(tokenizer.encode(PREFIX, add_special_tokens=False).ids)
    base = None
    branches = []
    prefix_ids = None
    for n in new_lengths:
        generator = InputGenerator(
            HeatPopulation({u: 1.0}, {"source": "independent-content matrix"}),
            tokenizer,
            text_config=TextConfig(
                user_lengths=(h - instruction,),
                user_probabilities=(1.0,),
                item_lengths=(n + instruction,),
                item_probabilities=(1.0,),
            ),
            schedule_config=ScheduleConfig(seed=42, sampling="sequential"),
        )
        if base is None:
            base = generator
        else:
            generator._history = base._history
        for i in range(3):
            name = case_name(h, u, n, i)
            source = root / "inputs" / name
            source.mkdir(parents=True, exist_ok=True)
            row = generator.for_user(u, item_variant=i)
            ids = row["input_ids"]
            assert len(ids) == h + n and row["stable_prefix_tokens"] == h
            if prefix_ids is None:
                prefix_ids = ids[:h]
            assert prefix_ids == ids[:h]
            suffix = ids[h:]
            suffix_hash = hashlib.sha256(bytes(str(suffix), "ascii")).hexdigest()
            row.update(history_variant=u, item_variant=i, candidate_suffix_sha256=suffix_hash)
            (source / "input.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
            (source / "prompt.txt").write_text(row["prompt"])
            branches.append(
                {
                    "name": name,
                    "source": source,
                    "new": n,
                    "item_variant": i,
                    "ids": suffix,
                    "history_sha256": row["history_sha256"],
                    "candidate_suffix_sha256": suffix_hash,
                }
            )
            print(f"Generated {name}", flush=True)
    return prefix_ids, branches


def forward_chunk(runner, norms, mlp, case, hidden, start, layer, check, history_only=False):
    """Forward exactly these query rows against the shared, causal prefix cache."""
    stop = start + len(hidden)
    residual = hidden.cuda()
    case.x = rmsnorm(residual.float(), norms[0], runner.cfg.norm_eps).bfloat16()
    positions = case.position_ids[start:stop]
    proj = runner.project(SimpleNamespace(batch=len(hidden), x=case.x, position_ids=positions))
    runner.write_chunk(case, proj, start)
    if layer == 2 and history_only and not check:
        return None, None, None
    q, qs = quantize_index(proj.idx_q[:, 0])
    weights = (proj.idx_weights * qs[..., 0] * runner.cfg.index_head_dim**-0.5).contiguous()
    ends = (positions + 1).int()
    indices, sampled = select_batched(
        q, case.index_keys[:stop], case.index_scales[:stop], weights, ends
    )
    valid = indices >= 0
    assert torch.equal(valid.sum(1), ends.clamp(max=2048).long())
    assert ((indices < ends[:, None]) | ~valid).all()
    sorted_indices = indices.sort(-1).values
    assert ((sorted_indices[:, 1:] > sorted_indices[:, :-1]) | (sorted_indices[:, 1:] < 0)).all()
    validation = {"layer": layer, "start": start, "stop": stop}
    if check:
        validation["index"] = validate_index(
            runner, case, proj, sampled, indices, start, stop, sampled_logits=True
        )
    del q, qs, weights, sampled, sorted_indices, valid, ends
    out = None
    if layer < 2 or check:
        attn = runner.sparse_prefill(
            proj.q_attn, case.kv_cache, indices, attention_scale(runner.cfg), 512, bf16_qk=False
        )
        if isinstance(attn, tuple):
            attn = attn[0]
        if check:
            validation["attention"] = validate_attention(
                runner, case, proj, indices, attn, start, stop, offload=not history_only
            )
        if layer < 2:
            summed = residual.bfloat16().float() + runner.post_wo(runner.post_wv_b(attn)).float()
            normalized = rmsnorm(summed, norms[1], runner.cfg.norm_eps).bfloat16()
            gate, up, down = mlp
            intermediate = (
                torch.nn.functional.silu(gate(normalized).float()) * up(normalized).float()
            ).bfloat16()
            ffn = down(intermediate)
            out = (summed.bfloat16().float() + ffn.float()).cpu()
            assert torch.isfinite(out).all()
            if check:
                validation["ffn"] = validate_ffn(mlp, normalized, ffn)
    return out, indices.cpu(), validation if check else None


@torch.inference_mode()
def measure_group(root, model, h, u, new_lengths, chunk=512):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    group = root / f"h{h}_u{u}"
    group.mkdir(parents=True, exist_ok=True)
    (group / "complete.json").unlink(missing_ok=True)
    prefix_ids, branches = generate_group(root, model, h, u, new_lengths)
    cfg_raw = json.loads((model / "config.json").read_text())
    assert cfg_raw["first_k_dense_replace"] >= 3
    cfg = config_from_checkpoint(cfg_raw)
    cfg = replace(cfg, max_seq_len=max(cfg.max_seq_len, h + max(new_lengths)))
    torch.cuda.reset_peak_memory_stats()
    with safe_open(
        model / "model-00001-of-000163.safetensors", framework="pt", device="cpu"
    ) as ckpt:
        embedding = ckpt.get_tensor("model.embed_tokens.weight")
        hidden = embedding[torch.tensor(prefix_ids)]
        for branch in branches:
            branch["hidden"] = embedding[torch.tensor(branch.pop("ids"))]
            branch["validations"] = []
        del embedding, prefix_ids
        history_validations = []
        for layer in range(3):
            runner, norms, mlp = load_layer(ckpt, cfg, layer)
            total = h + max(new_lengths)
            case = SimpleNamespace(
                history_len=0,
                new_tokens=total,
                position_ids=torch.arange(total, device="cuda"),
                kv_cache=torch.zeros(
                    ((total + 63) // 64, 64, 1, 656), dtype=torch.uint8, device="cuda"
                ),
                index_keys=torch.empty((total, 128), dtype=torch.float8_e4m3fn, device="cuda"),
                index_scales=torch.empty(total, device="cuda"),
            )
            output = torch.empty(hidden.shape, dtype=torch.float32) if layer < 2 else None
            for start in range(0, h, chunk):
                stop = min(start + chunk, h)
                out, _, validation = forward_chunk(
                    runner,
                    norms,
                    mlp,
                    case,
                    hidden[start:stop],
                    start,
                    layer,
                    check=start == 0 or stop == h,
                    history_only=True,
                )
                if output is not None:
                    output[start:stop] = out
                if validation is not None:
                    history_validations.append(validation)
                if stop % 65536 == 0 or stop == h:
                    print(f"h{h} u{u} layer{layer} prefix {stop}/{h}", flush=True)
            for branch in branches:
                n = branch["new"]
                out, indices, validation = forward_chunk(
                    runner,
                    norms,
                    mlp,
                    case,
                    branch["hidden"],
                    h,
                    layer,
                    check=True,
                )
                # Only [h:h+n] is overwritten; every branch sees the same [0:h]
                # prefix and its own causal suffix. No preceding branch is visible.
                if layer < 2:
                    branch["hidden"] = out
                branch["validations"].append(validation)
                target = root / f"layer{layer}" / branch["name"]
                save_hits(
                    target,
                    branch["source"],
                    indices,
                    h,
                    n,
                    layer,
                    torch.cuda.max_memory_allocated() / 2**20,
                    None,
                    index_query_batch=128,
                    metadata={
                        "history_variant": u,
                        "item_variant": branch["item_variant"],
                        "history_sha256": branch["history_sha256"],
                        "candidate_suffix_sha256": branch["candidate_suffix_sha256"],
                        "shared_history_prefill": True,
                        "mla_new_batch_tokens": n,
                        "rope_cache_tokens": cfg.max_seq_len,
                    },
                )
                print(f"Completed layer {layer}: {branch['name']}", flush=True)
            hidden = output
            del runner, norms, mlp, case, output, out, indices
            gc.collect()
            torch.cuda.empty_cache()
        for branch in branches:
            validations = history_validations + branch["validations"]
            assert len(validations) == 9
            (root / f"{branch['name']}_validation.json").write_text(
                json.dumps(validations, indent=2) + "\n"
            )
    (group / "complete.json").write_text(
        json.dumps(
            {
                "history": h,
                "history_variant": u,
                "new_lengths": list(new_lengths),
                "inputs": len(branches),
                "layer_results": 3 * len(branches),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Completed group h{h} u{u}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, choices=DEFAULT_USER_LENGTHS, required=True)
    parser.add_argument("--history-variant", type=int, choices=range(3), required=True)
    parser.add_argument(
        "--new-lengths",
        type=int,
        nargs="+",
        choices=DEFAULT_ITEM_LENGTHS,
        default=DEFAULT_ITEM_LENGTHS,
    )
    parser.add_argument("--model", type=Path, default=Path("models/DeepSeek-V3.2"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    measure_group(args.root, args.model, args.history, args.history_variant, args.new_lengths)


if __name__ == "__main__":
    main()
