"""Full 61-layer checkpoint prefill/extend comparison with identical KV semantics."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from tokenizers import Tokenizer

from GR.input_generator import INSTRUCTION, TextConfig, create_input_generator
from GR.scheduling import ScheduleConfig
from models.deepseek_v32 import request_format
from models.deepseek_v32.echo_infer import DeepSeekEchoModel
from models.deepseek_v32.echo_model import Config
from operators.sm90.echo_indexer import build_info


class Scopes:
    def __init__(self, phase):
        self.phase = phase
        self.events = []
        self.layer = "shared"
        self.stack = []

    @contextmanager
    def __call__(self, name):
        if name.startswith("layer_"):
            previous, self.layer = self.layer, name
            try:
                with torch.cuda.nvtx.range(f"echo/{self.phase}/{name}"):
                    yield
            finally:
                self.layer = previous
            return
        label = f"echo/{self.phase}/{self.layer}/{name}"
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        node = {
            "stage": name,
            "layer": self.layer,
            "device": torch.cuda.current_device(),
            "start": start,
            "stop": stop,
            "children": [],
        }
        if self.stack:
            self.stack[-1]["children"].append(node)
        self.stack.append(node)
        with torch.cuda.nvtx.range(label), torch.profiler.record_function(label):
            start.record()
            begin = time.perf_counter()
            try:
                yield
            finally:
                node["host_ms"] = (time.perf_counter() - begin) * 1000
                stop.record()
                self.stack.pop()
                self.events.append(node)

    def summary(self):
        totals = defaultdict(float)
        rows = []
        for node in self.events:
            elapsed = node["start"].elapsed_time(node["stop"])
            child_ms = sum(x["start"].elapsed_time(x["stop"]) for x in node["children"])
            exclusive = max(0, elapsed - child_ms)
            totals[node["stage"]] += exclusive
            rows.append(
                {k: node[k] for k in ("stage", "layer", "device", "host_ms")}
                | {
                    "cuda_inclusive_ms": elapsed,
                    "cuda_exclusive_ms": exclusive,
                }
            )
        return {"stages_cuda_exclusive_ms": dict(totals), "stage_calls": rows}


def make_request(model, prefix, extend, seed):
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    overhead = len(
        tokenizer.encode(request_format.prefix(INSTRUCTION), add_special_tokens=False).ids
    )
    if prefix <= overhead:
        raise ValueError("prefix must accommodate the GR instruction")
    generator = create_input_generator(
        model="deepseek_v32",
        tokenizer=tokenizer,
        num_users=1000,
        text_config=TextConfig(
            user_lengths=(prefix - overhead,),
            user_probabilities=(1.0,),
            item_lengths=(extend + overhead,),
            item_probabilities=(1.0,),
            max_input_tokens=prefix + extend,
        ),
        schedule_config=ScheduleConfig(seed=seed, sampling="weighted"),
    )
    request = next(generator.iter_generate(1))
    if request["stable_prefix_tokens"] != prefix or request["candidate_suffix_tokens"] != extend:
        raise ValueError("GR semantic boundary differs from the measured execution boundary")
    return request


def source_manifest():
    root = Path(__file__).resolve().parents[3]
    paths = [
        *root.glob("models/deepseek_v32/echo_*.py"),
        *root.glob("operators/sm90/deepseek_*.py"),
        *root.glob("operators/sm90/echo_*.py"),
        *root.glob("operators/sm90/csrc/echo_*"),
        root / "operators/sm90/kv_transfer.py",
        root / "operators/sm90/csrc/kv_transfer.cu",
        root / "cache/sparse_token_cache.py",
        Path(__file__).resolve(),
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def timed(model, ids, label, *, annotate=False):
    model.synchronize()
    scopes = Scopes(label) if annotate else None
    begin = time.perf_counter()
    output = model.forward(ids, scope=scopes)
    elapsed = (time.perf_counter() - begin) * 1000
    if not torch.isfinite(output).all():
        raise RuntimeError(f"nonfinite full-model output for {label}")
    return output, {"wall_ms": elapsed, **(scopes.summary() if scopes else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/mnt/user-ssd/chenkaiqi/DeepSeek-V3.2"))
    parser.add_argument("--devices", default="0,1,2,6,7")
    parser.add_argument("--prefix", type=int, default=65536)
    parser.add_argument("--extend", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--nsys", action="store_true", help="capture the annotated extend ranges")
    parser.add_argument("--chrome-trace", action="store_true")
    parser.add_argument(
        "--save-kernel-inputs",
        action="store_true",
        help="save untimed real layer 0/30/60 inputs for NCU",
    )
    args = parser.parse_args()
    if (
        min(
            args.prefix,
            args.extend,
            args.slots,
            args.chunk_size,
            args.repeats,
            args.prefill_repeats,
        )
        < 1
    ):
        parser.error("shape and repetition counts must be positive")
    if args.warmups < 1:
        parser.error("at least one warmup is required to exclude compilation")
    if Config.from_checkpoint(args.model).num_hidden_layers != 61:
        parser.error("this full-model experiment requires all 61 DeepSeek V3.2 layers")
    args.output.mkdir(parents=True, exist_ok=False)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    request = make_request(args.model, args.prefix, args.extend, args.seed)
    request_text = json.dumps(request, ensure_ascii=False)
    (args.output / "request.json").write_text(request_text + "\n")
    sources = source_manifest()
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    root = Path(__file__).resolve().parents[3]
    for relative in sources:
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    ids = request["input_ids"]
    print(f"loading complete checkpoint; input={args.prefix}+{args.extend}", flush=True)
    model = DeepSeekEchoModel(
        args.model,
        devices=[int(x) for x in args.devices.split(",")],
        capacity=len(ids),
        slots=args.slots,
        chunk_size=args.chunk_size,
    )
    result = {
        "run_id": args.run_id,
        "scope": "complete_checkpoint_61_transformer_layers_embedding_final_norm_lm_head",
        "model": str(args.model.resolve()),
        "layers": model.cfg.num_hidden_layers,
        "prefix_tokens": args.prefix,
        "extend_tokens": args.extend,
        "logits": "last_token_only",
        "checkpoint_weights": "resident_block_FP8_with_BF16_MLA_absorbed_KV_projections",
        "cache_record": "BF16_512_latent_plus_64_RoPE_1152_bytes",
        "indexer": "64x128_FP8_after_normalized_Hadamard_top2048",
        "offload": "mapped_pinned_local_DRAM_fused_indexer_prefetch_then_exact_recall",
        "placement": [str(x) for x in model.placement],
        "chunk_size": args.chunk_size,
        "slots": args.slots,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "prefill_repeats": args.prefill_repeats,
        "seed": args.seed,
        "request_sha256": hashlib.sha256(request_text.encode()).hexdigest(),
        "source_sha256": sources,
        "indexer_build": build_info(),
        "checkpoint_metadata_sha256": {
            name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json", "tokenizer.json")
        },
        "dependencies": {
            x: importlib.metadata.version(x)
            for x in ("torch", "triton", "safetensors", "apache-tvm-ffi")
        },
        "hardware": [str(torch.cuda.get_device_properties(d)) for d in model.devices],
        "measurements": {},
    }
    outputs = {}
    for offload in (False, True):
        mode = "offload" if offload else "resident"
        print(f"begin {mode} warm prefill", flush=True)
        model.set_cache_mode(offload)
        for _ in range(args.warmups):
            model.set_cache_mode(offload)
            timed(model, ids[: args.prefix], f"{mode}/warm_prefill")
        prefix_rows = []
        for iteration in range(args.prefill_repeats):
            model.set_cache_mode(offload)
            for device in model.devices:
                torch.cuda.reset_peak_memory_stats(device)
            _, row = timed(model, ids[: args.prefix], f"{mode}/prefill")
            row["peak_allocated_bytes"] = [
                torch.cuda.max_memory_allocated(d) for d in model.devices
            ]
            prefix_rows.append(row)
            print(f"{mode} prefill {iteration}: {row['wall_ms']:.3f} ms", flush=True)
        model.set_cache_mode(offload)
        _, annotated_prefix = timed(
            model, ids[: args.prefix], f"{mode}/prefill_annotated", annotate=True
        )
        prefix_cache_stats = [block.cache.metrics() for block in model.blocks]
        snapshot = model.snapshot_prefix()
        for _ in range(args.warmups):
            model.restore_prefix(snapshot)
            timed(model, ids[args.prefix :], f"{mode}/warm_extend")
        extend_rows = []
        for iteration in range(args.repeats):
            model.restore_prefix(snapshot)
            for device in model.devices:
                torch.cuda.reset_peak_memory_stats(device)
            output, row = timed(model, ids[args.prefix :], f"{mode}/extend")
            row["peak_allocated_bytes"] = [
                torch.cuda.max_memory_allocated(d) for d in model.devices
            ]
            extend_rows.append(row)
            print(f"{mode} extend {iteration}: {row['wall_ms']:.3f} ms", flush=True)
        model.restore_prefix(snapshot)
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStart()
        if args.chrome_trace:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                output, annotated = timed(
                    model, ids[args.prefix :], f"{mode}/extend_annotated", annotate=True
                )
            prof.export_chrome_trace(str(args.profile_dir / f"{mode}_extend.json"))
        else:
            output, annotated = timed(
                model, ids[args.prefix :], f"{mode}/extend_annotated", annotate=True
            )
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStop()
        outputs[mode] = output.cpu()
        torch.save(outputs[mode], args.output / f"{mode}_logits.pt")
        stats = [block.cache.metrics() for block in model.blocks]
        result["measurements"][mode] = {
            "prefill": prefix_rows,
            "extend": extend_rows,
            "prefill_median_ms": statistics.median(x["wall_ms"] for x in prefix_rows),
            "extend_median_ms": statistics.median(x["wall_ms"] for x in extend_rows),
            "annotated_extend": annotated,
            "annotated_prefill": annotated_prefix,
            "prefill_cache_per_layer": prefix_cache_stats,
            "cache_per_layer": stats,
        }
        (args.output / f"{mode}.json").write_text(
            json.dumps(result["measurements"][mode], indent=2) + "\n"
        )
    a, b = outputs["offload"], outputs["resident"]
    error = a.float() - b.float()
    result["correctness"] = {
        "max_abs": float(error.abs().max()),
        "nrmse": float(error.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)),
        "bitwise_equal": torch.equal(a, b),
        "same_next_token": int(a.argmax(-1)) == int(b.argmax(-1)),
    }
    torch.testing.assert_close(a, b, rtol=0.01, atol=0.02)
    if args.save_kernel_inputs:
        model.restore_prefix(snapshot)
        for layer in (0, 30, 60):
            if layer >= len(model.blocks):
                continue

            def capture(p, keys, scales, indices, cache, position, layer=layer):
                torch.save(
                    {
                        "layer": layer,
                        "query_start": position,
                        "attention_scale": model.cfg.attention_scale,
                        "q": p.q.cpu(),
                        "index_q": p.index_q.cpu(),
                        "index_weights": p.index_weights.cpu(),
                        "index_keys": keys.cpu(),
                        "index_scales": scales.cpu(),
                        "indices": indices.cpu(),
                        "kv": cache.host[: cache.written].clone(),
                        "source_run_id": args.run_id,
                    },
                    args.output / f"kernel_inputs_layer_{layer}.pt",
                )

            model.blocks[layer].attention.capture_hook = capture
        model.forward(ids[args.prefix :])
        for block in model.blocks:
            block.attention.capture_hook = None
    if source_manifest() != sources:
        raise RuntimeError("implementation changed during measurement; result cannot be accepted")
    result["accepted"] = True
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_id": args.run_id, "correctness": result["correctness"]}), flush=True)


if __name__ == "__main__":
    main()
