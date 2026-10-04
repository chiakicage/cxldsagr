"""Opt-in real checkpoint layers 0–2, 64K+1K acceptance, without result artifacts."""

import gc
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PREFIX_TOKENS = 65536
EXTEND_TOKENS = 1024
PREFILL_CHUNK = 2048
POOL_TOKENS = 32768
NUM_LAYERS = 3
SEED = 314159


def _sources():
    """Identify this execution's model/cache/kernel sources and local includes."""
    from experiments.deepseek_v32_echo_prefill.src.backend_provenance import source_files

    paths = set(source_files())
    paths.update((ROOT / "models/deepseek_v32").glob("echo_*.py"))
    for relative in (
        "models/deepseek_v32/cache_resources.py",
        "models/deepseek_v32/nonmatrix.py",
        "models/deepseek_v32/request_format.py",
        "operators/flashinfer.py",
        "cache/sparse_token_cache.py",
        "cache/sparse_token_pool.py",
        "cache/host_allocation.py",
        "GR/input_generator.py",
        "GR/dataset.py",
        "GR/heat.py",
        "GR/scheduling.py",
        "experiments/deepseek_v32_echo_prefill/src/measure.py",
        "experiments/deepseek_v32_echo_prefill/src/profile_layers.py",
    ):
        paths.add(ROOT / relative)
    for directory in ("operators/deepseek_v32", "operators/common"):
        paths.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and "tests" not in path.relative_to(ROOT / directory).parts
            and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
        )
    paths.add(Path(__file__).resolve())
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _source_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _assert_empty(model, *, offload):
    from cache.sparse_token_cache import MISSING

    assert model.num_layers == len(model.blocks) == NUM_LAYERS
    assert [block.layer_idx for block in model.blocks] == list(range(NUM_LAYERS))
    assert model.length == 0
    assert model.offload is offload
    for block in model.blocks:
        cache = block.cache
        assert cache.length == cache.written == cache.indexer_visible_end == 0
        assert cache._step_end is None
        assert cache.offload is offload
        if offload:
            assert block.attention.fused_prefetch, "ECHO acceptance requires fused indexer prefetch"
        assert (cache.host_to_device == MISSING).all()
        assert torch.count_nonzero(block.attention.offset) == 0
    if offload:
        assert len(model._shared_pools) == len(model.devices)
        for pool in model._shared_pools.values():
            assert pool.slots == POOL_TOKENS
            assert pool.host_capacity == PREFIX_TOKENS + EXTEND_TOKENS
            assert not pool._writes


def _progress(mode, phase):
    chunks = 0

    @contextmanager
    def scope(name):
        nonlocal chunks
        yield
        if name == f"layer_{NUM_LAYERS - 1}":
            chunks += 1
            print(
                f"checkpoint layers 0–2 {mode} {phase}: completed chunk {chunks}",
                flush=True,
            )

    return scope


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_ECHO_CHECKPOINT"),
    reason="set DEEPSEEK_ECHO_CHECKPOINT for real first-three-layer 64K+1K acceptance",
)
def test_checkpoint_three_layers_64k_1k_independent_resident_and_offload_sparse_prefixes():
    # Once opted in, missing files, GPUs, dependencies or memory must fail.
    from experiments.deepseek_v32_echo_prefill.src.measure import make_request
    from experiments.deepseek_v32_echo_prefill.src.profile_layers import comparison
    from models.deepseek_v32.echo_infer import DeepSeekEchoModel
    from models.deepseek_v32.echo_model import Config

    checkpoint = Path(os.environ["DEEPSEEK_ECHO_CHECKPOINT"])
    assert checkpoint.is_dir(), f"checkpoint directory is absent: {checkpoint}"
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (checkpoint / name).is_file(), f"checkpoint file is absent: {name}"
    assert Config.from_checkpoint(checkpoint).num_hidden_layers >= NUM_LAYERS
    assert torch.cuda.is_available(), "explicit checkpoint acceptance requires CUDA"
    devices = (int(os.environ.get("DEEPSEEK_ECHO_CHECKPOINT_DEVICE", "0")),)
    assert all(0 <= device < torch.cuda.device_count() for device in devices)
    for device in devices:
        assert torch.cuda.get_device_capability(device) == (9, 0), "SM90/Hopper is required"
    initial_sources = _sources()
    print(
        f"three-layer checkpoint source SHA256={_source_digest(initial_sources)} "
        f"files={len(initial_sources)}",
        flush=True,
    )
    request = make_request(checkpoint, PREFIX_TOKENS, EXTEND_TOKENS, SEED)
    ids = request["input_ids"]
    assert request["stable_prefix_tokens"] == PREFIX_TOKENS
    assert request["candidate_suffix_tokens"] == EXTEND_TOKENS
    assert len(ids) == PREFIX_TOKENS + EXTEND_TOKENS
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "layers": NUM_LAYERS,
                "devices": devices,
                "prefix_tokens": PREFIX_TOKENS,
                "extend_tokens": EXTEND_TOKENS,
                "prefill_chunk": PREFILL_CHUNK,
                "extend_chunk": EXTEND_TOKENS,
                "global_pool_tokens_per_layer": POOL_TOKENS,
                "host_arena_tokens": len(ids),
                "seed": SEED,
                "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "checkpoint_metadata_sha256": {
                    name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model = None
    threads = torch.get_num_threads()
    precision = torch.backends.cuda.matmul.fp32_precision
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        model = DeepSeekEchoModel(
            checkpoint,
            devices=devices,
            capacity=len(ids),
            num_layers=NUM_LAYERS,
            offload=False,
            slots=POOL_TOKENS,
            host_arena_tokens=len(ids),
            chunk_size=PREFILL_CHUNK,
            extend_chunk_size=EXTEND_TOKENS,
            workspace_query_tokens=PREFILL_CHUNK,
        )
        _assert_empty(model, offload=False)
        resident_prefix = model.forward(
            ids[:PREFIX_TOKENS], scope=_progress("resident", "prefix")
        ).cpu()
        assert torch.isfinite(resident_prefix).all()
        assert model.length == PREFIX_TOKENS
        resident = {
            name: tensor.cpu()
            for name, tensor in model.forward(
                ids[PREFIX_TOKENS:], return_hidden=True, scope=_progress("resident", "extend")
            ).items()
        }
        assert resident["hidden"].shape == (EXTEND_TOKENS, model.cfg.dim)
        assert resident["logits"].shape == (1, model.cfg.vocab_size)
        assert all(torch.isfinite(tensor).all() for tensor in resident.values())
        # Reuse weights only. The model drops every resident KV/indexer buffer
        # before installing a new empty shared pool and independent index state.
        model.set_cache_mode(True)
        _assert_empty(model, offload=True)
        offload_prefix = model.forward(
            ids[:PREFIX_TOKENS], scope=_progress("offload", "prefix")
        ).cpu()
        assert model.length == PREFIX_TOKENS
        offload = {
            name: tensor.cpu()
            for name, tensor in model.forward(
                ids[PREFIX_TOKENS:], return_hidden=True, scope=_progress("offload", "extend")
            ).items()
        }
        assert offload["hidden"].shape == resident["hidden"].shape
        assert model.length == len(ids)
        assert all(block.cache.length == block.cache.written == len(ids) for block in model.blocks)
        # Same contract as the existing checkpoint profile acceptance: no
        # tolerance widening, token subsampling, or last-hidden-only shortcut.
        results = {
            "all_extend_hidden": comparison(offload["hidden"], resident["hidden"]),
            "extend_last_logits": comparison(offload["logits"], resident["logits"]),
            "prefix_last_logits": comparison(offload_prefix, resident_prefix),
        }
        metrics = [block.cache.metrics() for block in model.blocks]
        traffic = {
            name: sum(layer[name] for layer in metrics)
            for name in (
                "prefetched_records",
                "recalled_records",
                "evicted_records",
                "capacity_splits",
                "host_to_device_bytes",
                "device_to_host_bytes",
            )
        }
        assert traffic["host_to_device_bytes"] > 0, "acceptance must exercise host KV reads"
        assert traffic["prefetched_records"] > 0, "acceptance must exercise actual fused prefetch"
        assert traffic["device_to_host_bytes"] == len(ids) * NUM_LAYERS * 576 * 2
        print(
            json.dumps(
                {"three_layer_checkpoint_numerical": results, "cache_traffic": traffic},
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        final_sources = _sources()
        changed = sorted(
            name
            for name in initial_sources.keys() | final_sources.keys()
            if initial_sources.get(name) != final_sources.get(name)
        )
        print(
            f"three-layer checkpoint final source SHA256={_source_digest(final_sources)} "
            f"changed={changed}",
            flush=True,
        )
        if model is not None:
            model.synchronize()
            model._release_shared_caches()
            model.blocks.clear()
            model.embedding_weight = model.final_norm = model.head_weight = None
            model.reader = None
        model = None
        gc.collect()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.fp32_precision = precision
        assert not changed, f"execution dependencies changed during acceptance: {changed}"
