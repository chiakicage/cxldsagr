"""Opt-in full-checkpoint offload correctness, not a performance experiment.

Run from the repository root on SM90/Hopper, for example::

    NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
        .venv/bin/python -m pytest -s -q -p no:cacheprovider \
        models/nosa/tests/test_offload_checkpoint.py

``NOSA_OFFLOAD_REQUEST`` optionally selects an existing 65,536 + 1,024 token
NOSA GR request JSON. No checkpoint is loaded unless the checkpoint environment
variable is set. Results are printed to the terminal; no report is written.
"""

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.nosa.model import NosaForCausalLM

PREFIX_TOKENS = 65536
EXTEND_TOKENS = 1024
CHUNK_SIZE = 1024
TOTAL_TOKENS = PREFIX_TOKENS + EXTEND_TOKENS
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REQUEST = (
    ROOT
    / "experiments/indexer_block_sparse_profile/output/data"
    / "sparse_native_h200_gpu1_20260929_01/request.json"
)


@torch.inference_mode()
def test_cuda_offloaded_checkpoint_matches_independent_resident_64k_1k(monkeypatch):
    """Compare final hidden states after independently propagated sparse prefixes."""
    checkpoint = os.environ.get("NOSA_OFFLOAD_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set NOSA_OFFLOAD_CHECKPOINT to opt into the full NOSA-8B check")
    checkpoint = Path(checkpoint).expanduser()
    assert checkpoint.is_dir(), f"Checkpoint directory does not exist: {checkpoint}"
    request_path = Path(os.environ.get("NOSA_OFFLOAD_REQUEST", DEFAULT_REQUEST)).expanduser()
    request = json.loads(request_path.read_text())
    assert request.get("model") == "nosa", "Expected a NOSA GR request"
    input_ids = request["input_ids"]
    assert len(input_ids) == TOTAL_TOKENS
    assert all(type(token) is int and token >= 0 for token in input_ids)
    assert request.get("stable_prefix_tokens") == PREFIX_TOKENS
    assert request.get("candidate_suffix_tokens") == EXTEND_TOKENS
    assert request.get("attention_mask", [1] * TOTAL_TOKENS) == [1] * TOTAL_TOKENS

    if not torch.cuda.is_available():
        pytest.fail("Explicit full-checkpoint offload validation requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.fail("Full-checkpoint NOSA offload validation requires SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    print(f"Loading full NOSA checkpoint for correctness: {checkpoint}", flush=True)
    model = NosaForCausalLM.from_pretrained(
        checkpoint,
        device=device,
        dtype=torch.bfloat16,
        attention_mode="sparse",
        sparse_backend="auto",
        cache_backend="offload",
        offload_query_tile_size=128,
        offload_overlap=True,
    )
    assert (
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.head_dim,
    ) == (32, 32, 2, 128), "This integration check requires the complete NOSA-8B model"
    assert len(model.model.layers) == 32
    assert model.ignored_checkpoint_keys == ()
    assert max(input_ids) < model.config.vocab_size
    # Only the runtime context limit changes; strict checkpoint shape checks
    # above still cover every original parameter, including the CIS tensors.
    model.config = replace(model.config, max_position_embeddings=TOTAL_TOKENS)
    tokens = torch.tensor(input_ids, device=device, dtype=torch.long)

    with ExitStack() as resources:
        caches = {}
        for backend in ("resident", "offload"):
            model.cache_backend = backend
            cache = model.new_cache(TOTAL_TOKENS)
            resources.callback(model.cache_manager.release, cache)
            caches[backend] = cache
        assert caches["resident"] is not caches["offload"]
        assert caches["resident"].length == caches["offload"].length == 0

        hidden_states, statistics = {}, {}
        for backend, cache in caches.items():
            model.cache_backend = backend
            # Never copy prefix KV or hidden states between the two runs. Each
            # request executes every checkpoint layer from its own empty cache.
            for start in range(0, PREFIX_TOKENS, CHUNK_SIZE):
                stop = start + CHUNK_SIZE
                model(tokens[start:stop], cache, return_hidden=True)
                assert cache.length == cache.indexer_cache.length == stop
                if stop % (16 * CHUNK_SIZE) == 0:
                    print(f"{backend}: built independent prefix {stop}/{PREFIX_TOKENS}", flush=True)
            assert cache.length == PREFIX_TOKENS
            hidden = model(tokens[PREFIX_TOKENS:], cache, return_hidden=True)
            torch.cuda.synchronize(device)
            assert cache.length == cache.indexer_cache.length == TOTAL_TOKENS
            assert hidden.shape == (EXTEND_TOKENS, model.config.hidden_size)
            assert hidden.dtype == torch.bfloat16
            assert torch.isfinite(hidden).all().item(), f"Nonfinite {backend} hidden states"
            hidden_states[backend] = hidden.cpu()
            statistics[backend] = cache.stats()

        expected, actual = hidden_states["resident"], hidden_states["offload"]
        bitwise_equal = torch.equal(actual, expected)
        max_abs = (actual.float() - expected.float()).abs().max().item()
        print(
            f"Full NOSA 64K + 1K correctness: bitwise_equal={bitwise_equal}, "
            f"hidden_max_abs={max_abs:.9g}, atol=0.016, rtol=0.016",
            flush=True,
        )
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)

        resident, offload = statistics["resident"], statistics["offload"]
        kv_bytes = (
            2
            * model.config.num_hidden_layers
            * TOTAL_TOKENS
            * model.config.num_key_value_heads
            * model.config.head_dim
            * torch.empty((), dtype=torch.bfloat16).element_size()
        )
        assert resident["host_bytes"] == 0
        assert offload["host_bytes"] == kv_bytes
        assert offload["resident_bytes"] < resident["resident_bytes"]
        print(
            "Committed cache allocation bytes (not peak process memory): "
            f"resident_HBM={resident['resident_bytes']}, "
            f"offload_HBM={offload['resident_bytes']}, "
            f"offload_host={offload['host_bytes']}",
            flush=True,
        )
