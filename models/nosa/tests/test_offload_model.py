"""NOSA offloaded requests retain independent resident-model semantics."""

import pytest
import torch

from models.nosa.model import NosaForCausalLM
from models.nosa.offload_cache import NosaOffloadCache
from models.nosa.tests.test_model import tiny_config, write_checkpoint
from models.nosa.tests.test_sparse_model import initialized_sparse_model


def _pair(*, device="cpu", dtype=torch.float32, backend="reference"):
    geometry = {}
    if device == "cuda":
        geometry = {
            "hidden_size": 128,
            "num_attention_heads": 16,
            "num_key_value_heads": 1,
            "head_dim": 128,
        }
    config = tiny_config(max_position_embeddings=512, **geometry)
    resident = initialized_sparse_model(config, device=device, dtype=dtype, backend=backend)
    offload = NosaForCausalLM(
        config,
        device=device,
        dtype=dtype,
        attention_mode="sparse",
        sparse_backend=backend,
        cache_backend="offload",
        offload_query_tile_size=32,
    )
    offload.load_state_dict(resident.state_dict())
    return resident, offload


def _compare_appends(resident, offload, *, atol, rtol):
    device = resident.model.embed_tokens.weight.device
    tokens = (torch.arange(225, device=device) * 7 + 3) % resident.config.vocab_size
    left, right = resident.new_cache(512), offload.new_cache(512)
    try:
        start = 0
        for count in (47, 33, 65, 80):
            chunk = tokens[start : start + count]
            expected = resident(chunk, left, return_hidden=True)
            actual = offload(chunk, right, return_hidden=True)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            start += count
            assert left.length == right.length == start
            assert right.indexer_cache.length == start
        # Rewind across both a compressed-window and block boundary, then append.
        left.truncate(79)
        right.truncate(79)
        expected = resident(tokens[79:112], left, return_hidden=True)
        actual = offload(tokens[79:112], right, return_hidden=True)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        assert left.length == right.length == 112
    finally:
        left.release()
        right.release()


def test_offloaded_cpu_request_matches_resident_after_unaligned_appends_and_rewind():
    _compare_appends(*_pair(), atol=1e-6, rtol=1e-5)


def test_offloaded_model_failure_rolls_back_every_layer_and_can_retry(monkeypatch):
    resident, offload = _pair()
    tokens = torch.arange(96) % offload.config.vocab_size
    cache = offload.new_cache(512)
    control = resident.new_cache(512)
    try:
        resident(tokens[:47], control, return_hidden=True)
        offload(tokens[:47], cache, return_hidden=True)
        attention = offload.main_attention

        def fail_second_layer(q, selection, access, context):
            if context.layer_idx == 1:
                raise RuntimeError("injected attention failure")
            return attention(q, selection, access, context)

        monkeypatch.setattr(offload, "main_attention", fail_second_layer)
        with pytest.raises(RuntimeError, match="injected"):
            offload(tokens[47:80], cache, return_hidden=True)
        assert cache.length == cache.indexer_cache.length == 47
        monkeypatch.setattr(offload, "main_attention", attention)
        actual = offload(tokens[47:80], cache, return_hidden=True)
        expected = resident(tokens[47:80], control, return_hidden=True)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert cache.length == 80
    finally:
        cache.release()
        control.release()


def test_dense_mode_rejects_offloaded_nosa_cache():
    with pytest.raises(ValueError, match="attention_mode"):
        NosaForCausalLM(tiny_config(), cache_backend="offload")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", None])
def test_offload_fetch_producer_count_rejects_invalid_values(value):
    config = tiny_config()
    with pytest.raises(ValueError, match="offload_fetch_ctas must be a positive integer"):
        NosaForCausalLM(config, offload_fetch_ctas=value)
    with pytest.raises(ValueError, match="fetch_ctas must be a positive integer"):
        NosaOffloadCache(config, 32, device="cpu", dtype=torch.float32, fetch_ctas=value)


@pytest.mark.parametrize("options,expected_ctas", [({}, 96), ({"offload_fetch_ctas": 6}, 6)])
def test_checkpoint_offload_options_reach_lazy_workspace(
    tmp_path, monkeypatch, options, expected_ctas
):
    import operators.nosa.attention.offload.api as offload_operator

    source = initialized_sparse_model(tiny_config())
    write_checkpoint(tmp_path, source.config, dict(source.state_dict()))
    calls = []
    workspace = object()

    def allocate_workspace(*args, **kwargs):
        calls.append((args, kwargs))
        return workspace

    monkeypatch.setattr(offload_operator, "NosaFetchWorkspace", allocate_workspace)
    model = NosaForCausalLM.from_pretrained(
        tmp_path,
        device="cpu",
        dtype=torch.float32,
        attention_mode="sparse",
        sparse_backend="reference",
        cache_backend="offload",
        offload_query_tile_size=17,
        offload_overlap=False,
        **options,
    )
    cache = model.new_cache(32)
    try:
        assert calls == []
        assert cache.attention_workspace is workspace
        assert cache.attention_workspace is workspace
        assert calls == [
            (
                (32, source.config.num_key_value_heads, source.config.head_dim),
                {
                    "device": torch.device("cpu"),
                    "dtype": torch.float32,
                    "query_tile_size": 17,
                    "fetch_ctas": expected_ctas,
                    "overlap": False,
                },
            )
        ]
    finally:
        cache.release()


def test_cuda_offloaded_model_matches_independent_resident_request(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; scripts/run_tests.sh gpu requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("NOSA offload checks require SM90/Hopper")
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")
    _compare_appends(
        *_pair(device="cuda", dtype=torch.bfloat16, backend="auto"), atol=0.02, rtol=0.02
    )
