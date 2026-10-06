from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import MISSING
from experiments.deepseek_v32_mfu.src.extend_graph_validation import (
    cache_state,
    compare_cache_state,
    tensor_identity,
)


def fixture_model(*, swap=False):
    host = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    records = torch.cat([torch.zeros_like(host[:1]), host.flip(0) if swap else host])
    shared = SimpleNamespace(
        free=torch.tensor([False, False, False]),
        priority=torch.tensor([0, 8, 8], dtype=torch.int64),
        clock_tensor=torch.tensor([9], dtype=torch.int64),
    )
    cache = SimpleNamespace(
        length=2,
        written=2,
        indexer_visible_end=2,
        _step_end=None,
        offload=True,
        device=torch.device("cpu"),
        slots=2,
        layer_id=0,
        _pool=SimpleNamespace(layers=[shared]),
        host_to_device=torch.tensor([2, 1] if swap else [1, 2], dtype=torch.int32),
        device_to_host=torch.tensor([MISSING, 1, 0] if swap else [MISSING, 0, 1]),
        records=records,
        host=host,
        host_records=lambda: host.clone(),
        metrics=lambda: {"written_records": 2},
        logical_to_global=lambda value: value,
    )
    attention = SimpleNamespace(
        index_keys=torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
        index_scales=torch.ones(2),
        offset=torch.zeros(1, dtype=torch.int64),
    )
    return SimpleNamespace(
        length=2,
        blocks=[SimpleNamespace(cache=cache, attention=attention)],
        synchronize=lambda: None,
    )


def test_cache_acceptance_allows_equivalent_tied_slot_permutations():
    first, swapped = cache_state(fixture_model()), cache_state(fixture_model(swap=True))
    assert compare_cache_state(first, swapped)["equal"]
    assert first["layers"][0]["map_invariants_passed"]


@pytest.mark.parametrize("corrupt", ["reverse", "free", "record", "commit"])
def test_cache_acceptance_rejects_invalid_maps_records_or_commit(corrupt):
    model = fixture_model()
    cache = model.blocks[0].cache
    if corrupt == "reverse":
        cache.device_to_host[1] = 1
    elif corrupt == "free":
        cache._pool.layers[0].free[1] = True
    elif corrupt == "record":
        cache.records[1, 0] = 7
    else:
        cache._step_end = 3
    with pytest.raises(AssertionError):
        cache_state(model)


def test_tensor_identity_hashes_scalar_and_low_precision_bytes():
    assert tensor_identity(torch.tensor(1.0))["shape"] == []
    assert tensor_identity(torch.ones(2, dtype=torch.bfloat16))["shape"] == [2]
