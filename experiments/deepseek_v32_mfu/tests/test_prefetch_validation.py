"""CPU checks for observer lifetime, static destinations and stage ordering."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import MISSING
from experiments.deepseek_v32_mfu.src.prefetch_validation import ColdPrefetchObserver
from operators.deepseek_v32.indexer import echo


def observer_model():
    layers = [
        SimpleNamespace(
            free=torch.tensor([False, *([True] * 6)]),
            priority=torch.full((7,), -1, dtype=torch.int64),
            clock_tensor=torch.tensor([10], dtype=torch.int64),
        )
        for _ in range(3)
    ]
    pool = SimpleNamespace(
        layers=layers,
        _sessions={"only": object()},
        counter=torch.zeros(1, dtype=torch.uint32),
        prefetch_stats=torch.zeros(3, dtype=torch.int64),
    )
    blocks = []
    for index in range(3):
        cache = SimpleNamespace(
            offload=True,
            transient_start=None,
            device=torch.device("cpu"),
            slots=6,
            layer_id=index,
            record_bytes=4,
            _pool=pool,
            _counter_totals=torch.zeros(8, dtype=torch.int64),
            host_to_device=torch.full((6,), MISSING, dtype=torch.int32),
            device_to_host=torch.full((7,), MISSING, dtype=torch.int64),
            logical_to_global=lambda ids: ids,
            prepare_prefetch=lambda *args, **kwargs: {"max_prefetch": 4},
            finalize_prefetch=lambda *_: None,
            append=lambda *_: None,
            _ensure_from_topk=lambda indices: indices,
        )
        blocks.append(
            SimpleNamespace(
                cache=cache,
                attention=SimpleNamespace(
                    offset=torch.zeros(16), cfg=SimpleNamespace(index_topk=2)
                ),
            )
        )
    closed = []
    return SimpleNamespace(
        cache_method="echo",
        length=4,
        slots=6,
        blocks=blocks,
        _shared_pools={"cpu": pool},
        synchronize=lambda: None,
        _close_extend_graphs=lambda: closed.append(True),
        closed=closed,
    )


def test_observer_keeps_preallocated_eager_and_capture_buffers_separate(tmp_path, monkeypatch):
    model = observer_model()
    capturing = {"active": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["active"])
    scores = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    monkeypatch.setattr(echo, "logits", lambda: scores)
    original = model.blocks[0].cache.prepare_prefetch
    with ColdPrefetchObserver(model, 4, 2, tmp_path) as observer:
        addresses = [target["scores"].data_ptr() for target in observer.eager + observer.captured]
        assert len(set(addresses)) == 6
        for capture in (False, True):
            capturing["active"] = capture
            for index, block in enumerate(model.blocks):
                cache = block.cache
                cache.prepare_prefetch(4, 2, block.attention.offset)
                returned = echo.logits()
                assert returned is scores
                cache.finalize_prefetch()
                cache.append(torch.ones(2, 2))
                indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
                cache._ensure_from_topk(indices)
                target = (observer.captured if capture else observer.eager)[index]
                assert target["complete"]
                assert torch.equal(target["indices"], indices)
                assert torch.equal(target["scores"], scores)
                scores.add_(1)
                assert not torch.equal(target["scores"], scores)
        assert addresses == [
            target["scores"].data_ptr() for target in observer.eager + observer.captured
        ]
        assert observer.active is None
    assert model.blocks[0].cache.prepare_prefetch is original
    assert model.closed == [True]
    assert not hasattr(model, "_prefetch_validation_owner")


def test_observer_rejects_nonempty_initial_hbm(tmp_path):
    model = observer_model()
    model.blocks[0].cache.host_to_device[0] = 1
    with pytest.raises(ValueError, match="initially empty"):
        ColdPrefetchObserver(model, 4, 2, tmp_path)


def test_observer_requires_stage_sequence_and_restores_hooks_on_error(tmp_path, monkeypatch):
    model = observer_model()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    original = model.blocks[0].cache.append
    with (
        pytest.raises(ValueError, match="matching layer"),
        ColdPrefetchObserver(model, 4, 2, tmp_path),
    ):
        model.blocks[0].cache.append(torch.ones(2, 2))
    assert model.blocks[0].cache.append is original
    assert model.closed == [True]


def test_failed_graph_close_retains_observer_and_both_original_exceptions(tmp_path):
    model = observer_model()
    original, cleanup = ValueError("body"), RuntimeError("drain")

    def fail_close():
        raise cleanup

    model._close_extend_graphs = fail_close
    observer = ColdPrefetchObserver(model, 4, 2, tmp_path)
    append = model.blocks[0].cache.append
    with pytest.raises(BaseExceptionGroup) as failure, observer:
        raise original
    assert failure.value.exceptions == (original, cleanup)
    assert model._prefetch_validation_owner is observer and model._poisoned
    assert model.blocks[0].cache.append is append
    assert observer.eager and observer.captured
