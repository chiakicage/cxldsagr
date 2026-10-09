"""CPU checks for observer lifetime, static destinations and stage ordering."""

from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import MISSING
from experiments.deepseek_v32_mfu.src.prefetch_validation import ColdPrefetchObserver
from operators.deepseek_v32.indexer import echo


def observer_model(*, history=4, append=2, slots=6):
    layers = [
        SimpleNamespace(
            free=torch.tensor([False, *([True] * slots)]),
            priority=torch.full((slots + 1,), -1, dtype=torch.int64),
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
            slots=slots,
            layer_id=index,
            record_bytes=4,
            session=SimpleNamespace(host_tokens=slots),
            _pool=pool,
            _counter_totals=torch.zeros(8, dtype=torch.int64),
            host_to_device=torch.full((slots,), MISSING, dtype=torch.int32),
            device_to_host=torch.full((slots + 1,), MISSING, dtype=torch.int64),
            logical_to_global=lambda ids: ids,
            prepare_prefetch=lambda *args, **kwargs: {"max_prefetch": min(8192, slots - append)},
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
        length=history,
        slots=slots,
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


def test_official_observer_snapshots_stage_before_finalize_and_selects_hint_one(
    tmp_path, monkeypatch
):
    from experiments.deepseek_v32_mfu.src.prefetch_transition_audit import OFFICIAL_POLICY

    model = observer_model(history=64, append=1, slots=128)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    block = model.blocks[0]
    block.attention.offset[:2] = torch.tensor([3.0, 7.0])
    original = block.cache.prepare_prefetch
    with ColdPrefetchObserver(model, 64, 1, tmp_path) as observer:
        lease = observer._prepare(original, 0, 64, 1, block.attention.offset)
        state = SimpleNamespace(
            host_ids=torch.arange(64, dtype=torch.int32).view(1, 64),
            records=torch.arange(128, dtype=torch.float32).to(torch.bfloat16).view(1, 64, 2),
        )
        lease.update(
            prefetch_policy=OFFICIAL_POLICY,
            official_prefetch_cap=64,
            _official_prefetch=state,
            free_slots=torch.arange(1, 129, dtype=torch.int32),
            allocation_log=torch.arange(129, dtype=torch.int64),
        )

        def finalize(prefetch):
            prefetch["_official_prefetch"].records.zero_()

        observer._stage(finalize, 0, "after_prefetch", lease)
        target = observer.eager[0]
        assert target["policy_metadata"] == {
            "prefetch_policy": OFFICIAL_POLICY,
            "max_prefetch": 64,
            "prepared_max_prefetch": 127,
            "hint_index": 1,
        }
        assert target["initial_hint"][1].item() == 7
        assert torch.equal(target["official_staging"]["host_ids"], torch.arange(64).int())
        assert target["official_staging"]["records"].sum() > 0
        assert state.records.sum() == 0


def bounded_observer_model():
    model = observer_model(history=65536, append=1, slots=65600)
    for block in model.blocks:
        cache, pool = block.cache, block.cache._pool
        cache.record_bytes = 1152
        cache.host_written_end = 65536
        cache.indexer_visible_end = 65537
        cache.session.owner = "only"
        block.attention.cfg.index_topk = 2048
        pool._active = ("only", 0)
        pool._depth = 1
        pool._pending_prefetch = ("only", 0)
        pool.free_slots = torch.arange(1, 65601, dtype=torch.int32)
        pool.allocation_log = torch.full((65601,), MISSING, dtype=torch.int64)
        pool.miss_scratch = torch.zeros(16, dtype=torch.int64)
    return model


def bounded_lease(cache):
    pool = cache._pool
    token = echo._BoundedPreparedPrefetch(
        1,
        pool.free_slots,
        pool.allocation_log,
        pool.counter,
        pool.prefetch_stats,
        pool.miss_scratch,
    )
    return {
        "_prepared": token,
        "prepared_limit": 64,
        "max_prefetch": 64,
        "history_length": cache.host_written_end,
        "transient_suffix": False,
        "free_slots": pool.free_slots,
        "allocation_log": pool.allocation_log,
        "counter": pool.counter,
        "prefetch_stats": pool.prefetch_stats,
    }


def official_stage(lease):
    from experiments.deepseek_v32_mfu.src.prefetch_transition_audit import OFFICIAL_POLICY

    lease.update(
        prefetch_policy=OFFICIAL_POLICY,
        official_prefetch_cap=64,
        _official_prefetch=SimpleNamespace(
            host_ids=torch.full((1, 64), -1, dtype=torch.int32),
            records=torch.zeros((1, 64, 576), dtype=torch.bfloat16),
        ),
    )


def test_observer_records_actual_bounded_dispatch_separately_from_full_capture(
    tmp_path, monkeypatch
):
    from experiments.deepseek_v32_mfu.src.prefetch_transition_audit import BOUNDED_FREE_PREPARATION

    model = bounded_observer_model()
    cache = model.blocks[0].cache
    capturing = {"active": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["active"])
    with ColdPrefetchObserver(model, 65536, 1, tmp_path) as observer:
        lease = bounded_lease(cache)
        observer._prepare(lambda *a, **k: lease, 0, 65536, 1, torch.zeros(16))
        assert not lease["_prepared"].used
        lease["_prepared"].consume(lease, 1, consumer_limit=64)
        official_stage(lease)
        observer._stage(lambda *_: None, 0, "after_prefetch", lease)
        observer._stage(lambda *_: None, 0, "after_append")
        observer._stage(lambda x: x, 0, "after_recall", torch.arange(2048).int()[None])
        bounded = observer.eager[0]["policy_metadata"]
        assert bounded["prepared_max_prefetch"] == 64
        assert bounded["preparation"]["kind"] == BOUNDED_FREE_PREPARATION
        assert bounded["preparation"]["requested_max_prefetch"] == 8192
        assert observer.metadata[0]["max_prefetch"] == 8192
        assert "prepared_token" not in observer.eager[0]
        capturing["active"] = True
        full = {
            key: value for key, value in lease.items() if key not in ("_prepared", "prepared_limit")
        }
        full["max_prefetch"] = 8192
        observer._prepare(lambda *a, **k: full, 0, 65536, 1, torch.zeros(16))
        observer._stage(lambda *_: None, 0, "after_prefetch", full)
        assert observer.captured[0]["policy_metadata"]["prepared_max_prefetch"] == 8192
        assert "preparation" not in observer.captured[0]["policy_metadata"]
        assert observer.eager[0]["policy_metadata"] == bounded
        assert observer.metadata[0]["max_prefetch"] == 8192


@pytest.mark.parametrize(
    "change",
    [
        "fake",
        "full",
        "consumed",
        "rows",
        "buffer",
        "scratch",
        "cap",
        "limit",
        "lease_history",
        "history",
        "columns",
        "headroom",
        "transient",
        "session",
        "owner",
        "depth",
        "pending",
        "requested",
    ],
)
def test_observer_rejects_bounded_dispatch_mismatches(change, tmp_path, monkeypatch):
    model = bounded_observer_model()
    cache, pool = model.blocks[0].cache, model.blocks[0].cache._pool
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with ColdPrefetchObserver(model, 65536, 1, tmp_path) as observer:
        lease = bounded_lease(cache)
        token = lease["_prepared"]
        if change == "fake":
            lease["_prepared"] = SimpleNamespace(**token.__dict__)
        elif change == "full":
            lease["_prepared"] = echo._PreparedPrefetch(1, *token.tensors, pool.miss_scratch)
        elif change == "consumed":
            token.used = True
        elif change == "rows":
            token.rows = 2
        elif change == "buffer":
            lease["free_slots"] = lease["free_slots"].clone()
        elif change == "scratch":
            pool.miss_scratch = pool.miss_scratch.clone()
        elif change in {"cap", "limit", "lease_history"}:
            key = {
                "cap": "max_prefetch",
                "limit": "prepared_limit",
                "lease_history": "history_length",
            }[change]
            lease[key] += 1
        elif change == "history":
            cache.host_written_end -= 1
            lease["history_length"] = cache.host_written_end
        elif change == "columns":
            cache.indexer_visible_end += 1
        elif change == "headroom":
            cache.slots = 65599
        elif change == "transient":
            cache.transient_start = 65536
        elif change == "session":
            pool._sessions["other"] = object()
        elif change == "owner":
            pool._active = ("other", 0)
        elif change == "depth":
            pool._depth = 0
        elif change == "pending":
            pool._pending_prefetch = ("other", 0)
        with pytest.raises(ValueError, match="observer|bounded-free"):
            observer._prepare(
                lambda *a, **k: lease,
                0,
                65536,
                1,
                torch.zeros(16),
                limit=63 if change == "requested" else 8192,
            )


@pytest.mark.parametrize("change", ["coarse", "unconsumed", "replaced", "cap", "limit_type"])
def test_bounded_observer_requires_actual_official_token_consumption(change, tmp_path, monkeypatch):
    model = bounded_observer_model()
    cache = model.blocks[0].cache
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with ColdPrefetchObserver(model, 65536, 1, tmp_path) as observer:
        lease = bounded_lease(cache)
        observer._prepare(lambda *a, **k: lease, 0, 65536, 1, torch.zeros(16))
        official_stage(lease)
        if change != "unconsumed":
            lease["_prepared"].consume(lease, 1, consumer_limit=64)
        if change == "coarse":
            del lease["prefetch_policy"]
        elif change == "replaced":
            lease["_prepared"] = bounded_lease(cache)["_prepared"]
        elif change == "cap":
            lease["max_prefetch"] = 8192
        elif change == "limit_type":
            lease["prepared_limit"] = 64.0
        with pytest.raises(ValueError, match="official Q1 policy"):
            observer._stage(lambda *_: None, 0, "after_prefetch", lease)
