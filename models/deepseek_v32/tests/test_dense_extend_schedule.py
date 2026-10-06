"""Dense full-extend selection must precede the IO wait without sharing layer leases."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_pool import SharedSparseTokenPool
from models.deepseek_v32.attention import EchoAttentionRunner
from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
from models.deepseek_v32.model import DeepSeekEchoModel


@pytest.fixture
def dense_attention(monkeypatch):
    import operators.deepseek_v32.indexer.echo as indexer

    pool = SharedSparseTokenPool(64, 4, 2, 8, device="cpu", dense_contiguous=True)
    session = pool.allocate_session(8)
    caches = [session.layer(layer) for layer in range(2)]
    for cache in caches:
        cache.begin_step(2)
        cache.append(torch.arange(2).bfloat16()[:, None].expand(-1, 4).contiguous())
        cache.commit()
    helper = PoolHistoryPrefetch("cpu")
    ticket = helper.prefetch(caches[0])
    cfg = SimpleNamespace(
        kv_lora_rank=2, qk_rope_head_dim=2, index_head_dim=2, index_topk=4, attention_scale=1.0
    )
    events = []

    def project(hidden, position, normalized=False):
        assert pool._active == (session.owner, 0)
        events.append("project")
        values = torch.arange(position, position + len(hidden)).bfloat16()
        return SimpleNamespace(
            q=values[:, None, None],
            kv=values[:, None].expand(-1, 4).contiguous(),
            index_k=values[:, None].expand(-1, 2).to(torch.float8_e4m3fn).contiguous(),
            index_q=values,
            index_scale=torch.ones(len(hidden)),
            index_weights=torch.ones(len(hidden)),
        )

    attention = SimpleNamespace(
        cfg=cfg, device=torch.device("cpu"), project=project, output=lambda x: x
    )
    runner = EchoAttentionRunner(
        attention, 8, offload=True, slots=8, chunk_size=2, cache=caches[0], fused_prefetch=False
    )

    def logits(q, k, weights, scales, start, prefetch=None):
        assert pool._active == (session.owner, 0)
        assert start == 2 and caches[0].indexer_visible_end == 4
        events.append("indexer")
        return torch.arange(len(k)).float().expand(len(q), -1).clone()

    def consume(q, selection, scope):
        assert pool._active == (session.owner, 0)
        assert caches[0].written == 4
        events.append("consume")
        physical = caches[0].ensure(selection.token_ids)
        valid = selection.token_ids >= 0
        torch.testing.assert_close(
            caches[0].records[physical[valid].long(), 0],
            selection.token_ids[valid].bfloat16(),
            rtol=0,
            atol=0,
        )
        return q

    @contextmanager
    def scope(stage):
        if stage == "cache_write":
            assert events[-1] == "prefetch_next" and pool._active == (session.owner, 0)
            events.append("append")
        yield
        if stage == "exact_topk":
            events.append("topk")

    monkeypatch.setattr(indexer, "logits", logits)
    runner._consume = consume
    caches[0].begin_step(2)
    yield SimpleNamespace(
        pool=pool,
        session=session,
        caches=caches,
        helper=helper,
        ticket=ticket,
        runner=runner,
        events=events,
        scope=scope,
    )
    helper.drain()
    if caches[0]._step_end is not None:
        caches[0].rollback()
    session.release()
    pool.close()


@pytest.mark.parametrize("callback_failure", [False, True])
def test_dense_selection_releases_lease_before_wait_and_next_layer_prefetch(
    dense_attention, callback_failure
):
    state = dense_attention
    error = RuntimeError("next layer scheduling failed")

    def ready():
        assert state.events == ["project", "indexer", "topk"]
        assert state.pool._active is None
        assert state.caches[0].written == 2  # No main-KV append before the wait.
        state.helper.wait(state.ticket)
        state.events.append("wait")
        state.helper.prefetch(state.caches[1])  # Acquires a different real pool layer lease.
        state.events.append("prefetch_next")
        assert state.pool._active is None
        if callback_failure:
            raise error

    if callback_failure:
        with pytest.raises(RuntimeError) as raised:
            state.runner.forward(torch.zeros(2, 2), scope=state.scope, before_kv_consume=ready)
        assert raised.value is error
        assert state.caches[0].written == 2 and len(state.helper._tickets) == 2
        assert state.events == ["project", "indexer", "topk", "wait", "prefetch_next"]
    else:

        def capture(*args):
            assert state.caches[0].written == 4 and state.caches[0].all_history_resident
            state.events.append("capture")

        state.runner.capture_hook = capture
        result = state.runner.forward(torch.zeros(2, 2), scope=state.scope, before_kv_consume=ready)
        assert result.shape == (2, 1, 1)
        assert state.events == [
            "project",
            "indexer",
            "topk",
            "wait",
            "prefetch_next",
            "append",
            "capture",
            "consume",
        ]
    assert state.pool._active is None and state.pool._depth == 0


@pytest.mark.parametrize("invalid", ["fused", "diagnostics", "not_callable"])
def test_deferred_wait_rejects_unsafe_runner_modes_before_projection(dense_attention, invalid):
    state = dense_attention
    callback = lambda: None
    if invalid == "fused":
        state.runner.fused_prefetch = True
    elif invalid == "diagnostics":
        state.runner.collect_cache_diagnostics = True
    else:
        callback = 0
    with pytest.raises((TypeError, ValueError)):
        state.runner.forward(torch.zeros(2, 2), before_kv_consume=callback)
    assert not state.events and state.pool._active is None


def scheduling_model(monkeypatch, *, history=2, method="dense_prefetch"):
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    events = []
    device = torch.device("cpu")

    class Runner:
        attention = object()

        def __init__(self, layer):
            self.layer = layer

        def forward(self, hidden, *, scope=None, before_kv_consume=None):
            events.append(("selection", self.layer))
            if before_kv_consume is not None:
                before_kv_consume()
            events.append(("consume", self.layer))
            return hidden

    class Block:
        def __init__(self, layer):
            self.attention, self.cache = Runner(layer), SimpleNamespace(layer_id=layer)

        def forward(self, hidden, residual, *, scope=None, attention=None):
            runner = self.attention if attention is None else attention
            return runner.forward(hidden, scope=scope), torch.zeros_like(hidden)

    class Helper:
        def prefetch(self, cache):
            events.append(("prefetch", cache.layer_id))
            return cache.layer_id

        def wait(self, ticket):
            events.append(("wait", ticket))

        def drain(self):
            events.append(("drain", None))

    model = object.__new__(DeepSeekEchoModel)
    model.devices = [device]
    model.placement = [device] * 3
    model.blocks = [Block(layer) for layer in range(3)]
    model._pool_prefetch = {device: Helper()}
    model.cache_method, model.length = method, history
    model.embedding_weight = torch.ones(8, 2, dtype=torch.bfloat16)
    model.final_norm = torch.ones(2)
    model.head_weight = torch.ones(8, 2, dtype=torch.bfloat16)
    model.cfg = SimpleNamespace(norm_eps=1e-6)
    return model, events


@pytest.mark.parametrize("deferred", [False, True])
def test_model_lookahead_starts_next_copy_at_selected_consumption_boundary(monkeypatch, deferred):
    model, events = scheduling_model(monkeypatch)
    result = model._execute_gpu_body(
        torch.tensor([1, 2]), 2, use_compute_graphs=False, dense_history_after_selection=deferred
    )
    assert result.shape == (1, 8)
    expected = [("prefetch", 0)]
    for layer in range(3):
        if deferred:
            expected.append(("selection", layer))
        expected.append(("wait", layer))
        if layer < 2:
            expected.append(("prefetch", layer + 1))
        if not deferred:
            expected.append(("selection", layer))
        expected.append(("consume", layer))
    assert events == [*expected, ("drain", None)]


@pytest.mark.parametrize("invalid", ["prefill", "method", "islands", "chunked", "layers"])
def test_private_late_wait_gate_rejects_other_execution_paths(monkeypatch, invalid):
    model, events = scheduling_model(monkeypatch)
    options = {"use_compute_graphs": False, "dense_history_after_selection": True}
    chunk = 2
    if invalid == "prefill":
        model.length = 0
    elif invalid == "method":
        model.cache_method = "serial_sparse"
    elif invalid == "islands":
        options["use_compute_graphs"] = True
    elif invalid == "chunked":
        chunk = 1
    else:
        model.blocks.pop()
    with pytest.raises(ValueError, match="direct three-layer full extend"):
        model._execute_gpu_body(torch.tensor([1, 2]), chunk, **options)
    assert not events
