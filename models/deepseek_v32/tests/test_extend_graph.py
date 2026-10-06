from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import SparseTokenCache
from models.deepseek_v32.execution.extend_graph import (
    DeepSeekExtendGraph,
    _cache_identity,
    _precision_policy,
    _ResidentRecipe,
    _weight_identity,
)
from models.deepseek_v32.tests.test_echo_infer import fake_model_factory as _fake_model_factory


@pytest.fixture
def fake_model_factory(monkeypatch):
    return _fake_model_factory.__wrapped__(monkeypatch)


def test_resident_recipe_replays_bookkeeping_without_repeating_device_writes():
    cache = SparseTokenCache(8, 4, device="cpu")
    cache.begin_step(2)
    cache.append(torch.ones(2, 4, dtype=torch.bfloat16))
    cache.commit()
    cache.begin_step(3)
    recipe = _ResidentRecipe(cache)
    cache.append(torch.full((3, 4), 7, dtype=torch.bfloat16))
    recipe.finish()
    assert cache.length == cache.written == 2 and cache._step_end == 5
    recipe.validate()
    cache.records[2:5].fill_(9)
    recipe.apply()
    assert cache.length == 2 and cache.written == cache.indexer_visible_end == 5
    assert cache.stats.written_records == 5
    assert torch.equal(cache.records[2:5], torch.full((3, 4), 9, dtype=torch.bfloat16))
    cache.commit()
    with pytest.raises(RuntimeError, match="range changed"):
        recipe.validate(pending=False)


class _ForwardGraph:
    def __init__(self, model, *, failure=None):
        self.model, self.failure = model, failure
        self.failed = self.pending = False
        self.saved_offsets = [None] * len(model.blocks)
        self.events = []
        self.output = torch.tensor([[123.0]])

    def validate(self):
        assert all(block.cache._step_end is None for block in self.model.blocks)
        self.events.append("validate")

    def replay(self, ids, *, scope=None):
        assert all(block.cache._step_end == 5 for block in self.model.blocks)
        self.events.append("replay")
        self.pending = True
        return self.output

    def apply(self):
        assert self.model.sync_observations[-1] == [(2, 2)] * 3
        self.events.append("apply")
        if self.failure is not None:
            self.failed = True
            raise self.failure
        for block in self.model.blocks:
            block.cache.written = block.cache._step_end
        self.pending = False


def test_model_completes_all_graph_effects_after_sync_before_commit(fake_model_factory):
    model = fake_model_factory()
    model.forward([1, 2])
    graph = _ForwardGraph(model)
    model._extend_graphs = {(3, False, False): graph}
    calls = list(model.call_log)
    assert model.forward([4, 5, 6]) is graph.output
    assert graph.events == ["validate", "replay", "apply"]
    assert model.call_log == calls  # No ordinary block invocation during replay.
    assert model.length == 5
    assert all(block.cache.length == 5 and block.cache.commits == 2 for block in model.blocks)


def test_model_explicit_graph_bypass_preserves_ordinary_computation(fake_model_factory):
    model = fake_model_factory()
    model.forward([1, 2])
    graph = _ForwardGraph(model)
    model._extend_graphs = {(3, False, False): graph}
    result = model.forward([4, 5, 6], use_extend_graph=False)
    assert not graph.events and result.shape == (1, 19)
    assert model.length == 5 and all(block.calls == 2 for block in model.blocks)


def test_model_rejects_unprepared_output_mode_before_starting_transaction(fake_model_factory):
    model = fake_model_factory()
    model.forward([1, 2])
    graph = _ForwardGraph(model)
    model._extend_graphs = {(3, False, False): graph}
    with pytest.raises(ValueError, match="query shape and output mode"):
        model.forward([4, 5, 6], return_hidden=True)
    assert not graph.events and model.length == 2
    assert all(block.cache._step_end is None for block in model.blocks)


def test_model_graph_effect_failure_retains_graph_and_prevents_reuse(fake_model_factory):
    model = fake_model_factory()
    model.forward([1, 2])
    failure = RuntimeError("graph host effects failed")
    graph = _ForwardGraph(model, failure=failure)
    model._extend_graphs = {(3, False, False): graph}
    with pytest.raises(RuntimeError) as raised:
        model.forward([4, 5, 6])
    assert raised.value is failure
    assert model._poisoned and model.length == 2 and model._extend_graph is graph
    assert all(block.cache.commits == 1 and block.cache.rollbacks == 1 for block in model.blocks)
    with pytest.raises(RuntimeError, match="poisoned"):
        model.forward([4, 5, 6])


def _identity_model():
    cache = SparseTokenCache(8, 4, device="cpu")
    cache.length = cache.written = cache.indexer_visible_end = 2
    weight = torch.ones(4, 4)
    attention = SimpleNamespace(weight=weight)
    runner = SimpleNamespace(
        attention=attention,
        index_keys=torch.empty(8, 4),
        index_scales=torch.empty(8),
        offset=torch.zeros(16),
        fused_prefetch=False,
    )
    block = SimpleNamespace(
        cache=cache,
        attention=runner,
        post_norm_weight=torch.ones(4),
        mlp=SimpleNamespace(weight=torch.ones(4, 4)),
    )
    return SimpleNamespace(
        devices=[torch.device("cpu")],
        length=2,
        _cache_generation=1,
        cache_method="hbm",
        capacity=8,
        slots=8,
        workspace_query_tokens=3,
        extend_chunk_size=3,
        placement=[torch.device("cpu")],
        blocks=[block],
        embedding_weight=torch.ones(8, 4),
        final_norm=torch.ones(4),
        head_weight=torch.ones(8, 4),
        _check_usable=lambda: None,
    )


@pytest.mark.parametrize("mutation", ["length", "generation", "cache", "weight", "weight_value"])
def test_graph_rejects_changed_capture_identity(mutation):
    model = _identity_model()
    graph = DeepSeekExtendGraph(model, 3)
    graph.allocated = True
    graph.precision = _precision_policy()
    graph.weights = _weight_identity(model)
    graph.cache_identity = _cache_identity(model)
    graph.validate()
    if mutation == "length":
        model.length = 3
    elif mutation == "generation":
        model._cache_generation = 2
    elif mutation == "cache":
        model.blocks[0].cache.records = model.blocks[0].cache.records.clone()
    elif mutation == "weight":
        model.head_weight = model.head_weight.clone()
    else:
        model.head_weight.add_(1)
    with pytest.raises(ValueError, match="changed"):
        graph.validate()


def test_close_drains_and_destroys_graph_before_releasing_io_sources(monkeypatch):
    graph = DeepSeekExtendGraph(_identity_model(), 3)
    calls = []
    graph.graph = SimpleNamespace(reset=lambda: calls.append("destroy"))
    graph.recipes = [SimpleNamespace(close=lambda: calls.append("release_sources"))]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: calls.append("drain"))
    graph.close()
    assert calls == ["drain", "destroy", "release_sources"] and graph.closed


def test_failed_close_retains_graph_and_io_sources(monkeypatch):
    graph = DeepSeekExtendGraph(_identity_model(), 3)
    graph.graph = object()
    graph.recipes = [object()]

    def fail(_):
        raise RuntimeError("drain failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="drain failed"):
        graph.close()
    assert graph.failed and not graph.closed and graph.graph is not None and graph.recipes
