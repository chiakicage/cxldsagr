import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v32.model import DeepSeekEchoModel, plan_layer_devices


def test_contiguous_placement_respects_unequal_memory_budgets():
    mapping = plan_layer_devices([1, 1, 8, 8, 8], [0, 1, 2], [3, 16, 8])
    assert mapping == [0, 0, 1, 1, 2]
    assert plan_layer_devices([2, 2], [7], [4]) == [7, 7]


def test_impossible_placement_fails_without_partial_model():
    with pytest.raises(RuntimeError, match="complete FP8"):
        plan_layer_devices([5, 5, 5], [0, 1], [9, 9])


@pytest.fixture
def constructor_fixture(monkeypatch, tmp_path):
    """Track checkpoint access and placement without constructing CUDA blocks."""
    config = SimpleNamespace(
        num_hidden_layers=61,
        max_seq_len=128,
        index_topk=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
    )
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    names += [f"model.layers.{layer}.input_layernorm.weight" for layer in range(61)]
    reader = SimpleNamespace(
        tensor_files=dict.fromkeys(names, "model.safetensors"),
        tensor_metadata={name: {"data_offsets": [0, 16]} for name in names},
        get_tensor=lambda _name: SimpleNamespace(to=lambda _device: None),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": reader.tensor_files})
    )
    loaded = []

    def block(_path, layer, _device, **_kwargs):
        loaded.append(layer)
        return SimpleNamespace()

    monkeypatch.setattr("models.deepseek_v32.model.Config.from_checkpoint", lambda _: config)
    monkeypatch.setattr("models.deepseek_v32.model.CheckpointReader", lambda _: reader)
    monkeypatch.setattr("models.deepseek_v32.layers.CheckpointBlock", block)
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (80 * 2**30, 80 * 2**30))
    monkeypatch.setattr(DeepSeekEchoModel, "synchronize", lambda _: None)
    return tmp_path, reader, loaded


@pytest.mark.parametrize("num_layers", [None, 3])
def test_constructor_loads_exactly_selected_checkpoint_prefix(constructor_fixture, num_layers):
    path, _, loaded = constructor_fixture
    model = DeepSeekEchoModel(path, devices=[0], capacity=16, num_layers=num_layers)
    expected_layers = 61 if num_layers is None else num_layers
    assert loaded == list(range(expected_layers))
    assert model.num_layers == len(model.blocks) == len(model.placement) == expected_layers
    assert model.cfg.num_hidden_layers == 61


def test_partial_model_allows_missing_unselected_layers_but_checks_selected(constructor_fixture):
    path, reader, loaded = constructor_fixture
    del reader.tensor_files["model.layers.60.input_layernorm.weight"]
    DeepSeekEchoModel(path, devices=[0], capacity=16, num_layers=3)
    assert loaded == [0, 1, 2]
    with pytest.raises(ValueError, match="incomplete checkpoint.*model.layers.60"):
        DeepSeekEchoModel(path, devices=[0], capacity=16)
    del reader.tensor_files["model.layers.2.input_layernorm.weight"]
    with pytest.raises(ValueError, match="incomplete checkpoint.*model.layers.2"):
        DeepSeekEchoModel(path, devices=[0], capacity=16, num_layers=3)


@pytest.mark.parametrize("num_layers", [0, -1, 62, True, 3.5])
def test_invalid_layer_limit_fails_before_loading(constructor_fixture, num_layers):
    path, _, loaded = constructor_fixture
    with pytest.raises(ValueError, match="num_layers"):
        DeepSeekEchoModel(path, devices=[0], capacity=16, num_layers=num_layers)
    assert loaded == []


class FakeCache:
    """Track committed history separately from speculative appended rows."""

    def __init__(self):
        self.length = self.written = 0
        self._step_end = None
        self.rows = []
        self.commits = self.rollbacks = 0

    def begin_step(self, count):
        if self._step_end is not None:
            raise RuntimeError("cache step already active")
        self._step_end = self.length + count
        self.written = self.length

    def append(self, rows):
        assert self._step_end is not None and self.written + len(rows) <= self._step_end
        self.rows.extend(row.clone() for row in rows.float())
        self.written += len(rows)

    def commit(self):
        assert self.written == self._step_end
        self.length = self.written
        self._step_end = None
        self.commits += 1

    def rollback(self):
        self.rows = self.rows[: self.length]
        self.written = self.length
        self._step_end = None
        self.rollbacks += 1


class FakeCausalBlock:
    """CPU causal mixer whose output depends on its own full layer history."""

    def __init__(self, layer, call_log):
        self.layer = layer
        self.call_log = call_log
        self.cache = FakeCache()
        self.calls = 0
        self.fail_on_call = None
        self.omit_append = False

    def forward(self, hidden, residual=None, *, scope=None):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected layer failure")
        start = self.cache.written
        self.call_log.append((self.layer, start, len(hidden), self.cache.length))
        combined = hidden.float() if residual is None else hidden.float() + residual.float()
        past = torch.stack(self.cache.rows) if self.cache.rows else combined[:0]
        history = torch.cat((past, combined))
        prefix_mean = (
            history.cumsum(0)[start:] / torch.arange(start + 1, start + len(hidden) + 1)[:, None]
        )
        if not self.omit_append:
            self.cache.append(combined)
        output = (combined * 0.75 + prefix_mean * 0.125 + (self.layer + 1) * 0.001).bfloat16()
        return output, (combined * 0.125).bfloat16()


@pytest.fixture
def fake_model_factory(monkeypatch):
    # Keep the real full-model scheduling, endpoint arithmetic, and transaction
    # implementation; only CUDA contexts and heavyweight transformer blocks vary.
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())

    def make(*, layers=3, chunk_size=2):
        model = object.__new__(DeepSeekEchoModel)
        model.devices = [torch.device("cpu")]
        model.placement = model.devices * layers
        model.capacity = 32
        model.length = 0
        model.chunk_size = chunk_size
        model.cfg = SimpleNamespace(vocab_size=19, norm_eps=1e-6)
        generator = torch.Generator().manual_seed(87)
        model.embedding_weight = torch.randn(19, 8, generator=generator).bfloat16()
        model.final_norm = torch.randn(8, generator=generator).bfloat16()
        model.head_weight = torch.randn(19, 8, generator=generator).bfloat16()
        model.call_log = []
        model.blocks = [FakeCausalBlock(layer, model.call_log) for layer in range(layers)]
        model.sync_observations = []

        def synchronize():
            model.sync_observations.append(
                [(block.cache.length, block.cache.written) for block in model.blocks]
            )

        model.synchronize = synchronize
        return model

    return make


def test_chunked_prefill_and_extend_preserve_causal_history_and_logit_selection(fake_model_factory):
    chunked = fake_model_factory(chunk_size=2)
    unchunked = fake_model_factory(chunk_size=32)
    last_only = fake_model_factory(chunk_size=2)
    prefix, extend = [1, 4, 2, 7, 3], [5, 8, 6]
    for ids in (prefix, extend):
        expected = unchunked.forward(ids, all_logits=True)
        actual = chunked.forward(ids, all_logits=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(last_only.forward(ids), expected[-1:], rtol=0, atol=0)
    assert chunked.length == unchunked.length == last_only.length == 8
    assert all(block.cache.length == block.cache.written == 8 for block in chunked.blocks)
    assert chunked.sync_observations == [[(0, 5)] * 3, [(5, 8)] * 3]
    assert all(block.cache.commits == 2 for block in chunked.blocks)


def test_graph_completion_failure_poison_model_after_successful_transaction(fake_model_factory):
    model = fake_model_factory()
    failure = RuntimeError("completion event failed")

    class Bank:
        failed = False
        eager_fallbacks = 0

        def supports(self, *args):
            return False

        @contextmanager
        def execution(self):
            yield
            self.failed = True
            raise failure

    bank = model._compute_graphs = Bank()
    with pytest.raises(RuntimeError) as raised:
        model.forward([1, 2])
    assert raised.value is failure
    assert model._poisoned and model._compute_graphs is bank
    assert all(block.cache.commits == 1 for block in model.blocks)
    with pytest.raises(RuntimeError, match="poisoned"):
        model.set_cache_mode(False)


def test_cold_eviction_drain_failure_disables_storage_reuse(fake_model_factory):
    model = fake_model_factory()
    model.offload, model._poisoned = True, False
    failure = RuntimeError("eviction drain failed")

    def fail():
        raise failure

    model.synchronize = fail
    with pytest.raises(RuntimeError) as raised:
        model.evict_prefix_residency()
    assert raised.value is failure and model._poisoned
    assert all(block.cache.commits == block.cache.rollbacks == 0 for block in model.blocks)


def test_model_close_retains_graph_and_cache_owners_when_drain_fails(fake_model_factory):
    model = fake_model_factory()
    bank = model._compute_graphs = object()
    pool = model._shared_pools = {"retained": object()}
    failure = RuntimeError("close drain failed")

    def fail():
        raise failure

    model.synchronize = fail
    with pytest.raises(RuntimeError) as raised:
        model.close()
    assert raised.value is failure and model._poisoned
    assert model._compute_graphs is bank and model._shared_pools is pool


@pytest.mark.parametrize("state", ["poisoned", "closed"])
@pytest.mark.parametrize(
    "entry,args",
    [
        ("forward", ([1, 2],)),
        ("prepare_compute_graphs", ([2],)),
        ("evict_prefix_residency", ()),
        ("snapshot_prefix", ()),
        ("restore_prefix", ({},)),
        ("set_cache_mode", (True,)),
        ("set_cache_method", ("echo",)),
        ("_allocate_shared_caches", ()),
    ],
)
def test_unusable_model_rejects_cache_and_graph_work_before_touching_owners(
    fake_model_factory, state, entry, args
):
    model = fake_model_factory()
    model._poisoned, model._closed = state == "poisoned", state == "closed"
    model.offload = True

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unusable model touched an execution or cache owner")

    model.synchronize = unexpected
    bank = model._compute_graphs = SimpleNamespace(execution=unexpected)
    pool = model._shared_pools = {"retained": object()}
    sessions = model._shared_sessions = {"retained": object()}
    helper = model._pool_prefetch = {"retained": object()}
    caches = [block.cache for block in model.blocks]
    with pytest.raises(RuntimeError, match=state):
        getattr(model, entry)(*args)
    assert model._compute_graphs is bank
    assert model._shared_pools is pool and model._shared_sessions is sessions
    assert model._pool_prefetch is helper
    assert [block.cache for block in model.blocks] == caches
    assert all(cache.commits == cache.rollbacks == cache.written == 0 for cache in caches)


def test_normal_model_close_releases_cache_references_after_all_owners_drain(fake_model_factory):
    model = fake_model_factory()
    events = []
    model._poisoned = model._closed = False
    model.synchronize = lambda: events.append("synchronize")
    model._compute_graphs = SimpleNamespace(close=lambda: events.append("graph"))
    model._pool_prefetch = {"cpu": SimpleNamespace(close=lambda: events.append("prefetch"))}
    model._shared_sessions = {"cpu": SimpleNamespace(release=lambda: events.append("session"))}
    model._shared_pools = {"cpu": SimpleNamespace(close=lambda: events.append("pool"))}

    model.close()

    assert events == ["synchronize", "graph", "prefetch", "session", "pool"]
    assert model._closed and not model._poisoned
    assert model._compute_graphs is None
    assert model._pool_prefetch == model._shared_sessions == model._shared_pools == {}
    assert all(block.cache is None and block.attention is None for block in model.blocks)
    model.close()
    assert events == ["synchronize", "graph", "prefetch", "session", "pool"]


@pytest.mark.parametrize("all_logits", [False, True])
def test_hidden_output_contains_every_token_and_preserves_logits(fake_model_factory, all_logits):
    chunked = fake_model_factory(chunk_size=2)
    unchunked = fake_model_factory(chunk_size=32)
    logits_model = fake_model_factory(chunk_size=2)
    for ids in ([1, 4, 2, 7, 3], [5, 8, 6]):
        actual = chunked.forward(ids, return_hidden=True, all_logits=all_logits)
        expected = unchunked.forward(ids, return_hidden=True, all_logits=all_logits)
        all_expected_logits = logits_model.forward(ids, all_logits=True)
        assert actual["hidden"].shape == (len(ids), 8)
        assert actual["hidden"].dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.nn.functional.linear(actual["hidden"], chunked.head_weight).float(),
            all_expected_logits,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual["logits"],
            all_expected_logits if all_logits else all_expected_logits[-1:],
            rtol=0,
            atol=0,
        )
    assert chunked.length == unchunked.length == logits_model.length == 8


def test_each_chunk_executes_all_61_layers_before_next_chunk(fake_model_factory):
    model = fake_model_factory(layers=61, chunk_size=2)
    output = model.forward([1, 2, 3, 4, 5])
    expected = [
        (layer, start, count, 0) for start, count in ((0, 2), (2, 2), (4, 1)) for layer in range(61)
    ]
    assert model.call_log == expected
    assert model.sync_observations == [[(0, 5)] * 61]
    assert output.shape == (1, 19)
    assert all(block.cache.commits == 1 for block in model.blocks)


def test_failed_begin_rolls_back_only_steps_owned_by_current_call(fake_model_factory):
    model = fake_model_factory()
    model.blocks[1].cache.begin_step(1)
    with pytest.raises(RuntimeError, match="already active"):
        model.forward([1, 2, 3])
    first, existing, untouched = (block.cache for block in model.blocks)
    assert first._step_end is None and first.rollbacks == 1
    assert existing._step_end == 1 and existing.rollbacks == 0
    assert untouched._step_end is None and untouched.rollbacks == 0
    assert model.length == 0 and not model.call_log


@pytest.mark.parametrize("failure", ["layer", "synchronize", "incomplete_write"])
def test_failed_model_step_preserves_prefix_and_can_retry(fake_model_factory, failure):
    model = fake_model_factory()
    expected_model = fake_model_factory()
    model.extend_chunk_size = expected_model.extend_chunk_size = 2
    prefix, extend = [1, 2, 3], [4, 5, 6]
    model.forward(prefix)
    expected_model.forward(prefix)
    saved = [torch.stack(block.cache.rows).clone() for block in model.blocks]
    synchronize = model.synchronize
    if failure == "layer":
        # Fail after all layers have appended the first extend chunk, and a
        # previous layer has already appended part of the second chunk.
        model.blocks[1].fail_on_call = model.blocks[1].calls + 2
    elif failure == "synchronize":
        attempts = 0

        def failed_synchronize():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected recoverable synchronization failure")
            synchronize()

        model.synchronize = failed_synchronize
    else:
        model.blocks[1].omit_append = True
    with pytest.raises(RuntimeError):
        model.forward(extend)
    assert model.length == len(prefix)
    for block, committed in zip(model.blocks, saved, strict=True):
        assert block.cache.length == block.cache.written == len(prefix)
        assert block.cache._step_end is None
        assert block.cache.commits == 1 and block.cache.rollbacks == 1
        torch.testing.assert_close(torch.stack(block.cache.rows), committed, rtol=0, atol=0)
    model.blocks[1].fail_on_call = None
    model.blocks[1].omit_append = False
    model.synchronize = synchronize
    actual = model.forward(extend, all_logits=True)
    expected = expected_model.forward(extend, all_logits=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unrecoverable_cuda_drain_poison_prevents_retry(fake_model_factory):
    model = fake_model_factory()
    model.forward([1, 2, 3])

    def failed_synchronize():
        raise RuntimeError("permanent CUDA failure")

    model.synchronize = failed_synchronize
    with pytest.raises(ExceptionGroup) as failed:
        model.forward([4, 5])
    assert len(failed.value.exceptions) == 2
    assert all("permanent CUDA" in str(error) for error in failed.value.exceptions)
    assert model._poisoned
    assert model.length == 3
    assert all(block.cache.commits == 1 for block in model.blocks)
    with pytest.raises(RuntimeError, match="poisoned"):
        model.forward([4, 5])


@pytest.mark.parametrize("drain_type", [OSError, KeyboardInterrupt])
def test_model_failed_drain_preserves_original_and_pending_cache(fake_model_factory, drain_type):
    model = fake_model_factory()
    model.forward([1, 2, 3])
    body, drain = ValueError("model execution failed"), drain_type("device drain failed")

    def fail_body(*args, **kwargs):
        raise body

    def fail_drain():
        raise drain

    model.blocks[0].forward = fail_body
    model.synchronize = fail_drain
    retained = [block.cache for block in model.blocks]
    with pytest.raises(BaseExceptionGroup) as failed:
        model.forward([4, 5])
    assert failed.value.exceptions == (body, drain)
    assert model._poisoned and model.length == 3
    assert [block.cache for block in model.blocks] == retained
    assert all(cache._step_end == 5 and cache.rollbacks == 0 for cache in retained)
    with pytest.raises(RuntimeError, match="poisoned"):
        model.forward([4])
    with pytest.raises(RuntimeError, match="poisoned"):
        model.set_cache_mode(False)


def test_model_collects_all_rollback_and_offset_failures(fake_model_factory):
    model = fake_model_factory()
    body = ValueError("model body failed")
    rollbacks = [OSError("rollback 0"), OSError("rollback 2")]
    restores = [RuntimeError(f"offset {layer}") for layer in range(3)]
    attempted = []

    class Offset:
        def __init__(self, layer):
            self.layer = layer

        def clone(self):
            return self

        def copy_(self, saved):
            assert saved is self
            attempted.append(("offset", self.layer))
            raise restores[self.layer]

    def fail_body(*args, **kwargs):
        raise body

    model.blocks[0].forward = fail_body
    for layer, block in enumerate(model.blocks):
        block.attention = SimpleNamespace(offset=Offset(layer))
        original = block.cache.rollback

        def rollback(layer=layer, original=original):
            attempted.append(("rollback", layer))
            if layer != 1:
                raise rollbacks[layer // 2]
            original()

        block.cache.rollback = rollback
    with pytest.raises(ExceptionGroup) as failed:
        model.forward([1, 2])
    assert failed.value.exceptions == (body, *rollbacks, *restores)
    assert attempted == [(kind, layer) for kind in ("rollback", "offset") for layer in range(3)]
    assert model._poisoned and model.length == 0
    assert model.blocks[1].cache.rollbacks == 1


def test_model_synchronizes_all_devices_and_preserves_each_error(monkeypatch):
    model = object.__new__(DeepSeekEchoModel)
    model.devices = [0, 1, 2]
    errors = [OSError("device 0"), KeyboardInterrupt("device 2")]
    observed = []

    def synchronize(device):
        observed.append(device)
        if device != 1:
            raise errors[device // 2]

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with pytest.raises(BaseExceptionGroup) as failed:
        model.synchronize()
    assert observed == model.devices
    assert failed.value.exceptions == tuple(errors)


@pytest.mark.parametrize("offload", [False, True])
def test_cache_hbm_budget_fails_before_any_layer_cache_allocation(constructor_fixture, offload):
    from cache.prefix_pool import CacheBudgetExceeded

    path, _, loaded = constructor_fixture
    with pytest.raises(CacheBudgetExceeded, match="cache needs"):
        DeepSeekEchoModel(
            path,
            devices=[0],
            capacity=16,
            num_layers=3,
            offload=offload,
            hbm_cache_budget_bytes=1,
        )
    assert loaded == []


def test_host_arena_dram_budget_fails_before_gpu_or_host_pool_allocation(
    constructor_fixture, monkeypatch
):
    from cache.prefix_pool import CacheBudgetExceeded
    from cache.sparse_token_pool import SharedSparseTokenPool

    path, _, loaded = constructor_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated a pool before checking DRAM budget")

    monkeypatch.setattr(SharedSparseTokenPool, "__init__", forbidden)
    with pytest.raises(CacheBudgetExceeded, match="DRAM bytes"):
        DeepSeekEchoModel(
            path,
            devices=[0],
            capacity=16,
            num_layers=3,
            offload=True,
            dram_cache_budget_bytes=1,
        )
    assert loaded == []


@pytest.mark.parametrize("failure_stage", ["pool", "session", "layer_view"])
def test_later_device_cache_allocation_failure_releases_earlier_resources(
    monkeypatch, failure_stage
):
    import cache.sparse_token_pool as cache_module

    model = object.__new__(DeepSeekEchoModel)
    model.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    model.placement = model.devices
    model.capacity, model.host_arena_tokens, model.slots = 32, 64, 4
    model.cfg = SimpleNamespace(kv_lora_rank=2, qk_rope_head_dim=2)
    model._shared_pools, model._shared_sessions = {}, {}
    model._poisoned = False
    pools, sessions = [], []
    fail = True

    class Session:
        def __init__(self, device):
            self.device = device
            self.released = False
            sessions.append(self)

        def layer(self, index):
            if fail and self.device.index == 1 and failure_stage == "layer_view":
                raise RuntimeError("later device allocation failed")
            return (self.device, index)

        def release(self):
            self.released = True

    class Pool:
        def __init__(self, *args, device, **kwargs):
            if fail and device.index == 1 and failure_stage == "pool":
                raise RuntimeError("later device allocation failed")
            self.device, self.closed = device, False
            pools.append(self)

        def allocate_session(self, capacity):
            if fail and self.device.index == 1 and failure_stage == "session":
                raise RuntimeError("later device allocation failed")
            return Session(self.device)

        def close(self):
            self.closed = True

    monkeypatch.setattr(cache_module, "SharedSparseTokenPool", Pool)
    with pytest.raises(RuntimeError, match="later device allocation"):
        model._allocate_shared_caches()
    assert pools and all(pool.closed for pool in pools)
    assert sessions and all(session.released for session in sessions)
    assert model._shared_pools == model._shared_sessions == {}
    assert not model._poisoned
    # Fully cleaned allocation failure may be retried without leaked ownership.
    fail = False
    allocated = model._allocate_shared_caches()
    assert set(allocated) == {0, 1}
    assert set(model._shared_pools) == set(model.devices)
    model._release_shared_caches()


@pytest.mark.parametrize("cleanup_type", [None, OSError, KeyboardInterrupt])
def test_failed_later_cache_runner_replacement_cleans_pools_and_poisons_model(
    monkeypatch, cleanup_type
):
    import models.deepseek_v32.attention as attention_module

    model = object.__new__(DeepSeekEchoModel)
    device = torch.device("cpu")
    model.devices = [device]
    model.capacity, model.slots, model.chunk_size = 32, 4, 4
    model.length, model.offload = 8, False
    model._cache_generation = 3
    model._poisoned = False
    model.synchronize = lambda: None
    model._plan_cache_resources = lambda offload: {"offload": offload}
    old_caches = [object() for _ in range(3)]
    model.blocks = [
        SimpleNamespace(
            attention=SimpleNamespace(attention=SimpleNamespace(device=device)), cache=cache
        )
        for cache in old_caches
    ]
    released = []
    closed = []
    body = RuntimeError("later indexer cache allocation failed")
    cleanup = None if cleanup_type is None else cleanup_type("replacement release failed")
    model._shared_pools, model._shared_sessions = {}, {}

    def release():
        released.append(True)
        if cleanup is not None:
            raise cleanup

    def allocate_replacements():
        assert all(block.attention is block.cache is None for block in model.blocks)
        model._shared_pools[device] = SimpleNamespace(close=lambda: closed.append(True))
        model._shared_sessions[device] = SimpleNamespace(release=release)
        return {layer: object() for layer in range(3)}

    model._allocate_shared_caches = allocate_replacements
    created = []

    def runner(*args, cache, **kwargs):
        if created:
            raise body
        result = SimpleNamespace(cache=cache)
        created.append(result)
        return result

    monkeypatch.setattr(attention_module, "EchoAttentionRunner", runner)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    if cleanup is None:
        with pytest.raises(RuntimeError) as failed:
            model.set_cache_mode(True)
        assert failed.value is body
        assert released == closed == [True]
        assert model._shared_pools == model._shared_sessions == {}
    else:
        with pytest.raises(BaseExceptionGroup) as failed:
            model.set_cache_mode(True)
        assert failed.value.exceptions == (body, cleanup)
        assert released == [True] and closed == []
        assert device in model._shared_pools and device in model._shared_sessions
    assert model._cache_generation == 4
    assert model._poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        model.forward([1])


def test_cache_allocation_and_release_errors_keep_original_objects_and_storage(monkeypatch):
    import cache.sparse_token_pool as cache_module

    model = object.__new__(DeepSeekEchoModel)
    device = torch.device("cuda:0")
    model.devices = model.placement = [device]
    model.capacity, model.host_arena_tokens, model.slots = 32, 64, 4
    model.cfg = SimpleNamespace(kv_lora_rank=2, qk_rope_head_dim=2)
    model._shared_pools, model._shared_sessions = {}, {}
    model._poisoned = False
    body, cleanup = ValueError("layer view failed"), KeyboardInterrupt("session release failed")
    releases = []

    class Session:
        def layer(self, index):
            raise body

        def release(self):
            releases.append(self)
            raise cleanup

    class Pool:
        def __init__(self, *args, **kwargs):
            self.session = Session()

        def allocate_session(self, capacity):
            return self.session

        def close(self):
            raise AssertionError("failed session still owns this pool")

    monkeypatch.setattr(cache_module, "SharedSparseTokenPool", Pool)
    with pytest.raises(BaseExceptionGroup) as failed:
        model._allocate_shared_caches()
    assert failed.value.exceptions == (body, cleanup)
    assert model._poisoned
    pool = model._shared_pools[device]
    assert model._shared_sessions[device] is pool.session
    assert releases == [pool.session]
    with pytest.raises(RuntimeError, match="poisoned"):
        model._allocate_shared_caches()
    assert releases == [pool.session]


def test_shared_cache_release_attempts_independent_devices_and_retains_failed_owners():
    model = object.__new__(DeepSeekEchoModel)
    model._poisoned = False
    errors = [OSError("session 0"), ValueError("session 2"), KeyboardInterrupt("pool 1")]
    calls = []

    def release(device):
        calls.append(("session", device))
        if device != 1:
            raise errors[device // 2]

    def close(device):
        calls.append(("pool", device))
        assert device == 1
        raise errors[2]

    model._shared_sessions = {
        device: SimpleNamespace(release=lambda device=device: release(device))
        for device in range(3)
    }
    model._shared_pools = {
        device: SimpleNamespace(close=lambda device=device: close(device)) for device in range(3)
    }
    with pytest.raises(BaseExceptionGroup) as failed:
        model._release_shared_caches()
    assert failed.value.exceptions == tuple(errors)
    assert calls == [("session", 0), ("session", 1), ("session", 2), ("pool", 1)]
    assert set(model._shared_sessions) == {0, 2} and set(model._shared_pools) == {0, 1, 2}
    assert model._poisoned


def test_single_query_capacity_is_checked_before_loading_checkpoint_layers(constructor_fixture):
    path, _, loaded = constructor_fixture
    with pytest.raises(ValueError, match="exact selection"):
        DeepSeekEchoModel(
            path, devices=[0], capacity=16, num_layers=3, offload=True, slots=8, chunk_size=4
        )
    assert loaded == []


def test_small_pool_mode_change_preserves_existing_resident_prefix(fake_model_factory):
    model, oracle = fake_model_factory(), fake_model_factory()
    prefix, candidate = [1, 2, 3], [4, 5]
    model.forward(prefix)
    oracle.forward(prefix)
    model.cfg.index_topk = 16
    model.slots, model.offload, model._poisoned = 8, False, False
    model._cache_generation = 7
    resource_plan = model._cache_resource_plan = {"resident": True}
    pools = model._shared_pools = {}
    sessions = model._shared_sessions = {}
    caches = [block.cache for block in model.blocks]
    contents = [torch.stack(cache.rows).clone() for cache in caches]
    syncs = len(model.sync_observations)
    with pytest.raises(ValueError, match="full exact selection"):
        model.set_cache_mode(True)
    assert model._cache_generation == 7 and not model._poisoned and not model.offload
    assert model._cache_resource_plan is resource_plan
    assert model._shared_pools is pools and model._shared_sessions is sessions
    assert len(model.sync_observations) == syncs
    assert model.length == len(prefix)
    for block, cache, expected in zip(model.blocks, caches, contents, strict=True):
        assert block.cache is cache and cache.length == cache.written == len(prefix)
        torch.testing.assert_close(torch.stack(cache.rows), expected, rtol=0, atol=0)
    # A rejected offload admission must leave the current resident model usable.
    torch.testing.assert_close(model.forward(candidate), oracle.forward(candidate), rtol=0, atol=0)


def test_extend_mlp_geometry_uses_outer_batch_not_prefill_chunk(fake_model_factory):
    model = fake_model_factory(chunk_size=2)
    model.forward([1, 2, 3])
    model.forward([4, 5, 6, 7, 8])
    assert all(block.chunk_size == 5 for block in model.blocks)
    model.extend_chunk_size = 3
    model.forward([9, 10, 11, 12])
    assert all(block.chunk_size == 1 for block in model.blocks)  # Actual final tail.


@pytest.mark.parametrize("oversized", ["prefill", "extend", "workspace"])
def test_oversized_query_mode_change_preserves_resident_prefix(fake_model_factory, oversized):
    model, oracle = fake_model_factory(), fake_model_factory()
    prefix, candidate = [1, 2, 3], [4, 5]
    model.forward(prefix)
    oracle.forward(prefix)
    model.cfg.index_topk = 4  # A single exact selection fits; only Q exceeds P.
    model.slots, model.offload, model._poisoned = 8, False, False
    model.execution_reservation = SimpleNamespace(
        query_tokens=16 if oversized == "workspace" else 8
    )
    if oversized == "prefill":
        model.chunk_size = 16
    elif oversized == "extend":
        model.extend_chunk_size = 16
    model._cache_generation = 9
    resource_plan = model._cache_resource_plan = {"resident": True}
    pools = model._shared_pools = {}
    sessions = model._shared_sessions = {}
    caches = [block.cache for block in model.blocks]
    contents = [torch.stack(cache.rows).clone() for cache in caches]
    syncs = len(model.sync_observations)
    with pytest.raises(ValueError, match="query batch and workspace"):
        model.set_cache_mode(True)
    assert model._cache_generation == 9 and not model._poisoned and not model.offload
    assert model._cache_resource_plan is resource_plan
    assert model._shared_pools is pools and model._shared_sessions is sessions
    assert len(model.sync_observations) == syncs and model.length == len(prefix)
    for block, cache, expected in zip(model.blocks, caches, contents, strict=True):
        assert block.cache is cache and cache.length == cache.written == len(prefix)
        torch.testing.assert_close(torch.stack(cache.rows), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.forward(candidate), oracle.forward(candidate), rtol=0, atol=0)
