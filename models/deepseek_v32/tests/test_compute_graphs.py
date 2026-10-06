import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from cache.prefix_pool import CacheBudgetExceeded, CacheFootprint
from models.deepseek_v32.config import Config
from models.deepseek_v32.execution.adapter import SCHEMES, DeepSeekServingBackend
from models.deepseek_v32.execution.compute_graphs import (
    DeepSeekComputeGraphs,
    _precision_policy,
    plan_compute_graphs,
)
from models.deepseek_v32.layers import CheckpointBlock
from models.deepseek_v32.nonmatrix import residual_rms_norm
from models.deepseek_v32.tests.test_echo_block import block_checkpoint as block_fixture
from serving.persistent import PersistentGRRunner, token_digest


@pytest.fixture
def graph_checkpoint(tmp_path):
    return block_fixture.__wrapped__(tmp_path)


def test_graph_plan_is_allocation_free_and_has_no_session_multiplier(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("graph planning must not allocate or query CUDA")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    cfg = Config()
    plan = plan_compute_graphs(cfg, 10, 1024, 65536, 128, 12 * 2**30)
    assert plan["compute_graph_query_sizes"] == [128, 1024]
    assert plan["compute_graph_count"] == 40
    # Layers 1/2 borrow both preceding finish outputs. Repeated source layers
    # keep independent activation inputs; positions are shared per query size.
    per_token = 8 * cfg.dim * 2 + 10 * cfg.n_heads * cfg.v_head_dim * 2 + 8 + 4
    per_token += 4 * cfg.dim * 2
    assert plan["compute_graph_static_storage_bytes"] == (1024 + 128) * per_token + 2 * 8
    assert plan["compute_graph_reserved_limit_bytes"] == (
        plan["compute_graph_static_allocation_limit_bytes"] + 12 * 2**30
    )
    tail = plan_compute_graphs(cfg, 10, 1024, 2051, 7, 12 * 2**30)
    assert tail["compute_graph_query_sizes"] == [3, 7, 1024]


def test_graph_private_capacity_counts_inactive_segments(monkeypatch):
    static = torch.zeros(8)
    address = static.untyped_storage().data_ptr()
    bank = object.__new__(DeepSeekComputeGraphs)
    bank.device = torch.device("cuda:0")
    bank.metadata = {
        "compute_graph_private_limit_bytes": 8 * 2**20,
        "compute_graph_static_allocation_limit_bytes": 512,
    }
    bank._shared_inputs = {}
    graph = SimpleNamespace(pool=lambda: (4, 0))
    bank.pairs = {
        (0, 1): SimpleNamespace(
            inputs={"hidden": static},
            owned_inputs={"hidden": static},
            projection_graph=graph,
            finish_graph=graph,
        )
    }
    snapshot = [
        {
            "device": 0,
            "segment_pool_id": (0, 0),
            "total_size": 2**20,
            "blocks": [{"address": address, "size": 512, "state": "active_allocated"}],
        },
        {
            "device": 0,
            "segment_pool_id": (4, 0),
            "total_size": 8 * 2**20,
            "blocks": [
                {"address": 1, "size": 2**20, "state": "active_allocated"},
                {"address": 2, "size": 7 * 2**20, "state": "inactive"},
            ],
        },
    ]
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: snapshot)
    bank._audit_capacity()
    assert bank.shared_bytes() == {"hbm": 8 * 2**20 + 512, "dram": 0}
    assert bank.static_storage_bytes == 32
    bank.metadata["compute_graph_private_limit_bytes"] -= 1
    with pytest.raises(CacheBudgetExceeded, match="private reserved"):
        bank._audit_capacity()


@pytest.mark.parametrize("setting", ["allow_tf32", "allow_bf16_reduced_precision_reduction"])
def test_graph_rejects_precision_changes_before_any_replay(monkeypatch, setting):
    bank = DeepSeekComputeGraphs([object()], [object()], "cuda:0", {})
    bank.allocated = True
    bank.precision_policy = _precision_policy()
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(matmul, setting, not getattr(matmul, setting))
    with pytest.raises(RuntimeError, match="precision policy changed"), bank.execution():
        raise AssertionError("changed precision must fail before execution")
    assert not bank._active and not bank.failed


def test_graph_failed_stream_dependency_retains_storage_and_prevents_reuse(monkeypatch):
    bank = DeepSeekComputeGraphs([object()], [object()], "cuda:0", {})
    bank.allocated = True
    bank.precision_policy = _precision_policy()
    bank.pairs = {(0, 1): object()}
    previous_event = object()
    bank._last_stream, bank._last_event = object(), previous_event

    def fail_wait(event):
        assert event is previous_event
        raise RuntimeError("dependency wait failed")

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(wait_event=fail_wait)
    )
    with pytest.raises(RuntimeError, match="dependency wait failed"), bank.execution():
        raise AssertionError("failed dependency must not permit graph execution")
    assert bank.failed and not bank._active and bank.pairs
    with pytest.raises(RuntimeError, match="closed or failed"), bank.execution():
        pass


def test_graph_failed_close_retains_storage(monkeypatch):
    bank = DeepSeekComputeGraphs([object()], [object()], "cuda:0", {})
    bank.pairs = {(0, 1): object()}

    def fail_sync(device):
        raise RuntimeError("graph consumers did not drain")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    with pytest.raises(RuntimeError, match="did not drain"):
        bank.close()
    assert bank.failed and not bank.closed and bank.pairs


@pytest.mark.parametrize("body_type", [None, ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("completion", ["success", "construct", "record"])
def test_graph_completion_preserves_body_errors_and_retains_failed_owner(
    monkeypatch, body_type, completion
):
    from models.deepseek_v32.tests.test_serving_backend import _model

    owner = object()
    model, session = _model(layers=3, owner=owner)
    bank = DeepSeekComputeGraphs([object()], [object()], "cuda:0", {})
    bank.allocated = True
    bank.precision_policy = _precision_policy()
    retained = object()
    bank.pairs = {(0, 1): retained}
    model._compute_graphs = bank
    current = object()
    body = None if body_type is None else body_type("execution failed")
    cleanup = OSError("completion failed")
    events = []

    def event():
        events.append("construct")
        if completion == "construct":
            raise cleanup

        def record(stream):
            assert stream is current
            events.append("record")
            if completion == "record":
                raise cleanup

        return SimpleNamespace(record=record)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "Event", event)

    def execute():
        with model._execution(session, owner=owner):
            if body is not None:
                raise body

    if body is not None and completion != "success":
        with pytest.raises(BaseExceptionGroup) as failed:
            execute()
        assert failed.value.exceptions == (body, cleanup)
    elif body is not None or completion != "success":
        expected = body if completion == "success" else cleanup
        with pytest.raises(type(expected)) as failed:
            execute()
        assert failed.value is expected
    else:
        execute()
    assert events == (["construct"] if completion == "construct" else ["construct", "record"])
    assert not bank._active and bank._stream is None
    assert bank.pairs[(0, 1)] is retained and not bank.closed
    assert bank.failed is model.lifecycle.poisoned is (completion != "success")
    assert model.lifecycle.admission_owner is owner
    if completion != "success":
        assert model.lifecycle.active_session is session
        with pytest.raises(RuntimeError, match="poisoned"):
            model.release_session(session, owner=owner)
    else:
        assert model.lifecycle.active_session is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("residual_present", [False, True])
@torch.inference_mode()
def test_graph_positions_outputs_and_delayed_owned_writeback(graph_checkpoint, residual_present):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek compute graph validation requires Hopper SM90")
    path, _ = graph_checkpoint
    layer = int(residual_present)
    blocks = [
        CheckpointBlock(path, 0, "cuda", capacity=16, chunk_size=5, linear_backend="bf16")
        for _ in range(layer + 1)
    ]
    block = blocks[layer]
    attention = block.attention.attention
    metadata = plan_compute_graphs(block.cfg, len(blocks), 5, 10, 2, 256 * 2**20)
    bank = DeepSeekComputeGraphs(
        [block.attention.attention for block in blocks], blocks, "cuda", metadata
    )
    stream = torch.cuda.Stream()
    sources, targets = [], []
    cache = SimpleNamespace(offload=True, transient_start=None, written=0)

    class Runner:
        def forward(self, hidden, *, project_callback, output_callback, **kwargs):
            projected = project_callback(hidden, cache.written, normalized=False)
            starts, ends = kwargs["indexer_bounds"]
            torch.testing.assert_close(starts, torch.zeros_like(starts), rtol=0, atol=0)
            torch.testing.assert_close(
                ends,
                torch.arange(
                    cache.written + 1,
                    cache.written + len(hidden) + 1,
                    device=hidden.device,
                    dtype=torch.int32,
                ),
                rtol=0,
                atol=0,
            )
            sources.append(projected.kv)
            if cache.transient_start is None:
                ready = torch.cuda.Event()
                ready.record()
                target = torch.empty_like(projected.kv, device="cpu", pin_memory=True)
                targets.append(target)
                stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    # Delay D2H while the same graph's next replay overwrites
                    # its private outputs on the caller stream.
                    torch.cuda._sleep(20_000_000)
                    target.copy_(projected.kv, non_blocking=True)
                projected.kv.record_stream(stream)
            return output_callback(projected.q[..., : block.cfg.kv_lora_rank])

    runner = Runner()
    runner.attention, runner.cache = attention, cache
    hidden = [torch.randn(5, block.cfg.dim, device="cuda").bfloat16() for _ in range(2)]
    incoming = [torch.randn_like(value) if residual_present else None for value in hidden]
    expected, expected_kv, expected_saved = [], [], []
    for start, value, residual_input in zip((0, 5), hidden, incoming, strict=True):
        normalized, saved = residual_rms_norm(
            value, residual_input, attention.input_norm_weight, block.cfg.norm_eps
        )
        expected_saved.append(saved)
        projected = attention.project(normalized, start, normalized=True)
        output = attention.output(projected.q[..., : block.cfg.kv_lora_rank])
        normalized, residual = residual_rms_norm(
            saved, output, block.post_norm_weight, block.cfg.norm_eps
        )
        expected.append((block.mlp(normalized), residual))
        expected_kv.append(projected.kv.cpu())
    try:
        bank.allocate()
        pair = bank.pairs[layer, 5]
        captured_saved = pair.saved
        saved_address = captured_saved.untyped_storage().data_ptr()
        assert "saved" not in pair.inputs and "attention" not in pair.inputs
        assert pair.inputs["expanded"].shape == (5, block.cfg.n_heads, block.cfg.v_head_dim)
        if residual_present:
            static_addresses = {
                tensor.untyped_storage().data_ptr()
                for graph_pair in bank.pairs.values()
                for tensor in graph_pair.inputs.values()
            }
            assert saved_address not in static_addresses
            owners = [
                segment
                for segment in torch.cuda.memory_snapshot()
                if segment["device"] == bank.device.index
                and any(
                    allocation["address"] == saved_address
                    and allocation["state"] == "active_allocated"
                    for allocation in segment["blocks"]
                )
            ]
            assert len(owners) == 1
            assert tuple(owners[0]["segment_pool_id"]) == pair.projection_graph.pool()
            assert pair.projection_graph.pool() != pair.finish_graph.pool()
        else:
            assert captured_saved is pair.inputs["hidden"]
        graph_pools = {
            graph.pool()
            for graph_pair in bank.pairs.values()
            for graph in (graph_pair.projection_graph, graph_pair.finish_graph)
        }
        assert bank.private_reserved_bytes == sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == bank.device.index
            and tuple(segment["segment_pool_id"]) in graph_pools
        )
        assert bank.static_storage_bytes == metadata["compute_graph_static_storage_bytes"]
        before = bank.shared_bytes()
        observed_saved = []
        with bank.execution():
            first = tuple(
                t.clone() for t in bank.forward_block(layer, runner, hidden[0], incoming[0])
            )
            observed_saved.append(pair.saved.clone())
            cache.written = 5
            second = bank.forward_block(layer, runner, hidden[1], incoming[1])
            observed_saved.append(pair.saved.clone())
        stream.synchronize()
        torch.testing.assert_close(observed_saved, expected_saved, rtol=0, atol=0)
        for actual, reference in zip((first, second), expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for actual, reference in zip(targets, expected_kv, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert sources[0].data_ptr() != sources[1].data_ptr() != pair.projected.kv.data_ptr()
        cache.transient_start = 5
        with bank.execution():
            bank.forward_block(layer, runner, hidden[0], incoming[0])
        assert sources[-1].data_ptr() == pair.projected.kv.data_ptr()
        with torch.cuda.stream(stream), bank.execution():
            switched = bank.forward_block(layer, runner, hidden[1], incoming[1])
        stream.synchronize()
        torch.testing.assert_close(switched, expected[1], rtol=0, atol=0)
        assert pair.saved is captured_saved
        assert pair.saved.untyped_storage().data_ptr() == saved_address
        torch.testing.assert_close(pair.saved, expected_saved[1], rtol=0, atol=0)
        bank._audit_capacity()
        assert bank.shared_bytes() == before
        assert pair.projection_replays == pair.finish_replays == 4
        assert bank.position_updates == 1
        assert not bank.supports(layer, hidden[0][:3], incoming[0])
        with pytest.raises(RuntimeError, match="execution stream"):
            bank.forward_block(layer, runner, hidden[0], incoming[0])
    finally:
        stream.synchronize()
        bank.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_graph_chain_borrows_outputs_without_repacking_or_scalar_rewrites(graph_checkpoint):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("DeepSeek compute graph validation requires Hopper SM90")
    path, _ = graph_checkpoint
    blocks = [
        CheckpointBlock(path, 0, "cuda", capacity=16, chunk_size=5, linear_backend="bf16")
        for _ in range(4)
    ]
    metadata = plan_compute_graphs(blocks[0].cfg, 4, 5, 10, 2, 512 * 2**20)
    bank = DeepSeekComputeGraphs(
        [block.attention.attention for block in blocks], blocks, "cuda", metadata
    )

    class Runner:
        def __init__(self, attention):
            self.attention = attention
            self.cache = SimpleNamespace(offload=False, written=5)

        def forward(self, hidden, *, project_callback, output_callback, **kwargs):
            projected = project_callback(hidden, self.cache.written, normalized=False)
            return output_callback(projected.q[..., : self.attention.cfg.kv_lora_rank])

    runners = [Runner(block.attention.attention) for block in blocks]
    expected, seeds = [], []
    for _ in range(2):
        hidden = torch.randn(5, blocks[0].cfg.dim, device="cuda").bfloat16()
        seeds.append(hidden)
        residual = None
        states = []
        for layer, block in enumerate(blocks):
            if layer == 3:
                hidden, residual = seeds[-1].clone(), None
            attention = block.attention.attention
            normalized, saved = residual_rms_norm(
                hidden, residual, attention.input_norm_weight, block.cfg.norm_eps
            )
            projected = attention.project(normalized, 5, normalized=True)
            attention_output = attention.output(projected.q[..., : block.cfg.kv_lora_rank])
            normalized, residual = residual_rms_norm(
                saved, attention_output, block.post_norm_weight, block.cfg.norm_eps
            )
            hidden = block.mlp(normalized)
            states.append((hidden, residual))
        expected.append(states)
    try:
        bank.allocate()
        for layer in (1, 2):
            pair = bank.pairs[layer, 5]
            assert pair.inputs["hidden"] is bank.pairs[layer - 1, 5].output[0]
            assert pair.inputs["residual"] is bank.pairs[layer - 1, 5].output[1]
        assert bank.pairs[3, 5].inputs["hidden"] is not bank.pairs[0, 5].inputs["hidden"]
        assert len({id(bank.pairs[layer, 5].inputs["start"]) for layer in range(4)}) == 1
        for visit, seed in enumerate(seeds):
            hidden, residual = seed, None
            with bank.execution():
                observed = []
                for layer, runner in enumerate(runners):
                    if layer == 3:
                        hidden, residual = seed, None
                    hidden, residual = bank.forward_block(layer, runner, hidden, residual)
                    observed.append((hidden.clone(), residual.clone()))
            torch.testing.assert_close(observed, expected[visit], rtol=0, atol=0)
        assert bank.input_copies == 4  # Initial embedding and independent repeated input only.
        assert bank.position_updates == 1  # All layers and the same-position revisit share it.
        bank._audit_capacity()
        assert bank.static_storage_bytes == metadata["compute_graph_static_storage_bytes"]
    finally:
        bank.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_graph_failed_capacity_check_retains_partial_bank_until_close(graph_checkpoint):
    path, _ = graph_checkpoint
    block = CheckpointBlock(path, 0, "cuda", capacity=16, chunk_size=5, linear_backend="bf16")
    bank = DeepSeekComputeGraphs(
        [block.attention.attention],
        [block],
        "cuda",
        plan_compute_graphs(block.cfg, 1, 5, 10, 2, 1),
    )
    try:
        with pytest.raises(CacheBudgetExceeded, match="private reserved capacity"):
            bank.allocate()
        assert bank.failed and bank.pairs and bank.private_reserved_bytes > 1
        assert not bank.allocated
    finally:
        bank.close()
    assert bank.closed and not bank.pairs and bank.shared_bytes() == {"hbm": 0, "dram": 0}


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_GRAPH_CHECKPOINT"),
    reason="set DEEPSEEK_GRAPH_CHECKPOINT for ten-copy eager/graph checkpoint validation",
)
@torch.inference_mode()
def test_checkpoint_graphs_match_eager_across_users_and_candidate_shapes():
    assert torch.cuda.is_available(), "explicit graph checkpoint validation requires CUDA"
    assert torch.cuda.get_device_capability() == (9, 0), "SM90/Hopper is required"
    path = os.environ["DEEPSEEK_GRAPH_CHECKPOINT"]
    history = int(os.environ.get("DEEPSEEK_GRAPH_HISTORY", "2304"))
    chunk_size = int(os.environ.get("DEEPSEEK_GRAPH_CHUNK_SIZE", "256"))
    candidate_size = int(os.environ.get("DEEPSEEK_GRAPH_CANDIDATE", "23"))
    assert history % 64 == 0 and candidate_size > 7
    fallback_size = candidate_size - 7
    prefix = [1 + (token * 137) % 100000 for token in range(history)]
    other_prefix = [1 + (token * 131 + 31) % 100000 for token in range(history)]
    candidates = (
        [2 + token * 53 for token in range(fallback_size)],
        [3 + token * 79 for token in range(candidate_size)],
    )
    models = [
        DeepSeekServingBackend(
            path,
            device="cuda:0",
            num_layers=10,
            chunk_size=chunk_size,
            sparse_pool_tokens=history,
            host_arena_tokens=2 * history,
            enable_compute_graphs=enabled,
        )
        for enabled in (False, True)
    ]
    copied_checks = [0, 0]
    retained_hook_outputs = [{}, {}]
    for model_index, model in enumerate(models):
        originals = {}

        def check_copy(layer, hidden, residual, originals=originals, model_index=model_index):
            if layer == 0 and not retained_hook_outputs[model_index]:
                retained_hook_outputs[model_index].update(
                    actual=(hidden, residual), expected=(hidden.clone(), residual.clone())
                )
            if layer < 3:
                originals[layer] = (hidden.clone(), residual.clone())
            else:
                torch.testing.assert_close((hidden, residual), originals[layer % 3], rtol=0, atol=0)
                copied_checks[model_index] += 1

        model.capture_hook = check_copy
        for layer in range(3, 10):
            assert (
                model.attentions[layer].wq_a.weight.data_ptr()
                != model.attentions[layer % 3].wq_a.weight.data_ptr()
            )
    try:
        scheme_reference = []
        for scheme in SCHEMES:
            sessions = []
            plans = []
            for retained in retained_hook_outputs:
                retained.clear()
            try:
                for model in models:
                    model.configure_scheme(scheme)
                    plan = model.plan_resources(
                        None,
                        {
                            "max_session_capacity": history + candidate_size,
                            "max_history_tokens": history,
                            "max_candidate_tokens": candidate_size,
                        },
                    )
                    model.allocate_shared(plan)
                    plans.append(plan)
                    sessions.append(model.create_session(history))
                bank = models[1]._compute_graphs
                assert bank.allocated and bank.setup_seconds > 0
                identities = {
                    key: (id(pair.projection_graph), id(pair.finish_graph))
                    for key, pair in bank.pairs.items()
                }
                graph_bytes = bank.shared_bytes()
                allocation_memory = bank.memory_at_allocation
                assert (
                    allocation_memory["pytorch_allocated"]
                    <= allocation_memory["pytorch_reserved"]
                    <= allocation_memory["device_used"]
                    <= allocation_memory["device_total"]
                )
                prefix_outputs_device = [
                    model.prefill(session, prefix)
                    for model, session in zip(models, sessions, strict=True)
                ]
                prefix_outputs = [value.cpu() for value in prefix_outputs_device]
                torch.testing.assert_close(prefix_outputs[0], prefix_outputs[1], rtol=0, atol=0)
                snapshots = [
                    [
                        (runner.index_keys.clone(), runner.index_scales.clone())
                        for runner in session.runners
                    ]
                    for session in sessions
                ]
                host_snapshots = [
                    [runner.cache.host_records().clone() for runner in session.runners]
                    for session in sessions
                ]
                # Interleave another owner to evict the original offload history.
                for model in models:
                    other = model.create_session(history)
                    try:
                        model.prefill(other, other_prefix)
                    finally:
                        model.release_session(other)
                for visit, candidate in enumerate(candidates):
                    before = sum(pair.projection_replays for pair in bank.pairs.values())
                    fallback_before = bank.eager_fallbacks
                    values = [
                        model.extend_candidate(session, candidate).cpu()
                        for model, session in zip(models, sessions, strict=True)
                    ]
                    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
                    torch.testing.assert_close(
                        models[0].last_logits, models[1].last_logits, rtol=0, atol=0
                    )
                    delta = sum(pair.projection_replays for pair in bank.pairs.values()) - before
                    assert delta == (10 if len(candidate) == candidate_size else 0)
                    assert bank.eager_fallbacks - fallback_before == (
                        10 if len(candidate) == fallback_size else 0
                    )
                    if scheme == "hbm":
                        scheme_reference.append((values[0], models[0].last_logits.cpu()))
                    else:
                        torch.testing.assert_close(
                            (values[0], models[0].last_logits.cpu()),
                            scheme_reference[visit],
                            rtol=0,
                            atol=0,
                        )
                    for model, session, saved in zip(models, sessions, snapshots, strict=True):
                        assert session.length == history
                        assert model.session_metrics(session)["candidate_device_to_host_bytes"] == 0
                        for runner, (keys, scales) in zip(session.runners, saved, strict=True):
                            assert torch.equal(
                                runner.index_keys.view(torch.uint8), keys.view(torch.uint8)
                            )
                            assert torch.equal(runner.index_scales, scales)
                        model.truncate(session, history)
                    for session, host_saved in zip(sessions, host_snapshots, strict=True):
                        for runner, saved in zip(session.runners, host_saved, strict=True):
                            torch.testing.assert_close(
                                runner.cache.host_records(), saved, rtol=0, atol=0
                            )
                    bank._audit_capacity()
                    assert bank.shared_bytes() == graph_bytes
                    for model, plan in zip(models, plans, strict=True):
                        assert CacheFootprint.from_mapping(model.shared_bytes()).fits(plan.shared)
                    assert identities == {
                        key: (id(pair.projection_graph), id(pair.finish_graph))
                        for key, pair in bank.pairs.items()
                    }
                for actual, saved in zip(prefix_outputs_device, prefix_outputs, strict=True):
                    torch.testing.assert_close(actual.cpu(), saved, rtol=0, atol=0)
                for retained in retained_hook_outputs:
                    torch.testing.assert_close(
                        retained["actual"], retained["expected"], rtol=0, atol=0
                    )
                print(
                    f"graph checkpoint {scheme}: all prefix/candidate hidden and logits exact; "
                    f"H={history}, C={chunk_size}, A{fallback_size} eager fallback, "
                    f"A{candidate_size} captured; private_reserved={bank.private_reserved_bytes}, "
                    f"static_allocated={bank.static_allocated_bytes}, setup_seconds={bank.setup_seconds:.3f}",
                    flush=True,
                )
                print(
                    f"graph checkpoint {scheme}: allocation_memory={allocation_memory}", flush=True
                )
            finally:
                for model, session in zip(models, sessions):
                    model.release_session(session)
    finally:
        for model in models:
            model.close()
    assert all(count > 0 for count in copied_checks)


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_GRAPH_CHECKPOINT"),
    reason="set DEEPSEEK_GRAPH_CHECKPOINT for packed-input runner lifecycle validation",
)
@torch.inference_mode()
def test_checkpoint_graph_runner_packed_admission_audits_and_failure_ownership(monkeypatch):
    import serving.persistent as serving_module

    history, chunk, candidate_size = 2304, 256, 23
    prefix = [1 + (token * 137) % 100000 for token in range(history)]
    other_prefix = [1 + (token * 131 + 31) % 100000 for token in range(history)]
    candidates = ([2 + token * 53 for token in range(16)], [3 + token * 79 for token in range(23)])
    model = DeepSeekServingBackend(
        os.environ["DEEPSEEK_GRAPH_CHECKPOINT"],
        device="cuda:0",
        num_layers=10,
        chunk_size=chunk,
        sparse_pool_tokens=2 * history,
        host_arena_tokens=2 * history,
        enable_compute_graphs=True,
    )
    references = []
    limits = {
        "max_session_capacity": history + candidate_size,
        "max_history_tokens": history,
        "max_candidate_tokens": candidate_size,
    }
    try:

        def check_scheme(scheme):
            nonlocal references
            model.configure_scheme(scheme)
            runner = PersistentGRRunner(model, resource_limits=limits)
            bank = model._compute_graphs
            audit_events, prepared, input_views = [], [], []
            current_request = {}
            original_prepare = serving_module._prepare_token_input
            original_acquire = runner.pool.acquire
            original_audit = runner.pool.audit
            original_truncate = model.truncate
            original_prefill, original_candidate = model.prefill, model.extend_candidate

            def prepare(ids, prefix):
                signature, tensor = original_prepare(ids, prefix)
                prepared.append(tensor)
                return signature, tensor

            def acquire(*args, **kwargs):
                # Admission follows the completed full-input device transfer.
                # Deliberately overwrite both host sources before GPU compute.
                assert args[1] == (history, token_digest(current_request["expected"][:history]))
                prepared.pop().zero_()
                current_request["request"]["input_ids"][:] = [0]
                return original_acquire(*args, **kwargs)

            def check_input(session, ids, *, prefill, **kwargs):
                expected = current_request["expected"]
                expected = expected[:history] if prefill else expected[history:]
                assert ids.device == model.device and ids.dtype == torch.long
                torch.testing.assert_close(ids.cpu(), torch.tensor(expected), rtol=0, atol=0)
                input_views.append((ids, ids.cpu()))
                return (original_prefill if prefill else original_candidate)(session, ids, **kwargs)

            def audit():
                measured = []
                original_measure = runner.pool._measure

                def measure(session):
                    measured.append(id(session))
                    return original_measure(session)

                with monkeypatch.context() as audit_patch:
                    audit_patch.setattr(runner.pool, "_measure", measure)
                    result = original_audit()
                assert measured == [id(entry.session) for entry in runner.pool._entries.values()]
                audit_events.append(("audit", tuple(measured), result))
                return result

            def truncate(session, length, **kwargs):
                audit_events.append(("truncate", id(session), length))
                return original_truncate(session, length, **kwargs)

            def execute(user, history_ids, candidate):
                request = {
                    "user_id": user,
                    "input_ids": history_ids + candidate,
                    "stable_prefix_tokens": history,
                }
                current_request.update(request=request, expected=list(request["input_ids"]))
                return runner.execute(request)

            try:
                with monkeypatch.context() as patch:
                    patch.setattr(serving_module, "_prepare_token_input", prepare)
                    patch.setattr(runner.pool, "acquire", acquire)
                    patch.setattr(runner.pool, "audit", audit)
                    patch.setattr(model, "truncate", truncate)
                    patch.setattr(
                        model,
                        "prefill",
                        lambda s, ids, **kwargs: check_input(s, ids, prefill=True, **kwargs),
                    )
                    patch.setattr(
                        model,
                        "extend_candidate",
                        lambda s, ids, **kwargs: check_input(s, ids, prefill=False, **kwargs),
                    )
                    first = execute("alice", prefix, candidates[0])
                    original = runner.pool._entries["alice"].session
                    first_values = (first.hidden.cpu(), model.last_logits.cpu())
                    execute("bob", other_prefix, candidates[0])
                    other = runner.pool._entries["bob"].session
                    audit_events.clear()
                    revisit = execute("alice", prefix, candidates[1])
                    revisit_values = (revisit.hidden.cpu(), model.last_logits.cpu())
                    assert runner.pool._entries["alice"].session is original
                    assert not first.metrics["is_revisit"] and revisit.metrics["is_revisit"]
                    assert (
                        revisit.metrics["prefix_cache_hit"] and not revisit.metrics["evicted_users"]
                    )
                    assert [event[0] for event in audit_events] == ["audit", "truncate", "audit"]
                    assert audit_events[0][1] == audit_events[2][1] == (id(other), id(original))
                    for _, _, actual in (audit_events[0], audit_events[2]):
                        assert actual == runner.pool.shared_bytes() + sum(
                            (
                                CacheFootprint.from_mapping(model.session_bytes(entry.session))
                                for entry in runner.pool._entries.values()
                            ),
                            CacheFootprint(),
                        )
                    assert revisit.metrics["cache_hbm_bytes"] == audit_events[2][2].hbm
                    assert revisit.metrics["cache_dram_bytes"] == audit_events[2][2].dram
                    assert original.capacity == original.length == history
                    assert model.session_metrics(original)["candidate_device_to_host_bytes"] == 0
                    if scheme == "hbm":
                        references = [first_values, revisit_values]
                    else:
                        torch.testing.assert_close(
                            [first_values, revisit_values], references, rtol=0, atol=0
                        )
                    changed = execute("alice", other_prefix, candidates[0])
                    changed_session = runner.pool._entries["alice"].session
                    assert original.released and changed_session is not original
                    assert changed.metrics["is_revisit"] and not changed.metrics["prefix_cache_hit"]
                    assert changed.metrics["evicted_users"] == ["alice"]
                    for actual, saved in input_views:
                        torch.testing.assert_close(actual.cpu(), saved, rtol=0, atol=0)

                    def fail_layer(layer, hidden, residual):
                        if layer == 2:
                            raise RuntimeError("injected candidate layer failure")

                    with monkeypatch.context() as failure_patch:
                        failure_patch.setattr(model, "capture_hook", fail_layer)
                        with pytest.raises(RuntimeError, match="candidate layer failure"):
                            execute("alice", other_prefix, candidates[1])
                    assert changed_session.released and list(runner.pool._entries) == ["bob"]
                    assert runner.visits["alice"] == 3
                    assert not model._execution_active() and not model.lifecycle.poisoned
                    assert model.lifecycle.admission_owner is runner
                    if model._shared_pool is not None:
                        assert not model._shared_pool._transient_owners

                    if scheme == "dense_prefetch":
                        original_execution = bank.execution

                        def failed_event():
                            raise RuntimeError("injected graph completion event failure")

                        @contextmanager
                        def fail_completion():
                            with monkeypatch.context() as completion_patch, original_execution():
                                yield
                                completion_patch.setattr(torch.cuda, "Event", failed_event)

                        try:
                            with monkeypatch.context() as completion_patch:
                                completion_patch.setattr(bank, "execution", fail_completion)
                                with pytest.raises(ExceptionGroup, match="cleanup failed") as error:
                                    execute("bob", other_prefix, candidates[1])
                            assert "graph completion event" in str(error.value.exceptions[0])
                            assert "poisoned" in str(error.value.exceptions[1])
                            assert bank.failed and bank.pairs and model.lifecycle.poisoned
                            assert model.lifecycle.active_session is other and not other.released
                            assert runner.pool._entries["bob"].session is other
                            assert model.lifecycle.admission_owner is runner
                            with pytest.raises(RuntimeError, match="poisoned"):
                                runner.close()
                            assert runner.closed and runner._owner_bound
                            with pytest.raises(RuntimeError, match="poisoned"):
                                original_candidate(other, candidates[1])
                            with pytest.raises(RuntimeError, match="poisoned"):
                                model._release_shared()
                        finally:
                            # This test injected a host-side Event-constructor
                            # error, not a device fault. Drain and reset only the
                            # synthetic fixture state to release its allocations.
                            torch.cuda.synchronize(model.device)
                            bank.failed = model.lifecycle.poisoned = False
                            model.lifecycle.active_session = None
                print(
                    f"graph runner {scheme}: packed GPU inputs exact, both full audits retained, "
                    "changed-A reuse and candidate failure release verified",
                    flush=True,
                )
            finally:
                runner.close()

        for scheme in SCHEMES:
            check_scheme(scheme)
    finally:
        model.close()
