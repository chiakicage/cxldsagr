"""Official dispatch remains between the optional common compute callbacks."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v32.execution.official import OfficialAttentionRunner, OfficialPipeline


@pytest.fixture
def official_runner(monkeypatch):
    from operators.deepseek_v32.indexer import official

    events = []
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    projection = SimpleNamespace(
        index_k=hidden.clone(),
        index_scale=torch.ones(2),
        index_q=hidden + 1,
        index_weights=torch.ones(2),
        kv=hidden + 2,
        q=hidden + 3,
    )
    raw_attention = hidden + 4
    output = (hidden + 5, hidden + 6)

    @contextmanager
    def operation():
        events.append("begin")
        try:
            yield
        finally:
            events.append("end")

    def project(x, position, *, normalized):
        assert x is hidden and position == 2 and normalized is runner.expected_normalized
        events.append("project")
        return projection

    def finish(x):
        assert x is raw_attention
        events.append("output")
        return output

    def prepare(position, count, offset):
        assert (position, count, offset) == (2, 2, 0.75)
        events.append("prepare")
        return {"arguments": {"official_probe": True}}

    def logits(offload, q, key_pair, weights, starts, ends, **kwargs):
        assert runner.cache.offload is offload
        assert q is projection.index_q and weights is projection.index_weights
        keys, scales = key_pair
        torch.testing.assert_close(keys[2:], projection.index_k)
        torch.testing.assert_close(scales[2:], projection.index_scale)
        assert starts.tolist() == [0, 0] and ends.tolist() == [3, 4]
        assert kwargs == {
            "clean_logits": False,
            **({"official_probe": True} if runner.cache.offload else {}),
        }
        events.append("fused_logits" if offload else "resident_logits")
        return torch.arange(8, dtype=torch.float32).reshape(2, 4)

    def resident_logits(*args, **kwargs):
        return logits(False, *args, **kwargs)

    def fused_logits(*args, **kwargs):
        return logits(True, *args, **kwargs)

    def topk(scores, ends, indices, table, cumulative, extra):
        assert scores.shape == (2, 4) and ends.tolist() == [3, 4]
        assert table.tolist() == [[0, 1, 2, 3]]
        assert cumulative.tolist() == [0, 2] and extra is None
        indices.fill_(-1)
        indices[:, :2] = torch.tensor([[1, 0], [3, 2]], dtype=torch.int32)
        events.append("official_topk")

    def consume(q, selection, scope):
        indices = selection.token_ids
        assert q is projection.q
        assert indices[:, :2].tolist() == [[1, 0], [3, 2]]
        assert bool((indices[:, 2:] == -1).all())
        events.append("consume")
        return raw_attention

    runner = OfficialAttentionRunner.__new__(OfficialAttentionRunner)
    runner.attention = SimpleNamespace(project=project, output=finish)
    runner.cfg = SimpleNamespace(index_topk=2048)
    runner.offset = 0.75
    runner.expected_normalized = True
    runner.collect_cache_diagnostics = False
    runner.capture_hook = None
    runner.index_keys = torch.zeros(4, 4)
    runner.index_scales = torch.zeros(4)
    runner._consume = consume
    runner.cache = SimpleNamespace(
        written=2,
        offload=False,
        operation=operation,
        reserve_append_source=lambda: events.append("reserve"),
        declare_indexer_visible=lambda end: events.append(("visible", end)),
        prepare_prefetch=prepare,
        finalize_prefetch=lambda prefetch, *, logits: events.append("finalize"),
        append=lambda kv: events.append(("append", kv is projection.kv)),
    )
    monkeypatch.setattr(
        official,
        "module",
        lambda: SimpleNamespace(
            fp8_mqa_logits=resident_logits, fp8_mqa_logits_fuse_prefetch=fused_logits
        ),
    )
    monkeypatch.setattr(official, "topk_module", lambda: SimpleNamespace(fast_topk_transform=topk))
    return runner, hidden, project, finish, output, events


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("callbacks", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_official_pipeline_with_common_compute_callbacks(
    official_runner, offload, callbacks, normalized
):
    runner, hidden, project, finish, output, events = official_runner
    runner.cache.offload = offload
    runner.expected_normalized = normalized
    kwargs = {}
    if callbacks:

        def forbidden(*args, **kwargs):
            pytest.fail("callback path invoked eager checkpoint computation")

        runner.attention = SimpleNamespace(project=forbidden, output=forbidden)
        kwargs = {"project_callback": project, "output_callback": finish}
    actual = runner.forward(hidden, normalized=normalized, capture_indices=True, **kwargs)
    assert actual is output
    assert runner.last_indices[0][:, :2].tolist() == [[1, 0], [3, 2]]
    assert events == [
        "begin",
        "reserve",
        "project",
        ("visible", 4),
        *(["prepare", "fused_logits", "finalize"] if offload else ["resident_logits"]),
        "official_topk",
        ("append", True),
        "consume",
        "output",
        "end",
    ]


@pytest.mark.parametrize("callback", ["project_callback", "output_callback"])
def test_official_invalid_callback_fails_before_cache_operation(official_runner, callback):
    runner, hidden, _, _, _, events = official_runner
    with pytest.raises(TypeError, match="callbacks must be callable"):
        runner.forward(hidden, **{callback: object()})
    assert events == []


@pytest.mark.parametrize("callback", ["project_callback", "output_callback"])
@pytest.mark.parametrize("offload", [False, True])
def test_official_callback_failure_exits_cache_operation(official_runner, callback, offload):
    runner, hidden, _, _, _, events = official_runner
    runner.cache.offload = offload

    def fail(*args, **kwargs):
        raise RuntimeError("injected compute callback failure")

    with pytest.raises(RuntimeError, match="injected compute callback failure"):
        runner.forward(hidden, normalized=True, **{callback: fail})
    assert events[0] == "begin" and events[-1] == "end"
    assert runner._diagnostic_state is None


def test_official_runner_factory_allocates_private_indexer_once_without_rebinding():
    attention = SimpleNamespace(
        cfg=SimpleNamespace(kv_lora_rank=4, qk_rope_head_dim=4, index_head_dim=8),
        device=torch.device("cpu"),
    )
    pipeline = OfficialPipeline()
    cache = object()
    first = pipeline.create_runner(
        attention,
        16,
        resources=None,
        scheme="hbm",
        slots=16,
        chunk_size=4,
        cache=cache,
        dense_backend=None,
    )
    second = pipeline.create_runner(
        attention,
        16,
        resources=None,
        scheme="hbm",
        slots=16,
        chunk_size=4,
        cache=cache,
        dense_backend=None,
    )
    assert first.cache is cache and second.cache is cache
    assert first.index_keys.shape == second.index_keys.shape == (16, 8)
    assert first.index_keys.data_ptr() != second.index_keys.data_ptr()
    assert first.index_scales.data_ptr() != second.index_scales.data_ptr()
    assert first.offset.data_ptr() != second.offset.data_ptr()


def test_official_metrics_preserve_uninstrumented_selection_boundary():
    metrics = {
        "host_to_device_bytes": 100,
        "device_to_host_bytes": 0,
        "prefetched_records": 1,
        "recalled_records": 2,
        "evicted_records": 3,
        "selection_records": None,
        "resident_selection_records": None,
    }
    session = SimpleNamespace(
        scheme="echo",
        length=128,
        last_candidate_transient=True,
        runners=[SimpleNamespace(cache=SimpleNamespace(metrics=lambda: metrics))],
    )
    actual = OfficialPipeline().session_metrics(session)
    assert actual["candidate_device_to_host_bytes"] == 0
    assert actual["selection_records"] is actual["resident_selection_records"] is None
    assert actual["hit_ratio_stage"] == "not_instrumented_in_official_implementation"


def test_official_factory_rejects_unsupported_scheme_before_model_loading():
    from models.deepseek_v32.execution.official import build_official_backend

    with pytest.raises(ValueError, match="only hbm and echo"):
        build_official_backend("/nonexistent", scheme="serial_sparse", num_layers=10)
