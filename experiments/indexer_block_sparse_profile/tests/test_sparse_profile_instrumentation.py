"""CPU clocks validate stage attribution/restoration, not GPU performance."""

import pytest
import torch

from experiments.indexer_block_sparse_profile.src.instrumentation import SparseScopes
from layers.attention import BlockSelection
from models.nosa.config import NosaConfig
from models.nosa.model import NosaForCausalLM


class FakeClock:
    def __init__(self):
        self.tick = 0
        self.synchronizations = 0

    def record(self):
        self.tick += 1
        return self.tick

    def synchronize(self):
        self.synchronizations += 1

    @staticmethod
    def elapsed_ms(start, end):
        return end - start


class FakeNVTX:
    def __init__(self):
        self.stack = []
        self.labels = []

    def range_push(self, label):
        self.stack.append(label)
        self.labels.append(label)

    def range_pop(self):
        self.stack.pop()


def tiny_sparse_model():
    config = NosaConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        max_position_embeddings=128,
    )
    model = NosaForCausalLM(config, attention_mode="sparse", sparse_backend="reference")
    generator = torch.Generator().manual_seed(775)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
            if "norm.weight" in name:
                parameter.add_(1)
    return model


def test_scopes_attribute_chunks_and_nested_stages_without_operation_sync():
    model = tiny_sparse_model()
    original_indexer, original_attention = model.indexer, model.main_attention
    clock, nvtx = FakeClock(), FakeNVTX()
    cache = model.new_cache(8)
    with SparseScopes(model, phase="full_prefill", clock=clock, nvtx=nvtx) as scopes:
        model(torch.tensor([1, 2, 3]), cache, return_hidden=True)
        model(torch.tensor([4, 5]), cache, return_hidden=True)
        assert model.indexer.policy is original_indexer.policy
        assert clock.synchronizations == 0
        records = scopes.collect()
        assert clock.synchronizations == 1
        assert scopes.collect() == records
        assert clock.synchronizations == 1
    assert model.indexer is original_indexer and model.main_attention is original_attention
    assert nvtx.stack == []
    assert len(records) == 20
    assert {record["phase"] for record in records} == {"full_prefill"}
    assert {record["call_id"] for record in records} == set(range(4))
    for layer_idx in range(2):
        totals = [
            r for r in records if r["layer_idx"] == layer_idx and r["stage"] == "indexer_total"
        ]
        assert [(r["query_start"], r["query_length"]) for r in totals] == [(0, 3), (3, 2)]
        for parent in totals:
            children = [r for r in records if r["parent_scope_id"] == parent["scope_id"]]
            assert {r["stage"] for r in children} == {"compression_k", "compression_cis"}
            assert parent["cuda_elapsed_ms"] > sum(r["cuda_elapsed_ms"] for r in children)
            assert parent["inclusive"] and parent["parent_scope_id"] is None
            assert all(r["call_id"] == parent["call_id"] for r in children)
    assert all("details" not in record and record["status"] == "ok" for record in records)


def test_audit_runs_without_clock_and_reports_resident_selection_geometry():
    model = tiny_sparse_model()
    cache = model.new_cache(8)
    model(torch.tensor([1, 2, 3]), cache, return_hidden=True)
    clock = FakeClock()
    with SparseScopes(model, timing=False, audit=True, clock=clock) as scopes:
        model(torch.tensor([4, 5]), cache, return_hidden=True)
    records = scopes.collect()
    attention = [r for r in records if r["stage"] == "block_sparse_attention"]
    assert len(attention) == 2
    assert clock.tick == clock.synchronizations == 0
    for record in attention:
        assert record["cuda_elapsed_ms"] is None
        assert record["query_start"] == 3 and record["query_length"] == 2
        details = record["details"]
        assert details["q_shape"] == [2, 4, 8]
        assert details["k_shape"] == details["v_shape"] == [5, 2, 8]
        assert details["selection_shape"] == [2, 2, 64]
        assert details["block_budget"] == details["block_size"] == 64
        assert details["valid_blocks_min"] == details["valid_blocks_max"] == 1
        assert details["valid_entries"] == 4


def test_operator_children_are_nested_in_the_matching_indexer_call(monkeypatch):
    import models.nosa.indexer as indexer_module
    import operators.nosa.indexer.api as operator_module

    model = tiny_sparse_model()

    def fake_scores(query, keys, positions):
        return torch.zeros((*query.shape[:2], len(keys)))

    def fake_selection(scores, cis, positions, total_length):
        return torch.zeros((*scores.shape[:2], 1), dtype=torch.long)

    def select(q, keys, cis, query_start, **kwargs):
        grouped = q.reshape(len(q), keys.shape[1], -1, q.shape[-1])
        compressed_k = indexer_module.compress_sequence(keys)
        compressed_cis = indexer_module.compress_sequence(cis)
        positions = torch.arange(query_start, query_start + len(q))
        scores = operator_module.compressed_scores(grouped, compressed_k, positions)
        ids = operator_module._select_validated_scores(scores, compressed_cis, positions, len(keys))
        return BlockSelection(ids, 64)

    monkeypatch.setattr(operator_module, "compressed_scores", fake_scores)
    monkeypatch.setattr(operator_module, "_select_validated_scores", fake_selection)
    monkeypatch.setattr(model.indexer, "_select_nosa", select)
    with SparseScopes(model, clock=FakeClock()) as scopes:
        model(torch.tensor([1, 2]), return_hidden=True)
    records = scopes.collect()
    parents = {r["scope_id"]: r for r in records if r["stage"] == "indexer_total"}
    children = [r for r in records if r["stage"] in ("compressed_scores", "select_from_scores")]
    assert len(children) == 4
    for child in children:
        parent = parents[child["parent_scope_id"]]
        assert child["layer_idx"] == parent["layer_idx"]
        assert child["call_id"] == parent["call_id"]
        assert child["query_start"] == parent["query_start"] == 0
        assert child["query_length"] == parent["query_length"] == 2
        assert 0 < child["cuda_elapsed_ms"] < parent["cuda_elapsed_ms"]
    assert operator_module.compressed_scores is fake_scores
    assert operator_module._select_validated_scores is fake_selection


def test_fused_pipeline_scopes_are_siblings_and_restore_launch_helpers(monkeypatch):
    import models.nosa.indexer as indexer_module
    import operators.nosa.indexer.api as operator_module
    import operators.nosa.indexer.validation as validation_module

    model = tiny_sparse_model()
    helpers = (
        (validation_module, "all_finite", "indexer_validate"),
        (indexer_module, "prepare_indexer_inputs", "indexer_cache_update"),
        *[
            (operator_module, name, name)
            for name in ("pooled_scores", "topk_qa", "prepare_cis", "topk_cis", "finish_selection")
        ],
    )
    sentinel = object()

    def launch(*args):
        return sentinel

    for module, name, _ in helpers:
        monkeypatch.setattr(module, name, launch)

    def select(q, keys, cis, query_start, **kwargs):
        for module, name, _ in helpers:
            assert getattr(module, name)(q) is sentinel
        return BlockSelection(torch.zeros((len(q), keys.shape[1], 1), dtype=torch.long), 64)

    monkeypatch.setattr(model.indexer, "_select_nosa", select)
    with SparseScopes(model, clock=FakeClock()) as scopes:
        model(torch.tensor([1, 2]), return_hidden=True)
    records = scopes.collect()
    for parent in (record for record in records if record["stage"] == "indexer_total"):
        children = [record for record in records if record["parent_scope_id"] == parent["scope_id"]]
        assert [record["stage"] for record in children] == [stage for _, _, stage in helpers]
    assert all(getattr(module, name) is launch for module, name, _ in helpers)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("ranked", [False, True])
def test_native_preparation_and_selection_scopes_record_prepared_ranking(
    monkeypatch, joint, ranked
):
    from experiments.indexer_block_sparse_profile.src.analyze import _validate_scopes
    from operators.nosa.indexer import _indexer_cuda as _nosa_indexer_cuda
    from operators.nosa.indexer import _prepare_cuda as _nosa_prepare_cuda
    from operators.nosa.indexer import _prepare_ranked_cuda as _nosa_prepare_ranked_cuda
    from operators.nosa.indexer import _selection_cuda as _nosa_selection_cuda

    model = tiny_sparse_model()
    preparation = _nosa_prepare_ranked_cuda if ranked else _nosa_prepare_cuda
    preparation_name = "prepare_ranked_out" if ranked else "prepare_out"
    selection = _nosa_indexer_cuda if joint else _nosa_selection_cuda
    selection_name = "select" if joint else "select_pooled_blocks"
    sentinel = object()

    def launch(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(preparation, preparation_name, launch)
    monkeypatch.setattr(selection, selection_name, launch)

    def select(q, keys, cis, query_start, **kwargs):
        assert getattr(preparation, preparation_name)(q) is sentinel
        assert (
            getattr(selection, selection_name)(q, prepared_ranking=sentinel if ranked else None)
            is sentinel
        )
        return BlockSelection(torch.zeros((len(q), keys.shape[1], 1), dtype=torch.long), 64)

    monkeypatch.setattr(model.indexer, "_select_nosa", select)
    with SparseScopes(model, clock=FakeClock()) as scopes:
        model(torch.tensor([1, 2]), return_hidden=True)
    records = scopes.collect()
    _validate_scopes(records)
    for parent in (record for record in records if record["stage"] == "indexer_total"):
        children = [record for record in records if record["parent_scope_id"] == parent["scope_id"]]
        assert [record["stage"] for record in children] == [
            "native_prepare_ranked" if ranked else "native_prepare",
            "native_indexer" if joint else "native_selection",
        ]
        assert children[-1]["prepared_ranking"] is ranked
        children[-1]["prepared_ranking"] = not ranked
        with pytest.raises(ValueError, match="prepared_ranking"):
            _validate_scopes(records)
        children[-1]["prepared_ranking"] = ranked
    assert getattr(preparation, preparation_name) is launch
    assert getattr(selection, selection_name) is launch


def test_exception_restores_modules_hooks_and_nvtx_and_keeps_layer_attribution(monkeypatch):
    import models.nosa.indexer as indexer_module
    import models.nosa.scoring as scoring_module

    model = tiny_sparse_model()
    original_indexer, original_attention = model.indexer, model.main_attention
    original_compress = indexer_module.compress_sequence
    original_cis = scoring_module.cis_scores
    invocations = []

    def fail_second_layer(*args, **kwargs):
        invocations.append(1)
        if len(invocations) == 2:
            raise RuntimeError("deliberate CIS failure")
        return original_cis(*args, **kwargs)

    monkeypatch.setattr(scoring_module, "cis_scores", fail_second_layer)
    clock, nvtx = FakeClock(), FakeNVTX()
    cache = model.new_cache(8)
    with (
        pytest.raises(RuntimeError, match="deliberate CIS"),
        SparseScopes(model, clock=clock, nvtx=nvtx) as scopes,
    ):
        model(torch.tensor([1, 2]), cache, return_hidden=True)
    assert cache.length == 0
    assert model.indexer is original_indexer and model.main_attention is original_attention
    assert indexer_module.compress_sequence is original_compress
    assert scoring_module.cis_scores is fail_second_layer
    assert nvtx.stack == []
    assert clock.synchronizations == 0
    errors = [r for r in scopes.collect() if r["status"] == "error"]
    assert [(r["stage"], r["layer_idx"]) for r in errors] == [("cis_projection", 1)]
    assert errors[0]["error_type"] == "RuntimeError"
    for layer in model.model.layers:
        assert not layer.self_attn._forward_pre_hooks
        assert not layer.self_attn._forward_hooks


def test_failed_context_install_restores_prior_patches(monkeypatch):
    import models.nosa.scoring as scoring_module

    model = tiny_sparse_model()
    indexer, attention, cis = model.indexer, model.main_attention, scoring_module.cis_scores

    def fail_hook(*args, **kwargs):
        raise RuntimeError("hook installation failed")

    monkeypatch.setattr(model.model.layers[1].self_attn, "register_forward_hook", fail_hook)
    with pytest.raises(RuntimeError, match="hook installation"), SparseScopes(model, timing=False):
        pass
    assert model.indexer is indexer and model.main_attention is attention
    assert scoring_module.cis_scores is cis
    assert SparseScopes._active_instance is None
    for layer in model.model.layers:
        assert not layer.self_attn._forward_pre_hooks
        assert not layer.self_attn._forward_hooks


def test_audit_timing_cpu_clock_and_overlap_contracts():
    model = tiny_sparse_model()
    with pytest.raises(ValueError, match="timing=False"):
        SparseScopes(model, audit=True)
    with pytest.raises(ValueError, match="CUDA"):
        SparseScopes(model)
    with (
        SparseScopes(model, timing=False),
        pytest.raises(RuntimeError, match="Overlapping"),
        SparseScopes(model, timing=False),
    ):
        pass


def test_checked_native_scope_is_one_container_with_explicit_prepared_ranking(monkeypatch):
    from experiments.indexer_block_sparse_profile.src.analyze import _validate_scopes
    from operators.nosa.indexer import _indexer_checked_cuda as _nosa_indexer_checked_cuda

    model = tiny_sparse_model()
    sentinel = object()

    def launch(*args, **kwargs):
        return sentinel

    def select(q, keys, cis, query_start, **kwargs):
        assert _nosa_indexer_checked_cuda.select_prepared_out(q) is sentinel
        return BlockSelection(torch.zeros((len(q), keys.shape[1], 1), dtype=torch.long), 64)

    monkeypatch.setattr(_nosa_indexer_checked_cuda, "select_prepared_out", launch)
    monkeypatch.setattr(model.indexer, "_select_nosa", select)
    with SparseScopes(model, clock=FakeClock()) as scopes:
        model(torch.tensor([1, 2]), return_hidden=True)
    records = scopes.collect()
    _validate_scopes(records)
    children = [record for record in records if record["stage"] == "native_checked_indexer"]
    assert len(children) == 2
    for child in children:
        assert child["prepared_ranking"] is True
        parent = next(
            record for record in records if record["scope_id"] == child["parent_scope_id"]
        )
        assert parent["stage"] == "indexer_total"
        assert 0 < child["cuda_elapsed_ms"] < parent["cuda_elapsed_ms"]
    assert _nosa_indexer_checked_cuda.select_prepared_out is launch
