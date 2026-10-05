import pytest
import torch

from experiments.nosa_motivation.src.flops import (
    matrix_flops,
    observe_selection,
    work_counts,
)
from experiments.nosa_motivation.src.matrix_baseline import invocation_geometry
from models.attention_contracts import BlockSelection


def test_reference_geometry_preserves_partial_history_and_whole_candidate():
    assert invocation_geometry(1536, 2000, 1024) == [(0, 1024), (1024, 512), (1536, 2000)]


def test_chunk_boundary_controls_indexer_bypass_and_candidate_is_not_chunked():
    assert work_counts(0, 4096, 1024)["compressed_pairs_per_query_head_layer"] == 0
    chunked = work_counts(0, 4224, 1024)
    whole = work_counts(0, 4224, 4224)
    assert chunked["indexer_scored_queries"] == 128
    assert whole["indexer_scored_queries"] == 4224
    assert (
        whole["compressed_pairs_per_query_head_layer"]
        > chunked["compressed_pairs_per_query_head_layer"]
    )
    assert (
        whole["attention_pairs_per_query_head_layer"]
        == chunked["attention_pairs_per_query_head_layer"]
    )


def test_flops_include_cis_compressed_score_and_causal_attention():
    cfg = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
    }
    flops = matrix_flops(cfg, 4096, 1, 1)
    assert flops["qkv_proj"] == 2 * 2 * 8 * 16
    assert flops["cis_projection"] == 2 * 2 * 4 * 2
    assert flops["indexer_qk"] == 2 * 2 * 4 * 2 * 255
    assert flops["block_sparse_attention"] == 4 * 2 * 4 * 2 * (63 * 64 + 1)
    assert "lm_head" not in flops


def test_actual_selection_validation_handles_partial_current_block_and_padding():
    ids = torch.full((2, 2, 64), -1, dtype=torch.long)
    ids[..., 0] = 0
    ids[1, :, 1] = 1
    selected = BlockSelection(ids, 64, ids >= 0)
    result = observe_selection(selected, query_start=63, query_heads=8)
    assert result["causal_pairs_all_query_heads"] == (64 + 65) * 8
    ids[1, 0, 1] = 0
    with pytest.raises(ValueError, match="current block"):
        observe_selection(selected, query_start=63, query_heads=8)


def test_actual_selection_rejects_duplicates_even_when_count_and_current_pass():
    ids = torch.full((1, 1, 64), -1, dtype=torch.long)
    ids[0, 0, :3] = torch.tensor([0, 0, 2])
    with pytest.raises(ValueError, match="duplicate"):
        observe_selection(BlockSelection(ids, 64, ids >= 0), query_start=128, query_heads=4)
