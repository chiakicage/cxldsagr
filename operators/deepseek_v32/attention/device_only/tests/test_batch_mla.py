"""Disjoint resident histories and explicit Q1 dispatch semantics."""

import pytest
import torch

from operators.deepseek_v32.attention.device_only import batch


def test_local_selection_translation_preserves_invalid_ids_without_cross_sequence_reads():
    ids = torch.tensor([[[0, 3, -1, 4, 2**32 + 1]], [[1, 2, -1, 7, 2**32 + 2]]])
    actual = batch.global_indices(ids, 4)
    assert actual.dtype == torch.int32
    assert actual.tolist() == [[[0, 3, -1, -1, -1]], [[5, 6, -1, -1, -1]]]


def test_batch_decode_uses_one_official_invocation_and_disjoint_history_spans(monkeypatch):
    q = torch.zeros(3, 1, 128, 576, dtype=torch.bfloat16)
    kv = torch.zeros(3, 4, 576, dtype=torch.bfloat16)
    for sequence in range(3):
        kv[sequence].fill_(sequence + 1)
    ids = torch.tensor([[[0, 3]], [[1, 2]], [[0, -1]]], dtype=torch.int32)
    calls = []

    def official(actual_q, actual_kv, actual_ids, scale, value_dim):
        calls.append(True)
        assert actual_q.shape == (3, 128, 576)
        assert actual_kv.data_ptr() == kv.data_ptr()
        assert actual_ids.tolist() == [[0, 3], [5, 6], [8, -1]]
        return torch.empty((3, 128, value_dim), dtype=torch.bfloat16)

    def forbidden(*args):
        raise AssertionError("batch decode must not loop over the single-query split16 adapter")

    monkeypatch.setattr(batch, "sparse_mla", official)
    monkeypatch.setattr(batch, "sparse_mla_decode", forbidden)
    actual = batch.sparse_mla_batch(q, kv, ids, 0.1)
    assert actual.shape == (3, 1, 128, 512)
    assert len(calls) == 1


def test_original_single_query_preserves_split16_dispatch(monkeypatch):
    q = torch.zeros(1, 1, 128, 576, dtype=torch.bfloat16)
    kv = torch.zeros(1, 4, 576, dtype=torch.bfloat16)
    ids = torch.zeros((1, 1, 2048), dtype=torch.int32)
    calls = []

    def decode(actual_q, actual_kv, actual_ids, scale, value_dim):
        calls.append(True)
        assert actual_q.shape == (1, 128, 576)
        assert actual_ids.data_ptr() == ids.data_ptr()
        return torch.empty((1, 128, value_dim), dtype=torch.bfloat16)

    monkeypatch.setattr(batch, "sparse_mla_decode", decode)
    actual = batch.sparse_mla_batch(q, kv, ids, 0.1)
    assert actual.shape == (1, 1, 128, 512) and len(calls) == 1


def test_flattened_token_count_cannot_overflow_int32():
    with pytest.raises(ValueError, match="fit int32"):
        batch.global_indices(torch.zeros((2, 1, 1), dtype=torch.int32), 2**30)
