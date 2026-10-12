"""Page-boundary and independent-sequence ABI checks without native launches."""

from contextlib import nullcontext
from itertools import pairwise

import pytest
import torch

from operators.deepseek_v32.indexer import resident_batch


def test_page64_preparation_never_merges_sequence_tail_with_next_history(monkeypatch):
    monkeypatch.setattr(resident_batch, "_check_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    keys = torch.arange(2 * 65 * 128, dtype=torch.int32).to(torch.uint8).view(2, 65, 128)
    keys = keys.view(torch.float8_e4m3fn)
    scales = torch.arange(130, dtype=torch.float32).view(2, 65)

    def fake_pack(k, s):
        assert k.shape == (256, 128) and s.shape == (256,)
        actual = k.view(2, 128, 128).view(torch.uint8)
        assert torch.equal(actual[:, :65], keys.view(torch.uint8))
        assert actual[:, 65:].eq(0).all()
        assert torch.equal(s.view(2, 128)[:, :65], scales)
        assert s.view(2, 128)[:, 65:].eq(0).all()
        return torch.empty((4, 8448), dtype=torch.uint8)

    monkeypatch.setattr(resident_batch, "pack_q1_keys", fake_pack)
    prepared = resident_batch.prepare_keys(keys, scales)
    assert prepared.packed.shape == (4, 64, 1, 132)
    assert prepared.block_table.tolist() == [[0, 1], [2, 3]]
    assert prepared.logical_tokens == 65


def test_explicit_padded_capacity_must_match_local_page_count(monkeypatch):
    monkeypatch.setattr(resident_batch, "_check_device", lambda _: None)
    keys = torch.empty((2, 192, 128), dtype=torch.float8_e4m3fn)
    scales = torch.empty((2, 192))
    with pytest.raises(ValueError, match="explicit padded capacity"):
        resident_batch.prepare_keys(keys, scales, logical_tokens=65)


def test_max_observation_packing_rebases_int32_offsets_without_copies(monkeypatch):
    from operators.deepseek_v32.indexer import page64

    monkeypatch.setattr(resident_batch, "_check_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    # Meta tensors reproduce B1024/H65536 storage geometry without allocating
    # multi-GiB operands in a CPU correctness test.
    keys = torch.empty((1024, 65600, 128), dtype=torch.float8_e4m3fn, device="meta")
    scales = torch.empty((1024, 65600), dtype=torch.float32, device="meta")
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(k, s, packed, *, TOKENS, num_warps):
                pages = grid[0]
                assert pages * 2112 <= 2**31
                assert pages * 2048 <= 2**31
                assert TOKENS == pages * 64
                assert k.shape == (TOKENS, 32)
                assert s.shape == (TOKENS,)
                assert packed.shape == (pages, 2112)
                assert num_warps == 4
                calls.append(
                    (pages, k.storage_offset(), s.storage_offset(), packed.storage_offset())
                )

            return launch

    monkeypatch.setattr(page64, "_kernel", lambda: Kernel())

    def forbidden_unbounded(*args):
        raise AssertionError("large packing cannot invoke unbounded original wrapper")

    monkeypatch.setattr(resident_batch, "pack_q1_keys", forbidden_unbounded)
    result = resident_batch.prepare_keys(keys, scales, logical_tokens=65537)
    assert result.packed.shape == (1049600, 64, 1, 132)
    assert len(calls) == 4
    assert [row[0] for row in calls] == [262400] * 4
    for index, (_, key_offset, scale_offset, output_offset) in enumerate(calls):
        first_page = index * 262400
        assert key_offset == first_page * 2048
        assert scale_offset == first_page * 64
        assert output_offset == first_page * 2112


def test_packing_ranges_respect_last_partial_batch_and_single_large_sequence():
    for total, per_sequence in ((1049600, 1025), (2000000, 2000000), (12345, 64)):
        ranges = resident_batch._packing_page_ranges(total, per_sequence)
        assert ranges[0][0] == 0 and ranges[-1][1] == total
        assert all(stop - start <= resident_batch._SAFE_PACK_PAGES for start, stop in ranges)
        assert all(left[1] == right[0] for left, right in pairwise(ranges))
