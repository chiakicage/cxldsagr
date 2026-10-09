"""Check paged backend delegation and byte-preserving GPU packing."""

import sys
from types import SimpleNamespace

import pytest
import torch

from operators.deepseek_v32.indexer import echo


@pytest.mark.parametrize("columns", [64, 65, 129])
def test_q1_forwards_current_packed_buffer_to_official_backend(columns, monkeypatch):
    key_bytes = torch.arange(columns * 128, dtype=torch.int64).remainder(256).to(torch.uint8)
    keys = key_bytes.view(columns, 128).view(torch.float8_e4m3fn)
    # Signed zero and noncanonical NaN payloads make accidental conversion visible.
    scale_bits = torch.tensor([0x00000000, -2147483648, 0x3F800000, 0x7FC01234], dtype=torch.int32)
    scales = scale_bits.repeat((columns + 3) // 4)[:columns].view(torch.float32)
    q = torch.empty((1, 64, 128), dtype=torch.float8_e4m3fn)
    weights = torch.empty((1, 64), dtype=torch.float32)
    ends = torch.tensor([columns], dtype=torch.int32)
    marker = object()
    metadata = torch.zeros((133, 2), dtype=torch.int32)
    packed_bytes = torch.arange(((columns + 63) // 64) * 8448).to(torch.uint8).view(-1, 8448)

    def pack(actual_keys, actual_scales):
        assert actual_keys is keys and actual_scales is scales
        return packed_bytes

    monkeypatch.setattr(echo, "_pack_q1_keys", pack)

    def make_metadata(context_lens, block_kv, num_sms, indices):
        assert context_lens.shape == (1, 1)
        assert context_lens.data_ptr() == ends.data_ptr()
        assert block_kv == 64 and num_sms == 132 and indices is None
        return metadata

    def paged(**kwargs):
        pages = (columns + 63) // 64
        packed = kwargs["kv_cache"]
        assert packed.shape == (pages, 64, 1, 132)
        assert packed.dtype == torch.uint8
        assert packed.data_ptr() == packed_bytes.data_ptr()
        assert kwargs["q"][0].data_ptr() == q.data_ptr()
        assert kwargs["q"][1] is None
        assert kwargs["weights"] is weights
        assert kwargs["schedule_meta"] is metadata
        assert kwargs["max_context_len"] == (columns + 255) // 256 * 256
        assert kwargs["indices"] is None
        assert kwargs["block_table"].tolist() == [list(range(pages))]
        return marker

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_num_sms=lambda: 132,
            get_paged_mqa_logits_metadata=make_metadata,
            fp8_fp4_paged_mqa_logits=paged,
        ),
    )
    result = echo._paged_q1_logits(q, keys, weights, scales, ends, (columns + 255) // 256 * 256)
    assert result is marker
    assert torch.equal(keys.view(torch.uint8).flatten(), key_bytes)
    assert torch.equal(scales.view(torch.int32), scale_bits.repeat((columns + 3) // 4)[:columns])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("columns", [1, 63, 64, 65, 32768, 65537])
def test_gpu_pack_preserves_special_bits_and_zeroes_padding(columns):
    key_bytes = torch.arange(columns * 128, device="cuda", dtype=torch.int64).to(torch.uint8)
    keys = key_bytes.view(columns, 128).view(torch.float8_e4m3fn)
    scale_bits = torch.tensor(
        [0, -2147483648, 0x3F800000, 0x7FC01234], dtype=torch.int32, device="cuda"
    )
    scales = scale_bits.repeat((columns + 3) // 4)[:columns].view(torch.float32)
    actual = echo._pack_q1_keys(keys, scales)
    packed_keys = actual[:, :8192].contiguous().flatten()
    packed_scales = actual[:, 8192:].contiguous().flatten()
    assert torch.equal(packed_keys[: columns * 128], key_bytes)
    assert torch.equal(packed_scales[: columns * 4], scales.view(torch.uint8))
    assert bool(packed_keys[columns * 128 :].eq(0).all())
    assert bool(packed_scales[columns * 4 :].eq(0).all())
