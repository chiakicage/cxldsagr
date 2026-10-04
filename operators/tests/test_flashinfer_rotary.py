"""CPU dispatch/ownership checks for the caller-supplied rotary destination."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from operators import flashinfer


def fixture(dtype=torch.bfloat16):
    q = torch.randn(5, 4, 192, dtype=dtype)[..., 128:]
    k = torch.randn(5, 576, dtype=dtype)[..., 512:]
    output = torch.full((5, 4, 576), 0.625, dtype=dtype)
    positions = torch.arange(5)
    trig = torch.randn(5, 64)
    return q, k, positions, trig, output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_direct_rotary_destination_preserves_layout_inputs_and_owned_key(monkeypatch, dtype):
    q, k, positions, trig, output = fixture(dtype)
    saved_q, saved_k = q.clone(), k.clone()
    calls = []

    def vendor(query, key, query_out, key_out, cache, ids, interleaved):
        assert query is q and cache is trig and ids is positions and interleaved is True
        assert key.shape == (5, 1, 64) and key.stride(0) == 576
        assert query_out.stride() == (4 * 576, 576, 1)
        assert key_out.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
        query_out.copy_(query + 1)
        key_out.copy_(key + 2)
        calls.append(True)

    monkeypatch.setitem(
        sys.modules, "flashinfer.rope", SimpleNamespace(_apply_rope_pos_ids_cos_sin_cache=vendor)
    )
    monkeypatch.setattr(flashinfer, "can_use_flashinfer", lambda _: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    destination = output[..., 512:]
    actual_q, actual_k = flashinfer.rotary_pair_into(
        q, k, positions, trig, destination, interleaved=True
    )
    assert actual_q is destination and calls == [True]
    torch.testing.assert_close(actual_q, q + 1, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, k + 2, rtol=0, atol=0)
    assert torch.all(output[..., :512] == 0.625)
    torch.testing.assert_close(q, saved_q, rtol=0, atol=0)
    torch.testing.assert_close(k, saved_k, rtol=0, atol=0)
    previous = actual_k.clone()
    _, next_key = flashinfer.rotary_pair_into(q, k, positions, trig, destination, interleaved=True)
    assert next_key.data_ptr() != actual_k.data_ptr()
    next_key.zero_()
    torch.testing.assert_close(actual_k, previous, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case", ["query_alignment", "key_alignment", "output_stride", "split_half", "partial_rotation"]
)
def test_unsupported_rotary_layout_uses_ordinary_adapter(monkeypatch, case):
    q, k, positions, trig, output = fixture()
    destination = output[..., 512:]
    interleaved = True
    if case == "query_alignment":
        q = torch.randn(5, 4, 195).bfloat16()[..., 131:]
    elif case == "key_alignment":
        k = torch.randn(5, 579).bfloat16()[..., 515:]
    elif case == "output_stride":
        destination = torch.empty(5, 4, 128).bfloat16()[..., ::2]
    elif case == "split_half":
        interleaved = False
    else:
        trig = trig[:, :32].contiguous()
    expected_q, expected_k = q + 3, k + 4
    calls = []

    def ordinary(query, key, ids, cache, *, interleaved):
        assert query is q and key is k and ids is positions and cache is trig
        calls.append(interleaved)
        return expected_q, expected_k

    monkeypatch.setattr(flashinfer, "can_use_flashinfer", lambda _: True)
    monkeypatch.setattr(flashinfer, "rotary_pair", ordinary)
    actual = flashinfer.rotary_pair_into(
        q, k, positions, trig, destination, interleaved=interleaved
    )
    assert actual[0] is destination and actual[1] is expected_k and calls == [interleaved]
    torch.testing.assert_close(destination, expected_q, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "q_shape",
        "key_shape",
        "key_dtype",
        "output_shape",
        "output_dtype",
        "positions_shape",
        "positions_dtype",
        "trig_dtype",
        "trig_stride",
        "alias_q",
        "alias_k",
    ],
)
def test_rotary_destination_rejects_invalid_metadata_before_vendor(monkeypatch, case):
    q, k, positions, trig, output = fixture()
    destination = output[..., 512:]
    if case == "q_shape":
        q = q[:, 0]
    elif case == "key_shape":
        k = k.unsqueeze(1)
    elif case == "key_dtype":
        k = k.float()
    elif case == "output_shape":
        destination = destination[:, :1]
    elif case == "output_dtype":
        destination = destination.float()
    elif case == "positions_shape":
        positions = positions[1:]
    elif case == "positions_dtype":
        positions = positions.float()
    elif case == "trig_dtype":
        trig = trig.bfloat16()
    elif case == "trig_stride":
        trig = trig[:, ::2]
    elif case == "alias_q":
        destination = q
    else:
        destination = k[:, None].expand(-1, 4, -1)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid metadata must not reach the vendor adapter")

    monkeypatch.setattr(flashinfer, "rotary_pair", forbidden)
    with pytest.raises(ValueError):
        flashinfer.rotary_pair_into(q, k, positions, trig, destination, interleaved=True)


def test_overlapping_output_cannot_enter_vectorized_vendor(monkeypatch):
    q, k, positions, trig, _ = fixture()
    destination = torch.empty(5, 1, 64).bfloat16().expand(-1, 4, -1)
    calls = []

    def ordinary(*args, **kwargs):
        calls.append(True)
        return torch.zeros_like(q), torch.zeros_like(k)

    monkeypatch.setattr(flashinfer, "can_use_flashinfer", lambda _: True)
    monkeypatch.setattr(flashinfer, "rotary_pair", ordinary)
    with pytest.raises(RuntimeError, match="single memory location"):
        flashinfer.rotary_pair_into(q, k, positions, trig, destination, interleaved=True)
    assert calls == [True]
