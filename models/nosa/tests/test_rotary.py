"""Static LongRoPE semantics, derived cache lifetime, and fused CUDA rotation."""

from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from models.nosa.rotary import NosaRotaryEmbedding, apply_rotary_qk
from models.nosa.tests.test_model import dense_attention, initialized_model, tiny_config
from operators import flashinfer


def rope_config(*, head_dim=8, use_longrope=True, max_position_embeddings=66560):
    scaling = None
    if use_longrope:
        factors = [1.0 + (index % 7) * 0.25 for index in range(head_dim // 2)]
        scaling = {
            "rope_type": "longrope",
            "original_max_position_embeddings": 32768,
            "short_factor": factors,
            "long_factor": list(factors),
            "attention_factor": 1.17,
        }
    return tiny_config(
        head_dim=head_dim,
        rope_theta=500000.0,
        rope_scaling=scaling,
        max_position_embeddings=max_position_embeddings,
    )


def complex_reference(x, config, positions):
    """FP32 phases on the input device, then independent FP64 complex rotation.

    FP32 phase construction is part of the model contract, including at long
    positions. Complex multiplication avoids sharing the split-half formula
    or reading the implementation's cached trigonometric values.
    """
    indices = torch.arange(0, config.head_dim, 2, device=x.device, dtype=torch.float32)
    denominator = config.rope_theta ** (indices / config.head_dim)
    scale = 1.0
    if config.rope_scaling is not None:
        denominator *= torch.tensor(
            config.rope_scaling["short_factor"], device=x.device, dtype=torch.float32
        )
        scale = config.rope_scaling["attention_factor"]
    angles = (positions.float()[:, None] * (1.0 / denominator)[None, :]).cpu().double()
    phases = torch.polar(torch.full_like(angles, scale), angles)
    real, imaginary = x.cpu().double().chunk(2, dim=-1)
    result = torch.complex(real, imaginary) * phases[:, None, :]
    return torch.cat((result.real, result.imag), dim=-1).to(x.dtype)


def packed_qkv_views(*, device, dtype, tokens=17, padded=False):
    """Actual NOSA 32-Q/2-KV heads, plus guard rows and optional aligned padding."""
    prefix = padding = 8 if padded else 0
    generator = torch.Generator(device=device).manual_seed(83)
    storage = torch.randn(
        tokens + 2, prefix + 4608 + padding, device=device, generator=generator
    ).to(dtype)
    packed = storage[1:-1, prefix : prefix + 4608]
    q, k, v = packed.split((4096, 256, 256), dim=-1)
    views = q.view(tokens, 32, 128), k.view(tokens, 2, 128), v.view(tokens, 2, 128)
    protected = torch.ones_like(storage, dtype=torch.bool)
    protected[1:-1, prefix : prefix + 4352] = False
    return storage, views, protected


def unsupported_qk_views(layout, *, device):
    def make(heads):
        if layout == "feature_stride":
            return torch.randn(3, heads, 256, device=device, dtype=torch.bfloat16)[..., ::2]
        if layout == "head_gap":
            return torch.randn(3, 2 * heads, 128, device=device, dtype=torch.bfloat16)[:, ::2]
        if layout == "overlapping_rows":
            return torch.randn(1, heads, 128, device=device, dtype=torch.bfloat16).expand(3, -1, -1)
        if layout == "unaligned_offset":
            return torch.randn(3 * heads * 128 + 1, device=device, dtype=torch.bfloat16)[1:].view(
                3, heads, 128
            )
        if layout == "unaligned_row":
            return torch.randn(3, heads * 128 + 1, device=device, dtype=torch.bfloat16)[
                :, :-1
            ].view(3, heads, 128)
        raise AssertionError(f"Unknown layout: {layout}")

    return make(32), make(2)


@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_cpu_packed_qkv_rotation_preserves_shared_input_storage(padded):
    config = rope_config(head_dim=128, max_position_embeddings=32)
    positions, table = NosaRotaryEmbedding()(config, 7, 24, device="cpu")
    storage, (q, k, v), _ = packed_qkv_views(device="cpu", dtype=torch.float32, padded=padded)
    before = storage.clone()
    assert q.stride() == k.stride() == v.stride() == (storage.shape[1], 128, 1)
    assert not q.is_contiguous() and k.storage_offset() > q.storage_offset()
    actual_q, actual_k = apply_rotary_qk(q, k, positions, table)
    torch.testing.assert_close(actual_q, complex_reference(q, config, positions))
    torch.testing.assert_close(actual_k, complex_reference(k, config, positions))
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_longrope", [False, True])
@torch.inference_mode()
def test_cpu_split_half_rotation_long_positions_and_input_preservation(dtype, use_longrope):
    config = rope_config(use_longrope=use_longrope)
    rotary = NosaRotaryEmbedding()
    _, table = rotary(config, 0, 1, device="cpu")
    positions = torch.tensor([0, 1, 32767, 32768, 65535, 65536, 66559])
    generator = torch.Generator().manual_seed(81)
    q = torch.randn(7, 4, config.head_dim, generator=generator).to(dtype)
    k = torch.randn(7, 2, config.head_dim, generator=generator).to(dtype)
    before = q.clone(), k.clone()
    actual = apply_rotary_qk(q, k, positions, table)
    tolerance = 2 * torch.finfo(dtype).eps
    for output, original, saved in zip(actual, (q, k), before, strict=True):
        assert output.shape == original.shape and output.dtype == dtype
        torch.testing.assert_close(
            output, complex_reference(saved, config, positions), atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(original, saved, atol=0, rtol=0)


@torch.inference_mode()
def test_derived_cache_reuses_positions_and_stays_out_of_checkpoint():
    config = rope_config(max_position_embeddings=64)
    rotary = NosaRotaryEmbedding()
    first_positions, first_table = rotary(config, 0, 5, device="cpu")
    later_positions, later_table = rotary(config, 31, 37, device="cpu")
    assert later_table is first_table
    assert later_table.dtype == torch.float32
    assert later_table.shape == (64, config.head_dim)
    assert first_positions.untyped_storage().data_ptr() == (
        later_positions.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(later_positions, torch.arange(31, 37))
    assert not rotary.state_dict()

    grown_positions, grown_table = rotary(config, 63, 70, device="cpu")
    torch.testing.assert_close(grown_positions, torch.arange(63, 70))
    torch.testing.assert_close(grown_table[:64], first_table, atol=0, rtol=0)
    assert grown_table.shape[0] >= 70


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_module_dtype_conversion_rebuilds_fp32_phases_without_rounding(dtype):
    config = rope_config(max_position_embeddings=128)
    rotary = NosaRotaryEmbedding()
    _, expected = rotary(config, 0, 128, device="cpu")
    expected = expected.clone()
    rotary.to(dtype=dtype)
    positions, actual = rotary(config, 0, 128, device="cpu")
    assert positions.dtype == torch.int64 and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_config_changes_invalidate_cached_longrope_values():
    config = rope_config(max_position_embeddings=64)
    rotary = NosaRotaryEmbedding()
    _, original = rotary(config, 0, 64, device="cpu")
    config.rope_scaling["short_factor"][1] = 3.75
    config.rope_scaling["long_factor"][1] = 3.75
    config.rope_scaling["attention_factor"] = 1.3
    positions, updated = rotary(config, 0, 64, device="cpu")
    assert updated is not original
    q = torch.ones(64, 4, config.head_dim)
    k = torch.ones(64, 2, config.head_dim)
    actual_q, actual_k = apply_rotary_qk(q, k, positions, updated)
    for actual, values in ((actual_q, q), (actual_k, k)):
        torch.testing.assert_close(actual, complex_reference(values, config, positions))

    config = replace(config, rope_theta=10000.0)
    _, changed_theta = rotary(config, 0, 64, device="cpu")
    assert changed_theta is not updated
    torch.testing.assert_close(
        apply_rotary_qk(q, k, positions, changed_theta)[0],
        complex_reference(q, config, positions),
    )


@torch.inference_mode()
def test_meta_checkpoint_assignment_materializes_only_derived_runtime_cache():
    config = rope_config(max_position_embeddings=16)
    original = initialized_model(config)
    tokens = torch.tensor([1, 3, 7, 11, 4])
    expected = original(tokens)
    checkpoint = original.state_dict()
    assert all("rotary" not in key for key in checkpoint)
    restored = type(original)(config, device="meta", attention=original.attention)
    restored.load_state_dict(checkpoint, assign=True)
    actual = restored(tokens)
    assert restored.model.rotary.cos_sin_cache.device.type == "cpu"
    assert restored.model.rotary.cos_sin_cache.dtype == torch.float32
    assert set(restored.state_dict()) == set(checkpoint)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("changed", ["attention_factor", "frequency_factor"])
@torch.inference_mode()
def test_changed_longrope_config_rejects_old_kv_and_accepts_fresh_cache(changed):
    config = rope_config(max_position_embeddings=16)
    model = initialized_model(config)
    tokens = torch.tensor([1, 3, 7, 11, 4])
    stale = model.new_cache(len(tokens))
    model(tokens[:3], stale)
    if changed == "attention_factor":
        config.rope_scaling["attention_factor"] = 1.3
    else:
        config.rope_scaling["short_factor"][1] = 3.75
        config.rope_scaling["long_factor"][1] = 3.75
    with pytest.raises(ValueError, match="KV cache must match the model config"):
        model(tokens[3:], stale)
    assert stale.length == 3

    expected = model(tokens)
    fresh = model.new_cache(len(tokens))
    actual = torch.cat((model(tokens[:3], fresh), model(tokens[3:], fresh)))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert fresh.length == len(tokens)
    model.cache_manager.release(stale)
    model.cache_manager.release(fresh)


class RejectHostTransfers(TorchDispatchMode):
    """A warm RoPE call must stay on device without reading CUDA scalars."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("RoPE read a device scalar on the host")
        if func == torch.ops.aten._to_copy.default and args[0].device.type == "cpu":
            target = kwargs.get("device")
            if target is not None and torch.device(target).type == "cuda":
                raise AssertionError("Warm RoPE copied host data to CUDA")
        return func(*args, **kwargs)


class RejectTensorCopies(RejectHostTransfers):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.ops.aten.clone.default,
            torch.ops.aten.copy_.default,
            torch.ops.aten._to_copy.default,
        ):
            raise AssertionError("Packed QKV RoPE copied a tensor")
        return super().__torch_dispatch__(func, types, args, kwargs)


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for actual FlashInfer RoPE"
)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_longrope", [False, True])
@torch.inference_mode()
def test_cuda_qk_longrope_matches_complex_reference_and_chunked_rotation(dtype, use_longrope):
    pytest.importorskip("flashinfer")
    config = rope_config(head_dim=128, use_longrope=use_longrope)
    rotary = NosaRotaryEmbedding()
    positions, table = rotary(config, 65528, 65545, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(82)
    q = torch.randn(17, 32, 128, device="cuda", generator=generator).to(dtype)
    k = torch.randn(17, 8, 128, device="cuda", generator=generator).to(dtype)
    q_chunks, k_chunks = q.clone(), k.clone()
    expected = complex_reference(q, config, positions), complex_reference(k, config, positions)
    actual = apply_rotary_qk(q, k, positions, table)
    assert actual[0] is q and actual[1] is k
    # One BF16/FP16 output rounding, plus FP32 trig/FMA rounding near ties.
    tolerance = torch.finfo(dtype).eps
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output.cpu(), reference, atol=tolerance, rtol=tolerance)

    with RejectHostTransfers():
        for start, end in ((0, 5), (5, 16), (16, 17)):
            chunk_positions, chunk_table = rotary(
                config, 65528 + start, 65528 + end, device=q.device
            )
            assert chunk_table is table
            apply_rotary_qk(q_chunks[start:end], k_chunks[start:end], chunk_positions, chunk_table)
    torch.testing.assert_close(q_chunks, q, atol=0, rtol=0)
    torch.testing.assert_close(k_chunks, k, atol=0, rtol=0)

    final_positions, table = rotary(config, 66559, 66560, device=q.device)
    q_last, k_last = q[:1].clone(), k[:1].clone()
    final_expected = complex_reference(q_last, config, final_positions)
    final_q, _ = apply_rotary_qk(q_last, k_last, final_positions, table)
    torch.testing.assert_close(final_q.cpu(), final_expected, atol=tolerance, rtol=tolerance)


@cuda_required
@torch.inference_mode()
def test_cuda_device_and_dtype_conversion_rebuilds_unrounded_cache():
    config = rope_config(head_dim=128, max_position_embeddings=64)
    rotary = NosaRotaryEmbedding()
    rotary(config, 0, 64, device="cpu")
    rotary.to(device="cuda", dtype=torch.bfloat16)
    positions, table = rotary(config, 0, 64, device="cuda")
    assert positions.is_cuda and table.is_cuda and table.dtype == torch.float32
    expected = table.clone()
    rotary.to(dtype=torch.float16)
    _, rebuilt = rotary(config, 0, 64, device="cuda")
    torch.testing.assert_close(rebuilt, expected, atol=0, rtol=0)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_cuda_packed_qkv_rope_uses_original_strides_and_preserves_v(dtype, padded):
    pytest.importorskip("flashinfer")
    config = rope_config(head_dim=128)
    positions, table = NosaRotaryEmbedding()(config, 65528, 65545, device="cuda")
    storage, (q, k, v), protected = packed_qkv_views(device="cuda", dtype=dtype, padded=padded)
    before = storage.clone()
    expected_q, expected_k = (
        complex_reference(q, config, positions),
        complex_reference(k, config, positions),
    )
    assert q.stride() == k.stride() == v.stride() == (storage.shape[1], 128, 1)
    assert not q.is_contiguous() and k.storage_offset() > q.storage_offset()
    # Resolve lazy kernel loading before asserting no repacking on this path.
    apply_rotary_qk(q.clone(), k.clone(), positions, table)
    with RejectTensorCopies():
        actual_q, actual_k = apply_rotary_qk(q, k, positions, table)
    assert actual_q is q and actual_k is k
    tolerance = torch.finfo(dtype).eps
    torch.testing.assert_close(actual_q.cpu(), expected_q, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_k.cpu(), expected_k, atol=tolerance, rtol=tolerance)
    # Includes V, row padding, and guard tokens at both ends of the allocation.
    torch.testing.assert_close(storage[protected], before[protected], atol=0, rtol=0)


@cuda_required
@pytest.mark.parametrize(
    "layout",
    ["feature_stride", "head_gap", "overlapping_rows", "unaligned_offset", "unaligned_row"],
)
@torch.inference_mode()
def test_cuda_unsupported_rope_layout_uses_nonmutating_fallback(monkeypatch, layout):
    config = rope_config(head_dim=128, max_position_embeddings=16)
    positions, table = NosaRotaryEmbedding()(config, 7, 10, device="cuda")
    q, k = unsupported_qk_views(layout, device="cuda")
    before = q.clone(), k.clone()

    def unsupported_kernel(*args):
        raise AssertionError("Unsupported RoPE view reached the vectorized CUDA kernel")

    monkeypatch.setattr(flashinfer, "apply_rope_with_cos_sin_cache", unsupported_kernel)
    actual = apply_rotary_qk(q, k, positions, table)
    tolerance = torch.finfo(q.dtype).eps
    for output, original, saved in zip(actual, (q, k), before, strict=True):
        torch.testing.assert_close(
            output.cpu(),
            complex_reference(saved, config, positions),
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(original, saved, atol=0, rtol=0)


@cuda_required
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cuda_attention_accepts_rotated_packed_qkv_views(tokens, dtype):
    pytest.importorskip("flashinfer")
    config = rope_config(head_dim=128, max_position_embeddings=32)
    positions, table = NosaRotaryEmbedding()(config, 0, tokens, device="cuda")
    storage, (q, k, v), protected = packed_qkv_views(device="cuda", dtype=dtype, tokens=tokens)
    before = storage.clone()
    q, k = apply_rotary_qk(q, k, positions, table)
    expected = dense_attention(q.cpu(), k.cpu(), v.cpu())
    actual = flashinfer.FlashInferFullAttention()(q, k, v)
    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(storage[protected], before[protected], atol=0, rtol=0)
