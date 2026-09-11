import math
from types import SimpleNamespace

import pytest
import torch

from model_run.deepseek_v32_decode import (
    CONFIG_PATH,
    V32DecodeRunner,
    index_cache_views,
    load_config,
    quantize_kv_v32,
)
from model_run.deepseek_v32_ops import (
    FlashInferV32Ops,
    attention_scale,
    quantize_index,
    yarn_cos_sin_cache,
)


def test_index_quantization():
    x = torch.randn(8, 128) * torch.arange(1, 9)[:, None]
    x[0] = 0
    data, scale = quantize_index(x)
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale.log2(), scale.log2().round())
    torch.testing.assert_close(data.float() * scale, x, atol=0.01, rtol=0.063)


def test_yarn_cache_and_scale():
    cfg = load_config(CONFIG_PATH)
    cache = yarn_cos_sin_cache(cfg, 8193, "cpu")
    torch.testing.assert_close(cache[0, :32], torch.ones(32))
    torch.testing.assert_close(cache[0, 32:], torch.zeros(32))
    # Highest frequency is unchanged; lowest frequency is interpolated by factor 40.
    angles = torch.tensor([8192.0, 8192 * 10000 ** (-62 / 64) / 40])
    torch.testing.assert_close(cache[8192, [0, 31]], angles.cos())
    assert attention_scale(cfg) == pytest.approx(192**-0.5 * (1 + 0.1 * math.log(40)) ** 2)


def test_cache_update_uses_projected_values():
    runner = object.__new__(V32DecodeRunner)
    runner.cfg = load_config(CONFIG_PATH)
    case = SimpleNamespace(
        kv_cache=torch.zeros(4, 64, 1, 656, dtype=torch.uint8),
        kv_current_packed=torch.empty(2, 1, 656, dtype=torch.uint8),
        index_kv_cache=torch.zeros(4, 64, 1, 132, dtype=torch.uint8),
        index_current_record=torch.empty(2, 132, dtype=torch.uint8),
        decode_block_ids=torch.tensor([0, 3]),
        decode_token_ids=torch.tensor([63, 0]),
    )
    projected = SimpleNamespace(
        kv_current=torch.randn(2, 576).bfloat16(), idx_k=torch.randn(2, 128).bfloat16()
    )
    runner.update_attention_cache(case, projected)
    runner.update_index_cache(case, projected)
    records = case.kv_cache[case.decode_block_ids, case.decode_token_ids]
    rope = records[:, 0, 528:].contiguous().view(torch.bfloat16)
    torch.testing.assert_close(rope, projected.kv_current[:, 512:])
    before = records.clone()
    projected.kv_current *= 2
    runner.update_attention_cache(case, projected)
    assert not torch.equal(before, case.kv_current_packed)
    assert not case.kv_cache[1].any()
    keys, scales = index_cache_views(case.index_kv_cache)
    values = (
        keys[case.decode_block_ids, case.decode_token_ids]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    scale = scales[case.decode_block_ids, case.decode_token_ids].contiguous().view(torch.float32)
    torch.testing.assert_close(values * scale, projected.idx_k.float(), atol=0.01, rtol=0.063)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@cuda
def test_flashinfer_norms():
    ops = FlashInferV32Ops(load_config(CONFIG_PATH))
    for dim, fn in [(1536, ops.q_norm), (512, ops.kv_norm)]:
        x = torch.randn(4, dim, device="cuda", dtype=torch.bfloat16)
        expected = (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        torch.testing.assert_close(fn(x), expected, atol=0.016, rtol=0.008)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) + 3
    expected = torch.nn.functional.layer_norm(x.float(), (128,), eps=1e-6).bfloat16()
    torch.testing.assert_close(ops.index_norm(x), expected, atol=0.016, rtol=0.008)


@cuda
@pytest.mark.parametrize("is_neox", [False, True])
def test_flashinfer_rope_against_complex_rotation(is_neox):
    ops = FlashInferV32Ops(load_config(CONFIG_PATH))
    positions = torch.tensor([0, 1, 4096, 8192, 163839], device="cuda")
    q = torch.randn(5, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    actual = ops.apply_rope(q, k, positions, is_neox=is_neox)
    cs = ops.cos_sin_cache[positions]
    for x, result in zip((q, k), actual):
        y = x.reshape(5, -1, 64).float()
        a, b = (y[..., :32], y[..., 32:]) if is_neox else (y[..., ::2], y[..., 1::2])
        z = torch.complex(a, b) * torch.complex(cs[:, None, :32], cs[:, None, 32:])
        ref = torch.cat((z.real, z.imag), -1) if is_neox else torch.view_as_real(z).flatten(-2)
        torch.testing.assert_close(result, ref.reshape_as(x).bfloat16(), atol=0.016, rtol=0.008)


@cuda
def test_decode_smoke():
    torch.manual_seed(0)
    runner = V32DecodeRunner(load_config(CONFIG_PATH), "fp8")
    case = runner.make_case(1, 64)
    projected = runner.project(case)
    indices = runner.run_indexer(case, projected)
    logits = runner.compute_index_logits(case, projected)
    keys, scales = index_cache_views(case.index_kv_cache)
    k = keys.contiguous().view(torch.float8_e4m3fn).float().view(-1, 128)[:65]
    ks = scales.contiguous().view(torch.float32).view(-1, 1)[:65]
    q, qs = quantize_index(projected.idx_q)
    dots = torch.relu(q[0, 0].float() @ k.T) * ks.T
    weights = projected.idx_weights[0] * qs[0, 0, :, 0] * 128**-0.5
    expected_logits = (dots * weights[:, None]).sum(0)
    torch.testing.assert_close(logits[0, :65], expected_logits, atol=0.003, rtol=0.01)
    out = runner.run_attention(case, projected, indices)
    torch.cuda.synchronize()
    assert out.shape == (1, 7168)
    assert torch.isfinite(out).all()
    assert ((indices[:, :65] >= 0) & (indices[:, :65] <= 64)).all()
    assert (indices[:, 65:] == -1).all()
    torch.testing.assert_close(
        case.kv_current_packed,
        quantize_kv_v32(projected.kv_current[:, None, None, :])[:, 0],
    )
