import pytest
import torch

from model_run.deepseek_v32_decode import CONFIG_PATH, load_config, quantize_index, quantize_kv_v32
from model_run.deepseek_v32_extend import V32ExtendRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("history", [7, 4096])
def test_extend_index_logits_and_chunk_causality(history):
    torch.manual_seed(12)
    runner = V32ExtendRunner(load_config(CONFIG_PATH), chunk_size=3)
    case = runner.make_case(history, 5)
    projected = runner.project_chunk(case, 0, 5)
    runner.write_chunk(case, projected, 0)
    expected_cache = quantize_kv_v32(projected.kv_current[:, None, None, :])[:, 0]
    expected_keys, expected_scales = quantize_index(projected.idx_k)
    torch.testing.assert_close(
        case.kv_cache.view(-1, 1, 656)[history : history + 5], expected_cache, rtol=0, atol=0
    )
    torch.testing.assert_close(
        case.index_keys[history:].float(), expected_keys.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        case.index_scales[history:], expected_scales.flatten(), rtol=0, atol=0
    )
    logits, ends = runner.index_logits(case, projected, 0, 5)
    q, qs = quantize_index(projected.idx_q[:, 0])
    weights = projected.idx_weights * qs[..., 0] * 128**-0.5
    dots = torch.relu(torch.einsum("qhd,kd->qhk", q.float(), case.index_keys.float()))
    expected = (dots * case.index_scales[None, None, :] * weights[:, :, None]).sum(1)
    valid = torch.arange(history + 5, device="cuda")[None, :] < ends[:, None]
    torch.testing.assert_close(logits[valid], expected[valid], atol=0.003, rtol=0.01)
    logits.masked_fill_(~valid, 1e20)  # Ragged top-k must never read future scores.
    selected = runner.select_indices(logits, ends, 2048)
    for row in range(5):
        actual_ids = selected[row][selected[row] >= 0].long()
        ref_ids = torch.topk(logits[row, : history + row + 1], min(2048, history + row + 1)).indices
        torch.testing.assert_close(
            logits[row, actual_ids].sort().values,
            logits[row, ref_ids].sort().values,
            rtol=0,
            atol=0,
        )
    for row in range(5):
        valid_indices = selected[row][selected[row] >= 0]
        assert (valid_indices < history + 1 + row).all()
        assert valid_indices.unique().numel() == min(2048, history + 1 + row)
    # Only the last query may see the last new key.
    case.index_keys[-1] = (-case.index_keys[-1].float()).to(torch.float8_e4m3fn)
    changed, _ = runner.index_logits(case, projected, 0, 5)
    torch.testing.assert_close(
        changed[:4][valid[:4]], expected[:4][valid[:4]], atol=0.003, rtol=0.01
    )
    out_chunked = runner.run_once(case)
    runner.chunk_size = 5
    out_full = runner.run_once(case)
    assert out_full.shape == (5, 7168)
    assert torch.isfinite(out_full).all()
    # Changing GEMM M tiling may change FP8 accumulation rounding slightly.
    torch.testing.assert_close(out_chunked, out_full, atol=0.035, rtol=0.06)
    history_snapshot = case.kv_cache.view(-1, 1, 656)[:history].clone()
    runner.run_once(case)
    torch.testing.assert_close(case.kv_cache.view(-1, 1, 656)[:history], history_snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [128, 193, 512, 7168])
def test_fused_activation_quantization(width):
    from deep_gemm.utils import per_token_cast_to_fp8

    from model_run.deepseek_v32_extend_kernels import quantize_activation

    x = torch.randn(34, width, device="cuda", dtype=torch.bfloat16)[::2]
    x[0] = 0
    x[1] *= 0.0001
    x[2] *= 1000
    actual, scale = quantize_activation(x)
    expected, expected_scale = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
