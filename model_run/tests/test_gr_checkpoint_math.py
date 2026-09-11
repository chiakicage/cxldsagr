"""Checkpoint layout, norm, projection and RoPE checks independent of custom kernels."""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from model_run.measure_gr_mla_cache_union import config_from_checkpoint
from model_run.measure_gr_multilayer_hits import load_layer
from model_run.validate_gr_numerics import error, ref_linear

MODEL = Path("models/DeepSeek-V3.2")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference_rope(x, positions, cfg, neox):
    # Independent float64 frequency construction and complex rotation.
    d = 64
    lo = max(
        math.floor(
            d * math.log(4096 / (cfg.beta_fast * 2 * math.pi)) / (2 * math.log(cfg.rope_theta))
        ),
        0,
    )
    hi = min(
        math.ceil(
            d * math.log(4096 / (cfg.beta_slow * 2 * math.pi)) / (2 * math.log(cfg.rope_theta))
        ),
        d - 1,
    )
    grid = torch.arange(d // 2, device=x.device, dtype=torch.float64)
    blend = ((grid - lo) / (hi - lo)).clamp(0, 1)
    freq = cfg.rope_theta ** (-2 * grid / d) * (1 - blend + blend / cfg.rope_factor)
    phases = torch.polar(
        torch.ones((len(positions), 32), device=x.device, dtype=torch.float64),
        positions[:, None] * freq,
    )
    xf = x.double().reshape(len(positions), -1, 64)
    real, imag = (xf[..., :32], xf[..., 32:]) if neox else (xf[..., ::2], xf[..., 1::2])
    rotated = torch.complex(real, imag) * phases[:, None, :]
    out = (
        torch.cat((rotated.real, rotated.imag), -1)
        if neox
        else torch.view_as_real(rotated).flatten(-2)
    )
    return out.reshape_as(x).float()


@pytest.mark.parametrize("layer", [0, 1, 2])
@torch.inference_mode()
def test_checkpoint_math(layer):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = config_from_checkpoint(json.loads((MODEL / "config.json").read_text()))
    with safe_open(MODEL / "model-00001-of-000163.safetensors", framework="pt") as ckpt:
        runner, _norms, _mlp = load_layer(ckpt, cfg, layer)
        attn = f"model.layers.{layer}.self_attn."
        # Decode by block reshape, independent of the loader's repeat_interleave.
        raw = ckpt.get_tensor(attn + "kv_b_proj.weight").cuda().float().reshape(256, 128, 4, 128)
        scales = ckpt.get_tensor(attn + "kv_b_proj.weight_scale_inv").cuda()
        w = (raw * scales[:, None, :, None]).reshape(128, 256, 512).bfloat16()
        torch.testing.assert_close(runner.wk_b.weight, w[:, :128].transpose(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(runner.wv_b.weight, w[:, 128:], rtol=0, atol=0)
        positions = torch.tensor([0, 1, 4095, 4096, 65536, 69631], device="cuda")
        for neox in (False, True):
            q = torch.randn(6, 4, 64, device="cuda").bfloat16()
            k = torch.randn(6, 64, device="cuda").bfloat16()
            outputs = runner.ops.apply_rope(q, k, positions, is_neox=neox)
            for x, out in zip((q, k), outputs):
                ref = reference_rope(x, positions, cfg, neox)
                metric = error(out, ref)
                assert metric["nrmse"] < 0.004, metric
        for size, weight, fn in [
            (1536, runner.ops.q_weight, runner.ops.q_norm),
            (512, runner.ops.kv_weight, runner.ops.kv_norm),
        ]:
            x = torch.randn(128, size, device="cuda").bfloat16()
            ref = torch.nn.functional.rms_norm(x.float(), (size,), weight, eps=cfg.norm_eps)
            assert error(fn(x), ref)["nrmse"] < 0.003
        x = torch.randn(128, 128, device="cuda").bfloat16()
        ref = torch.nn.functional.layer_norm(
            x.float(), (128,), runner.ops.index_weight, runner.ops.index_bias, cfg.norm_eps
        )
        assert error(runner.ops.index_norm(x), ref)["nrmse"] < 0.004
        for name in ("wq_a", "wq_b", "wkv_a", "index_wqi", "index_wki"):
            linear = getattr(runner, name)
            x = torch.randn(128, linear.in_features, device="cuda").bfloat16()
            actual = linear(x)[:3]
            ref = ref_linear(linear, x[:3])
            metric = error(actual, ref)
            assert metric["nrmse"] < 0.003, (name, metric)
        # O projection must preserve the non-power-of-two checkpoint scales.
        raw = ckpt.get_tensor(attn + "o_proj.weight").cuda().float().reshape(56, 128, 128, 128)
        scales = ckpt.get_tensor(attn + "o_proj.weight_scale_inv").cuda()
        w = (raw * scales[:, None, :, None]).reshape(7168, 16384).bfloat16()
        x = torch.randn(3, 16384, device="cuda").bfloat16()
        ref = torch.nn.functional.linear(x.float(), w.float())
        assert error(runner.wo(x), ref)["nrmse"] < 0.003


@pytest.mark.parametrize("neox", [False, True])
@torch.inference_mode()
def test_long_history_rope(neox):
    from dataclasses import replace

    from model_run.deepseek_v32_ops import FlashInferV32Ops

    torch.manual_seed(42)
    cfg = config_from_checkpoint(json.loads((MODEL / "config.json").read_text()))
    cfg = replace(cfg, max_seq_len=1048576 + 4096)
    ops = FlashInferV32Ops(cfg)
    positions = torch.tensor([524287, 528383, 1048575, 1052671], device="cuda")
    q = torch.randn(4, 4, 64, device="cuda").bfloat16()
    k = torch.randn(4, 64, device="cuda").bfloat16()
    for x, out in zip((q, k), ops.apply_rope(q, k, positions, is_neox=neox)):
        metric = error(out, reference_rope(x, positions, cfg, neox))
        # Existing FP32 phase construction loses more precision near 1M positions.
        # Bound that approximation explicitly; do not claim short-context accuracy.
        assert metric["nrmse"] < 0.01 and metric["cosine"] > 0.9999, metric
