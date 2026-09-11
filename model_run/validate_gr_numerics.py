"""Independent PyTorch references and exact CPU staging checks on real GR activations."""

import torch

from model_run.deepseek_v32_decode import attention_scale


def error(actual, expected):
    a, b = actual.float(), expected.float()
    diff = a - b
    return {
        "nrmse": float(diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)),
        "max_abs": float(diff.abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
    }


def sample_rows(m, device):
    return torch.tensor(sorted({0, m // 2, m - 1}), device=device)


def ref_linear(layer, x):
    # Independently quantize activations, decode checkpoint blocks and use FP32 GEMM.
    xf = x.float().reshape(x.shape[0], -1, 128)
    scale = torch.exp2(torch.ceil(torch.log2(xf.abs().amax(-1).clamp_min(1e-4) / 448)))
    quant = (xf / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
    data, ws = layer.weight
    weight = (
        data.float()
        * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)[: data.shape[0], : data.shape[1]]
    )
    return torch.nn.functional.linear(quant.flatten(1), weight)


def validate_index(runner, case, proj, logits, indices, start, stop, sampled_logits=False):
    rows = sample_rows(stop - start, logits.device)
    q = proj.idx_q[rows, 0].float()
    qs = torch.exp2(torch.ceil(torch.log2(q.abs().amax(-1).clamp_min(1e-4) / 448)))
    q8 = (q / qs[..., None]).to(torch.float8_e4m3fn).float()
    k = case.index_keys[:stop].float()
    dots = torch.matmul(q8, k.T).relu()
    weights = proj.idx_weights[rows] * qs * 128**-0.5
    ref = (dots * weights[..., None]).sum(1) * case.index_scales[:stop]
    ends = case.position_ids[start:stop][rows] + 1
    mask = torch.arange(stop, device=q.device)[None, :] < ends[:, None]
    actual_logits = logits[:, :stop] if sampled_logits else logits[rows, :stop]
    metric = error(actual_logits[mask], ref[mask])
    assert metric["nrmse"] < 1e-4, metric
    ref.masked_fill_(~mask, -torch.inf)
    top = ref.topk(min(2048, stop), -1).indices
    actual = indices[rows].long()
    # A near-tied cutoff may change the set, so also check the top-k objective gap.
    valid = actual >= 0
    selected_values = ref.gather(1, actual.clamp_min(0)).masked_fill(~valid, 0)
    ref_values = ref.gather(1, top).masked_fill(~torch.isfinite(ref.gather(1, top)), 0)
    gap = (ref_values.sum(1) - selected_values.sum(1)).abs() / ref_values.abs().sum(1).clamp_min(
        1e-12
    )
    metric["topk_objective_relative_gap"] = float(gap.max())
    assert metric["topk_objective_relative_gap"] < 1e-5, metric
    return metric


def validate_attention(runner, case, proj, indices, attn, start, stop, offload=False):
    rows = sample_rows(stop - start, indices.device)
    idx = indices[rows].long()
    records = case.kv_cache.view(-1, 656)[idx.clamp_min(0)].contiguous()
    latent = records[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scale = records[..., 512:528].contiguous().view(torch.float32).repeat_interleave(128, -1)
    latent *= scale
    rope = records[..., 528:].contiguous().view(torch.bfloat16).float()
    keys = torch.cat((latent, rope), -1)
    scores = torch.bmm(proj.q_attn[rows].float(), keys.transpose(1, 2)) * attention_scale(
        runner.cfg
    )
    scores.masked_fill_((idx < 0)[:, None, :], -torch.inf)
    ref = torch.bmm(scores.softmax(-1), latent)
    metric = error(attn[rows], ref)
    # Includes intentional latent-Q and PV FP8 quantization, unlike FP32 reference.
    metric["threshold_nrmse"] = 0.10
    assert metric["nrmse"] < 0.10 and metric["cosine"] > 0.99, metric
    if offload:
        unique, remap = torch.unique(indices.flatten().long(), sorted=True, return_inverse=True)
        assert unique.min() >= 0
        # Gather on CPU from actual packed cache, then stage precisely selected records.
        host_cache = case.kv_cache.view(-1, 656).cpu()
        gathered = host_cache[unique.cpu()].contiguous()
        staging = torch.zeros(
            ((len(unique) + 63) // 64, 64, 1, 656), device=indices.device, dtype=torch.uint8
        )
        staging.view(-1, 656)[: len(unique)].copy_(gathered.to(indices.device))
        torch.testing.assert_close(
            staging.view(-1, 656)[: len(unique)],
            case.kv_cache.view(-1, 656)[unique],
            rtol=0,
            atol=0,
        )
        remap = remap.view_as(indices).int()
        staged = runner.sparse_prefill(
            proj.q_attn, staging, remap, attention_scale(runner.cfg), 512, bf16_qk=False
        )[0]
        metric["cpu_staged_output_bitwise_equal"] = torch.equal(staged, attn)
        metric["cpu_staged_max_abs"] = float((staged.float() - attn.float()).abs().max())
        assert metric["cpu_staged_output_bitwise_equal"], metric
    return metric


def validate_ffn(mlp, x, actual):
    rows = sample_rows(x.shape[0], x.device)
    gate, up, down = mlp
    g = ref_linear(gate, x[rows]).bfloat16()
    u = ref_linear(up, x[rows]).bfloat16()
    inter = (torch.nn.functional.silu(g.float()) * u.float()).bfloat16()
    ref = ref_linear(down, inter)
    metric = error(actual[rows], ref)
    assert metric["nrmse"] < 0.02, metric
    return metric
