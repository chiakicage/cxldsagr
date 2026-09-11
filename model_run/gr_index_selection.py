"""Bound logits memory while preserving each query's full causal top-k search."""

import torch

from model_run.deepseek_v32_decode import deep_gemm
from model_run.deepseek_v32_extend import V32ExtendRunner


def select_batched(q, keys, scales, weights, ends, topk=2048, batch=128):
    indices, samples = [], []
    sample_rows = sorted({0, len(ends) // 2, len(ends) - 1})
    for start in range(0, len(ends), batch):
        stop = min(start + batch, len(ends))
        local_ends = ends[start:stop]
        logits = deep_gemm.fp8_fp4_mqa_logits(
            (q[start:stop].contiguous(), None),
            (keys, scales),
            weights[start:stop].contiguous(),
            torch.zeros_like(local_ends),
            local_ends,
            False,
            0,
            torch.float32,
        )
        valid = torch.arange(len(keys), device=q.device)[None, :] < local_ends[:, None]
        assert torch.isfinite(logits.masked_select(valid)).all()
        indices.append(V32ExtendRunner.select_indices(None, logits, local_ends, topk))
        for row in sample_rows:
            if start <= row < stop:
                samples.append(logits[row - start].clone())
        del logits, valid
    return torch.cat(indices), torch.stack(samples)
