"""Single-request FlashInfer dense attention adapter."""

import torch


class FlashInferFullAttention:
    """Single-request GQA over NHD K/V, already rotated by model LongRoPE."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashInfer attention requires CUDA and bfloat16 or float16 tensors")
        import flashinfer

        with torch.cuda.device(q.device):
            if q.shape[0] == 1:
                return flashinfer.single_decode_with_kv_cache(
                    q[0],
                    k,
                    v,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    use_tensor_cores=True,
                    sm_scale=q.shape[-1] ** -0.5,
                ).unsqueeze(0)
            # FlashInfer's causal mask aligns Q to the end of K. This also
            # handles a prefill chunk appended to an existing prefix cache.
            return flashinfer.single_prefill_with_kv_cache(
                q,
                k,
                v,
                causal=True,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=q.shape[-1] ** -0.5,
                backend="auto",
            )
