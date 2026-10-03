"""Checkpoint DeepSeek V3.2 transformer blocks with ECHO sparse attention.

Dense and routed expert weights retain the checkpoint FP8 representation.
MLP activations are chunked independently of context length. CUDA execution
uses the SM90 operators; explicit CPU execution of MLPs is a small-test oracle.
The caller owns cache begin/commit/rollback across all layers.
"""

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from models.deepseek_v32.echo_attention import EchoAttentionRunner
from models.deepseek_v32.echo_model import (
    CheckpointAttention,
    CheckpointLinear,
    CheckpointReader,
    Config,
    rms_norm,
)
from operators.deepseek_v32.linear.fp8 import grouped_fp8_linear, prepare_expert_routing


def route_experts(hidden, gate_weight, correction_bias, cfg):
    """V3.2 sigmoid scores, correction-biased grouped choice, original-score weights.

    Bias influences membership only. Group ranking uses the sum of its two
    largest corrected scores; weights use uncorrected sigmoid scores and are
    normalized before applying the model's routed scaling factor.
    """
    experts = cfg.n_routed_experts
    if cfg.scoring_func != "sigmoid":
        raise ValueError("DeepSeek V3.2 routing requires sigmoid scoring")
    if experts % cfg.n_group or experts // cfg.n_group < 2:
        raise ValueError("Expert groups must divide the experts and contain at least two")
    if not 1 <= cfg.topk_group <= cfg.n_group:
        raise ValueError("topk_group is outside the configured groups")
    if not 1 <= cfg.num_experts_per_tok <= cfg.topk_group * (experts // cfg.n_group):
        raise ValueError("Expert top-k must fit within the retained groups")
    if gate_weight.shape != (experts, cfg.dim) or correction_bias.shape != (experts,):
        raise ValueError("Gate weight or correction bias does not match the model")
    scores = F.linear(hidden.float(), gate_weight.float()).sigmoid()
    corrected = scores + correction_bias.float()
    grouped = corrected.reshape(len(hidden), cfg.n_group, -1)
    group_scores = grouped.topk(2, dim=-1).values.sum(-1)
    groups = group_scores.topk(cfg.topk_group, dim=-1).indices
    retained = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    corrected = grouped.masked_fill(~retained[..., None], -torch.inf).flatten(1)
    expert_ids = corrected.topk(cfg.num_experts_per_tok, dim=-1).indices.contiguous()
    weights = scores.gather(1, expert_ids)
    if cfg.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return expert_ids, (weights * cfg.routed_scaling_factor).contiguous()


class CheckpointMLP:
    """A dense or shared SwiGLU MLP loaded without dequantizing FP8 weights."""

    def __init__(self, reader, stem, cfg, *, device, linear_backend="fp8"):
        self.gate = CheckpointLinear(
            reader,
            stem + ".gate_proj",
            device=device,
            backend=linear_backend,
            block_size=cfg.weight_block_size,
        )
        self.up = CheckpointLinear(
            reader,
            stem + ".up_proj",
            device=device,
            backend=linear_backend,
            block_size=cfg.weight_block_size,
        )
        self.down = CheckpointLinear(
            reader,
            stem + ".down_proj",
            device=device,
            backend=linear_backend,
            block_size=cfg.weight_block_size,
        )

    def __call__(self, hidden):
        gate, up = self.gate(hidden), self.up(hidden)
        intermediate = (F.silu(gate.float()) * up.float()).bfloat16()
        return self.down(intermediate)


class CheckpointMoE:
    """GPU resident FP8 experts, grouped GEMMs, and a shared SwiGLU expert."""

    def __init__(self, reader, stem, cfg, *, device, linear_backend="fp8"):
        if tuple(cfg.weight_block_size) != (128, 128):
            raise ValueError("Grouped expert GEMM requires 128-by-128 weight blocks")
        self.cfg = cfg
        self.device = torch.device(device)
        self.gate_weight = reader.get_tensor(stem + ".gate.weight").to(
            device=device, dtype=torch.float32
        )
        self.correction_bias = reader.get_tensor(stem + ".gate.e_score_correction_bias").to(
            device=device, dtype=torch.float32
        )
        self.weights = {}
        self.scales = {}
        for projection, dimensions in (
            ("gate_proj", (cfg.moe_intermediate_size, cfg.dim)),
            ("up_proj", (cfg.moe_intermediate_size, cfg.dim)),
            ("down_proj", (cfg.dim, cfg.moe_intermediate_size)),
        ):
            weight = torch.empty(
                (cfg.n_routed_experts, *dimensions), device=device, dtype=torch.float8_e4m3fn
            )
            scale_shape = tuple((n + 127) // 128 for n in dimensions)
            scales = torch.empty(
                (cfg.n_routed_experts, *scale_shape), device=device, dtype=torch.float32
            )
            # Copy one mmap-backed expert directly into its final device slot.
            # Building a Python list followed by stack would double multi-GB allocations.
            for expert in range(cfg.n_routed_experts):
                prefix = f"{stem}.experts.{expert}.{projection}"
                source = reader.get_tensor(prefix + ".weight")
                source_scales = reader.get_tensor(prefix + ".weight_scale_inv")
                if source.dtype != torch.float8_e4m3fn or tuple(source.shape) != dimensions:
                    raise ValueError(f"Invalid FP8 expert weight {prefix}")
                if (
                    source_scales.dtype != torch.float32
                    or tuple(source_scales.shape) != scale_shape
                ):
                    raise ValueError(f"Invalid FP8 expert block scales {prefix}")
                weight[expert].copy_(source)
                scales[expert].copy_(source_scales)
            self.weights[projection], self.scales[projection] = weight, scales
        self.shared = (
            CheckpointMLP(
                reader, stem + ".shared_experts", cfg, device=device, linear_backend=linear_backend
            )
            if cfg.n_shared_experts
            else None
        )

    def __call__(self, hidden, *, scope=None):
        scope = scope or (lambda _: nullcontext())
        with scope("moe_routing"):
            expert_ids, weights = route_experts(
                hidden, self.gate_weight, self.correction_bias, self.cfg
            )
            routing = prepare_expert_routing(expert_ids, self.cfg.n_routed_experts)
        with scope("moe_experts"):
            gate = grouped_fp8_linear(
                hidden,
                expert_ids,
                self.weights["gate_proj"],
                self.scales["gate_proj"],
                routing=routing,
            )
            up = grouped_fp8_linear(
                hidden,
                expert_ids,
                self.weights["up_proj"],
                self.scales["up_proj"],
                routing=routing,
            )
            intermediate = (F.silu(gate.float()) * up.float()).bfloat16()
            del gate, up
            output = grouped_fp8_linear(
                intermediate,
                expert_ids,
                self.weights["down_proj"],
                self.scales["down_proj"],
                routing=routing,
            )
            output = (output.float() * weights[..., None]).sum(1)
        if self.shared is not None:
            with scope("moe_shared_experts"):
                output += self.shared(hidden).float()
        return output.bfloat16()


class CheckpointBlock:
    """Attention, residual, post-attention norm, and dense or routed SwiGLU.

    As in official inference, the MLP output and BF16 residual remain separate
    between blocks. Fused residual additions accumulate in FP32; normalization
    sees that unrounded sum before both outputs are converted back to BF16.
    """

    def __init__(
        self,
        model_path,
        layer_idx,
        device,
        reader=None,
        *,
        capacity,
        offload=False,
        slots=16384,
        chunk_size=1024,
        linear_backend="fp8",
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.cfg = Config.from_checkpoint(model_path)
        self.device = torch.device(device)
        self.layer_idx = layer_idx
        self.chunk_size = chunk_size
        reader = reader or CheckpointReader(model_path)
        layer = CheckpointAttention(
            model_path,
            layer_idx=layer_idx,
            device=device,
            reader=reader,
            linear_backend=linear_backend,
        )
        self.attention = EchoAttentionRunner(
            layer,
            capacity,
            offload=offload,
            slots=slots,
            chunk_size=chunk_size,
        )
        self.cache = self.attention.cache
        stem = f"model.layers.{layer_idx}."
        self.post_norm_weight = reader.get_tensor(stem + "post_attention_layernorm.weight").to(
            device=device, dtype=torch.float32
        )
        self.is_moe = layer_idx >= self.cfg.first_k_dense_replace
        cls = CheckpointMoE if self.is_moe else CheckpointMLP
        self.mlp = cls(reader, stem + "mlp", self.cfg, device=device, linear_backend=linear_backend)

    @torch.inference_mode()
    def forward(self, hidden, residual=None, *, scope=None):
        if hidden.ndim != 2 or hidden.shape[1] != self.cfg.dim or not len(hidden):
            raise ValueError("Block hidden input has an invalid shape")
        if hidden.dtype != torch.bfloat16:
            raise ValueError("Block hidden input must be BF16")
        if residual is not None and (
            residual.shape != hidden.shape
            or residual.device != hidden.device
            or residual.dtype != hidden.dtype
        ):
            raise ValueError("Residual must match hidden shape, device, and BF16 dtype")
        scope = scope or (lambda _: nullcontext())
        normalized_input = torch.empty_like(hidden)
        attention_residual = hidden if residual is None else torch.empty_like(hidden)
        for start in range(0, len(hidden), self.chunk_size):
            stop = min(start + self.chunk_size, len(hidden))
            with scope("input_residual_norm"):
                summed = hidden[start:stop].float()
                if residual is not None:
                    summed = summed + residual[start:stop].float()
                    attention_residual[start:stop] = summed.bfloat16()
                normalized_input[start:stop] = rms_norm(
                    summed, self.attention.attention.input_norm_weight, self.cfg.norm_eps
                ).bfloat16()
        attention = self.attention.forward(normalized_input, scope=scope, normalized=True)
        output = torch.empty_like(hidden)
        output_residual = torch.empty_like(hidden)
        for start in range(0, len(hidden), self.chunk_size):
            stop = min(start + self.chunk_size, len(hidden))
            with scope("post_attention_residual_norm"):
                summed = attention_residual[start:stop].float() + attention[start:stop].float()
                normalized = rms_norm(summed, self.post_norm_weight, self.cfg.norm_eps).bfloat16()
                output_residual[start:stop] = summed.bfloat16()
            with scope("moe" if self.is_moe else "dense_mlp"):
                ffn = self.mlp(normalized, scope=scope) if self.is_moe else self.mlp(normalized)
            output[start:stop] = ffn
        return output, output_residual
