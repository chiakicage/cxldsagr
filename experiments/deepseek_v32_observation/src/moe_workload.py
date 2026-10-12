"""One resident production MoE layer, with actual routing and operator fixtures.

Weights are the 256 routed experts and one shared expert of checkpoint layer 3.
Only selected route work contributes useful FLOPs. The grouped public APIs
include padding/packing/quantization; their observed runtime padding is not
substituted for logical model FLOPs.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.deepseek_v32_mfu.src.operator_flops import (
    OperatorWork,
    linear_work,
    non_matmul_work,
)
from experiments.deepseek_v32_observation.src.common_workload import (
    OperatorCase,
    add_fp8_linear_leaves,
    compare,
    compare_bits,
    float_linear_verifier,
    linear_verifier,
    quantization_verifier,
    sample_columns,
    sample_indices,
    tensor_metadata,
)
from models.deepseek_v32 import layers
from models.deepseek_v32.checkpoint import CheckpointLinear, CheckpointReader
from models.deepseek_v32.config import Config
from models.deepseek_v32.layers import CheckpointMoE
from operators import flashinfer
from operators.deepseek_v32.linear import fp8
from operators.deepseek_v32.linear.fp8 import (
    reference_fp8_linear,
    reference_quantize_fp8_activation,
)


def grouped_verifier(source, ids, weights, scales):
    def verify(actual):
        rows = sample_indices(len(source), 8, source.device)
        columns = sample_columns(weights.shape[1], source.device)
        blocks = torch.div(columns[::128], 128, rounding_mode="floor")
        expected = torch.empty(
            (len(rows), ids.shape[1], len(columns)), device=source.device, dtype=torch.bfloat16
        )
        selected_ids = ids[rows]
        # Reference math runs only selected routes, with independent activation
        # quantization and dequantization of each expert's real block weights.
        for expert in selected_ids.unique().cpu().tolist():
            selected = selected_ids == expert
            routed_source = (
                source[rows]
                if source.ndim == 3
                else source[rows, None].expand(-1, ids.shape[1], -1)
            )
            expected[selected] = reference_fp8_linear(
                routed_source[selected], weights[expert, columns], scales[expert, blocks]
            )
        return {
            **compare(actual[rows][:, :, columns].contiguous(), expected),
            "reference": "selected expert FP32 dequantized matmuls, independent UE8M0 activation quantization",
            "coverage": "sampled tokens, every active route, sampled aligned output-column blocks, all input channels",
            "row_indices": rows.cpu().tolist(),
            "column_indices": columns.cpu().tolist(),
            "route_ids": selected_ids.cpu().tolist(),
            "full_output_shape": list(actual.shape),
        }

    return verify


def prepared_grouped_verifier(data, activation_scales, weights, weight_scales, routing):
    """Verify selected routes using the exact prepared quantized GEMM operands."""

    def verify(actual):
        token_rows = sample_indices(len(routing.expert_ids), 8, data.device)
        ids = routing.expert_ids[token_rows]
        topk = ids.shape[1]
        routes = (token_rows[:, None] * topk + torch.arange(topk, device=data.device)).flatten()
        slots = routing.output_slots[routes]
        columns = sample_columns(weights.shape[1], data.device)
        # Byte gathers work on both CPU and CUDA even where FP8 indexing is absent.
        sampled_data = data.view(torch.uint8).index_select(0, slots).view(data.dtype).float()
        sampled_data *= activation_scales[slots].repeat_interleave(128, -1)
        expected = torch.empty((len(slots), len(columns)), device=data.device, dtype=torch.bfloat16)
        flat_ids = ids.flatten()
        for expert in flat_ids.unique().cpu().tolist():
            selected = flat_ids == expert
            weight = (
                weights[expert]
                .view(torch.uint8)
                .index_select(0, columns)
                .view(weights.dtype)
                .float()
            )
            weight *= weight_scales[expert, columns // 128].repeat_interleave(128, -1)
            expected[selected] = (sampled_data[selected] @ weight.T).bfloat16()
        return {
            **compare(actual[slots][:, columns].contiguous(), expected),
            "reference": "independent FP32 selected-expert matmuls of exact prepared FP8 data and block scales",
            "coverage": "sampled tokens, every active route, aligned output blocks, all input channels; padding excluded",
            "row_indices": token_rows.cpu().tolist(),
            "packed_row_indices": slots.cpu().tolist(),
            "column_indices": columns.cpu().tolist(),
            "route_ids": ids.cpu().tolist(),
            "full_output_shape": list(actual.shape),
        }

    return verify


def exact_verifier(expected, *, reference="independent CPU index/arithmetic reference"):
    """Keep reference execution outside timing and compare all logical bytes."""
    expected = expected.cpu()

    def verify(actual):
        return {
            **compare_bits(actual.cpu(), expected),
            "coverage": "full logical output",
            "reference": reference,
        }

    return verify


class MoEWorkload:
    def __init__(self, s, device, seed, checkpoint):
        if type(s) is not int or s < 1:
            raise ValueError("S must be a positive integer")
        self.s, self.device, self.seed = s, torch.device(device), seed
        if self.device.type != "cuda" or torch.cuda.get_device_capability(self.device) != (9, 0):
            raise RuntimeError("MoE observation requires SM90/Hopper")
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("MoE FP32 routing requires TF32 disabled")
        # Production SiLU imports FlashInfer after the first grouped linears.
        # FlashInfer's import initializes compiler environment, so complete it
        # before the quantizer freezes its compilation identity.
        from flashinfer.activation import silu_and_mul  # noqa: F401

        self.checkpoint = Path(checkpoint)
        self.reader = CheckpointReader(checkpoint)
        self.cfg = Config.from_checkpoint(checkpoint)
        self.layer_idx = self.cfg.first_k_dense_replace
        self.stem = f"model.layers.{self.layer_idx}.mlp"
        self.moe = CheckpointMoE(self.reader, self.stem, self.cfg, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.hidden = torch.randn(
            (s, self.cfg.dim), device=self.device, generator=generator
        ).bfloat16()
        self.cases = []
        self.captured = {}
        self._before_calls = {}
        self._capture_cases()
        self.metadata = {
            "family": "moe",
            "S": s,
            "source_layer": self.layer_idx,
            "checkpoint": str(checkpoint),
            "input_distribution": "seeded Gaussian BF16 activations, real checkpoint MoE weights",
            "route_histogram": torch.bincount(
                self.expert_ids.flatten(), minlength=self.cfg.n_routed_experts
            )
            .cpu()
            .tolist(),
            "logical_active_routes": s * self.cfg.num_experts_per_tok,
            "experts_resident": self.cfg.n_routed_experts,
            "shared_experts": self.cfg.n_shared_experts,
            "routing_dependency": "runtime depends on actual route distribution as well as S",
            "measurement_semantics": "independent operator fixtures; module aggregates overlap and are not additive",
        }

    def _add(self, name, call, work, verify, *, inputs=(), scope="operator", **metadata):
        kind = metadata.get("operator_kind", "non_gemm")
        if kind == "api_aggregate":
            scope = "module_aggregate"
        family = metadata.get("operator_family", "moe_reduce")
        if "cast" in name:
            family = metadata.get("operator_family", "moe_cast")
        self.cases.append(
            OperatorCase(
                name,
                call,
                work,
                verify,
                {
                    "family": "moe",
                    "S": self.s,
                    "source_layer": self.layer_idx,
                    "scope": scope,
                    "measure_scope": scope,
                    "operator_kind": kind,
                    "operator_family": family,
                    "parent_api": metadata.get("parent_api", name),
                    "not_additive": True,
                    "inputs": [tensor_metadata(value) for value in inputs],
                    **metadata,
                },
            )
        )

    def _leaf(self, name, call, expected, *, parent, inputs, family="moe_routing", boundary):
        self._add(
            name,
            call,
            non_matmul_work(name),
            exact_verifier(expected),
            inputs=inputs,
            operator_kind="non_gemm",
            operator_family=family,
            parent_api=parent,
            boundary=boundary,
        )

    def _primitive(self, name, function, inputs, *, parent, family="layout", boundary):
        result = function(*inputs)
        expected = function(*(value.cpu() for value in inputs))
        self._leaf(
            name,
            lambda: function(*inputs),
            expected,
            parent=parent,
            inputs=inputs,
            family=family,
            boundary=boundary,
        )
        return result

    def before_call(self, name):
        return self._before_calls.get(name)

    def _routing_leaves(self, hidden, gate, bias, cfg):
        parent = "moe_routing"
        h32, g32, b32 = hidden.float(), gate.float(), bias.float()
        self._leaf(
            "moe_routing_hidden_cast",
            lambda: hidden.float(),
            hidden.cpu().float(),
            parent=parent,
            inputs=(hidden,),
            family="moe_cast",
            boundary="production routing BF16-to-FP32 input conversion",
        )
        logits = F.linear(h32, g32)
        self._add(
            "moe_routing_gate_gemm",
            lambda: F.linear(h32, g32),
            linear_work(
                "moe_routing_gate_gemm",
                rows=self.s,
                in_features=cfg.dim,
                out_features=cfg.n_routed_experts,
                precision="fp32",
            ),
            float_linear_verifier(h32, g32),
            inputs=(h32, g32),
            operator_kind="gemm",
            operator_family="moe_routing",
            parent_api=parent,
            boundary="production FP32 routing linear with prepared FP32 operands; TF32 disabled; casts excluded",
        )
        scores = logits.sigmoid()
        self._add(
            "moe_routing_sigmoid",
            lambda: logits.sigmoid(),
            non_matmul_work("moe_routing_sigmoid"),
            lambda actual: {
                **compare(
                    actual.cpu(), logits.cpu().double().sigmoid().float(), rtol=2e-6, atol=1e-7
                ),
                "coverage": "all routing scores",
                "reference": "CPU FP64 sigmoid rounded to FP32",
            },
            inputs=(logits,),
            operator_kind="non_gemm",
            operator_family="moe_routing",
            parent_api=parent,
            boundary="production sigmoid of prepared FP32 gate logits",
        )
        corrected = scores + b32
        self._leaf(
            "moe_routing_correction",
            lambda: scores + b32,
            scores.cpu() + b32.cpu(),
            parent=parent,
            inputs=(scores, b32),
            boundary="correction bias addition; weights continue to use uncorrected sigmoid scores",
        )
        grouped = corrected.reshape(self.s, cfg.n_group, -1)
        values = grouped.topk(2, dim=-1).values
        self._leaf(
            "moe_routing_group_top2",
            lambda: grouped.topk(2, dim=-1).values,
            grouped.cpu().topk(2, dim=-1).values,
            parent=parent,
            inputs=(grouped,),
            boundary="production top-two corrected scores per expert group",
        )
        group_scores = values.sum(-1)
        self._leaf(
            "moe_routing_group_score_reduce",
            lambda: values.sum(-1),
            values.cpu().sum(-1),
            parent=parent,
            inputs=(values,),
            boundary="sum of each group's two corrected scores",
        )
        groups = group_scores.topk(cfg.topk_group, dim=-1).indices
        self._leaf(
            "moe_routing_group_topk",
            lambda: group_scores.topk(cfg.topk_group, dim=-1).indices,
            group_scores.cpu().topk(cfg.topk_group, dim=-1).indices,
            parent=parent,
            inputs=(group_scores,),
            boundary="top-k expert groups by corrected group score",
        )
        retained = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
        self._leaf(
            "moe_routing_group_mask_zero",
            lambda: torch.zeros_like(group_scores, dtype=torch.bool),
            torch.zeros_like(group_scores.cpu(), dtype=torch.bool),
            parent=parent,
            inputs=(group_scores,),
            boundary="production boolean group-mask zero initialization",
        )
        mask_output = torch.zeros_like(group_scores, dtype=torch.bool)
        self._leaf(
            "moe_routing_group_mask_scatter",
            lambda: mask_output.scatter_(1, groups, True),
            retained,
            parent=parent,
            inputs=(mask_output, groups),
            boundary="production retained-group scatter into prepared zero mask; repeated writes are idempotent",
        )
        excluded = ~retained[..., None]
        self._leaf(
            "moe_routing_group_exclude_mask",
            lambda: ~retained[..., None],
            ~retained.cpu()[..., None],
            parent=parent,
            inputs=(retained,),
            boundary="boolean inversion of retained expert groups",
        )
        masked = grouped.masked_fill(excluded, -torch.inf).flatten(1)
        self._leaf(
            "moe_routing_group_mask",
            lambda: grouped.masked_fill(excluded, -torch.inf).flatten(1),
            grouped.cpu().masked_fill(excluded.cpu(), -torch.inf).flatten(1),
            parent=parent,
            inputs=(grouped, excluded),
            boundary="mask corrected scores in excluded groups; flatten is a view",
        )
        ids = masked.topk(cfg.num_experts_per_tok, dim=-1).indices.contiguous()
        self._leaf(
            "moe_routing_expert_topk",
            lambda: masked.topk(cfg.num_experts_per_tok, dim=-1).indices.contiguous(),
            masked.cpu().topk(cfg.num_experts_per_tok, dim=-1).indices.contiguous(),
            parent=parent,
            inputs=(masked,),
            boundary="production active-expert selection and any required ID continuous layout",
        )
        weights = scores.gather(1, ids)
        self._leaf(
            "moe_routing_weight_gather",
            lambda: scores.gather(1, ids),
            scores.cpu().gather(1, ids.cpu()),
            parent=parent,
            inputs=(scores, ids),
            boundary="gather original uncorrected sigmoid weights for selected experts",
        )
        if cfg.norm_topk_prob:
            totals = weights.sum(-1, keepdim=True)
            self._add(
                "moe_routing_weight_sum",
                lambda weights=weights: weights.sum(-1, keepdim=True),
                non_matmul_work("moe_routing_weight_sum"),
                lambda actual, weights=weights: {
                    **compare(
                        actual.cpu(),
                        weights.cpu().double().sum(-1, keepdim=True).float(),
                        rtol=2e-6,
                        atol=1e-7,
                    ),
                    "coverage": "all tokens",
                    "reference": "CPU FP64 selected-weight sum rounded to FP32",
                },
                inputs=(weights,),
                operator_kind="non_gemm",
                operator_family="moe_routing",
                parent_api=parent,
                boundary="sum selected original weights for normalization",
            )
            denominator = totals.clamp_min(torch.finfo(torch.float32).tiny)
            self._leaf(
                "moe_routing_denominator_clamp",
                lambda: totals.clamp_min(torch.finfo(torch.float32).tiny),
                totals.cpu().clamp_min(torch.finfo(torch.float32).tiny),
                parent=parent,
                inputs=(totals,),
                boundary="clamp routing normalizer to the minimum normal FP32 value",
            )
            normalized = weights / denominator
            self._leaf(
                "moe_routing_weight_normalize",
                lambda weights=weights: weights / denominator,
                weights.cpu() / denominator.cpu(),
                parent=parent,
                inputs=(weights, denominator),
                boundary="normalize selected original sigmoid weights",
            )
            weights = normalized
        self._leaf(
            "moe_routing_weight_scale",
            lambda: (weights * cfg.routed_scaling_factor).contiguous(),
            (weights.cpu() * cfg.routed_scaling_factor).contiguous(),
            parent=parent,
            inputs=(weights,),
            boundary="model routed scaling factor and any required weight continuous layout",
        )

    def _metadata_leaves(self):
        ids, experts, routing = self.expert_ids, self.cfg.n_routed_experts, self.routing
        parent, block = "moe_prepare_routing", routing.block_m
        flat = ids.reshape(-1).long()
        valid_routes = (flat >= 0) & (flat < experts)
        self._leaf(
            "moe_metadata_route_valid",
            lambda: (flat >= 0) & (flat < experts),
            (flat.cpu() >= 0) & (flat.cpu() < experts),
            parent=parent,
            inputs=(flat,),
            family="layout",
            boundary="production route-ID bounds masks and boolean intersection",
        )
        keys = torch.where(valid_routes, flat, experts)
        self._leaf(
            "moe_metadata_sort_keys",
            lambda: torch.where(valid_routes, flat, experts),
            torch.where(valid_routes.cpu(), flat.cpu(), experts),
            parent=parent,
            inputs=(flat, valid_routes),
            family="layout",
            boundary="replace invalid route IDs with sentinel sort key",
        )
        sorted_routes = routing.sorted_routes

        def verify_sort(actual):
            actual = actual.cpu()
            compare_bits(actual.sort().values, torch.arange(len(flat)))
            ordered = keys.cpu()[actual]
            if not torch.all(ordered[1:] >= ordered[:-1]):
                raise AssertionError("Metadata argsort does not order all expert keys")
            return {
                "coverage": "all keys and permutation",
                "reference": "CPU permutation bijection and monotonic expert-key check",
            }

        self._add(
            "moe_metadata_argsort",
            lambda: torch.argsort(keys),
            non_matmul_work("moe_metadata_argsort"),
            verify_sort,
            inputs=(keys,),
            operator_kind="non_gemm",
            operator_family="layout",
            parent_api=parent,
            boundary="production expert-key argsort; equal-key order is unspecified",
        )
        counts = torch.bincount(keys, minlength=experts + 1).to(torch.int32)

        def count_routes():
            result = torch.zeros(experts + 1, device=keys.device, dtype=torch.int32)
            return result.scatter_add_(0, keys, torch.ones_like(keys, dtype=torch.int32))

        self._leaf(
            "moe_metadata_expert_counts",
            count_routes,
            torch.bincount(keys.cpu(), minlength=experts + 1).to(torch.int32),
            parent=parent,
            inputs=(keys,),
            family="layout",
            boundary="production zero counts, unit-route values and scatter-add histogram",
        )
        zero = counts.new_zeros(1)
        rows = counts[:-1].cumsum(0, dtype=torch.int32)
        self._leaf(
            "moe_metadata_row_cumsum",
            lambda: counts[:-1].cumsum(0, dtype=torch.int32),
            counts.cpu()[:-1].cumsum(0, dtype=torch.int32),
            parent=parent,
            inputs=(counts,),
            family="layout",
            boundary="expert row-offset prefix sum",
        )
        self._leaf(
            "moe_metadata_row_offsets",
            lambda: torch.cat((counts.new_zeros(1), rows)),
            torch.cat((counts.cpu().new_zeros(1), rows.cpu())),
            parent=parent,
            inputs=(counts, rows),
            family="layout",
            boundary="leading-zero initialization and row-offset concatenation",
        )
        tiles = torch.div(counts[:-1] + block - 1, block, rounding_mode="floor")
        self._leaf(
            "moe_metadata_tile_counts",
            lambda: torch.div(counts[:-1] + block - 1, block, rounding_mode="floor"),
            torch.div(counts.cpu()[:-1] + block - 1, block, rounding_mode="floor"),
            parent=parent,
            inputs=(counts,),
            family="layout",
            boundary="round each expert count up to official grouped GEMM tile count",
        )
        tile_prefix = tiles.cumsum(0, dtype=torch.int32)
        self._leaf(
            "moe_metadata_tile_cumsum",
            lambda: tiles.cumsum(0, dtype=torch.int32),
            tiles.cpu().cumsum(0, dtype=torch.int32),
            parent=parent,
            inputs=(tiles,),
            family="layout",
            boundary="expert tile-offset prefix sum",
        )
        offsets = torch.cat((zero, tile_prefix))
        self._leaf(
            "moe_metadata_tile_offsets",
            lambda: torch.cat((counts.new_zeros(1), tile_prefix)),
            torch.cat((counts.cpu().new_zeros(1), tile_prefix.cpu())),
            parent=parent,
            inputs=(counts, tile_prefix),
            family="layout",
            boundary="leading-zero initialization and tile-offset concatenation",
        )
        capacity = len(routing.packed_routes)
        positions = torch.arange(capacity, device=ids.device, dtype=torch.int32)
        self._leaf(
            "moe_metadata_packed_positions",
            lambda: torch.arange(capacity, device=ids.device, dtype=torch.int32),
            torch.arange(capacity, dtype=torch.int32),
            parent=parent,
            inputs=(),
            family="layout",
            boundary="static-capacity packed-row position generation",
        )
        ends = offsets[1:] * block
        self._leaf(
            "moe_metadata_tile_ends",
            lambda: offsets[1:] * block,
            offsets.cpu()[1:] * block,
            parent=parent,
            inputs=(offsets,),
            family="layout",
            boundary="convert expert tile offsets to packed-row boundaries",
        )
        groups = torch.searchsorted(ends, positions, right=True, out_int32=True)
        self._leaf(
            "moe_metadata_packed_experts",
            lambda: torch.searchsorted(ends, positions, right=True, out_int32=True),
            torch.searchsorted(ends.cpu(), positions.cpu(), right=True, out_int32=True),
            parent=parent,
            inputs=(ends, positions),
            family="layout",
            boundary="packed-row expert lookup by tile-boundary searchsorted",
        )
        safe = groups.clamp_max(experts - 1).long()
        self._leaf(
            "moe_metadata_safe_experts",
            lambda: groups.clamp_max(experts - 1).long(),
            groups.cpu().clamp_max(experts - 1).long(),
            parent=parent,
            inputs=(groups,),
            family="layout",
            boundary="clamp packed expert IDs then convert indexing dtype",
        )
        local = positions - offsets[safe] * block
        self._leaf(
            "moe_metadata_local_rows",
            lambda: positions - offsets[safe] * block,
            positions.cpu() - offsets.cpu()[safe.cpu()] * block,
            parent=parent,
            inputs=(positions, offsets, safe),
            family="layout",
            boundary="expert-local packed-row arithmetic and tile-offset gather",
        )
        valid = (groups < experts) & (local < counts[safe])
        self._leaf(
            "moe_metadata_packed_valid",
            lambda: (groups < experts) & (local < counts[safe]),
            (groups.cpu() < experts) & (local.cpu() < counts.cpu()[safe.cpu()]),
            parent=parent,
            inputs=(groups, local, counts, safe),
            family="layout",
            boundary="packed row bounds and expert-count masks",
        )
        sorted_positions = (routing.row_offsets[safe] + local).clamp(0, len(flat) - 1).long()
        self._leaf(
            "moe_metadata_sorted_positions",
            lambda: (routing.row_offsets[safe] + local).clamp(0, len(flat) - 1).long(),
            (routing.row_offsets.cpu()[safe.cpu()] + local.cpu()).clamp(0, len(flat) - 1).long(),
            parent=parent,
            inputs=(routing.row_offsets, safe, local),
            family="layout",
            boundary="sorted-route position gather/add/clamp and index conversion",
        )
        self._leaf(
            "moe_metadata_packed_routes",
            lambda: sorted_routes[sorted_positions],
            sorted_routes.cpu()[sorted_positions.cpu()],
            parent=parent,
            inputs=(sorted_routes, sorted_positions),
            family="layout",
            boundary="gather route permutation into static packed capacity",
        )
        self._leaf(
            "moe_metadata_grouped_layout",
            lambda: torch.where(valid, groups, -1).contiguous(),
            torch.where(valid.cpu(), groups.cpu(), -1).contiguous(),
            parent=parent,
            inputs=(valid, groups),
            family="layout",
            boundary="mark invalid packed rows with group minus one",
        )
        route_positions = torch.arange(len(flat), device=ids.device)

        def inverse_rank():
            rank = torch.empty_like(sorted_routes)
            return rank.scatter_(0, sorted_routes, torch.arange(len(flat), device=ids.device))

        rank = inverse_rank()
        cpu_rank = torch.empty_like(sorted_routes.cpu()).scatter_(
            0, sorted_routes.cpu(), route_positions.cpu()
        )
        self._leaf(
            "moe_metadata_inverse_rank",
            inverse_rank,
            cpu_rank,
            parent=parent,
            inputs=(sorted_routes,),
            family="layout",
            boundary="route arange generation and inverse-permutation scatter",
        )
        safe_keys = keys.clamp_max(experts - 1)
        self._leaf(
            "moe_metadata_safe_keys",
            lambda: keys.clamp_max(experts - 1),
            keys.cpu().clamp_max(experts - 1),
            parent=parent,
            inputs=(keys,),
            family="layout",
            boundary="clamp sentinel keys before offset indexing",
        )
        slots = offsets[safe_keys] * block + rank - routing.row_offsets[safe_keys]
        self._leaf(
            "moe_metadata_output_slot_arithmetic",
            lambda: offsets[safe_keys] * block + rank - routing.row_offsets[safe_keys],
            offsets.cpu()[safe_keys.cpu()] * block
            + rank.cpu()
            - routing.row_offsets.cpu()[safe_keys.cpu()],
            parent=parent,
            inputs=(offsets, safe_keys, rank, routing.row_offsets),
            family="layout",
            boundary="inverse output-slot offset gathers and integer arithmetic",
        )
        self._leaf(
            "moe_metadata_output_slots",
            lambda: torch.where(valid_routes, slots, 0).long(),
            torch.where(valid_routes.cpu(), slots.cpu(), 0).long(),
            parent=parent,
            inputs=(valid_routes, slots),
            family="layout",
            boundary="invalid-route slot masking and final index dtype",
        )

        compound_names = {
            "route_valid",
            "expert_counts",
            "row_offsets",
            "tile_counts",
            "tile_offsets",
            "safe_experts",
            "local_rows",
            "packed_valid",
            "sorted_positions",
            "inverse_rank",
            "output_slot_arithmetic",
        }
        for case in self.cases:
            if case.name.removeprefix("moe_metadata_") in compound_names:
                case.metadata.update(
                    operator_kind="api_aggregate",
                    scope="module_aggregate",
                    measure_scope="module_aggregate",
                    parent_api=case.name,
                )
        self._metadata_primitives()

    def _metadata_primitives(self):
        routing, experts = self.routing, self.cfg.n_routed_experts
        flat = self.expert_ids.reshape(-1).long()
        block, parent = routing.block_m, "moe_prepare_routing"

        def op(suffix, function, *inputs):
            return self._primitive(
                "moe_metadata_" + suffix,
                function,
                inputs,
                parent=parent,
                boundary="prepared production metadata primitive: " + suffix,
            )

        lower = op("route_ge_zero", lambda value: value >= 0, flat)
        upper = op("route_lt_experts", lambda value: value < experts, flat)
        valid_routes = op("route_valid_and", torch.bitwise_and, lower, upper)
        keys = torch.where(valid_routes, flat, experts)
        zeros = op(
            "counts_zero",
            lambda value: torch.zeros(experts + 1, device=value.device, dtype=torch.int32),
            keys,
        )
        ones = op("route_ones", lambda value: torch.ones_like(value, dtype=torch.int32), keys)
        counts_expected = torch.bincount(keys.cpu(), minlength=experts + 1).to(torch.int32)
        self._leaf(
            "moe_metadata_counts_scatter_add",
            lambda: zeros.scatter_add_(0, keys, ones),
            counts_expected,
            parent=parent,
            inputs=(zeros, keys, ones),
            family="layout",
            boundary="production route histogram scatter-add on prepared zeros/ones; reset outside interval",
        )
        self._before_calls["moe_metadata_counts_scatter_add"] = zeros.zero_
        counts = zeros.clone().scatter_add_(0, keys, ones)
        zero = op("offset_zero", lambda value: value.new_zeros(1), counts)
        row_prefix = counts[:-1].cumsum(0, dtype=torch.int32)
        op("row_offset_cat", lambda left, right: torch.cat((left, right)), zero, row_prefix)
        plus_block = op("tile_count_add", lambda value: value + block, counts[:-1])
        numerator = op("tile_count_sub_one", lambda value: value - 1, plus_block)
        tiles = op(
            "tile_count_div",
            lambda value: torch.div(value, block, rounding_mode="floor"),
            numerator,
        )
        tile_prefix = tiles.cumsum(0, dtype=torch.int32)
        offsets = op(
            "tile_offset_cat", lambda left, right: torch.cat((left, right)), zero, tile_prefix
        )
        positions = torch.arange(len(routing.packed_routes), device=flat.device, dtype=torch.int32)
        ends = offsets[1:] * block
        groups = torch.searchsorted(ends, positions, right=True, out_int32=True)
        clamped = op("expert_clamp", lambda value: value.clamp_max(experts - 1), groups)
        safe = op("expert_index_cast", lambda value: value.long(), clamped)
        tile_starts = op("tile_offset_gather", lambda value, indices: value[indices], offsets, safe)
        start_rows = op("tile_start_mul", lambda value: value * block, tile_starts)
        local = op("local_row_sub", torch.sub, positions, start_rows)
        group_valid = op("packed_expert_lt", lambda value: value < experts, groups)
        per_expert_count = op(
            "expert_count_gather", lambda value, indices: value[indices], counts, safe
        )
        local_valid = op("local_row_lt", torch.lt, local, per_expert_count)
        op("packed_valid_and", torch.bitwise_and, group_valid, local_valid)
        row_starts = op(
            "row_offset_gather", lambda value, indices: value[indices], routing.row_offsets, safe
        )
        sorted_unclamped = op("sorted_position_add", torch.add, row_starts, local)
        sorted_clamped = op(
            "sorted_position_clamp", lambda value: value.clamp(0, len(flat) - 1), sorted_unclamped
        )
        op("sorted_index_cast", lambda value: value.long(), sorted_clamped)
        route_positions = op(
            "rank_arange", lambda value: torch.arange(len(flat), device=value.device), flat
        )
        rank_output = torch.empty_like(routing.sorted_routes)
        rank_expected = torch.empty_like(routing.sorted_routes.cpu()).scatter_(
            0, routing.sorted_routes.cpu(), route_positions.cpu()
        )
        self._leaf(
            "moe_metadata_rank_scatter",
            lambda: rank_output.scatter_(0, routing.sorted_routes, route_positions),
            rank_expected,
            parent=parent,
            inputs=(rank_output, routing.sorted_routes, route_positions),
            family="layout",
            boundary="inverse-permutation scatter on prepared arange; every destination is overwritten",
        )
        rank = rank_output.scatter_(0, routing.sorted_routes, route_positions)
        safe_keys = keys.clamp_max(experts - 1)
        slot_tile = op(
            "slot_tile_gather", lambda value, indices: value[indices], offsets, safe_keys
        )
        slot_base = op("slot_base_mul", lambda value: value * block, slot_tile)
        slot_rank = op("slot_rank_add", torch.add, slot_base, rank)
        slot_row = op(
            "slot_row_gather", lambda value, indices: value[indices], routing.row_offsets, safe_keys
        )
        op("slot_row_sub", torch.sub, slot_rank, slot_row)

    def _grouped_leaves(self, name, source, ids, weights, scales, kwargs):
        import deep_gemm

        prepared = self.captured[name + "_prepared"]
        activation = prepared["activation"]
        data, activation_scales = prepared["quantized"]
        packed_data, packed_scales = prepared["a"]
        padded_weight, weight_scales = prepared["b"]
        routing = kwargs["routing"]
        topk = ids.shape[1]
        k, n = data.shape[-1], weights.shape[1]
        if k != source.shape[-1] or padded_weight.shape != weights.shape:
            raise ValueError("Checkpoint grouped decomposition expects no input/weight padding")
        if (
            padded_weight.data_ptr() != weights.data_ptr()
            or weight_scales.data_ptr() != scales.data_ptr()
        ):
            raise ValueError(
                "Checkpoint grouped decomposition expects identity weight layout preparation"
            )
        self._add(
            name + "_activation_quantize",
            lambda: fp8.quantize_fp8_activation(activation),
            non_matmul_work(name + "_activation_quantize"),
            quantization_verifier(activation, reference_quantize_fp8_activation),
            inputs=(activation,),
            operator_kind="non_gemm",
            operator_family="quantization",
            parent_api=name,
            boundary="production activation quantization on exact grouped input; route gather and GEMM excluded",
        )
        input_rows = routing.packed_routes if source.ndim == 3 else routing.packed_routes // topk
        if source.ndim != 3:
            self._leaf(
                name + "_input_rows",
                lambda: routing.packed_routes // topk,
                routing.packed_routes.cpu() // topk,
                parent=name,
                inputs=(routing.packed_routes,),
                family="layout",
                boundary="map packed active routes to source token rows",
            )
        valid = routing.grouped_layout >= 0
        self._leaf(
            name + "_packed_valid",
            lambda: routing.grouped_layout >= 0,
            routing.grouped_layout.cpu() >= 0,
            parent=name,
            inputs=(routing.grouped_layout,),
            family="layout",
            boundary="valid-row mask from prepared grouped layout",
        )
        gathered_data = data.view(torch.uint8).index_select(0, input_rows)
        self._leaf(
            name + "_data_gather",
            lambda: data.view(torch.uint8).index_select(0, input_rows),
            data.cpu().view(torch.uint8).index_select(0, input_rows.cpu()),
            parent=name,
            inputs=(data, input_rows),
            family="layout",
            boundary="production FP8 byte gather into static packed rows",
        )
        self._leaf(
            name + "_data_mask",
            lambda: torch.where(valid[:, None], gathered_data, 0).view(data.dtype),
            torch.where(valid.cpu()[:, None], gathered_data.cpu(), 0).view(data.dtype),
            parent=name,
            inputs=(gathered_data, valid),
            family="layout",
            boundary="zero invalid packed FP8 rows; FP8 view has no kernel",
        )
        gathered_scales = activation_scales.index_select(0, input_rows)
        self._leaf(
            name + "_scale_gather",
            lambda: activation_scales.index_select(0, input_rows),
            activation_scales.cpu().index_select(0, input_rows.cpu()),
            parent=name,
            inputs=(activation_scales, input_rows),
            family="layout",
            boundary="production activation-scale row gather",
        )
        self._leaf(
            name + "_scale_mask",
            lambda: torch.where(valid[:, None], gathered_scales, 1.0),
            torch.where(valid.cpu()[:, None], gathered_scales.cpu(), 1.0),
            parent=name,
            inputs=(gathered_scales, valid),
            family="layout",
            boundary="replace invalid packed activation scales with one",
        )
        transformed = deep_gemm.transform_sf_into_required_layout(
            packed_scales, len(packed_data), k, (1, 128, 128), is_sfa=True
        )
        layout_changed = transformed.data_ptr() != packed_scales.data_ptr()
        if layout_changed:
            self._add(
                name + "_scale_layout",
                lambda: deep_gemm.transform_sf_into_required_layout(
                    packed_scales, len(packed_data), k, (1, 128, 128), is_sfa=True
                ),
                non_matmul_work(name + "_scale_layout"),
                exact_verifier(packed_scales),
                inputs=(packed_scales,),
                operator_kind="non_gemm",
                operator_family="scale_layout",
                parent_api=name,
                boundary="official SM90 grouped activation-scale layout conversion; GEMM excluded",
            )
        # Both the official wrapper and the isolated call skip group -1 rows.
        # Initialize undefined padding once outside the call for deterministic
        # complete tensor snapshots without adding a kernel to GEMM timing.
        output = torch.empty_strided(
            prepared["output"].shape,
            prepared["output"].stride(),
            device=source.device,
            dtype=torch.bfloat16,
        )
        output.zero_()

        def gemm():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (packed_data, transformed),
                (padded_weight, weight_scales),
                output,
                routing.grouped_layout,
                **prepared["kwargs"],
            )
            return output

        self._add(
            name + "_gemm",
            gemm,
            OperatorWork(
                name + "_gemm",
                "fp8",
                2 * self.s * topk * n * k,
                None,
                "2 * S * selected_experts_per_token * N * K",
                dimensions={
                    "S": self.s,
                    "active_routes": self.s * topk,
                    "N": n,
                    "K": k,
                    "packed_capacity": len(packed_data),
                },
                notes="Useful FLOPs count actual selected routes only; official grouped kernel executes its supported padding schedule.",
            ),
            prepared_grouped_verifier(
                packed_data, packed_scales, padded_weight, weight_scales, routing
            ),
            inputs=(
                packed_data,
                transformed,
                padded_weight,
                weight_scales,
                output,
                routing.grouped_layout,
            ),
            operator_kind="gemm",
            operator_family="moe_gemm",
            parent_api=name,
            quantization_operation=name + "_activation_quantize",
            scale_layout_operation=name + "_scale_layout" if layout_changed else None,
            scale_layout_no_kernel=not layout_changed,
            weight_layout_no_kernel=True,
            input_rows_no_kernel=source.ndim == 3,
            view_reshape_no_kernel=True,
            boundary="official grouped DeepGEMM with prepared production FP8 operands, layout and output stride; quantization, gathers, masks and scale conversion excluded",
        )
        gemm()
        gathered_output = output.index_select(0, routing.output_slots)[:, :n]
        self._leaf(
            name + "_output_gather",
            lambda: output.index_select(0, routing.output_slots)[:, :n],
            output.cpu().index_select(0, routing.output_slots.cpu())[:, :n],
            parent=name,
            inputs=(output, routing.output_slots),
            family="layout",
            boundary="inverse gather from grouped packed output; column trim is a view",
        )
        self._leaf(
            name + "_output_mask",
            lambda: torch.where(routing.valid_routes[:, None], gathered_output, 0).reshape(
                self.s, topk, n
            ),
            torch.where(routing.valid_routes.cpu()[:, None], gathered_output.cpu(), 0).reshape(
                self.s, topk, n
            ),
            parent=name,
            inputs=(gathered_output, routing.valid_routes),
            family="layout",
            boundary="zero invalid logical routes; final token/route reshape has no kernel",
        )

    def _silu_leaves(self, name, left, right):
        packed = torch.cat((left, right), dim=-1)
        expected_packed = torch.cat((left.cpu(), right.cpu()), dim=-1)
        self._leaf(
            name + "_pack",
            lambda: torch.cat((left, right), dim=-1),
            expected_packed,
            parent=name,
            inputs=(left, right),
            family="layout",
            boundary="production gate/up concatenation into FlashInfer activation layout",
        )
        self._add(
            name + "_kernel",
            lambda: flashinfer.silu_and_mul(packed),
            non_matmul_work(name + "_kernel"),
            lambda actual: {
                **compare(
                    actual,
                    (F.silu(left.float()) * right.float()).bfloat16(),
                    rtol=8e-3,
                    atol=1e-5,
                    relative_l2_limit=1e-5,
                ),
                "coverage": "full activation output",
                "reference": "FP32 SiLU and multiplication followed by BF16 conversion",
            },
            inputs=(packed,),
            operator_kind="non_gemm",
            operator_family="moe_activation",
            parent_api=name,
            output_cast_no_kernel=True,
            boundary="official FlashInfer SiLU-and-multiply on prepared BF16 gate/up packing; adapter's BF16 identity cast omitted",
        )

    def _capture_cases(self):
        import deep_gemm

        moe, cfg, captured = self.moe, self.cfg, self.captured
        original_route, original_prepare = layers.route_experts, layers.prepare_expert_routing
        original_grouped, original_silu = layers.grouped_fp8_linear, layers.silu_mul
        original_linear = CheckpointLinear.__call__
        original_mul, original_sum = torch.Tensor.__mul__, torch.Tensor.sum
        original_iadd = torch.Tensor.__iadd__
        original_bfloat16 = torch.Tensor.bfloat16
        original_float = torch.Tensor.float
        shared_names = {
            id(getattr(moe.shared, name)): "moe_shared_" + name for name in ("gate", "up", "down")
        }

        def route(hidden, gate, bias, config):
            result = original_route(hidden, gate, bias, config)
            captured["route_experts"] = (hidden, gate, bias, config, result)
            return result

        def prepare(ids, experts):
            result = original_prepare(ids, experts)
            captured["prepare_routing"] = (ids, experts, result)
            return result

        def grouped(x, ids, weights, scales, **kwargs):
            name = next(name for name, value in moe.weights.items() if value is weights)
            name = "moe_grouped_" + name.removesuffix("_proj")
            captured[name] = (
                x,
                ids,
                weights,
                scales,
                kwargs,
            )
            prepared = {}
            quantizer = fp8.quantize_fp8_activation
            gemm = deep_gemm.m_grouped_fp8_gemm_nt_contiguous

            def quantize(activation):
                result = quantizer(activation)
                prepared.update(activation=activation, quantized=result)
                return result

            def grouped_gemm(a, b, output, layout, **gemm_kwargs):
                prepared.update(a=a, b=b, output=output, layout=layout, kwargs=gemm_kwargs)
                return gemm(a, b, output, layout, **gemm_kwargs)

            with (
                patch.object(fp8, "quantize_fp8_activation", quantize),
                patch.object(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous", grouped_gemm),
            ):
                result = original_grouped(x, ids, weights, scales, **kwargs)
            captured[name + "_prepared"] = prepared
            return result

        def linear(instance, x, *args, **kwargs):
            if id(instance) in shared_names:
                captured[shared_names[id(instance)]] = (instance, x)
            return original_linear(instance, x, *args, **kwargs)

        def silu(gate, up):
            captured["moe_routed_silu" if gate.ndim == 3 else "moe_shared_silu"] = (gate, up)
            return original_silu(gate, up)

        def mul(left, right):
            if (
                left.shape == (self.s, cfg.num_experts_per_tok, cfg.dim)
                and isinstance(right, torch.Tensor)
                and right.shape == (self.s, cfg.num_experts_per_tok, 1)
            ):
                captured["moe_weighted_multiply"] = (left, right)
            return original_mul(left, right)

        def sum_(value, *args, **kwargs):
            if value.shape == (self.s, cfg.num_experts_per_tok, cfg.dim) and args == (1,):
                captured["moe_route_reduce"] = value
            return original_sum(value, *args, **kwargs)

        def iadd(left, right):
            if (
                left.shape == right.shape == (self.s, cfg.dim)
                and left.dtype == right.dtype == torch.float32
            ):
                # The production addition is in-place; retained independent
                # fixtures must use a fresh destination on each invocation.
                captured["moe_shared_add"] = (left.clone(), right)
            return original_iadd(left, right)

        def bfloat16(value, *args, **kwargs):
            if value.shape == (self.s, cfg.dim) and value.dtype == torch.float32:
                captured["moe_output_cast"] = value
            return original_bfloat16(value, *args, **kwargs)

        def float_(value, *args, **kwargs):
            if value.dtype == torch.bfloat16 and value.shape == (
                self.s,
                cfg.num_experts_per_tok,
                cfg.dim,
            ):
                captured["moe_routed_output_cast"] = value
            elif value.dtype == torch.bfloat16 and value.shape == (self.s, cfg.dim):
                # Final such conversion is the shared expert output; routing
                # hidden conversion is recorded by its complete API boundary.
                captured["moe_shared_output_cast"] = value
            return original_float(value, *args, **kwargs)

        with ExitStack() as stack:
            for target, name, replacement in (
                (layers, "route_experts", route),
                (layers, "prepare_expert_routing", prepare),
                (layers, "grouped_fp8_linear", grouped),
                (CheckpointLinear, "__call__", linear),
                (layers, "silu_mul", silu),
                (torch.Tensor, "__mul__", mul),
                (torch.Tensor, "sum", sum_),
                (torch.Tensor, "__iadd__", iadd),
                (torch.Tensor, "bfloat16", bfloat16),
                (torch.Tensor, "float", float_),
            ):
                stack.enter_context(patch.object(target, name, replacement))
            moe(self.hidden)
        hidden, gate, bias, config, routing_output = captured["route_experts"]
        self.expert_ids, self.route_weights = routing_output
        self.routing = captured["prepare_routing"][2]

        def verify_route(actual):
            from models.deepseek_v32.tests.test_echo_block import routing_reference

            expected_ids, expected_weights = routing_reference(
                hidden.cpu(), gate.cpu(), bias.cpu(), config
            )
            return {
                "ids": compare_bits(actual[0].cpu(), expected_ids),
                "weights": compare(actual[1].cpu(), expected_weights, rtol=2e-4, atol=1e-5),
                "coverage": "all token routes and weights",
                "reference": "existing independent CPU routing oracle: sigmoid, corrected top-two groups, unbiased normalized weights",
            }

        self._add(
            "moe_routing",
            lambda: original_route(hidden, gate, bias, config),
            linear_work(
                "moe_routing",
                rows=self.s,
                in_features=cfg.dim,
                out_features=cfg.n_routed_experts,
                precision="fp32",
            ),
            verify_route,
            inputs=(hidden, gate, bias),
            operator_kind="api_aggregate",
            operator_family="moe_routing",
            parent_api="moe_routing",
            boundary="complete production routing API: FP32 gate linear, sigmoid, grouped top-k, normalized original-score weights",
        )
        self._routing_leaves(hidden, gate, bias, config)

        def verify_prepare(actual):
            ids = self.expert_ids.cpu().flatten()
            experts = cfg.n_routed_experts
            counts = torch.bincount(ids, minlength=experts).to(torch.int32)
            row_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
            tiles = torch.div(counts + actual.block_m - 1, actual.block_m, rounding_mode="floor")
            tile_offsets = torch.cat((tiles.new_zeros(1), tiles.cumsum(0, dtype=torch.int32)))
            sorted_routes = actual.sorted_routes.cpu()
            compare_bits(actual.row_offsets.cpu(), row_offsets)
            compare_bits(actual.tile_offsets.cpu(), tile_offsets)
            compare_bits(sorted_routes.sort().values, torch.arange(len(ids)))
            sorted_ids = ids[sorted_routes]
            if not torch.all(sorted_ids[1:] >= sorted_ids[:-1]):
                raise AssertionError("Expert routes are not ordered by expert")
            slots = actual.output_slots.cpu()
            if not torch.equal(actual.packed_routes.cpu()[slots], torch.arange(len(ids))):
                raise AssertionError("Inverse route mapping does not recover each original route")
            if not torch.equal(actual.grouped_layout.cpu()[slots], ids.to(torch.int32)):
                raise AssertionError("Packed route expert assignments differ")
            return {
                "coverage": "all route permutation, row/tile offsets and inverse mapping",
                "block_m": actual.block_m,
                "packed_capacity": len(actual.packed_routes),
                "valid_routes": len(ids),
            }

        self._add(
            "moe_prepare_routing",
            lambda: original_prepare(self.expert_ids, cfg.n_routed_experts),
            non_matmul_work("moe_prepare_routing"),
            verify_prepare,
            inputs=(self.expert_ids,),
            operator_kind="api_aggregate",
            operator_family="layout",
            parent_api="moe_prepare_routing",
            boundary="production on-device route sort/pad/packing metadata API",
        )
        self._metadata_leaves()
        for name in ("moe_grouped_gate", "moe_grouped_up", "moe_grouped_down"):
            source, ids, weights, scales, kwargs = captured[name]
            n, k = weights.shape[-2:]
            work = OperatorWork(
                name,
                "fp8",
                2 * self.s * cfg.num_experts_per_tok * n * k,
                None,
                "2 * S * selected_experts_per_token * N * K",
                dimensions={
                    "S": self.s,
                    "active_routes": self.s * cfg.num_experts_per_tok,
                    "N": n,
                    "K": k,
                },
                notes="Actual selected routes only; DeepGEMM runtime padding unknown. Packing, quantization, GEMM and inverse mapping included in API time.",
            )
            self._add(
                name,
                lambda source=source, ids=ids, weights=weights, scales=scales, kwargs=kwargs: (
                    original_grouped(source, ids, weights, scales, **kwargs)
                ),
                work,
                grouped_verifier(source, ids, weights, scales),
                inputs=(source, ids, weights),
                operator_kind="api_aggregate",
                operator_family="moe_gemm",
                parent_api=name,
                boundary="production grouped FP8 API using prebuilt routing; route preparation excluded",
            )
            self._grouped_leaves(name, source, ids, weights, scales, kwargs)
        for name in shared_names.values():
            instance, source = captured[name]
            n, k = instance.weight.shape
            self._add(
                name,
                lambda instance=instance, source=source: instance(source),
                linear_work(
                    name,
                    rows=self.s,
                    in_features=k,
                    out_features=n,
                    precision="fp8",
                    local_fp8_kernel=True,
                ),
                linear_verifier(instance, source),
                inputs=(source, instance.weight),
                operator_kind="api_aggregate",
                operator_family="moe_gemm",
                parent_api=name,
                boundary="production shared-expert CheckpointLinear API including activation quantization",
            )
            add_fp8_linear_leaves(self, name, instance, source, family="moe_gemm")
        for name in ("moe_routed_silu", "moe_shared_silu"):
            left, right = captured[name]
            self._add(
                name,
                lambda left=left, right=right: original_silu(left, right),
                non_matmul_work(name),
                lambda actual, left=left, right=right: {
                    **compare(
                        actual,
                        (F.silu(left.float()) * right.float()).bfloat16(),
                        rtol=8e-3,
                        atol=1e-5,
                        relative_l2_limit=1e-5,
                    ),
                    "coverage": "full output",
                    "reference": "FP32 Torch SiLU and multiply, single BF16 conversion",
                },
                inputs=(left, right),
                operator_kind="api_aggregate",
                operator_family="moe_activation",
                parent_api=name,
                boundary="production SiLU adapter including gate/up packing",
            )
            self._silu_leaves(name, left, right)
        left, right = captured["moe_weighted_multiply"]
        self._add(
            "moe_weighted_multiply",
            lambda left=left, right=right: original_mul(left, right),
            non_matmul_work("moe_weighted_multiply"),
            lambda actual, left=left, right=right: {
                **compare(actual, (left.double() * right.double()).float(), rtol=0, atol=0),
                "coverage": "full output",
                "reference": "FP64 multiplication rounded to FP32",
            },
            inputs=(left, right),
            parent_api="moe",
            operator_family="moe_reduce",
            boundary="production FP32 route-output weighting leaf; BF16-to-FP32 cast excluded",
        )
        summed = captured["moe_route_reduce"]
        self._add(
            "moe_route_reduce",
            lambda: original_sum(summed, 1),
            non_matmul_work("moe_route_reduce"),
            lambda actual: {
                **compare(actual, summed.double().sum(1).float(), rtol=2e-4, atol=1e-4),
                "coverage": "full output",
                "reference": "FP64 route sum rounded to FP32",
            },
            inputs=(summed,),
            parent_api="moe",
            operator_family="moe_reduce",
            boundary="production FP32 active-route sum leaf; weighting separately measured",
        )
        cast_source = captured["moe_output_cast"]
        self._add(
            "moe_output_cast",
            lambda: original_bfloat16(cast_source),
            non_matmul_work("moe_output_cast"),
            lambda actual: {
                **compare_bits(actual, cast_source.bfloat16()),
                "coverage": "full output",
            },
            inputs=(cast_source,),
            parent_api="moe",
            operator_family="moe_cast",
            boundary="production final FP32-to-BF16 cast",
        )
        for name in ("moe_routed_output_cast", "moe_shared_output_cast"):
            source = captured[name]
            self._add(
                name,
                lambda source=source: original_float(source),
                non_matmul_work(name),
                lambda actual, source=source: {
                    **compare_bits(actual, source.float()),
                    "coverage": "full output",
                },
                inputs=(source,),
                parent_api="moe",
                operator_family="moe_cast",
                boundary="captured production BF16-to-FP32 expert output conversion",
            )
        add_left, add_right = captured["moe_shared_add"]
        add_output = torch.empty_like(add_left)
        self._add(
            "moe_shared_add",
            lambda: torch.add(add_left, add_right, out=add_output),
            non_matmul_work("moe_shared_add"),
            lambda actual: {
                **compare_bits(actual, (add_left.double() + add_right.double()).float()),
                "coverage": "full output",
                "reference": "FP64 addition rounded to FP32",
            },
            inputs=(add_left, add_right, add_output),
            parent_api="moe",
            operator_family="moe_reduce",
            boundary="production FP32 expert addition arithmetic with independent output destination; original model destination is in-place",
        )

        def verify_shared(actual):
            # Independent sampled-reference MLP consumes exactly the same input
            # and real weights; down output blocks are sampled, inner matrices
            # and the nonlinear intermediate are complete for sampled rows.
            rows = sample_indices(self.s, 8, self.device)
            columns = sample_columns(cfg.dim, self.device)
            blocks = columns[::128] // 128
            x = self.hidden[rows]
            shared = moe.shared
            gate_output = reference_fp8_linear(x, shared.gate.weight, shared.gate.scales)
            up_output = reference_fp8_linear(x, shared.up.weight, shared.up.scales)
            intermediate = (F.silu(gate_output.float()) * up_output.float()).bfloat16()
            expected = reference_fp8_linear(
                intermediate, shared.down.weight[columns], shared.down.scales[blocks]
            )
            return {
                **compare(actual[rows][:, columns].contiguous(), expected),
                "coverage": "sampled token rows/output blocks; complete shared-expert nonlinear intermediate",
                "reference": "independent activation quantization/dequantized FP32 matmuls and FP32 SwiGLU",
            }

        shared_flops = 6 * self.s * cfg.dim * cfg.moe_intermediate_size * cfg.n_shared_experts
        self._add(
            "moe_shared_expert",
            lambda: moe.shared(self.hidden),
            OperatorWork(
                "moe_shared_expert",
                "fp8",
                shared_flops,
                None,
                "6 * S * dim * shared_intermediate_size",
                dimensions={
                    "S": self.s,
                    "dim": cfg.dim,
                    "shared_intermediate_size": cfg.moe_intermediate_size,
                },
                notes="Whole shared expert including quantization and SwiGLU.",
            ),
            verify_shared,
            inputs=(self.hidden,),
            scope="module_aggregate",
            operator_kind="api_aggregate",
            operator_family="moe_gemm",
            parent_api="moe_shared_expert",
            boundary="complete production shared MLP; overlaps shared linears and SiLU",
        )

        def verify_moe(actual):
            rows = sample_indices(self.s, 4, self.device)
            columns = sample_columns(cfg.dim, self.device)
            blocks = columns[::128] // 128
            routed_reference = torch.empty(
                (len(rows), cfg.num_experts_per_tok, len(columns)),
                device=self.device,
                dtype=torch.bfloat16,
            )
            ids = self.expert_ids[rows]
            for expert in ids.unique().cpu().tolist():
                selected = ids == expert
                x = self.hidden[rows, None].expand(-1, cfg.num_experts_per_tok, -1)[selected]
                gate_output = reference_fp8_linear(
                    x, moe.weights["gate_proj"][expert], moe.scales["gate_proj"][expert]
                )
                up_output = reference_fp8_linear(
                    x, moe.weights["up_proj"][expert], moe.scales["up_proj"][expert]
                )
                intermediate = (F.silu(gate_output.float()) * up_output.float()).bfloat16()
                routed_reference[selected] = reference_fp8_linear(
                    intermediate,
                    moe.weights["down_proj"][expert, columns],
                    moe.scales["down_proj"][expert, blocks],
                )
            shared = moe.shared
            x = self.hidden[rows]
            gate_output = reference_fp8_linear(x, shared.gate.weight, shared.gate.scales)
            up_output = reference_fp8_linear(x, shared.up.weight, shared.up.scales)
            intermediate = (F.silu(gate_output.float()) * up_output.float()).bfloat16()
            shared_reference = reference_fp8_linear(
                intermediate, shared.down.weight[columns], shared.down.scales[blocks]
            )
            expected = (
                (routed_reference.float() * self.route_weights[rows, :, None]).sum(1)
                + shared_reference.float()
            ).bfloat16()
            return {
                **compare(actual[rows][:, columns].contiguous(), expected, rtol=0.02, atol=0.0625),
                "coverage": "sampled token rows/output blocks, all active experts and complete nonlinear intermediates; routing independently checked in full",
                "row_indices": rows.cpu().tolist(),
                "column_indices": columns.cpu().tolist(),
                "reference": "independent selected-expert FP32 dequantized MLPs, exact actual routing weights, shared expert and FP32 reduction",
                "full_output_shape": list(actual.shape),
            }

        routed_flops = 6 * self.s * cfg.num_experts_per_tok * cfg.dim * cfg.moe_intermediate_size
        routing_flops = 2 * self.s * cfg.dim * cfg.n_routed_experts
        self._add(
            "moe",
            lambda: moe(self.hidden),
            OperatorWork(
                "moe",
                "mixed",
                routed_flops + shared_flops + routing_flops,
                None,
                "FP8: 6*S*dim*moe_intermediate*(active_experts+shared_experts); FP32: 2*S*dim*routed_experts",
                dimensions={
                    "S": self.s,
                    "fp8_flops": routed_flops + shared_flops,
                    "fp32_flops": routing_flops,
                },
                notes="Mixed precision module MFU needs per-precision ideal-time sum; no single Tensor Core peak applies.",
            ),
            verify_moe,
            inputs=(self.hidden,),
            scope="module_aggregate",
            operator_kind="api_aggregate",
            operator_family="moe_gemm",
            parent_api="moe",
            precision_flops={"fp8": routed_flops + shared_flops, "fp32": routing_flops},
            boundary="complete production MoE including routing, metadata, active experts, shared expert and reduction; overlaps all MoE components",
        )

    def input_identity(self):
        inputs = {}
        for name, tensor in (
            ("hidden", self.hidden),
            ("expert_ids", self.expert_ids),
            ("route_weights", self.route_weights),
        ):
            raw = tensor.contiguous().view(torch.uint8).flatten()
            positions = sample_indices(len(raw), 4096, tensor.device)
            payload = raw[positions].cpu().numpy().tobytes()
            metadata = tensor_metadata(tensor)
            metadata.pop("device")
            inputs[name] = {
                **metadata,
                "sample_sha256": hashlib.sha256(payload).hexdigest(),
                "sampled_bytes": len(payload),
                "hash_method": "4096 or fewer evenly spaced bytes of contiguous logical tensor",
            }
        checkpoint = {
            name: {
                "shard": self.reader.tensor_files[name].name,
                "shape": item["shape"],
                "dtype": item["dtype"],
            }
            for name, item in self.reader.tensor_metadata.items()
            if name.startswith(self.stem + ".")
        }
        return {
            "workload": self.metadata,
            "seed": self.seed,
            "inputs": inputs,
            "checkpoint_tensors": checkpoint,
        }

    def close(self):
        torch.cuda.synchronize(self.device)
        self.cases.clear()
        self.captured.clear()
        self._before_calls.clear()
        self.moe = self.hidden = self.routing = self.expert_ids = self.route_weights = None


@torch.inference_mode()
def build_moe(s, device="cuda:0", seed=20261011, checkpoint="/preset-models"):
    return MoEWorkload(s, device, seed, checkpoint)
