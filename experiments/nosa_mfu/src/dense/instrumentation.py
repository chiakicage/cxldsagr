"""Module scopes for the NOSA nsys experiment."""

from collections import Counter
from unittest.mock import patch

import torch

from models.nosa import layers as nosa_layers
from models.nosa import model as nosa
from models.nosa.rotary import NosaRotaryEmbedding


class ModuleScopes:
    def __init__(self):
        self.phase = "prefill"
        self.flops = Counter()
        self.calls = Counter()

    def scope(self, category, layer="shared"):
        return torch.profiler.record_function(f"nosa::{self.phase}/{category}/{layer}")

    def wrap(self, fn, category, layer="shared", flop_fn=None):
        def call(*args, **kwargs):
            current_category = category(*args, **kwargs) if callable(category) else category
            key = f"{self.phase}/{current_category}/{layer}"
            self.calls[key] += 1
            if flop_fn is not None:
                self.flops[key] += flop_fn(*args, **kwargs)
            with self.scope(current_category, layer):
                return fn(*args, **kwargs)

        return call

    def install(self, model, stack):
        for name, module in model.named_modules():
            parts = name.split(".")
            layer = parts[2] if name.startswith("model.layers.") else "shared"
            category = parts[-1]
            flop_fn = None
            if isinstance(module, torch.nn.Linear):
                # Keep combined QKV and gate/up GEMMs as one real module each.
                # Their full output width already includes every projection;
                # neither time nor FLOPs are attributed to synthetic submodules.
                in_features, out_features = module.in_features, module.out_features

                def flop_fn(x, *, in_features=in_features, out_features=out_features):
                    return 2 * (x.numel() // in_features) * in_features * out_features

            elif isinstance(module, torch.nn.Embedding):
                category = "embedding"
            elif isinstance(module, nosa.NosaRMSNorm):
                category = "final_norm" if name == "model.norm" else category

                def category(x, residual=None, *, base_category=category):
                    # This marks the combined residual/norm execution boundary
                    # on both CUDA and the CPU reference. kernel_names tells
                    # whether that boundary actually used a fused GPU kernel.
                    return (
                        f"{base_category}_add_residual" if residual is not None else base_category
                    )

            elif isinstance(module, nosa.NosaAttention):
                category = "kv_cache_and_layout"
            elif isinstance(module, nosa.NosaMLP):
                category = "swiglu_elementwise"
            elif isinstance(module, nosa.NosaDecoderLayer):
                category = "decoder"
            elif isinstance(module, NosaRotaryEmbedding):
                category = "rope_prepare"
            else:
                continue
            stack.enter_context(
                patch.object(module, "forward", self.wrap(module.forward, category, layer, flop_fn))
            )
        stack.enter_context(
            patch.object(
                nosa_layers, "apply_rotary_qk", self.wrap(nosa_layers.apply_rotary_qk, "rope_apply")
            )
        )

        def attention_flops(q, k, v):
            queries, heads, dim = q.shape
            prefix = k.shape[0] - queries
            pairs = queries * prefix + queries * (queries + 1) // 2
            return 4 * heads * dim * pairs

        stack.enter_context(
            patch.object(
                model,
                "attention",
                self.wrap(model.attention, "attention_core", flop_fn=attention_flops),
            )
        )
