"""Module scopes for the NOSA nsys experiment and archived profiler checks."""

from collections import Counter
from unittest.mock import patch

import torch

from models.nosa import layers as nosa_layers
from models.nosa import model as nosa


class ModuleScopes:
    def __init__(self):
        self.phase = "prefill"
        self.flops = Counter()
        self.calls = Counter()

    def scope(self, category, layer="shared"):
        return torch.profiler.record_function(f"nosa::{self.phase}/{category}/{layer}")

    def wrap(self, fn, category, layer="shared", flop_fn=None):
        def call(*args, **kwargs):
            key = f"{self.phase}/{category}/{layer}"
            self.calls[key] += 1
            if flop_fn is not None:
                self.flops[key] += flop_fn(*args, **kwargs)
            with self.scope(category, layer):
                return fn(*args, **kwargs)

        return call

    def install(self, model, stack):
        for name, module in model.named_modules():
            parts = name.split(".")
            layer = parts[2] if name.startswith("model.layers.") else "shared"
            category = parts[-1]
            flop_fn = None
            if isinstance(module, torch.nn.Linear):
                in_features, out_features = module.in_features, module.out_features

                def flop_fn(x, in_features=in_features, out_features=out_features):
                    return 2 * (x.numel() // in_features) * in_features * out_features

            elif isinstance(module, torch.nn.Embedding):
                category = "embedding"
            elif isinstance(module, nosa.NosaRMSNorm):
                category = "final_norm" if name == "model.norm" else category
            elif isinstance(module, nosa.NosaAttention):
                category = "kv_cache_and_layout"
            elif isinstance(module, nosa.NosaMLP):
                category = "swiglu_elementwise"
            elif isinstance(module, nosa.NosaDecoderLayer):
                category = "residual"
            else:
                continue
            stack.enter_context(
                patch.object(module, "forward", self.wrap(module.forward, category, layer, flop_fn))
            )
        stack.enter_context(
            patch.object(
                nosa_layers, "apply_rotary", self.wrap(nosa_layers.apply_rotary, "rope_apply")
            )
        )
        stack.enter_context(
            patch.object(nosa, "rotary_cos_sin", self.wrap(nosa.rotary_cos_sin, "rope_prepare"))
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
