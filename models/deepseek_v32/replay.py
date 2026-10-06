"""Physical checkpoint copies and source-input propagation for replay workloads.

The first three blocks propagate real checkpoint hidden/residual streams. Each
subsequent block receives independent clones of its corresponding source input.
This does not define a trained model or the sequential three-layer benchmark.
"""

import math
from dataclasses import dataclass, field


def replay_parameter_count(reader, num_layers):
    """Count model parameters, excluding FP8 scaling metadata."""
    if type(num_layers) is not int or num_layers < 3:
        raise ValueError("the replay model requires at least three physical layers")
    counts = []
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        names = [name for name in reader.tensor_metadata if name.startswith(prefix)]
        if not names:
            raise ValueError(f"checkpoint is missing source layer {layer}")
        counts.append(
            sum(
                math.prod(reader.tensor_metadata[name]["shape"])
                for name in names
                if not name.endswith(".weight_scale_inv")
            )
        )
    endpoints = sum(
        math.prod(reader.tensor_metadata[name]["shape"])
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    )
    return {
        "source_layer_parameters": counts,
        "backbone_parameters": sum(counts[layer % 3] for layer in range(num_layers)),
        "endpoint_parameters": endpoints,
        "total_parameters": endpoints + sum(counts[layer % 3] for layer in range(num_layers)),
    }


@dataclass(frozen=True)
class ReplayLayout:
    source_layers: tuple[int, ...]

    @classmethod
    def repeated_sources(cls, physical_layers):
        if type(physical_layers) is not int or physical_layers < 3:
            raise ValueError("the replay model requires at least three physical layers")
        return cls(tuple(layer % 3 for layer in range(physical_layers)))

    def __post_init__(self):
        if len(self.source_layers) < 3 or self.source_layers != tuple(
            layer % 3 for layer in range(len(self.source_layers))
        ):
            raise ValueError("replay requires checkpoint sources 0, 1, 2 in repeating order")

    def load_blocks(self, model_path, device, reader, *, chunk_size, linear_backend):
        from models.deepseek_v32.layers import CheckpointBlock

        # Each constructor loads distinct weights, including copies of one source.
        return [
            CheckpointBlock(
                model_path,
                source,
                device,
                reader,
                capacity=1,
                chunk_size=chunk_size,
                linear_backend=linear_backend,
                allocate_cache=False,
            )
            for source in self.source_layers
        ]

    def begin_chunk(self):
        return ReplayInputs(self)


@dataclass
class ReplayInputs:
    layout: ReplayLayout
    sources: list = field(default_factory=list)

    def for_layer(self, layer, hidden, residual, *, scope, copy_source=True):
        if layer < 3:
            self.sources.append((hidden, residual))
            return hidden, residual
        with scope("replay_input_copy"):
            source_hidden, source_residual = self.sources[self.layout.source_layers[layer]]
            if not copy_source:
                # Graph replay copies these into the physical block's owned
                # static inputs. Avoid cloning once here and copying again.
                return source_hidden, source_residual
            return (
                source_hidden.clone(),
                source_residual.clone() if source_residual is not None else None,
            )
