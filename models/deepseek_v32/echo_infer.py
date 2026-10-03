"""Standalone complete DeepSeek V3.2 with layer placement across CUDA devices."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from models.deepseek_v32.echo_model import CheckpointReader, Config, rms_norm


def layer_parameter_bytes(reader, layers):
    sizes = [0] * layers
    for name, metadata in reader.tensor_metadata.items():
        parts = name.split(".")
        if len(parts) > 3 and parts[:2] == ["model", "layers"]:
            layer = int(parts[2])
            if layer < layers:
                a, b = metadata["data_offsets"]
                sizes[layer] += b - a
    if not all(sizes):
        raise ValueError("checkpoint is missing one or more transformer layers")
    return sizes


def plan_layer_devices(sizes, devices, budgets):
    """Contiguous partitions minimizing the largest resident weight allocation."""
    n, groups = len(sizes), len(devices)
    if not 1 <= groups <= n or len(budgets) != groups:
        raise ValueError("one memory budget per nonempty device partition is required")
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + size)
    dp = {(0, 0): (0, [])}
    for g in range(1, groups + 1):
        for end in range(g, n + 1):
            options = []
            for start in range(g - 1, end):
                previous = dp.get((g - 1, start))
                weight = prefix[end] - prefix[start]
                if previous is not None and weight <= budgets[g - 1]:
                    options.append((max(previous[0], weight), previous[1] + [(start, end)]))
            if options:
                dp[g, end] = min(options, key=lambda item: item[0])
    if (groups, n) not in dp:
        raise RuntimeError(
            "available GPU memory cannot hold the complete FP8 checkpoint plus reserve"
        )
    mapping = []
    for device, (start, stop) in zip(devices, dp[groups, n][1]):
        mapping.extend([device] * (stop - start))
    return mapping


class DeepSeekEchoModel:
    """Checkpoint transformer layers, embedding, final norm, and LM head.

    Weights remain on their assigned devices. Hidden states cross NVLink at
    placement boundaries; neither weights nor historical main KV are implicitly
    copied wholesale during an extend step. This is sequential layer placement,
    not tensor-parallel serving or continuous batching. By default all checkpoint
    layers execute; ``num_layers`` explicitly selects a contiguous prefix for
    diagnostics, with each layer consuming the preceding layer's real output.
    """

    def __init__(
        self,
        model_path,
        *,
        devices=(0, 1, 2, 6, 7),
        capacity=66560,
        offload=False,
        slots=16384,
        chunk_size=1024,
        reserve_gib=5,
        num_layers=None,
    ):
        from models.deepseek_v32.echo_block import CheckpointBlock

        self.path = Path(model_path)
        self.cfg = Config.from_checkpoint(self.path)
        self.num_layers = self.cfg.num_hidden_layers if num_layers is None else num_layers
        if (
            not isinstance(self.num_layers, int)
            or isinstance(self.num_layers, bool)
            or not 1 <= self.num_layers <= self.cfg.num_hidden_layers
        ):
            raise ValueError("num_layers must select 1 through all checkpoint transformer layers")
        if capacity > self.cfg.max_seq_len:
            raise ValueError("requested context exceeds checkpoint max_position_embeddings")
        self.devices = [torch.device("cuda", int(index)) for index in devices]
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must be distinct")
        self.reader = CheckpointReader(self.path)
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            expected = json.loads(index.read_text())["weight_map"]
            if self.num_layers < self.cfg.num_hidden_layers:
                prefixes = tuple(f"model.layers.{layer}." for layer in range(self.num_layers))
                expected = {
                    name: shard
                    for name, shard in expected.items()
                    if name.startswith(prefixes)
                    or name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
                }
            missing = [name for name in expected if name not in self.reader.tensor_files]
            if missing:
                raise ValueError(
                    f"incomplete checkpoint: {len(missing)} tensors missing; first={missing[0]}"
                )
        # Fail before allocating hundreds of GB if required endpoint tensors are absent.
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if name not in self.reader.tensor_files:
                raise ValueError(f"incomplete checkpoint: missing {name}")
        sizes = layer_parameter_bytes(self.reader, self.num_layers)
        budgets = []
        for device in self.devices:
            with torch.cuda.device(device):
                if torch.cuda.get_device_capability(device)[0] != 9:
                    raise ValueError("this implementation requires Hopper SM90 GPUs")
                free, _ = torch.cuda.mem_get_info(device)
                budgets.append(free - int(reserve_gib * 2**30))
        # Embedding and LM head remain BF16 on endpoint devices.
        for index, name in ((0, "model.embed_tokens.weight"), (-1, "lm_head.weight")):
            offsets = self.reader.tensor_metadata[name]["data_offsets"]
            budgets[index] -= offsets[1] - offsets[0]
        self.placement = plan_layer_devices(sizes, self.devices, budgets)
        self.capacity, self.slots, self.chunk_size = capacity, slots, chunk_size
        self.blocks = []
        self.embedding_weight = self.reader.get_tensor("model.embed_tokens.weight").to(
            self.devices[0]
        )
        self.final_norm = self.reader.get_tensor("model.norm.weight").to(self.devices[-1])
        self.head_weight = self.reader.get_tensor("lm_head.weight").to(self.devices[-1])
        for layer, device in enumerate(self.placement):
            with torch.cuda.device(device):
                block = CheckpointBlock(
                    self.path,
                    layer,
                    device,
                    reader=self.reader,
                    capacity=capacity,
                    offload=offload,
                    slots=slots,
                    chunk_size=chunk_size,
                )
                self.blocks.append(block)
            print(f"loaded layer {layer + 1}/{self.num_layers} on {device}", flush=True)
        self.length = 0
        self.offload = offload
        self.synchronize()

    def synchronize(self):
        for device in self.devices:
            torch.cuda.synchronize(device)

    def set_cache_mode(self, offload):
        from models.deepseek_v32.echo_attention import EchoAttentionRunner

        self.synchronize()
        for block in self.blocks:
            old = block.attention
            device = old.attention.device
            with torch.cuda.device(device):
                block.attention = EchoAttentionRunner(
                    old.attention,
                    self.capacity,
                    offload=offload,
                    slots=self.slots,
                    chunk_size=self.chunk_size,
                )
                block.cache = block.attention.cache
        self.length, self.offload = 0, offload

    @torch.inference_mode()
    def forward(self, token_ids, *, scope=None, all_logits=False, return_hidden=False):
        """Execute a cache step, optionally returning every token's normalized hidden.

        The default returns last-token logits, or all logits with ``all_logits``.
        ``return_hidden=True`` returns ``{"hidden": ..., "logits": ...}``, where
        hidden includes all input tokens after the final RMSNorm. For a truncated
        diagnostic model these are outputs after its selected transformer prefix,
        not the full checkpoint's hidden states or language-model predictions.
        """
        scope = scope or (lambda _: nullcontext())
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.devices[0])
        if ids.ndim != 1 or ids.numel() < 1 or self.length + ids.numel() > self.capacity:
            raise ValueError("expected a nonempty single sequence fitting cache capacity")
        if int(ids.min()) < 0 or int(ids.max()) >= self.cfg.vocab_size:
            raise ValueError("token ID is outside the checkpoint vocabulary")
        started = []
        try:
            for block in self.blocks:
                block.cache.begin_step(ids.numel())
                started.append(block.cache)
            logits_parts = []
            hidden_parts = []
            for chunk_start in range(0, ids.numel(), self.chunk_size):
                chunk_stop = min(ids.numel(), chunk_start + self.chunk_size)
                with torch.cuda.device(self.devices[0]), scope("embedding"):
                    hidden = F.embedding(ids[chunk_start:chunk_stop], self.embedding_weight)
                    residual = None
                for layer, (block, device) in enumerate(zip(self.blocks, self.placement)):
                    with torch.cuda.device(device):
                        with scope("hidden_transfer"):
                            hidden = hidden.to(device, non_blocking=True)
                            if residual is not None:
                                residual = residual.to(device, non_blocking=True)
                        with scope(f"layer_{layer}"):
                            hidden, residual = block.forward(hidden, residual, scope=scope)
                if return_hidden or all_logits or chunk_stop == ids.numel():
                    with torch.cuda.device(self.devices[-1]), scope("final_norm_lm_head"):
                        hidden = hidden.to(self.devices[-1], non_blocking=True)
                        residual = residual.to(self.devices[-1], non_blocking=True)
                        all_hidden = return_hidden or all_logits
                        selected = hidden if all_hidden else hidden[-1:]
                        selected_residual = residual if all_hidden else residual[-1:]
                        normalized = rms_norm(
                            selected.float() + selected_residual.float(),
                            self.final_norm,
                            self.cfg.norm_eps,
                        ).bfloat16()
                        if return_hidden:
                            hidden_parts.append(normalized)
                        if all_logits or chunk_stop == ids.numel():
                            head_input = normalized if all_logits else normalized[-1:]
                            logits_parts.append(F.linear(head_input, self.head_weight).float())
            output = torch.cat(logits_parts)
            if return_hidden:
                output = {"hidden": torch.cat(hidden_parts), "logits": output}
            # Synchronization surfaces asynchronous execution failures before
            # advancing valid length for any layer.
            self.synchronize()
            if any(
                cache._step_end is None or cache.written != cache._step_end for cache in started
            ):
                raise RuntimeError("one or more model layers did not finish the cache step")
            for block in self.blocks:
                block.cache.commit()
            self.length += ids.numel()
            return output
        except BaseException:
            for cache in started:
                if cache._step_end is not None:
                    cache.rollback()
            raise

    def snapshot_prefix(self):
        """Save exact HBM residency to CPU, outside measurement iterations."""
        self.synchronize()
        states = []
        for block in self.blocks:
            cache = block.cache
            states.append(
                {
                    "length": cache.length,
                    "records": cache.records.cpu() if self.offload else None,
                    "host_to_device": cache.host_to_device.cpu(),
                    "device_to_host": cache.device_to_host.cpu(),
                    "age": cache.age.cpu(),
                    "clock": cache._clock,
                    "offset": block.attention.offset.cpu(),
                }
            )
        return states

    def restore_prefix(self, states):
        self.synchronize()
        for block, state in zip(self.blocks, states):
            cache = block.cache
            with torch.cuda.device(cache.device):
                if state["records"] is not None:
                    cache.records.copy_(state["records"])
                cache.host_to_device.copy_(state["host_to_device"])
                cache.device_to_host.copy_(state["device_to_host"])
                cache.age.copy_(state["age"])
                cache._clock = state["clock"]
                cache.length = cache.written = state["length"]
                cache._step_end = None
                cache.reset_stats()
                block.attention.offset.copy_(state["offset"])
        self.length = states[0]["length"]
        self.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,6,7")
    parser.add_argument("--input-ids", type=Path, required=True, help="JSON token ID list")
    parser.add_argument("--history", type=int, default=65536)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--slots", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--num-layers", type=int, help="execute only the first N checkpoint layers for diagnostics"
    )
    args = parser.parse_args()
    ids = json.loads(args.input_ids.read_text())
    model = DeepSeekEchoModel(
        args.model,
        devices=[int(x) for x in args.devices.split(",")],
        capacity=len(ids),
        offload=args.offload,
        slots=args.slots,
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
    )
    model.forward(ids[: args.history])
    result = model.forward(ids[args.history :])
    print(json.dumps({"length": model.length, "next_token_id": int(result.argmax(-1))}))


if __name__ == "__main__":
    main()
