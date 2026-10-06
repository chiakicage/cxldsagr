"""Standalone replay retaining production cache stages and the real MLA consumer."""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from cache.sparse_token_cache import SparseTokenCache
from cache.sparse_token_pool import MISSING, PAGE_SIZE, SharedSparseTokenPool
from experiments.deepseek_v32_mfu.src.kernel_profile import file_sha256, load_inputs
from models.deepseek_v32.attention import EchoAttentionRunner
from models.deepseek_v32.cache.prefetch import PoolHistoryPrefetch
from operators.deepseek_v32.indexer import cache_ops

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")
PHASES = ("prefill_last_chunk", "extend_cold", "extend_warm")
# A failed asynchronous completion must keep borrowed host/device storage alive.
_FAILED_REPLAYS = []


@dataclass(frozen=True)
class Config:
    history: int = 65536
    append: int = 128
    chunk: int = 1024
    layers: int = 3

    def __post_init__(self):
        if min(self.history, self.append, self.chunk, self.layers) < 1:
            raise ValueError("workload dimensions must be positive")
        if self.chunk > self.history or self.history % self.chunk:
            raise ValueError("history must contain whole prefill chunks")

    @property
    def capacity(self):
        return self.history + self.append

    @property
    def slots(self):
        return (self.capacity + 63) // 64 * 64


def load_workload(directory, config):
    directory = Path(directory).resolve(strict=True)
    source_result = directory / "result.json"
    source = json.loads(source_result.read_text())
    if source.get("accepted") is not True or not source.get("run_id"):
        raise ValueError("input capture must belong to an accepted MFU source run")
    if (
        source.get("prefix_tokens") != config.history
        or source.get("extend_tokens") != config.append
        or source.get("num_layers") != config.layers
    ):
        raise ValueError("accepted source geometry differs from the replay")
    captures, identity = [], []
    for layer in range(config.layers):
        path = directory / f"kernel_inputs_layer_{layer}.pt"
        data = load_inputs(path)
        if data.get("layer") != layer or data["source_run_id"] != source["run_id"]:
            raise ValueError("capture layer or source run does not match its source result")
        if data["query_start"] != config.history or len(data["q"]) != config.append:
            raise ValueError("capture H/A differs from the requested replay")
        if len(data["kv"]) != config.capacity:
            raise ValueError("capture does not cover the complete pending append")
        captures.append(data)
        identity.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "source_run_id": data["source_run_id"],
                "layer": layer,
                "source_result_sha256": file_sha256(source_result),
                "source_validation_receipt": source.get("validation_receipt"),
            }
        )
    return captures, identity


def phase_input(capture, phase, config):
    """Synthetic prefill Q repeats captured extend queries; KV/keys stay captured."""
    if phase not in PHASES:
        raise ValueError("unknown phase")
    start = config.history - config.chunk if phase == PHASES[0] else config.history
    count = config.chunk if phase == PHASES[0] else config.append
    end = start + count
    result = {
        name: capture[name][:end].contiguous() for name in ("kv", "index_keys", "index_scales")
    }
    for name in ("q", "index_q", "index_weights"):
        tensor = capture[name]
        repeat = (count + len(tensor) - 1) // len(tensor)
        result[name] = tensor.repeat((repeat,) + (1,) * (tensor.ndim - 1))[:count].contiguous()
    result.update(
        start=start,
        count=count,
        end=end,
        input_kind="synthetic_repeated_extend_queries"
        if phase == PHASES[0]
        else "captured_extend_queries",
    )
    return result


class Replay:
    def __init__(self, captures, scheme, phase, config, *, device="cuda:0"):
        self.pool = self.prefetch = None
        self.runners, self.outputs = [], []
        try:
            self._initialize(captures, scheme, phase, config, device=device)
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve failed owners and errors.
                _FAILED_REPLAYS.append(self)
                raise BaseExceptionGroup(
                    "replay construction and cleanup failed", [original, cleanup]
                ) from None
            raise

    def _initialize(self, captures, scheme, phase, config, *, device):
        if scheme not in SCHEMES or phase not in PHASES:
            raise ValueError("unknown scheme or phase")
        self.scheme, self.phase, self.config = scheme, phase, config
        self.reference_indices = [capture["indices"] for capture in captures]
        self.device = torch.device(device)
        self.inputs = [
            {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in phase_input(data, phase, config).items()
            }
            for data in captures
        ]
        self.pool = None
        self.prefetch = None
        self.runners = []
        if scheme != "hbm":
            self.pool = SharedSparseTokenPool(
                config.slots,
                576,
                config.layers,
                config.slots,
                device=self.device,
                dtype=torch.bfloat16,
                metadata_ops=cache_ops,
                dense_contiguous=scheme == "dense_prefetch",
            )
            session = self.pool.allocate_session(config.capacity)
            if scheme == "dense_prefetch":
                self.prefetch = PoolHistoryPrefetch(self.device)
        for layer, (data, original) in enumerate(zip(self.inputs, captures, strict=True)):
            cache = (
                session.layer(layer)
                if self.pool is not None
                else SparseTokenCache(config.capacity, 576, device=self.device)
            )
            cfg = SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                index_topk=2048,
                attention_scale=original["attention_scale"],
            )
            attention = SimpleNamespace(cfg=cfg, device=self.device)
            runner = EchoAttentionRunner(
                attention,
                config.capacity,
                cache=cache,
                offload=scheme != "hbm",
                slots=config.slots,
                chunk_size=config.chunk,
                fused_prefetch=scheme == "echo",
            )
            runner.index_keys[: data["start"]].copy_(data["index_keys"][: data["start"]])
            runner.index_scales[: data["start"]].copy_(data["index_scales"][: data["start"]])
            self._append_prefix(cache, data)
            self.runners.append(runner)
        self.snapshot = (
            self.pool.snapshot()
            if self.pool is not None and self.phase != "prefill_last_chunk"
            else None
        )
        self.outputs = []
        self.reset()

    def _append_prefix(self, cache, data):
        if not data["start"]:
            return
        cache.begin_step(data["start"])
        for start in range(0, data["start"], self.config.chunk):
            cache.append(data["kv"][start : min(start + self.config.chunk, data["start"])])
        cache.commit()

    def reset(self):
        if self.prefetch is not None:
            self.prefetch.drain()
        if self.phase == "prefill_last_chunk":
            # Continuous prefill retains the residency proof produced by real
            # appends. Snapshot restoration deliberately invalidates that proof.
            for runner, data in zip(self.runners, self.inputs, strict=True):
                runner.cache.truncate(0)
                self._append_prefix(runner.cache, data)
                if self.pool is not None and not runner.cache.all_history_resident:
                    raise RuntimeError("append-built prefill prefix lost its residency proof")
        elif self.pool is not None:
            self.pool.restore(self.snapshot)
        else:
            for runner, data in zip(self.runners, self.inputs, strict=True):
                runner.cache.truncate(data["start"])
        if self.phase == "extend_cold" and self.pool is not None:
            for runner in self.runners:
                cache = runner.cache
                with cache.operation():
                    self.pool.release_ids(cache.layer_id, cache.session.global_ids())
        for runner in self.runners:
            runner.offset.zero_()
            runner.cache.reset_stats()
        self.outputs.clear()
        torch.cuda.synchronize(self.device)

    def run(self, scope=None):
        scope = scope or (lambda name: nullcontext())
        tickets = {}
        try:
            with scope("transaction_begin"):
                for runner, data in zip(self.runners, self.inputs, strict=True):
                    runner.cache.begin_step(data["count"])
            if self.prefetch is not None:
                with scope("dense_prefetch_0"):
                    tickets[0] = self.prefetch.prefetch(self.runners[0].cache)
            for layer, (runner, data) in enumerate(zip(self.runners, self.inputs, strict=True)):
                with scope(f"layer_{layer}"):
                    if self.prefetch is not None:
                        with scope("dense_wait"):
                            self.prefetch.wait(tickets[layer])
                        if layer + 1 < len(self.runners):
                            with scope(f"dense_prefetch_{layer + 1}"):
                                tickets[layer + 1] = self.prefetch.prefetch(
                                    self.runners[layer + 1].cache
                                )

                    def project(hidden, position, *, normalized=False, data=data):
                        if position != data["start"]:
                            raise RuntimeError("replay projection position changed")
                        return SimpleNamespace(
                            q=data["q"],
                            index_q=data["index_q"],
                            index_weights=data["index_weights"],
                            index_k=data["index_keys"][position : data["end"]],
                            index_scale=data["index_scales"][position : data["end"]],
                            kv=data["kv"][position : data["end"]],
                        )

                    # This tensor only communicates query count; projection is
                    # the captured input provider, not timed model computation.
                    hidden = data["index_weights"][:, :1]
                    attention = runner.forward(
                        hidden,
                        scope=scope,
                        capture_indices=True,
                        normalized=True,
                        project_callback=project,
                        output_callback=lambda x: x,
                    )
                    self.outputs.append((runner.last_indices[0], attention))
            with scope("transaction_complete"):
                if self.prefetch is not None:
                    self.prefetch.drain()
                torch.cuda.synchronize(self.device)
                for runner in self.runners:
                    runner.cache.commit()
        except BaseException as original:
            errors = []
            for cleanup in [self.prefetch.drain] if self.prefetch is not None else []:
                try:
                    cleanup()
                except BaseException as error:  # noqa: BLE001 -- retain cleanup failures.
                    errors.append(error)
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as error:  # noqa: BLE001 -- retain cleanup failures.
                errors.append(error)
            if not errors:
                for runner in self.runners:
                    if runner.cache._step_end is not None:
                        try:
                            runner.cache.rollback()
                        except BaseException as error:  # noqa: BLE001 -- retain cleanup failures.
                            errors.append(error)
            if errors:
                raise BaseExceptionGroup("manager replay and cleanup failed", [original, *errors])
            raise

    def verify(self, reference=None):
        """Check completed outputs and cache maps without changing the timed path.

        Physical IDs are reconstructed from the completed cache state on CPU;
        they are not a captured copy of the IDs passed to MLA. Bitwise output
        equality against the resident replay additionally checks the actual
        attention consumption of the same logical selection and captured KV.
        """
        if reference is not None and (
            not isinstance(reference, dict)
            or set(reference) != {"indices", "attention"}
            or any(len(reference[key]) != len(self.runners) for key in reference)
        ):
            raise ValueError("attention reference must contain one selection and output per layer")
        checks, outputs = [], {"indices": [], "attention": []}
        for layer, (runner, data, output) in enumerate(
            zip(self.runners, self.inputs, self.outputs, strict=True)
        ):
            indices, attention = (tensor.detach().cpu() for tensor in output)
            if self.phase != "prefill_last_chunk":
                torch.testing.assert_close(indices, self.reference_indices[layer], rtol=0, atol=0)
            if reference is not None:
                torch.testing.assert_close(indices, reference["indices"][layer], rtol=0, atol=0)
            expected_shape = (data["count"], data["q"].shape[1], runner.cfg.kv_lora_rank)
            if attention.shape != expected_shape or attention.dtype != data["q"].dtype:
                raise AssertionError("MLA output shape or dtype differs from the captured query")
            if not bool(torch.isfinite(attention).all()):
                raise AssertionError("MLA output contains non-finite values")
            if reference is not None:
                torch.testing.assert_close(attention, reference["attention"][layer], rtol=0, atol=0)
            causal = torch.arange(data["start"] + 1, data["end"] + 1)[:, None]
            if bool(((indices < -1) | (indices >= causal)).any()):
                raise AssertionError("selection is not causal or contains invalid padding")
            valid = indices >= 0
            expected_counts = causal.flatten().clamp_max(2048)
            if not torch.equal(valid.sum(dim=1), expected_counts):
                raise AssertionError("selection cardinality differs from exact top-k")
            ordered = indices.sort(dim=1).values
            if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
                raise AssertionError("selection contains duplicate logical IDs in one query")
            logical = torch.unique(indices[valid], sorted=True).long()
            if self.pool is not None:
                pages = runner.cache.page_table.detach().cpu().long()
                global_ids = pages[logical // PAGE_SIZE] * PAGE_SIZE + logical % PAGE_SIZE
                mapped = runner.cache.host_to_device.detach().cpu()[global_ids].long()
                if bool(((mapped < 1) | (mapped >= len(runner.cache.records))).any()):
                    raise AssertionError("selected logical record has no valid published pool slot")
                if len(mapped.unique()) != len(logical):
                    raise AssertionError("different logical records share a published pool slot")
            else:
                mapped = logical
            actual = runner.cache.records[mapped.to(self.device)].cpu()
            torch.testing.assert_close(
                actual, data["kv"][logical.to(self.device)].cpu(), rtol=0, atol=0
            )
            if self.pool is not None:
                state = self.pool.layers[layer]
                h2d, d2h, free = (
                    tensor.cpu()
                    for tensor in (state.host_to_device, state.device_to_host, state.free)
                )
                live = torch.where(d2h != MISSING)[0]
                if not torch.equal(h2d[d2h[live]].long(), live):
                    raise AssertionError("pool maps are not inverse")
                if not torch.equal(free[1:], d2h[1:] == MISSING):
                    raise AssertionError("free bitmap differs from published maps")
                if int(state.clock_tensor.cpu()) != state.clock:
                    raise AssertionError("GPU/CPU FIFO clocks differ")
            checks.append(
                {
                    "layer": layer,
                    "unique_selected": len(logical),
                    "exact_records": True,
                    "selected_record_source": "post_consume_cache_maps",
                    "attention_shape": list(attention.shape),
                    "attention_finite": True,
                    "attention_matches_resident": reference is not None,
                    "metrics": runner.cache.metrics(),
                }
            )
            outputs["indices"].append(indices)
            outputs["attention"].append(attention)
        return checks, outputs

    def close(self):
        try:
            self._close()
        except BaseException:
            _FAILED_REPLAYS.append(self)
            raise

    def _close(self):
        if self.prefetch is not None:
            self.prefetch.close()
        if self.pool is not None:
            self.pool.close()
        elif hasattr(self, "device"):
            torch.cuda.synchronize(self.device)
        self.runners.clear()
        self.outputs.clear()
