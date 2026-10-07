"""Independent cold-ECHO stage observation; never installed in timing/profile runs."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src.extend_graph_validation import tensor_identity


class ColdPrefetchObserver:
    """Copy actual eager/captured stages into preallocated diagnostic storage.

    CUDA Graph captures only copy nodes into these buffers. Buffer allocation is
    outside graph capture and outside model timing; the observer outlives every
    diagnostic graph, which its caller destroys before releasing the observer.
    """

    def __init__(self, model, history, append, output):
        if (
            model.cache_method != "echo"
            or model.length != history
            or history + append > model.slots
            or len(model.blocks) != 3
            or len(model._shared_pools) != 1
            or any(len(pool._sessions) != 1 for pool in model._shared_pools.values())
        ):
            raise ValueError("stage observation requires bounded, single-session cold ECHO")
        self.model, self.history, self.append = model, history, append
        self.output = Path(output)
        self.active = None
        self.hooks = ExitStack()
        self.metadata, self.eager, self.captured = [], [], []
        if getattr(model, "_prefetch_validation_owner", None) is not None:
            raise RuntimeError("a prior prefetch observer still owns diagnostic storage")
        model._prefetch_validation_owner = self
        try:
            self._initialize_buffers()
        except BaseException as error:
            try:
                model.synchronize()
            except BaseException as cleanup:  # noqa: BLE001 -- retain unsafe storage owner
                model._poisoned = True
                raise BaseExceptionGroup(
                    "prefetch observer allocation and drain failed", [error, cleanup]
                ) from None
            del model._prefetch_validation_owner
            raise

    def _initialize_buffers(self):
        history, append, model = self.history, self.append, self.model
        for index, block in enumerate(model.blocks):
            cache = block.cache
            if not cache.offload or cache.transient_start is not None:
                raise ValueError("stage observation requires persistent offloaded KV")
            global_ids = cache.logical_to_global(
                torch.arange(history + append, device=cache.device, dtype=torch.int64)
            ).cpu()
            if bool((cache.host_to_device != torch.iinfo(torch.int32).max).any()):
                raise ValueError("cold stage observation requires initially empty HBM maps")
            self.metadata.append(
                {
                    "layer": index,
                    "H": history,
                    "A": append,
                    "slots": cache.slots,
                    "record_bytes": cache.record_bytes,
                    "topk": block.attention.cfg.index_topk,
                    "max_prefetch": min(8192, cache.slots - append),
                    "global_ids": global_ids,
                }
            )
            self.eager.append(self._allocate(block))
            self.captured.append(self._allocate(block))
        model.synchronize()

    def _allocate(self, block):
        cache, layer = block.cache, block.cache._pool.layers[block.cache.layer_id]
        stages = {}
        for name in ("initial", "after_prefetch", "after_append", "after_recall"):
            stages[name] = {
                "host_to_device": torch.empty_like(cache.host_to_device),
                "reverse": torch.empty_like(cache.device_to_host),
                "free": torch.empty_like(layer.free),
                "priority": torch.empty_like(layer.priority),
                "clock": torch.empty_like(layer.clock_tensor),
                "counter_totals": torch.empty_like(cache._counter_totals),
            }
        stages["after_prefetch"].update(
            prefetch_counter=torch.empty_like(cache._pool.counter),
            prefetch_stats=torch.empty_like(cache._pool.prefetch_stats),
        )
        return {
            "stages": stages,
            "initial_hint": torch.empty_like(block.attention.offset),
            "scores": torch.empty(
                (self.append, self.history + self.append), device=cache.device, dtype=torch.float32
            ),
            "indices": torch.empty(
                (self.append, min(block.attention.cfg.index_topk, self.history + self.append)),
                device=cache.device,
                dtype=torch.int32,
            ),
            "complete": False,
        }

    def _snapshot(self, index, target, stage):
        cache = self.model.blocks[index].cache
        layer = cache._pool.layers[cache.layer_id]
        sources = {
            "host_to_device": cache.host_to_device,
            "reverse": cache.device_to_host,
            "free": layer.free,
            "priority": layer.priority,
            "clock": layer.clock_tensor,
            "counter_totals": cache._counter_totals,
        }
        if stage == "after_prefetch":
            sources.update(
                prefetch_counter=cache._pool.counter, prefetch_stats=cache._pool.prefetch_stats
            )
        for name, source in sources.items():
            target["stages"][stage][name].copy_(source)

    def _prepare(self, original, index, start, count, offset, **kwargs):
        if self.active is not None or (start, count) != (self.history, self.append):
            raise ValueError("observed execution must contain one complete batch per layer")
        target = (self.captured if torch.cuda.is_current_stream_capturing() else self.eager)[index]
        target["complete"] = False
        self.active = (index, target)
        self._snapshot(index, target, "initial")
        target["initial_hint"].copy_(offset)
        prefetch = original(start, count, offset, **kwargs)
        if prefetch is None or prefetch["max_prefetch"] != self.metadata[index]["max_prefetch"]:
            raise ValueError("cold observer did not execute the declared bounded prefetch")
        return prefetch

    def _stage(self, original, index, stage, *args, **kwargs):
        if self.active is None or self.active[0] != index:
            raise ValueError("prefetch stage lacks its matching layer invocation")
        target = self.active[1]
        if stage == "after_recall":
            indices = args[0]
            if indices.shape != target["indices"].shape or indices.dtype != torch.int32:
                raise ValueError("exact selection differs from the observed shape")
            target["indices"].copy_(indices)
        result = original(*args, **kwargs)
        self._snapshot(index, target, stage)
        if stage == "after_recall":
            target["complete"] = True
            self.active = None
        return result

    def __enter__(self):
        from operators.deepseek_v32.indexer import echo

        for index, block in enumerate(self.model.blocks):
            cache = block.cache
            original = cache.prepare_prefetch
            self.hooks.enter_context(
                patch.object(
                    cache,
                    "prepare_prefetch",
                    lambda start, count, offset, _original=original, _index=index, **kwargs: (
                        self._prepare(_original, _index, start, count, offset, **kwargs)
                    ),
                )
            )
            for method, stage in (
                ("finalize_prefetch", "after_prefetch"),
                ("append", "after_append"),
                ("_ensure_from_topk", "after_recall"),
            ):
                original = getattr(cache, method)
                self.hooks.enter_context(
                    patch.object(
                        cache,
                        method,
                        lambda *args, _original=original, _index=index, _stage=stage, **kwargs: (
                            self._stage(_original, _index, _stage, *args, **kwargs)
                        ),
                    )
                )
        original_logits = echo.logits

        def logits(*args, **kwargs):
            result = original_logits(*args, **kwargs)
            if self.active is None:
                raise ValueError("observed logits have no matching prefetch invocation")
            target = self.active[1]["scores"]
            if result.shape[0] != self.append or result.shape[1] < self.history + self.append:
                raise ValueError("observed logits have unexpected token dimensions")
            target.copy_(result[:, : self.history + self.append])
            return result

        self.hooks.enter_context(patch.object(echo, "logits", logits))
        return self

    def __exit__(self, error_type, error, traceback):
        # The caller closes graphs first; only then may their copy destinations
        # and patched objects lose their final references.
        cleanup = []
        closed = False
        try:
            self.model._close_extend_graphs()
            closed = True
        except BaseException as exc:  # noqa: BLE001 -- preserve graph cleanup failures
            self.model._poisoned = True
            cleanup.append(exc)
        try:
            self.hooks.__exit__(error_type, error, traceback)
        except BaseException as exc:  # noqa: BLE001 -- preserve every cleanup failure
            cleanup.append(exc)
        if closed:
            del self.model._prefetch_validation_owner
        # On an uncertain graph close, model retains this observer and all
        # external memcpy destinations even after Python patches are removed.
        if cleanup:
            if error is not None:
                cleanup.insert(0, error)
            if len(cleanup) == 1:
                raise cleanup[0]
            raise BaseExceptionGroup("prefetch observation and cleanup failed", cleanup)
        return False

    def audit(self, label, final_cache_state, *, captured=False):
        from experiments.deepseek_v32_mfu.src.prefetch_transition_audit import (
            compact_evidence,
            validate_execution,
        )

        self.model.synchronize()
        layers = []
        for index, target in enumerate(self.captured if captured else self.eager):
            if not target["complete"]:
                raise ValueError("incomplete observer buffers cannot validate an execution")
            metadata = self.metadata[index]
            stages = {}
            for name, stage in target["stages"].items():
                copied = {key: value.detach().cpu().clone() for key, value in stage.items()}
                mapping = copied.pop("host_to_device")
                copied["logical_to_slot"] = mapping[metadata["global_ids"]].clone()
                if mapping.numel() != self.history + self.append:
                    raise ValueError("cold evidence must cover the entire sole-session host arena")
                stages[name] = copied
            layers.append(
                {
                    **metadata,
                    "initial_hint": target["initial_hint"][:1].detach().cpu().clone(),
                    "scores": target["scores"].detach().cpu().clone(),
                    "indices": target["indices"].detach().cpu().clone(),
                    "stages": stages,
                    "metrics": self.model.blocks[index].cache.metrics(),
                    "final_cache_state": final_cache_state["layers"][index],
                }
            )
        evidence = {
            "schema_version": 1,
            "method": "echo",
            "residency": "cold",
            "single_session": True,
            "layers": layers,
        }
        proof = validate_execution(evidence)
        if proof.get("passed") is not True:
            raise ValueError("transition auditor did not accept the observed execution")
        compact = compact_evidence(evidence)
        compact_proof = validate_execution(compact)
        if proof != compact_proof:
            raise ValueError("saved compact transition evidence changes the acceptance result")
        path = self.output / f"echo_{label}_prefetch_evidence.pt"
        torch.save(compact, path)
        receipt = {
            "schema": "cold-echo-stage-acceptance-v1",
            "passed": True,
            "scope": {
                "method": "echo",
                "residency": "cold",
                "single_session": True,
                "num_layers": len(layers),
                "H": self.history,
                "A": self.append,
                "slots": self.model.slots,
                "max_prefetch": min(8192, self.model.slots - self.append),
            },
            "proof": proof,
            "evidence_file": path.name,
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "final_cache_state_sha256": identity_digest(final_cache_state),
            "indices": [tensor_identity(layer["indices"]) for layer in layers],
            "scores": [tensor_identity(layer["scores"]) for layer in layers],
            "initial_hints": [tensor_identity(layer["initial_hint"]) for layer in layers],
            "boundary": (
                "Actual independent-check eager or diagnostic-graph execution; graph copy nodes "
                "write preallocated buffers. No observer in clean timings or profile. "
                "Saved eligibility binds runtime score hashes; full scores are not retained."
            ),
        }
        (self.output / f"echo_{label}_prefetch_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
        return receipt
