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
from experiments.deepseek_v32_mfu.src.prefetch_transition_audit import (
    BOUNDED_FREE_PREPARATION,
    COARSE_POLICY,
    OFFICIAL_POLICY,
    preparation_capacity,
)


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
                    "host_arena_tokens": cache.host_to_device.numel(),
                    "session_host_tokens": cache.session.host_tokens,
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
            "official_staging": {
                "host_ids": torch.empty(64, device=cache.device, dtype=torch.int32),
                "records": torch.empty(
                    (64, cache.record_bytes // 2), device=cache.device, dtype=torch.bfloat16
                ),
                "prepared_slots": torch.empty(64, device=cache.device, dtype=torch.int32),
                "allocation_log": torch.empty_like(cache.device_to_host),
            },
            "policy_metadata": {},
            "preparation_metadata": {},
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
        from operators.deepseek_v32.indexer import echo

        if self.active is not None or (start, count) != (self.history, self.append):
            raise ValueError("observed execution must contain one complete batch per layer")
        target = (self.captured if torch.cuda.is_current_stream_capturing() else self.eager)[index]
        target["complete"] = False
        target["policy_metadata"] = {}
        target["preparation_metadata"] = {}
        self.active = (index, target)
        self._snapshot(index, target, "initial")
        target["initial_hint"].copy_(offset)
        prefetch = original(start, count, offset, **kwargs)
        if prefetch is None:
            raise ValueError("cold observer did not execute the declared bounded prefetch")
        token = prefetch.get("_prepared")
        if type(token) is echo._BoundedPreparedPrefetch:
            cache = self.model.blocks[index].cache
            pool = cache._pool
            expected_owner = (cache.session.owner, cache.layer_id)
            actual = tuple(
                prefetch[name]
                for name in ("free_slots", "allocation_log", "counter", "prefetch_stats")
            )
            if (
                token.used is not False
                or type(token.rows) is not int
                or token.rows != count
                or type(token.prepared_limit) is not int
                or token.prepared_limit != 64
                or type(prefetch.get("prepared_limit")) is not int
                or prefetch["prepared_limit"] != token.prepared_limit
                or len(token.tensors) != len(actual)
                or any(left is not right for left, right in zip(token.tensors, actual, strict=True))
                or prefetch.get("history_length") != cache.host_written_end
                or prefetch.get("transient_suffix") is not False
                or cache.slots != self.metadata[index]["slots"]
                or token.scratch is not pool.miss_scratch
                or any(
                    left is not right
                    for left, right in zip(
                        actual,
                        (pool.free_slots, pool.allocation_log, pool.counter, pool.prefetch_stats),
                        strict=True,
                    )
                )
            ):
                raise ValueError("bounded observer token does not bind the returned lease")
            preparation = {
                "kind": BOUNDED_FREE_PREPARATION,
                "query_start": start,
                "query_count": count,
                "visible_columns": cache.indexer_visible_end,
                "initialized_history": cache.host_written_end,
                "prepared_limit": token.prepared_limit,
                "requested_max_prefetch": min(kwargs.get("limit", 8192), 8192, cache.slots - 1),
                "single_session": len(pool._sessions) == 1,
                "exclusive_operation": pool._active == expected_owner and pool._depth > 0,
                "pending_owner_matches": pool._pending_prefetch == expected_owner,
                "persistent_append": cache.transient_start is None,
            }
            metadata = {
                **self.metadata[index],
                "preparation": preparation,
                "prefetch_policy": OFFICIAL_POLICY,
                "max_prefetch": prefetch.get("max_prefetch"),
                "prepared_max_prefetch": token.prepared_limit,
                "hint_index": 1,
            }
            preparation_capacity(metadata)
            target["preparation_metadata"] = {
                "preparation": preparation,
                "prepared_max_prefetch": token.prepared_limit,
            }
            target["prepared_token"] = token
        elif (
            (token is not None and type(token) is not echo._PreparedPrefetch)
            or "prepared_limit" in prefetch
            or prefetch["max_prefetch"] != self.metadata[index]["max_prefetch"]
        ):
            raise ValueError("cold observer did not execute the declared full prefetch")
        target["prefetch_lease"] = prefetch
        return prefetch

    def _prefetch_policy(self, index, target, lease):
        policy = lease.get("prefetch_policy", COARSE_POLICY)
        preparation = target["preparation_metadata"]
        if preparation and (
            policy != OFFICIAL_POLICY
            or lease is not target["prefetch_lease"]
            or lease.get("_prepared") is not target["prepared_token"]
            or target["prepared_token"].used is not True
            or type(lease.get("prepared_limit")) is not int
            or lease.get("prepared_limit") != 64
            or type(lease.get("max_prefetch")) is not int
            or lease.get("max_prefetch") != 64
        ):
            raise ValueError("bounded observer requires consumption by the official Q1 policy")
        if policy == COARSE_POLICY:
            target["policy_metadata"] = {}
            return
        if policy != OFFICIAL_POLICY:
            raise ValueError("observed prefetch used an unknown policy")
        if (
            self.append != 1
            or lease.get("official_prefetch_cap") != 64
            or lease["max_prefetch"] < 64
            or "_official_prefetch" not in lease
        ):
            raise ValueError("official Q1 observation lacks cap or staging ownership")
        state = lease["_official_prefetch"]
        sources = {
            "host_ids": state.host_ids.reshape(64),
            "records": state.records.reshape(64, -1),
            "prepared_slots": lease["free_slots"][:64],
            "allocation_log": lease["allocation_log"],
        }
        for name, source in sources.items():
            target["official_staging"][name].copy_(source)
        target["policy_metadata"] = {
            "prefetch_policy": OFFICIAL_POLICY,
            "max_prefetch": 64,
            "prepared_max_prefetch": self.metadata[index]["max_prefetch"],
            "hint_index": 1,
            **preparation,
        }

    def _stage(self, original, index, stage, *args, **kwargs):
        if self.active is None or self.active[0] != index:
            raise ValueError("prefetch stage lacks its matching layer invocation")
        target = self.active[1]
        if stage == "after_prefetch":
            lease = args[0] if args and isinstance(args[0], dict) else target["prefetch_lease"]
            self._prefetch_policy(index, target, lease)
        if stage == "after_recall":
            indices = args[0]
            if indices.shape != target["indices"].shape or indices.dtype != torch.int32:
                raise ValueError("exact selection differs from the observed shape")
            target["indices"].copy_(indices)
        result = original(*args, **kwargs)
        self._snapshot(index, target, stage)
        if stage == "after_recall":
            target["complete"] = True
            # Captured memcpy nodes retain their source storages in the graph;
            # the observer need not keep the Python lease after recording them.
            target.pop("prefetch_lease", None)
            target.pop("prepared_token", None)
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
            metadata = {**self.metadata[index], **target["policy_metadata"]}
            stages = {}
            for name, stage in target["stages"].items():
                copied = {key: value.detach().cpu().clone() for key, value in stage.items()}
                mapping = copied.pop("host_to_device")
                copied["logical_to_slot"] = mapping[metadata["global_ids"]].clone()
                if mapping.numel() != metadata["host_arena_tokens"]:
                    raise ValueError("host arena capacity changed during stage observation")
                # The auditor separately proves global_ids cover exactly the
                # valid prefix. Retain every remaining map entry, including
                # page padding, so compact reread can reject phantom residency.
                copied["padding_to_slot"] = mapping[self.history + self.append :].clone()
                stages[name] = copied
            extra = {}
            if metadata.get("prefetch_policy") == OFFICIAL_POLICY:
                official = {
                    name: value.detach().cpu().clone()
                    for name, value in target["official_staging"].items()
                }
                count = min(int(stages["after_prefetch"]["prefetch_counter"][0]), 64)
                host_ids = official["host_ids"][:count].long()
                host = self.model.blocks[index].cache.host
                if not bool(((host_ids >= 0) & (host_ids < len(host))).all()):
                    raise ValueError("official staged IDs exceed host storage")
                official["records"] = official["records"][:count].clone()
                official["expected_records"] = host[host_ids].clone()
                extra["official_staging"] = official
            hint_index = metadata.get("hint_index", 0)
            layers.append(
                {
                    **metadata,
                    "initial_hint": target["initial_hint"][hint_index : hint_index + 1]
                    .detach()
                    .cpu()
                    .clone(),
                    "scores": target["scores"].detach().cpu().clone(),
                    "indices": target["indices"].detach().cpu().clone(),
                    "stages": stages,
                    "metrics": self.model.blocks[index].cache.metrics(),
                    "final_cache_state": final_cache_state["layers"][index],
                    **extra,
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
                "max_prefetch": proof["scope"]["max_prefetch"],
                **{
                    name: proof["scope"][name]
                    for name in (
                        "prefetch_policy",
                        "prepared_max_prefetch",
                        "hint_index",
                        "preparation",
                    )
                    if name in proof["scope"]
                },
                **(
                    {name: proof["scope"][name] for name in ("record_bytes", "topk")}
                    if "preparation" in proof["scope"]
                    else {}
                ),
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
                "Saved eligibility binds runtime score hashes; full scores are not retained. "
                "Official stage KV is compared with host records at runtime; compact evidence "
                "retains identities, not those stage or host KV bytes."
            ),
        }
        (self.output / f"echo_{label}_prefetch_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
        return receipt
