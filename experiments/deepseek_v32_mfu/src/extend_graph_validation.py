"""Independent full-extend graph checks, outside benchmark samples."""

from __future__ import annotations

import hashlib

import torch

from cache.sparse_token_cache import MISSING
from evaluation.validation import identity_digest


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def cache_state(model):
    """Audit maps and compare logical records without requiring stable tied slots."""
    model.synchronize()
    layers = []
    for block in model.blocks:
        cache, runner = block.cache, block.attention
        if cache.length != model.length or cache.written != model.length:
            raise AssertionError("full graph did not commit every layer")
        if cache._step_end is not None:
            raise AssertionError("full graph left an active cache transaction")
        row = {
            "length": cache.length,
            "written": cache.written,
            "indexer_visible_end": cache.indexer_visible_end,
            "records": tensor_identity(cache.host_records()),
            "index_keys": tensor_identity(runner.index_keys[: model.length]),
            "index_scales": tensor_identity(runner.index_scales[: model.length]),
            "hint": tensor_identity(runner.offset),
            "metrics": cache.metrics(),
        }
        if cache.offload:
            pool_layer = cache._pool.layers[cache.layer_id]
            ids = cache.logical_to_global(
                torch.arange(model.length, dtype=torch.int64, device=cache.device)
            ).cpu()
            forward = cache.host_to_device.detach().cpu()
            reverse = cache.device_to_host.detach().cpu()
            free = pool_layer.free.detach().cpu()
            priority = pool_layer.priority.detach().cpu()
            slots = forward[ids].long()
            resident = slots != MISSING
            used = (reverse != MISSING).nonzero().flatten()
            used = used[used != 0]
            if used.numel() and (
                not torch.equal(forward[reverse[used].long()].long(), used)
                or bool(free[used].any())
            ):
                raise AssertionError("full graph left inconsistent reverse/free maps")
            if resident.any():
                selected = slots[resident]
                if not torch.equal(reverse[selected].long(), ids[resident]):
                    raise AssertionError("full graph left inconsistent forward maps")
                records = cache.records.detach().cpu()[selected]
                host = cache.host.detach().cpu()[ids[resident]]
                if not torch.equal(records, host):
                    raise AssertionError("resident KV differs from logical host backing")
            if bool(free[0]) or int(free[1:].sum()) + used.numel() != cache.slots:
                raise AssertionError("free bitmap does not partition the slot pool")
            logical_priority = torch.zeros(model.length, dtype=priority.dtype)
            logical_priority[resident] = priority[slots[resident]]
            row.update(
                resident=tensor_identity(resident),
                logical_priority=tensor_identity(logical_priority),
                free_count=int(free.sum()),
                clock=tensor_identity(pool_layer.clock_tensor),
                map_invariants_passed=True,
            )
        layers.append(row)
    return {"length": model.length, "layers": layers}


def compare_cache_state(actual, expected, *, actual_prefetch=None, expected_prefetch=None):
    from experiments.deepseek_v32_mfu.src.prefetch_transition_audit import (
        COARSE_POLICY,
        OFFICIAL_ELIGIBILITY,
        OFFICIAL_POLICY,
        OFFICIAL_SCHEMA,
        SCHEMA,
        preparation_capacity,
    )

    schedule_fields = set()
    if actual_prefetch is not None or expected_prefetch is not None:
        for state, proof in ((actual, actual_prefetch), (expected, expected_prefetch)):
            if (
                proof is None
                or proof.get("schema") != "cold-echo-stage-acceptance-v1"
                or proof.get("passed") is not True
                or proof.get("final_cache_state_sha256") != identity_digest(state)
                or proof.get("proof", {}).get("passed") is not True
            ):
                raise AssertionError("prefetch transition proof does not bind this cache state")
            scope = proof.get("scope", {})
            if (
                scope.get("method") != "echo"
                or scope.get("residency") != "cold"
                or scope.get("single_session") is not True
                or scope.get("num_layers") != 3
                or len(state["layers"]) != 3
                or any(type(scope.get(key)) is not int for key in ("H", "A", "slots"))
                or min(scope["H"], scope["A"]) < 1
                or scope["H"] + scope["A"] > scope["slots"]
                or scope["H"] + scope["A"] != state["length"]
            ):
                raise AssertionError("prefetch transition proof has unsupported execution scope")
            policy = scope.get("prefetch_policy", COARSE_POLICY)
            try:
                prepared_cap = preparation_capacity(scope)
            except ValueError as error:
                raise AssertionError(
                    "prefetch transition proof has unsupported preparation scope"
                ) from error
            if policy == OFFICIAL_POLICY:
                valid_policy = (
                    scope["A"] == 1
                    and scope.get("max_prefetch") == 64
                    and scope.get("prepared_max_prefetch") == prepared_cap
                    and prepared_cap >= 64
                    and scope.get("hint_index") == 1
                )
            else:
                valid_policy = policy == COARSE_POLICY and scope.get("max_prefetch") == prepared_cap
            if not valid_policy:
                raise AssertionError("prefetch transition proof has unsupported policy scope")
            audit = proof["proof"]
            layers = audit.get("layers", [])
            if (
                audit.get("schema") != (OFFICIAL_SCHEMA if policy == OFFICIAL_POLICY else SCHEMA)
                or any(audit.get("scope", {}).get(key) != value for key, value in scope.items())
                or len(layers) != 3
                or any(
                    len(proof.get(field, [])) != 3
                    for field in ("indices", "scores", "initial_hints")
                )
            ):
                raise AssertionError("prefetch acceptance lacks its scoped transition audit")
            for index, layer in enumerate(layers):
                if (
                    layer.get("layer") != index
                    or any(
                        layer.get(flag) is not True
                        for flag in (
                            "passed",
                            "stage_maps_free_priorities_clocks_verified",
                            "physical_owners_preserved_until_final",
                            "no_eviction",
                        )
                    )
                    or layer.get("final_cache_state_identity")
                    != identity_digest(state["layers"][index])
                    or layer.get("indices_identity") != proof["indices"][index]
                    or layer.get("eligibility", {}).get("score_identity") != proof["scores"][index]
                    or layer.get("eligibility", {}).get("hint_identity")
                    != proof["initial_hints"][index]
                ):
                    raise AssertionError("prefetch layer audit is incomplete or mismatched")
                if policy == OFFICIAL_POLICY and (
                    layer.get("prefetch_policy") != OFFICIAL_POLICY
                    or layer.get("effective_max_prefetch") != 64
                    or layer.get("prepared_max_prefetch") != prepared_cap
                    or layer.get("preparation") != scope.get("preparation")
                    or layer.get("hint_index") != 1
                    or layer.get("eligibility", {}).get("schema") != OFFICIAL_ELIGIBILITY
                    or layer.get("official_staging", {}).get("temporary_tags_cleared") is not True
                    or layer.get("official_staging", {}).get("copy_order_matches_publication")
                    is not True
                    or layer.get("official_staging", {})
                    .get("record_validation", {})
                    .get("checked_from_records")
                    is not True
                ):
                    raise AssertionError("official prefetch stage proof is incomplete")
        if actual_prefetch["scope"] != expected_prefetch["scope"]:
            raise AssertionError("prefetch transition proofs cover different scopes")
        for field, description in (
            ("indices", "exact top-k indices"),
            ("scores", "exact indexer scores"),
            ("initial_hints", "initial prefetch hints"),
        ):
            if len(actual_prefetch.get(field, [])) != 3 or actual_prefetch.get(
                field
            ) != expected_prefetch.get(field):
                raise AssertionError(f"full graph changed {description}")
        # Each accepted stage proof independently recomputes these fields from
        # the actual capped prefetch/recall execution. Other cache metadata,
        # numerical identities and clocks remain strict. Predictive false
        # positives can change final residency only under the official proof,
        # which independently derives every resident token and the free count.
        schedule_fields = {
            "prefetched_records",
            "prefetch_capacity_failures",
            "recalled_records",
            "resident_selection_records",
            "host_to_device_bytes",
        }
        schedule_state_fields = {"logical_priority"}
        if actual_prefetch["scope"].get("prefetch_policy") == OFFICIAL_POLICY:
            schedule_state_fields.update(("resident", "free_count"))
        actual = {
            **actual,
            "layers": [
                {
                    key: {k: v for k, v in value.items() if k not in schedule_fields}
                    if key == "metrics"
                    else value
                    for key, value in layer.items()
                    if key not in schedule_state_fields
                }
                for layer in actual["layers"]
            ],
        }
        expected = {
            **expected,
            "layers": [
                {
                    key: {k: v for k, v in value.items() if k not in schedule_fields}
                    if key == "metrics"
                    else value
                    for key, value in layer.items()
                    if key not in schedule_state_fields
                }
                for layer in expected["layers"]
            ],
        }
    if actual != expected:
        changed = [
            f"layer_{layer}.{key}"
            for layer, (left, right) in enumerate(
                zip(actual["layers"], expected["layers"], strict=True)
            )
            for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)
        ]
        raise AssertionError(f"full graph cache state differs: {changed}")
    return {
        "equal": True,
        "logical_slot_order": True,
        "layers": len(actual["layers"]),
        **(
            {
                "bounded_prefetch_transitions_validated": True,
                "schedule_dependent_metrics": sorted(schedule_fields),
                "exact_topk_equal": True,
                **(
                    {"schedule_dependent_state_fields": sorted(schedule_state_fields)}
                    if "resident" in schedule_state_fields
                    else {}
                ),
            }
            if schedule_fields
            else {}
        ),
    }
