"""CPU proof of the bounded cold-ECHO prefetch, append and recall transitions.

Raw scores establish policy-specific eligibility during independent acceptance.
Compact evidence retains the eligibility bitmap and source tensor identities;
its reread verifies transitions, not the omitted score-to-bitmap computation.
Final KV bytes retain the separate runtime cache-state check's boundary.
"""

from __future__ import annotations

import hashlib

import torch

from evaluation.validation import identity_digest

MISSING = torch.iinfo(torch.int32).max
SCHEMA = "cold-echo-prefetch-transition-v1"
COARSE_POLICY = "native-half-ordered-high-byte-strict-bin-v1"
OFFICIAL_POLICY = "official-q1-predictive-staging-promotion-v1"
OFFICIAL_ELIGIBILITY = "official-fp32-strict-predictive-threshold-v1"
OFFICIAL_SCHEMA = "cold-echo-official-q1-prefetch-transition-v1"
BOUNDED_FREE_PREPARATION = "bounded-free-q1-v1"
STAGES = ("initial", "after_prefetch", "after_append", "after_recall")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_identity(value):
    require(
        isinstance(value, torch.Tensor) and value.device.type == "cpu", "Proof requires CPU tensors"
    )
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(
            value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def tensor(value, shape, dtype, label):
    require(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and tuple(value.shape) == tuple(shape)
        and value.dtype == dtype,
        f"Invalid {label} tensor ABI",
    )
    return value


def equal(actual, expected, label):
    require(torch.equal(actual, expected), f"Invalid {label}")


def ordered_bins(values):
    """Native convert_to_uint8: FP32 -> RN FP16 -> ordered half key's high byte."""
    half_bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    ordered = torch.where((half_bits & 0x8000) != 0, (~half_bits) & 0xFFFF, half_bits | 0x8000)
    return (ordered >> 8).to(torch.uint8)


def _eligibility(scores, hint, history, topk):
    require(
        type(history) is int and history > 0 and type(topk) is int and topk > 0,
        "Invalid eligibility dimensions",
    )
    require(scores.ndim == 2 and scores.shape[0] > 0, "Invalid score shape")
    queries = scores.shape[0]
    tensor(scores, (queries, history + queries), torch.float32, "scores")
    tensor(hint, (1,), torch.float32, "initial hint")
    require(bool(torch.isfinite(hint).all()), "Nonfinite initial hint")
    eligible = torch.zeros(history, dtype=torch.bool)
    thresholds = torch.empty(queries, dtype=torch.int16)
    occurrences = 0
    for query in range(queries):
        end = history + query + 1
        values = scores[query, :end]
        require(bool(torch.isfinite(values).all()), "Nonfinite causal indexer score")
        require(
            bool(torch.isneginf(scores[query, end:]).all()),
            "Noncausal score tail is not negative infinity",
        )
        bins = ordered_bins(values - hint[0])
        # C[b] counts bins >= b. Native chooses b with C[b]>K>=C[b+1].
        # Therefore b is the (K+1)-th largest bin; only strictly larger bins
        # are candidates. If N<=K, threshold -1 includes bin zero as native does.
        threshold = (
            -1 if end <= topk else int(torch.kthvalue(bins.to(torch.int16), end - topk).values)
        )
        thresholds[query] = threshold
        selected = bins[:history].to(torch.int16) > threshold
        eligible |= selected
        occurrences += int(selected.sum())
    metadata = {
        "schema": "native-half-ordered-high-byte-strict-bin-v1",
        "score_identity": tensor_identity(scores),
        "hint_identity": tensor_identity(hint),
        "eligible_identity": tensor_identity(eligible),
        "threshold_bins_identity": tensor_identity(thresholds),
        "candidate_occurrences": occurrences,
        "checked_from_scores": True,
    }
    return eligible, metadata


def eligible_mask(scores, hint, history, topk):
    return _eligibility(scores, hint, history, topk)[0]


def _policy(layer):
    policy = layer.get("prefetch_policy", COARSE_POLICY)
    require(policy in (COARSE_POLICY, OFFICIAL_POLICY), "Unknown prefetch policy")
    return policy


def _raw_eligibility(layer):
    if _policy(layer) == COARSE_POLICY:
        return _eligibility(layer["scores"], layer["initial_hint"], layer["H"], layer["topk"])
    history = layer["H"]
    scores = tensor(layer["scores"], (1, history + 1), torch.float32, "official scores")
    hint = tensor(layer["initial_hint"], (1,), torch.float32, "official initial hint")
    require(bool(torch.isfinite(scores).all()), "Nonfinite causal indexer score")
    require(bool(torch.isfinite(hint).all()), "Nonfinite initial hint")
    eligible = scores[0, :history] > hint[0]
    return eligible, {
        "schema": OFFICIAL_ELIGIBILITY,
        "score_identity": tensor_identity(scores),
        "hint_identity": tensor_identity(hint),
        "eligible_identity": tensor_identity(eligible),
        "candidate_occurrences": int(eligible.sum()),
        "checked_from_scores": True,
    }


def _selection(layer, scores=None):
    history, queries, topk = layer["H"], layer["A"], layer["topk"]
    ids = tensor(
        layer["indices"], (queries, min(topk, history + queries)), torch.int32, "exact indices"
    )
    selected = torch.zeros(history + queries, dtype=torch.bool)
    for query in range(queries):
        end = history + query + 1
        row = ids[query].long()
        count = min(topk, end)
        valid = row[row != -1]
        require(
            valid.numel() == count and bool(((valid >= 0) & (valid < end)).all()),
            "Invalid exact selection causal coverage",
        )
        require(torch.unique(valid).numel() == count, "Exact selection repeats a logical ID")
        selected[valid] = True
        if scores is not None:
            remaining = torch.ones(end, dtype=torch.bool)
            remaining[valid] = False
            if bool(remaining.any()):
                require(
                    float(scores[query, valid].min())
                    >= float(scores[query, :end][remaining].max()),
                    "Selection is not an exact top-k",
                )
    return selected


def preparation_capacity(metadata):
    """Check an observed preparation contract without inferring it from cap64.

    Existing full-FIFO evidence has no descriptor. Bounded-free evidence must
    retain the actual Q1 dispatch and ownership facts observed at preparation.
    """
    history, queries, slots = (metadata.get(name) for name in ("H", "A", "slots"))
    require(
        all(type(value) is int and value > 0 for value in (history, queries, slots))
        and history + queries <= slots,
        "Invalid preparation dimensions",
    )
    full_capacity = min(8192, slots - queries)
    if "preparation" not in metadata:
        return full_capacity
    preparation = metadata["preparation"]
    expected = {
        "kind": BOUNDED_FREE_PREPARATION,
        "query_start": history,
        "query_count": 1,
        "visible_columns": history + 1,
        "initialized_history": history,
        "prepared_limit": 64,
        "single_session": True,
        "exclusive_operation": True,
        "pending_owner_matches": True,
        "persistent_append": True,
    }
    require(
        isinstance(preparation, dict)
        and set(preparation) == {*expected, "requested_max_prefetch"}
        and all(
            type(preparation[name]) is type(value) and preparation[name] == value
            for name, value in expected.items()
        ),
        "Unsupported bounded-free preparation descriptor",
    )
    requested = preparation["requested_max_prefetch"]
    require(
        queries == 1
        and history + 1 >= 32768
        and slots - history >= 64
        and type(requested) is int
        and 64 <= requested <= full_capacity
        and type(metadata.get("record_bytes")) is int
        and metadata["record_bytes"] == 1152
        and type(metadata.get("topk")) is int
        and metadata["topk"] == 2048
        and metadata.get("prefetch_policy") == OFFICIAL_POLICY
        and type(metadata.get("max_prefetch")) is int
        and metadata["max_prefetch"] == 64
        and type(metadata.get("prepared_max_prefetch")) is int
        and metadata["prepared_max_prefetch"] == 64
        and type(metadata.get("hint_index")) is int
        and metadata["hint_index"] == 1,
        "Unsupported bounded-free preparation scope",
    )
    return 64


def _metadata(layer):
    history, queries, slots = layer["H"], layer["A"], layer["slots"]
    require(
        all(
            type(value) is int and value > 0
            for value in (history, queries, slots, layer["record_bytes"], layer["topk"])
        ),
        "Invalid execution dimensions",
    )
    require(history + queries <= slots, "Cold bounded proof requires H+A<=P")
    prepared_cap = preparation_capacity(layer)
    if _policy(layer) == OFFICIAL_POLICY:
        require(
            queries == 1
            and layer["record_bytes"] == 1152
            and layer["max_prefetch"] == 64
            and layer.get("prepared_max_prefetch") == prepared_cap
            and prepared_cap >= 64
            and layer.get("hint_index") == 1,
            "Official Q1 policy requires effective cap64, prepared headroom and hint slot1",
        )
    else:
        require(
            layer["max_prefetch"] == prepared_cap,
            "Prefetch cap differs from actual supported policy",
        )
    global_ids = tensor(layer["global_ids"], (history + queries,), torch.int64, "global IDs")
    equal(
        global_ids.sort().values,
        torch.arange(history + queries),
        "single-session complete valid host prefix",
    )
    _host_geometry(layer)
    hint = tensor(layer["initial_hint"], (1,), torch.float32, "initial hint")
    require(bool(torch.isfinite(hint).all()), "Nonfinite initial hint")
    return history, queries, slots, global_ids


def _host_geometry(layer):
    end = layer["H"] + layer["A"]
    fields = ("host_arena_tokens", "session_host_tokens")
    if not any(name in layer for name in fields):
        # Existing evidence covered an arena with exactly H+A entries.
        return end, end
    require(all(name in layer for name in fields), "Incomplete host capacity metadata")
    arena, session = (layer[name] for name in fields)
    require(
        type(arena) is int
        and type(session) is int
        and arena % 64 == 0
        and session == (end + 63) // 64 * 64
        and arena >= session,
        "Invalid page-aligned host capacities",
    )
    return arena, session


def _stage(stage, history, queries, slots, global_ids, label, *, host_padding_tokens):
    logical = tensor(
        stage["logical_to_slot"], (history + queries,), torch.int32, label + " logical map"
    ).long()
    reverse = tensor(stage["reverse"], (slots + 1,), torch.int64, label + " reverse map")
    free = tensor(stage["free"], (slots + 1,), torch.bool, label + " free bitmap")
    priority = tensor(stage["priority"], (slots + 1,), torch.int64, label + " priority")
    clock = tensor(stage["clock"], (1,), torch.int64, label + " clock")
    counters = tensor(stage["counter_totals"], (8,), torch.int64, label + " counters")
    padding = tensor(
        stage.get("padding_to_slot", torch.empty(0, dtype=torch.int32)),
        (host_padding_tokens,),
        torch.int32,
        label + " host padding map",
    )
    equal(padding, torch.full_like(padding, MISSING), label + " unmapped host padding")
    present = logical != MISSING
    mapped = logical[present]
    require(bool(((mapped > 0) & (mapped <= slots)).all()), f"{label}: invalid physical slot")
    require(
        torch.unique(mapped).numel() == mapped.numel(), f"{label}: two logical IDs share a slot"
    )
    expected_reverse = torch.full_like(reverse, MISSING)
    expected_reverse[mapped] = global_ids[present]
    equal(reverse, expected_reverse, label + " bidirectional ownership")
    expected_free = torch.ones_like(free)
    expected_free[0] = False
    expected_free[mapped] = False
    equal(free, expected_free, label + " free partition")
    require(int(priority[0]) == MISSING, f"{label}: sentinel priority changed")
    equal(priority[free], torch.full_like(priority[free], -1), label + " free priority")
    require(
        bool(((priority[mapped] >= 0) & (priority[mapped] < int(clock[0]))).all()),
        f"{label}: invalid resident priority",
    )
    require(int(clock[0]) >= 0 and bool((counters >= 0).all()), f"{label}: negative clock/counter")
    return {
        "logical": logical,
        "reverse": reverse,
        "free": free,
        "priority": priority,
        "clock": int(clock[0]),
        "counters": counters,
        "present": present,
    }


def _preserve(before, after, label):
    present = before["present"]
    equal(
        after["logical"][present],
        before["logical"][present],
        label + " preserves physical ownership",
    )


def _priority(actual, present, logical_priority, label):
    expected = torch.full_like(actual["priority"], -1)
    expected[0] = MISSING
    expected[actual["logical"][present]] = logical_priority[present]
    equal(actual["priority"], expected, label + " logical/physical priority")


def _metrics(layer, selected, prefetched, recalls, rejected):
    queries, slots = layer["A"], layer["slots"]
    size, moved, missing = int(selected.sum()), int(prefetched.sum()), int(recalls.sum())
    host_arena, session_host = _host_geometry(layer)
    expected = {
        "written_records": queries,
        "transient_written_records": 0,
        "recalled_records": missing,
        "evicted_records": 0,
        "max_working_set": size,
        "capacity_splits": 0,
        "selection_records": size,
        "resident_selection_records": size - missing,
        "host_written_records": queries,
        "prefetched_records": moved,
        "prefetch_capacity_failures": rejected,
        "host_to_device_bytes": (moved + missing) * layer["record_bytes"],
        "device_to_host_bytes": queries * layer["record_bytes"],
        "record_bytes": layer["record_bytes"],
        "device_slots": slots,
        "host_token_capacity": host_arena,
        "session_host_tokens": session_host,
        "padding_slots": 1,
        "candidate_slots": 0,
    }
    for name, value in expected.items():
        require(layer["metrics"].get(name) == value, f"Invalid runtime metric {name}")
    require(layer["metrics"].get("pool_scope") == "shared_per_layer", "Unexpected pool scope")
    return expected


def _final_state(layer, final, logical_priority):
    state = layer["final_cache_state"]
    end = layer["H"] + layer["A"]
    require(state["map_invariants_passed"] is True, "Final runtime KV/map validation missing")
    require(
        state["length"] == state["written"] == state["indexer_visible_end"] == end,
        "Final cache lengths differ",
    )
    require(
        state["resident"] == tensor_identity(final["present"]),
        "Final resident proof is not bound to actual cache",
    )
    require(
        state["logical_priority"] == tensor_identity(logical_priority),
        "Final priority proof is not bound to actual cache",
    )
    require(
        state["clock"] == tensor_identity(layer["stages"]["after_recall"]["clock"]),
        "Final clock proof is not bound to actual cache",
    )
    require(state["free_count"] == int(final["free"].sum()), "Final free count differs")
    require(state["metrics"] == layer["metrics"], "Final runtime metric binding differs")
    for field in ("records", "index_keys", "index_scales", "hint"):
        require(
            isinstance(state[field], dict)
            and set(state[field]) == {"shape", "dtype", "sha256"}
            and len(state[field]["sha256"]) == 64,
            "Final content identity missing",
        )


def _official_staging(layer, prefetch, copied):
    require("official_staging" in layer, "Official staging observation missing")
    stage = layer["official_staging"]
    hosts = tensor(stage["host_ids"], (64,), torch.int32, "official stage host IDs")
    slots = tensor(stage["prepared_slots"], (64,), torch.int32, "official prepared slot order")
    journal = tensor(
        stage["allocation_log"], (layer["slots"] + 1,), torch.int64, "official allocation log"
    )
    equal(slots, torch.arange(1, 65, dtype=torch.int32), "official cold prepared FIFO ranks")
    equal(hosts[copied:], torch.full_like(hosts[copied:], -1), "unused official stage IDs")
    require(
        bool(((hosts[:copied] >= 0) & (hosts[:copied] < layer["H"] + 1)).all())
        and hosts[:copied].unique().numel() == copied,
        "Invalid or duplicate official stage host IDs",
    )
    expected_hosts = prefetch["reverse"][slots[:copied].long()]
    equal(hosts[:copied].long(), expected_hosts, "official stage-to-pool publication")
    expected_log = torch.full_like(journal, MISSING)
    expected_log[slots[:copied].long()] = hosts[:copied].long()
    equal(journal, expected_log, "official allocation journal")
    # Final _stage validation proves every forward/reverse map is legal and no
    # temporary tag escaped. These explicit stage IDs also bind the copy order.
    expected_shape = (copied, layer["record_bytes"] // 2)
    if "records" in stage or "expected_records" in stage:
        records = tensor(stage["records"], expected_shape, torch.bfloat16, "official stage KV")
        expected = tensor(
            stage["expected_records"], expected_shape, torch.bfloat16, "official host KV"
        )
        equal(records.view(torch.int16), expected.view(torch.int16), "official staged KV bytes")
        checked = {
            "checked_from_records": True,
            "records_identity": tensor_identity(records),
            "host_records_identity": tensor_identity(expected),
        }
    else:
        checked = stage.get("record_validation", {})
        require(checked.get("checked_from_records") is True, "Official staged KV check missing")
        record_identity = checked.get("records_identity", {})
        require(
            record_identity == checked.get("host_records_identity")
            and record_identity.get("shape") == list(expected_shape)
            and record_identity.get("dtype") == "torch.bfloat16"
            and len(record_identity.get("sha256", "")) == 64,
            "Invalid saved official staged KV identity",
        )
    return {
        "host_ids_identity": tensor_identity(hosts),
        "prepared_slots_identity": tensor_identity(slots),
        "allocation_log_identity": tensor_identity(journal),
        "record_validation": checked,
        "temporary_tags_cleared": True,
        "copy_order_matches_publication": True,
    }


def validate_layer(layer):
    history, queries, slots, global_ids = _metadata(layer)
    host_arena, session_host = _host_geometry(layer)
    host_padding_tokens = host_arena - history - queries
    scores = layer.get("scores")
    if scores is not None:
        eligible, eligibility = _raw_eligibility(layer)
        _selection(layer, scores)
    else:
        eligible = tensor(layer["eligible_mask"], (history,), torch.bool, "saved eligibility")
        eligibility = layer["eligibility"]
        require(
            eligibility["schema"]
            == (OFFICIAL_ELIGIBILITY if _policy(layer) == OFFICIAL_POLICY else COARSE_POLICY)
            and eligibility["checked_from_scores"] is True,
            "Saved eligibility lacks runtime derivation",
        )
        require(
            eligibility["eligible_identity"] == tensor_identity(eligible),
            "Saved eligible bitmap changed",
        )
        require(
            eligibility["hint_identity"] == tensor_identity(layer["initial_hint"]),
            "Saved initial hint changed",
        )
        require(
            eligibility["score_identity"]["shape"] == [queries, history + queries]
            and eligibility["score_identity"]["dtype"] == "torch.float32",
            "Saved score identity ABI differs",
        )
    selected = _selection(layer)
    if _policy(layer) == COARSE_POLICY:
        require(
            not bool((eligible & ~selected[:history]).any()),
            "Coarse eligibility includes a token outside exact selection",
        )
    require(set(layer["stages"]) == set(STAGES), "Incomplete stage observation")
    stages = {
        name: _stage(
            layer["stages"][name],
            history,
            queries,
            slots,
            global_ids,
            name,
            host_padding_tokens=host_padding_tokens,
        )
        for name in STAGES
    }
    initial, prefetch, append, final = (stages[name] for name in STAGES)
    require(not bool(initial["present"].any()), "Initial HBM is not cold/empty")
    equal(initial["counters"], torch.zeros(8, dtype=torch.int64), "initial zero counters")
    start = initial["clock"]
    require(
        [stages[name]["clock"] for name in STAGES] == [start, start + 1, start + 2, start + 4],
        "Stage clock transitions differ",
    )
    prefetched = prefetch["present"]
    require(not bool(prefetched[history:].any()), "Prefetch loads unwritten candidate records")
    require(
        not bool((prefetched[:history] & ~eligible).any()),
        "Prefetch copied an ineligible logical ID",
    )
    copied = int(prefetched.sum())
    cap = layer["max_prefetch"]
    require(
        copied == min(cap, int(eligible.sum())), "Prefetch did not exhaust eligibility or its cap"
    )
    if copied < cap:
        equal(prefetched[:history], eligible, "uncapped complete eligible set")
    # Native FIFO sort is stable over identical cold ages. Atomic claim rank
    # changes the logical owner ordering, but occupies precisely slots 1..N.
    equal(
        prefetch["logical"][prefetched].sort().values,
        torch.arange(1, copied + 1),
        "cold prefetch slot ranks",
    )
    counters = layer["stages"]["after_prefetch"]
    attempts = int(
        tensor(counters["prefetch_counter"], (1,), torch.uint32, "prefetch attempt counter")[0]
    )
    stats = tensor(counters["prefetch_stats"], (3,), torch.int64, "prefetch statistics")
    rejected = attempts - copied
    occurrences = eligibility["candidate_occurrences"]
    require(
        type(occurrences) is int and int(eligible.sum()) <= occurrences <= history * queries,
        "Invalid candidate occurrence count",
    )
    require(
        0 <= rejected and attempts <= occurrences, "Prefetch reservation accounting is impossible"
    )
    require(copied == cap or rejected == 0, "Rejected reservations below cap")
    equal(
        stats,
        torch.tensor([copied, 0, rejected], dtype=torch.int64),
        "prefetch copied/evicted/rejected accounting",
    )
    pre_counters = torch.tensor([copied, 0, rejected, 0, 0, 0, 0, 0], dtype=torch.int64)
    equal(prefetch["counters"], pre_counters, "prefetch counter accumulation")
    pre_priority = torch.full((history + queries,), start, dtype=torch.int64)
    _priority(prefetch, prefetched, pre_priority, "prefetch")
    candidates = torch.arange(history + queries) >= history
    expected_append = prefetched | candidates
    equal(append["present"], expected_append, "append adds exactly candidate records")
    _preserve(prefetch, append, "append")
    free_slots = prefetch["free"].nonzero().flatten()[:queries]
    equal(append["logical"][history:], free_slots, "candidate free-slot append order")
    append_priority = pre_priority.clone()
    append_priority[candidates] = start + 1
    _priority(append, expected_append, append_priority, "append")
    equal(append["counters"], pre_counters, "append leaves zero eviction and transport counters")
    recalls = selected & ~expected_append
    expected_final = expected_append | selected
    equal(final["present"], expected_final, "recall adds exactly residual exact-selection misses")
    _preserve(append, final, "recall")
    missing_slots = final["logical"][recalls]
    require(
        bool(append["free"][missing_slots].all()),
        "Recall evicts an occupied slot despite free capacity",
    )
    priorities = append_priority.clone()
    priorities[selected & expected_append] = start + 2
    priorities[recalls] = start + 3
    _priority(final, expected_final, priorities, "recall")
    size, missing = int(selected.sum()), int(recalls.sum())
    final_counters = torch.tensor(
        [copied, 0, rejected, size, size - missing, size, 0, missing], dtype=torch.int64
    )
    equal(final["counters"], final_counters, "exact selection/recall counters")
    metrics = _metrics(layer, selected, prefetched, recalls, rejected)
    logical_priority = torch.zeros(history + queries, dtype=torch.int64)
    logical_priority[expected_final] = priorities[expected_final]
    _final_state(layer, final, logical_priority)
    official = _policy(layer) == OFFICIAL_POLICY
    stage_proof = _official_staging(layer, prefetch, copied) if official else None
    return {
        "layer": layer["layer"],
        "passed": True,
        "indices_identity": tensor_identity(layer["indices"]),
        "eligibility": eligibility,
        "initial_clock": start,
        "eligible_records": int(eligible.sum()),
        "prefetched_records": copied,
        "reservation_attempts": attempts,
        "rejected_reservations": rejected,
        "exact_selection_records": size,
        "recalled_records": missing,
        "prefetched_set_identity": tensor_identity(prefetched),
        "recalled_set_identity": tensor_identity(recalls),
        "final_cache_state_identity": identity_digest(layer["final_cache_state"]),
        "metrics": metrics,
        "stage_maps_free_priorities_clocks_verified": True,
        "physical_owners_preserved_until_final": True,
        "no_eviction": True,
        **(
            {
                "prefetch_policy": OFFICIAL_POLICY,
                "effective_max_prefetch": cap,
                "prepared_max_prefetch": layer["prepared_max_prefetch"],
                **({"preparation": dict(layer["preparation"])} if "preparation" in layer else {}),
                "hint_index": 1,
                "prefetch_false_positive_records": int((prefetched & ~selected).sum()),
                "official_staging": stage_proof,
            }
            if official
            else {}
        ),
        **(
            {
                "host_padding": {
                    "host_arena_tokens": host_arena,
                    "session_host_tokens": session_host,
                    "padding_tokens": host_padding_tokens,
                    "all_stages_unmapped": True,
                    "stage_mapping_identities": {
                        name: tensor_identity(layer["stages"][name]["padding_to_slot"])
                        for name in STAGES
                    },
                }
            }
            if host_padding_tokens
            else {}
        ),
        "kv_boundary": "Final cache_state checks actual resident KV against host at runtime. All observed physical owners persist through the final check; per-stage KV bytes are not independently retained.",
    }


def validate_execution(evidence):
    require(
        evidence["schema_version"] == 1
        and evidence["method"] == "echo"
        and evidence["residency"] == "cold"
        and evidence["single_session"] is True,
        "Unsupported prefetch proof scope",
    )
    layers = evidence["layers"]
    require(
        len(layers) == 3 and [layer["layer"] for layer in layers] == [0, 1, 2],
        "Expected exactly L0-L2 evidence",
    )
    geometry = {
        (
            layer["H"],
            layer["A"],
            layer["slots"],
            layer["record_bytes"],
            layer["topk"],
            layer["max_prefetch"],
            _policy(layer),
            layer.get("prepared_max_prefetch"),
            layer.get("hint_index"),
            identity_digest(layer["preparation"]) if "preparation" in layer else None,
            *_host_geometry(layer),
        )
        for layer in layers
    }
    require(len(geometry) == 1, "Layer proof geometries differ")
    first = layers[0]
    return {
        "schema": OFFICIAL_SCHEMA if _policy(first) == OFFICIAL_POLICY else SCHEMA,
        "passed": True,
        "scope": {
            "method": "echo",
            "residency": "cold",
            "single_session": True,
            "num_layers": 3,
            **{
                name: first[name]
                for name in ("H", "A", "slots", "record_bytes", "topk", "max_prefetch")
            },
            **(
                {
                    "prefetch_policy": OFFICIAL_POLICY,
                    "prepared_max_prefetch": first["prepared_max_prefetch"],
                    **(
                        {"preparation": dict(first["preparation"])}
                        if "preparation" in first
                        else {}
                    ),
                    "hint_index": 1,
                }
                if _policy(first) == OFFICIAL_POLICY
                else {}
            ),
        },
        "layers": [validate_layer(layer) for layer in layers],
        "eligibility_boundary": (
            "Official strict FP32 predictive-threshold eligibility and exact top-k validity are checked against full scores during runtime acceptance. Predictive false positives remain resident and count as real H2D. Staged KV is compared against host records at runtime; compact reread checks saved identities and transitions, not omitted score or KV bytes. Matching executions separately require identical scores, initial hints and exact index tensors."
            if _policy(first) == OFFICIAL_POLICY
            else "Native coarse-bin eligibility and exact top-k validity are checked against full scores during runtime acceptance. Compact reread checks saved eligibility identities and stage transitions; it cannot re-evaluate omitted scores. Matching executions separately require identical scores, initial hints and exact index tensors."
        ),
    }


def compact_evidence(evidence):
    layers = []
    for layer in evidence["layers"]:
        if "scores" not in layer:
            layers.append(dict(layer))
            continue
        eligible, metadata = _raw_eligibility(layer)
        _selection(layer, layer["scores"])
        extra = {}
        if _policy(layer) == OFFICIAL_POLICY:
            stage_proof = validate_layer(layer)["official_staging"]
            extra["official_staging"] = {
                **{
                    key: value
                    for key, value in layer["official_staging"].items()
                    if key not in ("records", "expected_records")
                },
                "record_validation": stage_proof["record_validation"],
            }
        layers.append(
            {
                **{key: value for key, value in layer.items() if key != "scores"},
                "eligible_mask": eligible,
                "eligibility": metadata,
                **extra,
            }
        )
    return {**evidence, "layers": layers}
