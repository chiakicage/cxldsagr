"""Cold ECHO legality is checked by token sets and transitions, not count equality."""

import struct
from copy import deepcopy

import pytest
import torch

from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as proof


def bounded_preparation(history, *, requested=8192):
    return {
        "kind": proof.BOUNDED_FREE_PREPARATION,
        "query_start": history,
        "query_count": 1,
        "visible_columns": history + 1,
        "initialized_history": history,
        "prepared_limit": 64,
        "requested_max_prefetch": requested,
        "single_session": True,
        "exclusive_operation": True,
        "pending_owner_matches": True,
        "persistent_append": True,
    }


def layer_fixture(
    *,
    saturated=False,
    different_winners=False,
    padded=False,
    decode=False,
    official=False,
    bounded=False,
    legacy=False,
    partial_selection=False,
):
    official, padded = official or bounded or legacy, padded or bounded
    history, queries, topk = (8448, 6, 2048) if saturated else (64, 2, 2048)
    if decode:
        history, queries, topk = 65536, 1, 2048
    if official:
        history, queries, topk = 65536 if bounded else 4096, 1, 2048
    end, start = history + queries, 64
    capacity = (end + 63) // 64 * 64 if padded else end
    scores = torch.full((queries, end), -torch.inf)
    for query in range(queries):
        scores[query, : history + query + 1] = 0
        if official:
            scores[query, : (topk if saturated else 17)] = 3
            if saturated:
                scores[query, topk : topk + 128] = 2
            scores[query, history] = 4
        elif saturated:
            scores[query, query * 1400 : (query + 1) * 1400] = 1
        elif decode:
            scores[query, :1024] = 1
    values, indices = torch.topk(scores, min(topk, end), dim=-1)
    indices = torch.where(torch.isfinite(values), indices, -1).int()
    selected = torch.zeros(end, dtype=torch.bool)
    selected[indices[indices >= 0].long()] = True
    eligible = torch.ones(history, dtype=torch.bool)
    if official:
        eligible = scores[0, :history] > 1
    elif saturated:
        eligible[8400:] = False
    elif decode:
        eligible[1024:] = False
    prepared_cap = min(8192, capacity - queries)
    cap = 64 if official else prepared_cap
    winner_ids = eligible.nonzero().flatten()
    winner_ids = winner_ids[-cap:] if different_winners else winner_ids[:cap]
    if partial_selection:
        eligible_selected = (eligible & selected[:history]).nonzero().flatten()
        eligible_unselected = (eligible & ~selected[:history]).nonzero().flatten()
        winner_ids = torch.cat((eligible_selected[:32], eligible_unselected[:32]))
    staged_winner_ids = winner_ids.clone()
    if official and not legacy:
        winner_ids = winner_ids[selected[winner_ids]]
    prefetched = torch.zeros(end, dtype=torch.bool)
    prefetched[winner_ids] = True
    candidates = torch.arange(end) >= history
    appended = prefetched | candidates
    recalls = selected & ~appended
    copied, published = len(staged_winner_ids), len(winner_ids)
    missing, size = int(recalls.sum()), int(selected.sum())
    rejected = 3 if saturated else 0
    global_ids = torch.arange(end).flip(0)
    mapping = torch.full((end,), proof.MISSING, dtype=torch.int32)

    def stage(present, priorities, clock, counters):
        reverse = torch.full((capacity + 1,), proof.MISSING, dtype=torch.int64)
        free = torch.ones(capacity + 1, dtype=torch.bool)
        free[0] = False
        priority = torch.full((capacity + 1,), -1, dtype=torch.int64)
        priority[0] = proof.MISSING
        slots = mapping[present].long()
        reverse[slots] = global_ids[present]
        free[slots] = False
        priority[slots] = priorities[present]
        return {
            "logical_to_slot": mapping.clone(),
            "reverse": reverse,
            "free": free,
            "priority": priority,
            "clock": torch.tensor([clock]),
            "counter_totals": torch.tensor(counters),
            **(
                {"padding_to_slot": torch.full((capacity - end,), proof.MISSING, dtype=torch.int32)}
                if padded
                else {}
            ),
        }

    priorities = torch.zeros(end, dtype=torch.int64)
    initial = stage(torch.zeros(end, dtype=torch.bool), priorities, start, [0] * 8)
    mapping[winner_ids] = torch.arange(1, published + 1, dtype=torch.int32)
    priorities[prefetched] = start
    pre_counters = [copied, 0, rejected, 0, 0, 0, 0, 0]
    prefetch = stage(prefetched, priorities, start + 1, pre_counters)
    prefetch.update(
        prefetch_counter=torch.tensor([copied + rejected], dtype=torch.uint32),
        prefetch_stats=torch.tensor([copied, 0, rejected]),
    )
    mapping[history:] = torch.arange(published + 1, published + queries + 1, dtype=torch.int32)
    priorities[candidates] = start + 1
    append = stage(appended, priorities, start + 2, pre_counters)
    free_slots = append["free"].nonzero().flatten()[:missing].flip(0)
    mapping[recalls] = free_slots.int()
    priorities[selected & appended] = start + 2
    priorities[recalls] = start + 3
    final_present = appended | selected
    final_counters = [copied, 0, rejected, size, size - missing, size, 0, missing]
    final = stage(final_present, priorities, start + 4, final_counters)
    metrics = {
        "written_records": queries,
        "transient_written_records": 0,
        "recalled_records": missing,
        "evicted_records": 0,
        "max_working_set": size,
        "capacity_splits": 0,
        "selection_records": size,
        "resident_selection_records": size - missing,
        "host_written_records": queries,
        "prefetched_records": copied,
        "prefetch_capacity_failures": rejected,
        "host_to_device_bytes": (copied + missing) * 1152,
        "device_to_host_bytes": queries * 1152,
        "record_bytes": 1152,
        "device_slots": capacity,
        "host_token_capacity": capacity,
        "session_host_tokens": capacity,
        "padding_slots": 1,
        "candidate_slots": 0,
        "pool_scope": "shared_per_layer",
    }
    logical_priorities = torch.zeros(end, dtype=torch.int64)
    logical_priorities[final_present] = priorities[final_present]
    state = {
        "map_invariants_passed": True,
        "length": end,
        "written": end,
        "indexer_visible_end": end,
        "resident": proof.tensor_identity(final_present),
        "logical_priority": proof.tensor_identity(logical_priorities),
        "clock": proof.tensor_identity(final["clock"]),
        "free_count": int(final["free"].sum()),
        "metrics": metrics,
        **{
            name: {"shape": [end, 576], "dtype": "torch.bfloat16", "sha256": "a" * 64}
            for name in ("records", "index_keys", "index_scales", "hint")
        },
    }
    result = {
        "layer": 0,
        "H": history,
        "A": queries,
        "slots": capacity,
        **({"host_arena_tokens": capacity, "session_host_tokens": capacity} if padded else {}),
        "record_bytes": 1152,
        "topk": topk,
        "max_prefetch": cap,
        "global_ids": global_ids,
        "initial_hint": torch.ones(1) if official else torch.zeros(1),
        "scores": scores,
        "indices": indices,
        "stages": {
            "initial": initial,
            "after_prefetch": prefetch,
            "after_append": append,
            "after_recall": final,
        },
        "metrics": metrics,
        "final_cache_state": state,
    }
    if official:
        host_ids = torch.full((64,), -1, dtype=torch.int32)
        host_ids[:copied] = global_ids[staged_winner_ids].int()
        selected_ranks = (
            torch.ones(copied, dtype=torch.bool) if legacy else selected[staged_winner_ids]
        )
        promotion_slots = torch.full((64,), proof.MISSING, dtype=torch.int32)
        promotion_slots[:copied][selected_ranks] = torch.arange(1, published + 1, dtype=torch.int32)
        journal = torch.full((capacity + 1,), proof.MISSING, dtype=torch.int64)
        journal[1 : published + 1] = host_ids[:copied][selected_ranks].long()
        raw = deepcopy(initial)
        raw["logical_to_slot"][staged_winner_ids] = (
            capacity + 1 + torch.arange(copied, dtype=torch.int32)
        )
        raw.update(
            prefetch_counter=torch.tensor([copied + rejected], dtype=torch.uint32),
            prefetch_stats=torch.zeros(3, dtype=torch.int64),
        )
        promotion = deepcopy(raw)
        promotion["logical_to_slot"][winner_ids] = torch.arange(1, published + 1, dtype=torch.int32)
        promotion["reverse"] = prefetch["reverse"].clone()
        promotion["prefetch_stats"] = torch.tensor([copied, 0, rejected])
        if not legacy:
            result["stages"].update(after_staging=raw, after_promotion=promotion)
        records = host_ids[:copied, None].to(torch.bfloat16).expand(copied, 576).clone()
        result.update(
            prefetch_policy=proof.LEGACY_OFFICIAL_POLICY if legacy else proof.OFFICIAL_POLICY,
            prepared_max_prefetch=prepared_cap,
            hint_index=1,
            official_staging={
                "host_ids": host_ids,
                "prepared_slots": torch.arange(1, 65, dtype=torch.int32),
                "promotion_slots": promotion_slots,
                "page_table": torch.cat(
                    (global_ids[:history].int(), torch.zeros(1, dtype=torch.int32))
                ),
                "promotion_indices": indices.clone(),
                "raw_allocation_log": torch.full_like(journal, proof.MISSING),
                "allocation_log": journal,
                "records": records,
                "expected_records": records.clone(),
            },
        )
        if legacy:
            for field in (
                "promotion_slots",
                "page_table",
                "promotion_indices",
                "raw_allocation_log",
            ):
                del result["official_staging"][field]
        if bounded:
            result.update(prepared_max_prefetch=64, preparation=bounded_preparation(history))
    return result


def execution_fixture(**kwargs):
    layer = layer_fixture(**kwargs)
    return {
        "schema_version": 1,
        "method": "echo",
        "residency": "cold",
        "single_session": True,
        "layers": [{**deepcopy(layer), "layer": index} for index in range(3)],
    }


def test_ordered_bins_match_independent_ieee_half_encoding():
    values = [-65504, -100, -1, -0.0, 0.0, 2**-24, 0.9998, 1.0004, 100, 65504]
    expected = []
    for value in values:
        bits = int.from_bytes(struct.pack("<e", value), "little")
        ordered = ((~bits) & 0xFFFF) if bits & 0x8000 else bits | 0x8000
        expected.append(ordered >> 8)
    assert proof.ordered_bins(torch.tensor(values)).tolist() == expected


def test_eligibility_excludes_entire_boundary_bin():
    scores = torch.tensor([[8.0, 4.0, 2.0, 2.0, 1.0]])
    # Top-3 raw scores include one 2.0, but the coarse bucket containing the
    # boundary tie is excluded in full. This differs from ordinary top-k IDs.
    assert proof.eligible_mask(scores, torch.zeros(1), 4, 3).tolist() == [True, True, False, False]


def test_raw_and_compact_proofs_identical_and_weights_only_reload(tmp_path):
    raw = execution_fixture()
    direct = proof.validate_execution(raw)
    compact = proof.compact_evidence(raw)
    assert all("scores" not in layer for layer in compact["layers"])
    path = tmp_path / "compact.pt"
    torch.save(compact, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert proof.validate_execution(reloaded) == direct


def test_h64k_single_decode_preserves_padding_proof_after_compact_reload(tmp_path):
    raw = execution_fixture(padded=True, decode=True)
    direct = proof.validate_execution(raw)
    for layer in direct["layers"]:
        assert layer["host_padding"]["padding_tokens"] == 63
        assert layer["host_padding"]["all_stages_unmapped"]
        assert layer["metrics"]["session_host_tokens"] == 65600
        assert layer["metrics"]["host_token_capacity"] == 65600
        assert layer["metrics"]["written_records"] == 1
    path = tmp_path / "decode_compact.pt"
    torch.save(proof.compact_evidence(raw), path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert proof.validate_execution(reloaded) == direct


@pytest.mark.parametrize("stage", proof.STAGES)
@pytest.mark.parametrize("index", [0, -1])
def test_any_stage_padding_residency_is_rejected_in_raw_and_compact_evidence(stage, index):
    raw = execution_fixture(padded=True)
    compact = proof.compact_evidence(raw)
    for evidence in (raw, compact):
        evidence["layers"][0]["stages"][stage]["padding_to_slot"][index] = 1
        with pytest.raises(ValueError, match="unmapped host padding"):
            proof.validate_execution(evidence)


@pytest.mark.parametrize(
    "corruption", ["missing", "shape", "host_capacity", "session_capacity", "reverse"]
)
def test_incomplete_or_mismatched_padding_evidence_is_rejected(corruption):
    layer = layer_fixture(padded=True)
    if corruption == "missing":
        del layer["stages"]["after_append"]["padding_to_slot"]
    elif corruption == "shape":
        layer["stages"]["initial"]["padding_to_slot"] = torch.empty(0, dtype=torch.int32)
    elif corruption == "host_capacity":
        layer["host_arena_tokens"] -= 1
    elif corruption == "session_capacity":
        layer["session_host_tokens"] = layer["H"] + layer["A"]
    elif corruption == "reverse":
        layer["stages"]["after_recall"]["reverse"][-1] = layer["H"] + layer["A"]
    with pytest.raises(ValueError):
        proof.validate_layer(layer)


def test_two_saturated_schedules_pass_with_distinct_logical_priorities():
    first = layer_fixture(saturated=True)
    second = layer_fixture(saturated=True, different_winners=True)
    first_proof, second_proof = proof.validate_layer(first), proof.validate_layer(second)
    assert first_proof["prefetched_records"] == second_proof["prefetched_records"] == 8192
    assert first_proof["reservation_attempts"] == 8195
    assert first_proof["prefetched_set_identity"] != second_proof["prefetched_set_identity"]
    assert first["final_cache_state"]["resident"] == second["final_cache_state"]["resident"]
    assert (
        first["final_cache_state"]["logical_priority"]
        != second["final_cache_state"]["logical_priority"]
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "reverse",
        "free",
        "priority",
        "clock",
        "counter",
        "initial_resident",
        "recall_owner",
        "final_binding",
        "metrics",
    ],
)
def test_stage_and_final_corruption_is_rejected(corruption):
    layer = layer_fixture(saturated=True)
    before, after = layer["stages"]["after_append"], layer["stages"]["after_recall"]
    if corruption == "reverse":
        after["reverse"][1] = proof.MISSING
    elif corruption == "free":
        after["free"][1] = True
    elif corruption == "priority":
        after["priority"][1] += 1
    elif corruption == "clock":
        after["clock"][0] += 1
    elif corruption == "counter":
        after["counter_totals"][7] += 1
    elif corruption == "initial_resident":
        layer["stages"]["initial"] = deepcopy(layer["stages"]["after_prefetch"])
    elif corruption == "recall_owner":
        # Swap two already-resident owners and repair inverse metadata. Local
        # maps are still valid; the no-eviction transition must reject it.
        ids = (before["logical_to_slot"] != proof.MISSING).nonzero().flatten()[:2]
        slots = after["logical_to_slot"][ids].long()
        after["logical_to_slot"][ids] = slots.flip(0).int()
        after["reverse"][slots] = layer["global_ids"][ids.flip(0)]
    elif corruption == "final_binding":
        layer["final_cache_state"]["logical_priority"]["sha256"] = "b" * 64
    elif corruption == "metrics":
        layer["metrics"]["host_to_device_bytes"] += 1152
    with pytest.raises(ValueError):
        proof.validate_layer(layer)


def test_ineligible_prefetch_rejected_even_when_all_counts_unchanged():
    layer = layer_fixture(saturated=True)
    stage = layer["stages"]["after_prefetch"]
    # Replace copied token 0 with token 8447, which is outside all strict high
    # bins. Maintain the same number of copies and a valid inverse map.
    slot = int(stage["logical_to_slot"][0])
    stage["logical_to_slot"][0] = proof.MISSING
    stage["logical_to_slot"][8447] = slot
    stage["reverse"][slot] = layer["global_ids"][8447]
    with pytest.raises(ValueError, match="ineligible"):
        proof.validate_layer(layer)


def test_compact_eligibility_hash_and_raw_selection_corruption_rejected():
    raw = execution_fixture()
    compact = proof.compact_evidence(raw)
    compact["layers"][0]["eligible_mask"][0] = False
    with pytest.raises(ValueError, match="eligible bitmap changed"):
        proof.validate_execution(compact)
    layer = layer_fixture(saturated=True)
    layer["indices"][0, 0] = layer["indices"][0, 1]
    with pytest.raises(ValueError, match="repeats"):
        proof.validate_layer(layer)


def test_wrong_scope_rejected():
    evidence = execution_fixture()
    evidence["residency"] = "warm"
    with pytest.raises(ValueError, match="scope"):
        proof.validate_execution(evidence)


@pytest.mark.parametrize("saturated", [False, True])
def test_official_policy_charges_false_positives_and_promotes_exact_intersection(
    saturated, tmp_path
):
    raw = execution_fixture(official=True, saturated=saturated, different_winners=True, padded=True)
    result = proof.validate_execution(raw)
    assert result["schema"] == proof.OFFICIAL_SCHEMA
    assert result["scope"]["max_prefetch"] == 64
    assert result["scope"]["prepared_max_prefetch"] == 4159
    for layer in result["layers"]:
        assert layer["eligibility"]["schema"] == proof.OFFICIAL_ELIGIBILITY
        assert layer["prefetched_records"] == (64 if saturated else 17)
        assert layer["prefetch_false_positive_records"] == (64 if saturated else 0)
        assert layer["promoted_records"] == (0 if saturated else 17)
        assert layer["official_staging"]["exact_selected_intersection_verified"]
        assert layer["official_staging"]["raw_temporary_tags_verified"]
        assert layer["official_staging"]["temporary_tags_cleared"]
        assert layer["host_padding"]["padding_tokens"] == 63
        assert (
            layer["metrics"]["host_to_device_bytes"]
            == (layer["prefetched_records"] + layer["recalled_records"]) * 1152
        )
    compact = proof.compact_evidence(raw)
    assert "records" not in compact["layers"][0]["official_staging"]
    path = tmp_path / "official.pt"
    torch.save(compact, path)
    assert proof.validate_execution(torch.load(path, weights_only=True)) == result


@pytest.mark.parametrize(
    "corruption",
    [
        "policy",
        "cap",
        "prepared_cap",
        "hint_slot",
        "eligibility",
        "stage_id",
        "stage_padding",
        "slot_order",
        "journal",
        "kv",
        "counter",
        "unused_padding",
    ],
)
def test_official_policy_corruptions_are_rejected(corruption):
    layer = layer_fixture(official=True, padded=True)
    stage = layer["official_staging"]
    if corruption == "policy":
        layer["prefetch_policy"] = "unknown"
    elif corruption == "cap":
        layer["max_prefetch"] = 8192
    elif corruption == "prepared_cap":
        layer["prepared_max_prefetch"] = 64
    elif corruption == "hint_slot":
        layer["hint_index"] = 0
    elif corruption == "eligibility":
        layer["initial_hint"].fill_(4)
    elif corruption == "stage_id":
        stage["host_ids"][0] = layer["H"] + 1
    elif corruption == "stage_padding":
        stage["host_ids"][-1] = 0
    elif corruption == "slot_order":
        stage["prepared_slots"][0] = 0
    elif corruption == "journal":
        stage["allocation_log"][1] = proof.MISSING
    elif corruption == "kv":
        stage["records"][0, 0] = -1
    elif corruption == "counter":
        layer["stages"]["after_prefetch"]["prefetch_counter"][0] = 18
    elif corruption == "unused_padding":
        layer["stages"]["after_prefetch"]["padding_to_slot"][0] = 1
    with pytest.raises(ValueError):
        proof.validate_layer(layer)


def test_official_compact_staged_kv_and_eligibility_identity_corruptions_rejected():
    compact = proof.compact_evidence(execution_fixture(official=True))
    layer = compact["layers"][0]
    layer["official_staging"]["record_validation"]["records_identity"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="staged KV identity"):
        proof.validate_execution(compact)
    compact = proof.compact_evidence(execution_fixture(official=True))
    compact["layers"][0]["eligibility"]["schema"] = proof.COARSE_POLICY
    with pytest.raises(ValueError, match="runtime derivation"):
        proof.validate_execution(compact)


def test_official_false_positive_fixture_cannot_pass_as_coarse_policy():
    layer = layer_fixture(official=True, saturated=True, different_winners=True)
    del layer["prefetch_policy"]
    layer["max_prefetch"] = layer["prepared_max_prefetch"]
    with pytest.raises(ValueError):
        proof.validate_layer(layer)


def test_official_threshold_is_strict_fp32_without_coarse_rounding():
    above = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    scores = torch.tensor([[1.0, above, 0.0, 10.0]])
    eligible, metadata = proof._raw_eligibility(
        {
            "prefetch_policy": proof.OFFICIAL_POLICY,
            "H": 3,
            "scores": scores,
            "initial_hint": torch.ones(1),
        }
    )
    assert eligible.tolist() == [False, True, False]
    assert metadata["candidate_occurrences"] == 1
    assert scores[0, 0].half() == scores[0, 1].half()


@pytest.mark.parametrize("saturated", [False, True])
def test_bounded_free_raw_and_compact_proofs_preserve_actual_capacity(saturated, tmp_path):
    raw = execution_fixture(bounded=True, saturated=saturated, different_winners=True)
    direct = proof.validate_execution(raw)
    assert direct["scope"]["prepared_max_prefetch"] == 64
    assert direct["scope"]["preparation"] == bounded_preparation(65536)
    assert all(layer["preparation"] == direct["scope"]["preparation"] for layer in direct["layers"])
    assert all(layer["official_staging"]["temporary_tags_cleared"] for layer in direct["layers"])
    if saturated:
        assert all(layer["prefetch_false_positive_records"] == 64 for layer in direct["layers"])
    path = tmp_path / "bounded.pt"
    torch.save(proof.compact_evidence(raw), path)
    assert proof.validate_execution(torch.load(path, weights_only=True)) == direct


def bounded_scope(*, history=65536, slots=65600):
    return {
        "H": history,
        "A": 1,
        "slots": slots,
        "record_bytes": 1152,
        "topk": 2048,
        "prefetch_policy": proof.OFFICIAL_POLICY,
        "max_prefetch": 64,
        "prepared_max_prefetch": 64,
        "hint_index": 1,
        "preparation": bounded_preparation(history),
    }


@pytest.mark.parametrize("history,slots", [(32767, 32831), (65536, 65600)])
def test_bounded_capacity_accepts_exact_context_and_headroom_boundaries(history, slots):
    assert proof.preparation_capacity(bounded_scope(history=history, slots=slots)) == 64


@pytest.mark.parametrize(
    "change",
    [
        "short_context",
        "headroom",
        "queries",
        "record",
        "topk",
        "coarse",
        "cap",
        "prepared_cap",
        "hint",
        "unknown",
        "missing",
        "null",
        "extra",
        "start",
        "columns",
        "history",
        "rows",
        "limit",
        "requested",
        "requested_type",
        "session",
        "exclusive",
        "owner",
        "persistent",
        "boolean_geometry",
    ],
)
def test_bounded_capacity_rejects_unsupported_or_incomplete_dispatch(change):
    scope = bounded_scope()
    preparation = scope["preparation"]
    if change == "short_context":
        scope = bounded_scope(history=32766, slots=32830)
    elif change == "headroom":
        scope["slots"] = scope["H"] + 63
    elif change in {"queries", "record", "topk", "coarse", "cap", "prepared_cap", "hint"}:
        key, value = {
            "queries": ("A", 2),
            "record": ("record_bytes", 656),
            "topk": ("topk", 1024),
            "coarse": ("prefetch_policy", proof.COARSE_POLICY),
            "cap": ("max_prefetch", 8192),
            "prepared_cap": ("prepared_max_prefetch", 8192),
            "hint": ("hint_index", 0),
        }[change]
        scope[key] = value
    elif change == "unknown":
        preparation["kind"] = "unknown"
    elif change == "missing":
        del preparation["exclusive_operation"]
    elif change == "null":
        scope["preparation"] = None
    elif change == "extra":
        preparation["unverified"] = True
    elif change == "boolean_geometry":
        scope["A"] = True
    else:
        key, value = {
            "start": ("query_start", 0),
            "columns": ("visible_columns", 65538),
            "history": ("initialized_history", 65535),
            "rows": ("query_count", 2),
            "limit": ("prepared_limit", 65),
            "requested": ("requested_max_prefetch", 63),
            "requested_type": ("requested_max_prefetch", 8192.0),
            "session": ("single_session", False),
            "exclusive": ("exclusive_operation", False),
            "owner": ("pending_owner_matches", False),
            "persistent": ("persistent_append", False),
        }[change]
        preparation[key] = value
    with pytest.raises(ValueError):
        proof.preparation_capacity(scope)


def test_bounded_evidence_cannot_omit_descriptor_or_mix_layer_contracts():
    layer = layer_fixture(bounded=True)
    del layer["preparation"]
    with pytest.raises(ValueError, match="prepared headroom"):
        proof.validate_layer(layer)
    raw = execution_fixture(bounded=True)
    raw["layers"][1]["preparation"]["requested_max_prefetch"] = 64
    with pytest.raises(ValueError, match="geometries differ"):
        proof.validate_execution(raw)


@pytest.mark.parametrize("corruption", ["slot_order", "journal", "kv", "counter", "priority"])
def test_bounded_preparation_keeps_existing_transition_checks(corruption):
    layer = layer_fixture(bounded=True)
    stage = layer["official_staging"]
    if corruption == "slot_order":
        stage["prepared_slots"][0] = 0
    elif corruption == "journal":
        stage["allocation_log"][1] = proof.MISSING
    elif corruption == "kv":
        stage["records"][0, 0] = -1
    elif corruption == "counter":
        layer["stages"]["after_prefetch"]["prefetch_counter"][0] = 18
    else:
        layer["stages"]["after_prefetch"]["priority"][1] = 500
    with pytest.raises(ValueError):
        proof.validate_layer(layer)


def test_legacy_official_full_promotion_retains_v1_schema_and_compact_contract(tmp_path):
    raw = execution_fixture(legacy=True, saturated=True, different_winners=True, padded=True)
    result = proof.validate_execution(raw)
    assert result["schema"] == proof.LEGACY_OFFICIAL_SCHEMA
    assert result["scope"]["prefetch_policy"] == proof.LEGACY_OFFICIAL_POLICY
    assert "Predictive false positives remain resident" in result["eligibility_boundary"]
    for layer in result["layers"]:
        assert layer["prefetch_false_positive_records"] == 64
        assert "promoted_records" not in layer
        assert set(layer["official_staging"]) == {
            "host_ids_identity",
            "prepared_slots_identity",
            "allocation_log_identity",
            "record_validation",
            "temporary_tags_cleared",
            "copy_order_matches_publication",
        }
    path = tmp_path / "legacy_v1.pt"
    torch.save(proof.compact_evidence(raw), path)
    assert proof.validate_execution(torch.load(path, weights_only=True)) == result


def test_selective_promotion_compacts_stage_indices_and_charges_every_actual_copy():
    layer = layer_fixture(official=True, saturated=True, partial_selection=True, padded=True)
    result = proof.validate_layer(layer)
    assert result["prefetched_records"] == 64
    assert result["promoted_records"] == 32
    assert result["prefetch_false_positive_records"] == 32
    assert result["metrics"]["host_to_device_bytes"] == (64 + result["recalled_records"]) * 1152
    promotion_slots = layer["official_staging"]["promotion_slots"]
    assert promotion_slots[:32].tolist() == list(range(1, 33))
    assert (promotion_slots[32:] == proof.MISSING).all()
    assert (layer["stages"]["after_prefetch"]["logical_to_slot"] != proof.MISSING).sum() == 32


def test_pending_candidate_host_placeholder_does_not_promote_a_false_positive():
    layer = layer_fixture(official=True, saturated=True, different_winners=True)
    layer["official_staging"]["page_table"][-1] = layer["official_staging"]["host_ids"][0]
    result = proof.validate_layer(layer)
    assert result["promoted_records"] == 0
    assert result["prefetch_false_positive_records"] == 64


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        "raw_tag",
        "raw_reverse",
        "raw_free",
        "raw_priority",
        "raw_clock",
        "raw_counter",
        "raw_stats",
        "raw_journal",
        "raw_padding",
        "page_table",
        "promotion_indices",
        "promotion_map",
        "promotion_reverse",
        "promotion_clock",
        "promotion_stats",
        "promotion_slot",
        "promotion_journal_marker",
        "final_tag",
        "undercount_h2d",
    ],
)
def test_selective_raw_and_promotion_transition_corruptions_are_rejected(corruption, saved):
    evidence = execution_fixture(official=True, saturated=True, partial_selection=True, padded=True)
    if saved:
        evidence = proof.compact_evidence(evidence)
    layer = evidence["layers"][0]
    official = layer["official_staging"]
    raw, promotion = (layer["stages"][name] for name in ("after_staging", "after_promotion"))
    history = layer["H"]
    inverse = torch.empty(history + 1, dtype=torch.int64)
    inverse[layer["global_ids"]] = torch.arange(history + 1)
    first = int(inverse[official["host_ids"][0].long()])
    if corruption == "raw_tag":
        raw["logical_to_slot"][first] += 1
    elif corruption == "raw_reverse":
        raw["reverse"][1] = official["host_ids"][0]
    elif corruption == "raw_free":
        raw["free"][1] = False
    elif corruption == "raw_priority":
        raw["priority"][1] = 0
    elif corruption == "raw_clock":
        raw["clock"][0] += 1
    elif corruption == "raw_counter":
        raw["counter_totals"][0] = 64
    elif corruption == "raw_stats":
        raw["prefetch_stats"][0] = 64
    elif corruption == "raw_journal":
        official["raw_allocation_log"][1] = 0
    elif corruption == "raw_padding":
        raw["padding_to_slot"][0] = layer["slots"] + 1
    elif corruption == "page_table":
        official["page_table"][first] = official["host_ids"][1]
    elif corruption == "promotion_indices":
        official["promotion_indices"][0, 0] = official["promotion_indices"][0, 1]
    elif corruption == "promotion_map":
        promotion["logical_to_slot"][first] = proof.MISSING
    elif corruption == "promotion_reverse":
        promotion["reverse"][1] = proof.MISSING
    elif corruption == "promotion_clock":
        promotion["clock"][0] += 1
    elif corruption == "promotion_stats":
        promotion["prefetch_stats"][0] = 32
    elif corruption == "promotion_slot":
        official["promotion_slots"][0] = 2
    elif corruption == "promotion_journal_marker":
        official["allocation_log"][-1] = -1
    elif corruption == "final_tag":
        layer["stages"]["after_prefetch"]["logical_to_slot"][first] = layer["slots"] + 1
    elif corruption == "undercount_h2d":
        layer["metrics"]["host_to_device_bytes"] -= 32 * 1152
    with pytest.raises(ValueError):
        proof.validate_execution(evidence)
