"""Cold ECHO legality is checked by token sets and transitions, not count equality."""

import struct
from copy import deepcopy

import pytest
import torch

from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as proof


def layer_fixture(*, saturated=False, different_winners=False):
    history, queries, topk = (8448, 6, 2048) if saturated else (64, 2, 2048)
    end, start = history + queries, 64
    scores = torch.full((queries, end), -torch.inf)
    for query in range(queries):
        scores[query, : history + query + 1] = 0
        if saturated:
            scores[query, query * 1400 : (query + 1) * 1400] = 1
    values, indices = torch.topk(scores, min(topk, end), dim=-1)
    indices = torch.where(torch.isfinite(values), indices, -1).int()
    selected = torch.zeros(end, dtype=torch.bool)
    selected[indices[indices >= 0].long()] = True
    eligible = torch.ones(history, dtype=torch.bool)
    if saturated:
        eligible[8400:] = False
    cap = min(8192, history)
    winner_ids = eligible.nonzero().flatten()
    winner_ids = winner_ids[-cap:] if different_winners else winner_ids[:cap]
    prefetched = torch.zeros(end, dtype=torch.bool)
    prefetched[winner_ids] = True
    candidates = torch.arange(end) >= history
    appended = prefetched | candidates
    recalls = selected & ~appended
    copied, missing, size = len(winner_ids), int(recalls.sum()), int(selected.sum())
    rejected = 3 if saturated else 0
    global_ids = torch.arange(end).flip(0)
    mapping = torch.full((end,), proof.MISSING, dtype=torch.int32)

    def stage(present, priorities, clock, counters):
        reverse = torch.full((end + 1,), proof.MISSING, dtype=torch.int64)
        free = torch.ones(end + 1, dtype=torch.bool)
        free[0] = False
        priority = torch.full((end + 1,), -1, dtype=torch.int64)
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
        }

    priorities = torch.zeros(end, dtype=torch.int64)
    initial = stage(torch.zeros(end, dtype=torch.bool), priorities, start, [0] * 8)
    mapping[winner_ids] = torch.arange(1, copied + 1, dtype=torch.int32)
    priorities[prefetched] = start
    pre_counters = [copied, 0, rejected, 0, 0, 0, 0, 0]
    prefetch = stage(prefetched, priorities, start + 1, pre_counters)
    prefetch.update(
        prefetch_counter=torch.tensor([copied + rejected], dtype=torch.uint32),
        prefetch_stats=torch.tensor([copied, 0, rejected]),
    )
    mapping[history:] = torch.arange(copied + 1, copied + queries + 1, dtype=torch.int32)
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
        "device_slots": end,
        "host_token_capacity": end,
        "session_host_tokens": end,
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
    return {
        "layer": 0,
        "H": history,
        "A": queries,
        "slots": end,
        "record_bytes": 1152,
        "topk": topk,
        "max_prefetch": cap,
        "global_ids": global_ids,
        "initial_hint": torch.zeros(1),
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
