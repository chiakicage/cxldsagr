from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cache.sparse_token_cache import MISSING
from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src.extend_graph_validation import (
    cache_state,
    compare_cache_state,
    tensor_identity,
)


def fixture_model(*, swap=False):
    host = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    records = torch.cat([torch.zeros_like(host[:1]), host.flip(0) if swap else host])
    shared = SimpleNamespace(
        free=torch.tensor([False, False, False]),
        priority=torch.tensor([0, 8, 8], dtype=torch.int64),
        clock_tensor=torch.tensor([9], dtype=torch.int64),
    )
    cache = SimpleNamespace(
        length=2,
        written=2,
        indexer_visible_end=2,
        _step_end=None,
        offload=True,
        device=torch.device("cpu"),
        slots=2,
        layer_id=0,
        _pool=SimpleNamespace(layers=[shared]),
        host_to_device=torch.tensor([2, 1] if swap else [1, 2], dtype=torch.int32),
        device_to_host=torch.tensor([MISSING, 1, 0] if swap else [MISSING, 0, 1]),
        records=records,
        host=host,
        host_records=lambda: host.clone(),
        metrics=lambda: {"written_records": 2},
        logical_to_global=lambda value: value,
    )
    attention = SimpleNamespace(
        index_keys=torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
        index_scales=torch.ones(2),
        offset=torch.zeros(1, dtype=torch.int64),
    )
    return SimpleNamespace(
        length=2,
        blocks=[SimpleNamespace(cache=cache, attention=attention)],
        synchronize=lambda: None,
    )


def test_cache_acceptance_allows_equivalent_tied_slot_permutations():
    first, swapped = cache_state(fixture_model()), cache_state(fixture_model(swap=True))
    assert compare_cache_state(first, swapped)["equal"]
    assert first["layers"][0]["map_invariants_passed"]


@pytest.mark.parametrize("corrupt", ["reverse", "free", "record", "commit"])
def test_cache_acceptance_rejects_invalid_maps_records_or_commit(corrupt):
    model = fixture_model()
    cache = model.blocks[0].cache
    if corrupt == "reverse":
        cache.device_to_host[1] = 1
    elif corrupt == "free":
        cache._pool.layers[0].free[1] = True
    elif corrupt == "record":
        cache.records[1, 0] = 7
    else:
        cache._step_end = 3
    with pytest.raises(AssertionError):
        cache_state(model)


def test_tensor_identity_hashes_scalar_and_low_precision_bytes():
    assert tensor_identity(torch.tensor(1.0))["shape"] == []
    assert tensor_identity(torch.ones(2, dtype=torch.bfloat16))["shape"] == [2]


def proof_for(state, *, indices="same-exact-indices"):
    # A deliberate accepted-auditor stub isolates the comparison/binding logic;
    # the real transition auditor has its own corrupt-state rejection tests.
    receipt = {
        "schema": "cold-echo-stage-acceptance-v1",
        "passed": True,
        "scope": {
            "method": "echo",
            "residency": "cold",
            "single_session": True,
            "num_layers": 3,
            "H": 1,
            "A": 1,
            "slots": 2,
            "max_prefetch": 1,
        },
        "final_cache_state_sha256": identity_digest(state),
        "indices": [indices] * 3,
        "scores": ["same-exact-scores"] * 3,
        "initial_hints": ["same-hints"] * 3,
    }
    receipt["proof"] = {
        "schema": "cold-echo-prefetch-transition-v1",
        "passed": True,
        "scope": dict(receipt["scope"]),
        "layers": [
            {
                "layer": index,
                "passed": True,
                "stage_maps_free_priorities_clocks_verified": True,
                "physical_owners_preserved_until_final": True,
                "no_eviction": True,
                "final_cache_state_identity": identity_digest(layer),
                "indices_identity": indices,
                "eligibility": {
                    "score_identity": "same-exact-scores",
                    "hint_identity": "same-hints",
                },
            }
            for index, layer in enumerate(state["layers"])
        ],
    }
    return receipt


def prefetch_state():
    state = cache_state(fixture_model())
    state["layers"] = [deepcopy(state["layers"][0]) for _ in range(3)]
    return state


def test_priority_difference_requires_two_bound_transition_proofs():
    expected = prefetch_state()
    actual = deepcopy(expected)
    actual["layers"][0]["logical_priority"]["sha256"] = "different-schedule"
    with pytest.raises(AssertionError, match="logical_priority"):
        compare_cache_state(actual, expected)
    result = compare_cache_state(
        actual,
        expected,
        actual_prefetch=proof_for(actual),
        expected_prefetch=proof_for(expected),
    )
    assert result["bounded_prefetch_transitions_validated"] and result["exact_topk_equal"]
    with pytest.raises(AssertionError, match="does not bind"):
        compare_cache_state(actual, expected, actual_prefetch=proof_for(expected))


@pytest.mark.parametrize("field", ["records", "index_keys", "hint", "resident", "clock"])
def test_transition_proofs_do_not_relax_numerical_residency_or_clock_fields(field):
    expected = prefetch_state()
    actual = deepcopy(expected)
    actual["layers"][0][field]["sha256"] = "corrupt"
    with pytest.raises(AssertionError, match=field):
        compare_cache_state(
            actual,
            expected,
            actual_prefetch=proof_for(actual),
            expected_prefetch=proof_for(expected),
        )


def test_transition_proofs_require_exact_indices_and_keep_other_metrics_strict():
    expected = prefetch_state()
    with pytest.raises(AssertionError, match="exact top-k"):
        compare_cache_state(
            expected,
            expected,
            actual_prefetch=proof_for(expected, indices="changed"),
            expected_prefetch=proof_for(expected),
        )
    actual = deepcopy(expected)
    actual["layers"][0]["metrics"]["written_records"] += 1
    with pytest.raises(AssertionError, match="metrics"):
        compare_cache_state(
            actual,
            expected,
            actual_prefetch=proof_for(actual),
            expected_prefetch=proof_for(expected),
        )


@pytest.mark.parametrize("change", ["generic_passed", "warm", "multi_session", "scores", "hints"])
def test_transition_proof_cannot_be_a_generic_priority_exemption(change):
    state = prefetch_state()
    proof = proof_for(state)
    if change == "generic_passed":
        proof = {"passed": True, "final_cache_state_sha256": identity_digest(state)}
    elif change == "warm":
        proof["scope"]["residency"] = "warm"
    elif change == "multi_session":
        proof["scope"]["single_session"] = False
    elif change == "scores":
        proof["scores"][0] = "changed"
    else:
        proof["initial_hints"][0] = "changed"
    with pytest.raises(AssertionError):
        compare_cache_state(state, state, actual_prefetch=proof, expected_prefetch=proof_for(state))


def official_proof_and_state(*, different_winners=False, bounded=False):
    from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as transitions
    from experiments.deepseek_v32_mfu.tests.test_prefetch_transition_audit import execution_fixture

    evidence = execution_fixture(
        official=True,
        saturated=True,
        padded=True,
        different_winners=different_winners,
        bounded=bounded,
    )
    audit = transitions.validate_execution(evidence)
    state = {
        "length": evidence["layers"][0]["H"] + 1,
        "layers": [layer["final_cache_state"] for layer in evidence["layers"]],
    }
    receipt = {
        "schema": "cold-echo-stage-acceptance-v1",
        "passed": True,
        "scope": {
            key: value
            for key, value in audit["scope"].items()
            if bounded or key not in ("record_bytes", "topk")
        },
        "proof": audit,
        "final_cache_state_sha256": identity_digest(state),
        "indices": [transitions.tensor_identity(layer["indices"]) for layer in evidence["layers"]],
        "scores": [transitions.tensor_identity(layer["scores"]) for layer in evidence["layers"]],
        "initial_hints": [
            transitions.tensor_identity(layer["initial_hint"]) for layer in evidence["layers"]
        ],
    }
    return state, receipt


@pytest.mark.parametrize("bounded", [False, True])
def test_official_bound_false_positives_can_change_residency_and_free_count(bounded):
    expected, expected_proof = official_proof_and_state(bounded=bounded)
    actual, actual_proof = official_proof_and_state(different_winners=True, bounded=bounded)
    assert actual["layers"][0]["resident"] != expected["layers"][0]["resident"]
    assert actual["layers"][0]["free_count"] != expected["layers"][0]["free_count"]
    compared = compare_cache_state(
        actual, expected, actual_prefetch=actual_proof, expected_prefetch=expected_proof
    )
    assert compared["schedule_dependent_state_fields"] == [
        "free_count",
        "logical_priority",
        "resident",
    ]
    actual_proof["proof"]["layers"][0]["official_staging"]["temporary_tags_cleared"] = False
    with pytest.raises(AssertionError, match="official prefetch stage proof"):
        compare_cache_state(
            actual, expected, actual_prefetch=actual_proof, expected_prefetch=expected_proof
        )


@pytest.mark.parametrize("field", ["records", "index_keys", "hint", "clock"])
@pytest.mark.parametrize("bounded", [False, True])
def test_official_proof_keeps_numerical_and_clock_comparisons_strict(field, bounded):
    expected, expected_proof = official_proof_and_state(bounded=bounded)
    actual, actual_proof = official_proof_and_state(different_winners=True, bounded=bounded)
    actual["layers"][0][field]["sha256"] = "c" * 64
    actual_proof["final_cache_state_sha256"] = identity_digest(actual)
    actual_proof["proof"]["layers"][0]["final_cache_state_identity"] = identity_digest(
        actual["layers"][0]
    )
    with pytest.raises(AssertionError, match=field):
        compare_cache_state(
            actual, expected, actual_prefetch=actual_proof, expected_prefetch=expected_proof
        )


@pytest.mark.parametrize(
    "change", ["missing", "cap", "descriptor", "layer_descriptor", "mixed_full", "different_scope"]
)
def test_bounded_graph_comparison_requires_matching_complete_preparation_scopes(change):
    state, expected = official_proof_and_state(bounded=True)
    actual = deepcopy(expected)
    if change == "missing":
        del actual["scope"]["preparation"]
    elif change == "cap":
        actual["scope"]["prepared_max_prefetch"] = 8192
    elif change == "descriptor":
        actual["scope"]["preparation"]["exclusive_operation"] = False
    elif change == "layer_descriptor":
        actual["proof"]["layers"][0]["preparation"]["persistent_append"] = False
    elif change == "mixed_full":
        for scope in (actual["scope"], actual["proof"]["scope"], *actual["proof"]["layers"]):
            del scope["preparation"]
            scope["prepared_max_prefetch"] = 8192
    else:
        for scope in (actual["scope"], actual["proof"]["scope"], *actual["proof"]["layers"]):
            scope["preparation"]["requested_max_prefetch"] = 64
    with pytest.raises(AssertionError):
        compare_cache_state(state, state, actual_prefetch=actual, expected_prefetch=expected)


@pytest.mark.parametrize("field", ["indices", "scores", "initial_hints"])
def test_bounded_proofs_do_not_relax_exact_selection_scores_or_hints(field):
    state, expected = official_proof_and_state(bounded=True)
    actual = deepcopy(expected)
    actual[field][0] = "changed"
    layer = actual["proof"]["layers"][0]
    if field == "indices":
        layer["indices_identity"] = "changed"
    else:
        key = "score_identity" if field == "scores" else "hint_identity"
        layer["eligibility"][key] = "changed"
    with pytest.raises(AssertionError, match="full graph changed"):
        compare_cache_state(state, state, actual_prefetch=actual, expected_prefetch=expected)
