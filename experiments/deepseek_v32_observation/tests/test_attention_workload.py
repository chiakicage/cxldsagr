"""The measured decomposition must not change the complete attention workload."""

import pytest

from experiments.deepseek_v32_observation.src.attention_workload import AttentionWorkload


@pytest.mark.parametrize("paged", (False, True))
def test_complete_run_does_not_execute_measurement_leaves(paged):
    owner = AttentionWorkload.__new__(AttentionWorkload)
    owner.paged = paged
    owner.output = object()
    observed = []
    names = ("indexer_prepare", "indexer_metadata") if paged else ()
    names += ("indexer", "indexer_causal_mask", "exact_topk", "sparse_mla")
    owner.operations = {name: lambda name=name: observed.append(name) for name in names}

    def forbidden_leaf():
        raise AssertionError("Standalone measurement leaves cannot execute in the complete run")

    owner.operations["sparse_mla_core"] = forbidden_leaf
    owner.operations["page64_pack"] = forbidden_leaf
    assert owner.run() is owner.output
    assert observed == list(names)
