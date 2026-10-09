import copy

import pytest

from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_profile as audit


@pytest.fixture
def runs():
    names = {
        "zero": "void at::native::FillFunctor<int>",
        "arange": "void at::native::arange_cuda_out",
        "pack": "pack_page64",
        "schedule": "void deep_gemm::smxx_paged_mqa_logits_metadata<1>",
        "stage": "official_prefetch::prepare_kernel()",
        "fused": "void q1_fused_prepare::prepare_kernel<true>()",
        "core": "void deep_gemm::sm90_fp8_paged_mqa_logits_fused_v2<1>",
        "clean": "void deep_gemm::smxx_clean_logits<1>",
        "validate": "official_prefetch::validate_promotion_kernel()",
        "copy": "official_prefetch::copy_publish_kernel()",
    }
    sequences = {
        "baseline": (
            "zero",
            "arange",
            "pack",
            "arange",
            "schedule",
            "stage",
            "core",
            "clean",
            "validate",
            "copy",
        ),
        "candidate": ("zero", "arange", "fused", "schedule", "core", "clean", "validate", "copy"),
    }
    result = {}
    for arm, sequence in sequences.items():
        nodes = []
        for layer_index, layer in enumerate(audit.LAYERS):
            for index, name in enumerate(sequence):
                nodes.append(
                    {
                        "kind": "kernel",
                        "name": names[name],
                        "start": 1000 * (layer_index * 20 + index),
                        "end": 1000 * (layer_index * 20 + index) + 500,
                        "capture_node": layer_index * 100 + index,
                        "owner": {
                            "layer": layer,
                            "stage": "indexer_fused",
                            "source_stage": "indexer_prefetch",
                            "stage_path": ["extend_graph_body", layer, "indexer_prefetch"],
                        },
                        **dict.fromkeys(audit.FIELDS, 1),
                    }
                )
                if arm == "baseline" and index == 3:
                    nodes[-1]["gridX"] = 1025
        nodes.append(
            {
                "kind": "memcpy",
                "name": "copy",
                "bytes": 8,
                "copyKind": 1,
                "capture_node": 999,
                "owner": {"layer": "shared", "stage": "inputs"},
                "start": 0,
                "end": 1,
            }
        )
        result[arm] = {"nodes": nodes}
    return result


def test_exact_preparation_replacement_preserves_other_work(runs):
    result = audit.preparation_change(runs)
    assert result["baseline_preparation_nodes"] == 9
    assert result["candidate_preparation_nodes"] == 3
    assert result["unchanged_node_count"] == 22
    assert result["baseline_preparation_sum_us"] == 4.5
    assert result["candidate_preparation_sum_us"] == 1.5


@pytest.mark.parametrize(
    "change", ("bounds", "core_grid", "copy_bytes", "owner", "overlap", "duplicate")
)
def test_wrong_order_ownership_or_nonpreparation_work_fails(runs, change):
    rows = runs["candidate"]["nodes"]
    if change == "bounds":
        rows.pop(1)
    elif change == "core_grid":
        rows[4]["gridX"] += 1
    elif change == "copy_bytes":
        rows[-1]["bytes"] += 8
    elif change == "owner":
        rows[0]["owner"]["source_stage"] = "other"
    elif change == "overlap":
        rows[0]["end"] = rows[1]["start"] + 1
    else:
        rows.append(copy.deepcopy(rows[2]))
    with pytest.raises(ValueError):
        audit.preparation_change(runs)


def test_only_the_block_table_arange_is_removed(runs):
    runs["candidate"]["nodes"][1]["gridX"] = 1025
    with pytest.raises(ValueError, match="Non-preparation"):
        audit.preparation_change(runs)


def test_unknown_replacement_is_rejected(runs):
    runs["baseline"]["nodes"][2]["name"] = "pack_page64_wrong"
    with pytest.raises(ValueError, match="Unknown indexer-stage"):
        audit.preparation_change(runs)
