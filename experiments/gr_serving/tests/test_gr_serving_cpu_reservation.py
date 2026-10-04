import pytest

from experiments.gr_serving.src import audit
from experiments.gr_serving.tests.test_gr_serving_audit import metadata_fixture


def cpu_workspace_metadata():
    meta = metadata_fixture(models=["deepseek_v32"], deepseek_layers=10, deepseek_slots=32768)
    meta["models"] = {
        "deepseek_v32": {
            "device": "cuda:0",
            "physical_layers": 10,
            "chunk_size": 1024,
            "sparse_slots": 32768,
            "source_layers": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "total_parameters": 7827793408,
            "input_semantics": "copy_source_layer_hidden_and_residual_for_each_physical_copy",
            "output": "all_candidate_normalized_hidden_and_last_token_lm_head",
        }
    }
    for case in meta["cases"]:
        case["shared_reservation"] = {"hbm": 100, "dram": 40}
        case["cache_resource_plan"] = {
            "workspace_cpu_indexer_bytes": 8,
            "workspace_cpu_scalar_bytes": 8,
            "workspace_cpu_metrics_bytes": 24,
            "workspace_cpu_bytes": 40,
        }
    return meta


def test_resident_cpu_scratch_is_distinct_from_host_backing():
    meta = cpu_workspace_metadata()
    audit.audit_metadata(meta, "unit", None)
    meta["cases"][0]["session_reservation"]["dram"] = 1
    with pytest.raises(audit.AuditError, match="host backing"):
        audit.audit_metadata(meta, "unit", None)


@pytest.mark.parametrize("change", ["missing_term", "wrong_sum", "unreserved"])
def test_cpu_workspace_requires_complete_terms_and_shared_capacity(change):
    meta = cpu_workspace_metadata()
    case = meta["cases"][0]
    if change == "missing_term":
        del case["cache_resource_plan"]["workspace_cpu_metrics_bytes"]
    elif change == "wrong_sum":
        case["cache_resource_plan"]["workspace_cpu_bytes"] = 39
    else:
        case["shared_reservation"]["dram"] = 39
    with pytest.raises(audit.AuditError):
        audit.audit_metadata(meta, "unit", None)
