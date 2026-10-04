"""Metadata acceptance distinguishes retained history from committed suffixes."""

import pytest

from experiments.gr_serving.src import audit
from experiments.gr_serving.tests.test_gr_serving_cpu_reservation import cpu_workspace_metadata


def pooled_metadata(persistence):
    meta = cpu_workspace_metadata()
    params = meta["parameters"]
    params.pop("deepseek_slots")
    params.update(sparse_pool_tokens=32768, workspace_query_tokens=1024, extend_chunk_size=None)
    backend = meta["models"]["deepseek_v32"]
    backend.pop("sparse_slots")
    backend.update(
        cache_policy_revision="echo-global-pages-fifo-v1",
        sparse_pool_tokens=32768,
        extend_chunk_size=None,
    )
    retained = params["history_tokens"]
    if persistence != "gpu_transient":
        retained += params["candidate_tokens"]
    for case in meta["cases"]:
        plan = case["cache_resource_plan"]
        plan["cache_policy_revision"] = backend["cache_policy_revision"]
        if case["scheme"] not in ("echo", "serial_sparse"):
            continue
        plan.update(
            pool_scope="backend_per_layer",
            sparse_pool_tokens=32768,
            workspace_query_tokens=1024,
            host_arena_tokens=65536,
        )
        if persistence is not None:
            plan["candidate_persistence"] = persistence
        case["host_page_capacity"] = 1024
        case["session_host_pages"] = (retained + 63) // 64
        case["session_reservation"]["dram"] = case["session_host_pages"] * 4
    return meta


@pytest.mark.parametrize("persistence", [None, "committed", "gpu_transient"])
def test_page_charge_follows_explicit_candidate_persistence(persistence):
    meta = pooled_metadata(persistence)
    audit.audit_metadata(meta, "unit", None)


@pytest.mark.parametrize("persistence", [None, "committed", "gpu_transient"])
def test_page_charge_rejects_the_other_candidate_lifecycle(persistence):
    meta = pooled_metadata(persistence)
    case = next(case for case in meta["cases"] if case["scheme"] == "echo")
    candidate_pages = meta["parameters"]["candidate_tokens"] // 64
    case["session_host_pages"] += candidate_pages * (1 if persistence == "gpu_transient" else -1)
    case["session_reservation"]["dram"] = case["session_host_pages"] * 4
    with pytest.raises(audit.AuditError, match="session page charge"):
        audit.audit_metadata(meta, "unit", None)
