"""CPU execution histories and adversarial cross-process receipt contracts."""

import copy
import json

import pytest

from experiments.deepseek_v32_mfu.src import backend_provenance, profile_layers, run_contract
from experiments.deepseek_v32_mfu.tests import test_profile_layers as fixtures

cpu_annotation = fixtures.cpu_annotation
driver = fixtures.driver


@pytest.mark.parametrize("method", run_contract.METHODS)
def test_isolated_driver_runs_only_selected_method_and_preserves_cpu_reference(driver, method):
    execute, _, _ = driver
    reference = []
    if method != "hbm":
        hbm_dir, hbm, _ = execute("check", "--method", "hbm", run_name="hbm_check")
        reference = ["--hbm-check-receipt", str(hbm_dir / "receipt.json")]
    check_dir, check, _ = execute("check", "--method", method, *reference)
    assert check["schema_version"] == 4 and check["methods"] == [method]
    assert set(check["correctness"]) == run_contract.expected_check_fields(check)
    assert check["process_provenance"]["completed_method_warmups"] == [method]
    assert set(check["process_provenance"]["cache_method_selections"]) == {method}
    if method != "hbm":
        assert check["hbm_reference"]["execution_identity"] == hbm["execution_identity"]
        assert run_contract.common_workload_identity(
            check
        ) == run_contract.common_workload_identity(hbm)
        assert not (check_dir / "hbm_control.pt").exists()
    receipt = str(check_dir / "receipt.json")
    bench_dir, bench, _ = execute("bench", "--method", method, "--validation-receipt", receipt)
    profile_dir, profile, _ = execute(
        "profile",
        "--method",
        method,
        "--nsys",
        "--validation-receipt",
        receipt,
        "--benchmark-run",
        str(bench_dir),
    )
    for directory, result, mode in (
        (check_dir, check, "check"),
        (bench_dir, bench, "bench"),
        (profile_dir, profile, "profile"),
    ):
        assert set(result["process_provenance"]["cache_method_selections"]) == {method}
        assert (
            run_contract.validate_completed_result(directory / "result.json", expected_mode=mode)
            == result
        )
    assert set(bench["measurements"]) == {method} and not bench["correctness"]
    assert profile["nsys_capture_order"] == [
        f"{method}/prefill_annotated",
        f"{method}/extend_annotated",
    ]
    assert set(profile["correctness"]) == run_contract.expected_check_fields(
        profile, include_profile=True
    )
    assert (
        run_contract.benchmark_view(profile_dir, profile)["measurements"] == bench["measurements"]
    )


@pytest.mark.parametrize(
    "reference_kind", ["mixed", "other_method", "changed_output", "different_warmup"]
)
def test_offload_check_rejects_incompatible_hbm_reference(driver, reference_kind):
    execute, _, _ = driver
    flags = [] if reference_kind == "mixed" else ["--method", "hbm"]
    hbm_dir, _, _ = execute("check", *flags, run_name="hbm_check")
    reference = hbm_dir / "receipt.json"
    match = "isolated HBM"
    if reference_kind == "other_method":
        other_dir, _, _ = execute(
            "check",
            "--method",
            "serial_sparse",
            "--hbm-check-receipt",
            str(reference),
            run_name="serial_check",
        )
        reference = other_dir / "receipt.json"
    elif reference_kind == "changed_output":
        (hbm_dir / "hbm_control.pt").write_bytes(b"changed")
        match = "evidence changed"
    extra = []
    if reference_kind == "different_warmup":
        extra = ["--warmups", "2"]
        match = "common workload"
    with pytest.raises(ValueError, match=match):
        execute("check", "--method", "echo", "--hbm-check-receipt", str(reference), *extra)


@pytest.mark.parametrize("bad_method", [None, "serial_sparse"])
def test_isolated_benchmark_rejects_another_method_receipt(driver, bad_method):
    execute, _, _ = driver
    hbm_dir, _, _ = execute("check", "--method", "hbm")
    flags = [] if bad_method is None else ["--method", bad_method]
    with pytest.raises(ValueError, match="kind differs|does not cover"):
        execute("bench", *flags, "--validation-receipt", str(hbm_dir / "receipt.json"))


def test_isolated_hbm_does_not_require_or_load_offload_providers(driver, monkeypatch):
    execute, _, _ = driver
    original = backend_provenance.collect_flashinfer_runtime_artifacts
    calls = []

    def collect(*, require_local_native):
        calls.append(require_local_native)
        return original(require_local_native=require_local_native)

    monkeypatch.setattr(backend_provenance, "collect_flashinfer_runtime_artifacts", collect)
    _, result, _ = execute("check", "--method", "hbm")
    assert calls == [False, False]
    assert result["execution_runtime_artifacts"]["local_native_jit"] == []


@pytest.mark.parametrize(
    "mutation", ["empty_runtime", "missing_native", "missing_norm", "missing_quantizer"]
)
def test_isolated_runtime_requires_actual_participating_adapters(driver, monkeypatch, mutation):
    execute, _, _ = driver
    original = backend_provenance.collect_flashinfer_runtime_artifacts

    def collect(**kwargs):
        runtime = original(**kwargs)
        if mutation == "empty_runtime":
            return {}
        if mutation == "missing_native":
            runtime["native_jit"].pop()
        elif mutation == "missing_norm":
            runtime["cute_jit"] = []
        else:
            runtime["linear_quantization_triton"] = None
        return runtime

    monkeypatch.setattr(backend_provenance, "collect_flashinfer_runtime_artifacts", collect)
    with pytest.raises(ValueError, match="runtime adapters"):
        execute("check", "--method", "hbm")


def test_isolated_contract_binds_warmups_and_rejects_foreign_process_history(driver):
    execute, _, _ = driver
    directory, result, _ = execute("check", "--method", "hbm")
    assert result["execution_identity"]["warmups"] == 1
    changed = copy.deepcopy(result)
    changed["warmups"] = 2
    with pytest.raises(ValueError, match="preparation contract"):
        run_contract.execution_identity(changed)
    changed = copy.deepcopy(result)
    changed["process_provenance"]["cache_method_selections"].append("dense_prefetch")
    path = directory / "altered.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="process provenance"):
        run_contract.validate_completed_result(path, expected_mode="check")


@pytest.mark.parametrize(
    "mode,method,reference",
    [
        ("check", "echo", False),
        ("check", "hbm", True),
        ("check", None, True),
        ("bench", "echo", True),
        ("profile", "echo", True),
    ],
)
def test_hbm_reference_cli_only_allows_offload_check(tmp_path, mode, method, reference):
    argv = ["--run-id", "invalid", "--output", str(tmp_path / "out")]
    if mode != "profile":
        argv += ["--mode", mode]
    if method:
        argv += ["--method", method]
    if reference:
        argv += ["--hbm-check-receipt", str(tmp_path / "receipt.json")]
    with pytest.raises(SystemExit):
        profile_layers.parse_run_args("profile" if mode == "profile" else None, argv)


def _large_q1_result(*, method="echo", fused=True, page64=False):
    adapter = {
        "policy": "official-q1-predictive-staging-promotion-v1",
        "source_sha256": {"adapter": "sha"},
    }
    if fused:
        adapter["preparation"] = run_contract.Q1_PREFETCH_PREPARATION
    local = [
        {
            "name": "cxldsagr_echo_indexer_fixture.so",
            "category": "echo_indexer",
            "library": {"sha256": "echo"},
        },
        {
            "name": "cxldsagr_kv_transfer.so",
            "category": "record_transfer",
            "library": {"sha256": "transfer"},
        },
        {
            "name": "cxldsagr_official_echo_decode_fixture.so",
            "category": "other_local_native",
            "library": {"sha256": "decode"},
        },
        {
            "name": "cxldsagr_official_prefetch_fixture.so",
            "category": "other_local_native",
            "library": {"sha256": "prepare", "path": "/native/prepare.so"},
        },
        {"name": "topk.so", "category": "other_local_native", "library": {"sha256": "topk"}},
        {"name": "hint.so", "category": "other_local_native", "library": {"sha256": "hint"}},
    ]
    native = {
        "artifact_name": "cxldsagr_official_prefetch_fixture.so",
        "artifact_path": "/native/prepare.so",
        "artifact_sha256": "prepare",
        "build_identity": {"source_identity": copy.deepcopy(adapter)},
    }
    return {
        "schema_version": 4,
        "selected_method": method,
        "methods": [method],
        "method_isolation": run_contract.METHOD_ISOLATION,
        "backend_provenance": {"official_echo_prefetch_adapter": adapter},
        "prefix_tokens": 65536,
        "extend_tokens": 1,
        "chunk_size": 1024,
        "extend_chunk_size": 1,
        "slots": 65600,
        "extend_residency": "cold",
        "execution_runtime_artifacts": {
            "native_jit": [
                {"name": name, "loaded_in_this_process": True, "library": {"sha256": name}}
                for name in ("rope", "silu_and_mul", "topk")
            ],
            "cute_jit": [{}],
            "linear_quantization_triton": {"specializations": [{}]},
            "local_native_jit": local,
            "attention_decode_triton": {"specializations": [{}]},
            "q1_topk_cub_native": {"artifact_name": "topk.so", "artifact_sha256": "topk"},
            "q1_hint_native": {"artifact_name": "hint.so", "artifact_sha256": "hint"},
            "indexer_adaptation_triton": {"page64": [{}] if page64 else [], "decode_hint": [{}]},
            "q1_prefetch_preparation": {
                "preparation": run_contract.Q1_PREFETCH_PREPARATION,
                "entry_point": "logits_from_keys",
                "native": native,
            }
            if fused and method == "echo"
            else None,
        },
    }


@pytest.mark.parametrize("headroom", [1, 64])
def test_isolated_fused_q1_requires_observed_entry_and_actual_native_without_unused_packer(
    headroom,
):
    result = _large_q1_result()
    result["slots"] = result["prefix_tokens"] + headroom
    run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["q1_prefetch_preparation"] = None
    result["execution_runtime_artifacts"]["indexer_adaptation_triton"]["page64"] = [{}]
    with pytest.raises(ValueError, match="observed fused preparation"):
        run_contract.validate_runtime_participation(result)


@pytest.mark.parametrize(
    "corruption", ["entry", "preparation", "source", "sha", "path", "name", "unmapped", "duplicate"]
)
def test_isolated_fused_q1_rejects_unbound_native_participation(corruption):
    result = _large_q1_result()
    runtime = result["execution_runtime_artifacts"]
    observed = runtime["q1_prefetch_preparation"]
    if corruption == "entry":
        observed["entry_point"] = "logits"
    elif corruption == "preparation":
        observed["preparation"] = "unknown"
    elif corruption == "source":
        observed["native"]["build_identity"]["source_identity"]["source_sha256"] = {}
    elif corruption in {"sha", "path", "name"}:
        observed["native"][
            {"sha": "artifact_sha256", "path": "artifact_path", "name": "artifact_name"}[corruption]
        ] = "wrong"
    elif corruption == "unmapped":
        runtime["local_native_jit"].pop(3)
    else:
        runtime["local_native_jit"].append(copy.deepcopy(runtime["local_native_jit"][3]))
    with pytest.raises(ValueError, match="fused preparation|Fused preparation"):
        run_contract.validate_runtime_participation(result)


@pytest.mark.parametrize("method", run_contract.METHODS)
def test_legacy_large_q1_reports_keep_original_page64_requirement(method):
    result = _large_q1_result(method=method, fused=False, page64=True)
    run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["indexer_adaptation_triton"]["page64"] = []
    with pytest.raises(ValueError, match="page64 specialization"):
        run_contract.validate_runtime_participation(result)


@pytest.mark.parametrize("method", ("hbm", "serial_sparse", "dense_prefetch"))
def test_current_resident_q1_still_requires_page64(method):
    result = _large_q1_result(method=method, fused=True)
    with pytest.raises(ValueError, match="page64 specialization"):
        run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["indexer_adaptation_triton"]["page64"] = [{}]
    run_contract.validate_runtime_participation(result)


def test_fused_extend_does_not_replace_resident_prefill_tail_evidence():
    result = _large_q1_result()
    result["prefix_tokens"] = 65537
    result["slots"] = 65601
    with pytest.raises(ValueError, match="page64 specialization"):
        run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["indexer_adaptation_triton"]["page64"] = [{}]
    run_contract.validate_runtime_participation(result)


@pytest.mark.parametrize("headroom", [1, 64])
def test_warm_uncertified_residency_binds_only_observed_fused_entry(headroom):
    result = _large_q1_result()
    result["extend_residency"] = "warm"
    result["slots"] = result["prefix_tokens"] + headroom
    run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["q1_prefetch_preparation"] = None
    with pytest.raises(ValueError, match="page64 specialization"):
        run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["indexer_adaptation_triton"]["page64"] = [{}]
    run_contract.validate_runtime_participation(result)


def test_legacy_declaration_cannot_claim_new_fused_entry():
    result = _large_q1_result()
    del result["backend_provenance"]["official_echo_prefetch_adapter"]["preparation"]
    with pytest.raises(ValueError, match="selected source path"):
        run_contract.validate_runtime_participation(result)


@pytest.mark.parametrize("extend,chunk", [(2, 1), (1025, 1024)])
def test_chunked_extend_q1_binds_only_observed_fused_entry(extend, chunk):
    result = _large_q1_result()
    result.update(extend_tokens=extend, extend_chunk_size=chunk, slots=65536 + extend)
    run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["q1_prefetch_preparation"] = None
    with pytest.raises(ValueError, match="page64 specialization"):
        run_contract.validate_runtime_participation(result)
    result["execution_runtime_artifacts"]["indexer_adaptation_triton"]["page64"] = [{}]
    run_contract.validate_runtime_participation(result)
