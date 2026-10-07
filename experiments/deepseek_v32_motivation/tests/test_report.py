import hashlib
import json
import shutil

import pytest
import torch

from experiments.deepseek_v32_motivation.src.measure import (
    LEGACY_WARMUP_POLICY,
    SCHEMA,
    SCHEMES,
    configuration,
    parser,
    warmup_stages,
    write_json,
)
from experiments.deepseek_v32_motivation.src.report import (
    audit_run,
    digest,
    summarize,
    write_report,
)
from GR.workload import token_sha256


@pytest.fixture
def saved_run(tmp_path, request):
    parameters = getattr(request, "param", {})
    users = parameters.get("users", 2)
    pool_tokens = parameters.get("pool_tokens", 2048)
    config = configuration(
        parser().parse_args(
            [
                "--run-id",
                "fixture",
                "--num-users",
                str(users),
                "--history-tokens",
                "2048",
                "--candidate-tokens",
                "2",
                "--chunk-size",
                "32",
                "--sparse-pool-tokens",
                str(pool_tokens),
                "--host-arena-tokens",
                str(users * 2048),
            ]
        )
    )
    if parameters.get("legacy", False):
        config.update(
            warmup_policy=LEGACY_WARMUP_POLICY,
            warmup_requests_per_scheme=3,
            warmup_request_indices=[0, 1, users],
        )
    (tmp_path / "source").mkdir()
    (tmp_path / "source/example.py").write_text("fixture source\n")
    source = {"example.py": digest(tmp_path / "source/example.py")}
    write_json(tmp_path / "source_manifest.json", source)
    requests = []
    for index in range(users * 2):
        ids = [index % users + 1] * 2048 + [10 + index, 20 + index]
        requests.append(
            {
                "request_id": index,
                "user_id": index % users,
                "visit_index": index // users,
                "is_revisit": index >= users,
                "input_ids": ids,
                "input_sha256": token_sha256(ids),
            }
        )
    identity = {
        "config": config,
        "heat_sha256": None,
        "tokenizer_sha256": "fixture",
        "requests": [
            {key: value for key, value in row.items() if key != "input_ids"} for row in requests
        ],
    }
    workload_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    (tmp_path / "workload").mkdir()
    (tmp_path / "workload/requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests)
    )
    write_json(tmp_path / "workload/workload.json", {**identity, "workload_sha256": workload_id})
    memory = {
        "torch_allocated_bytes": 100,
        "torch_reserved_bytes": 200,
        "torch_peak_allocated_bytes": 100,
        "torch_peak_reserved_bytes": 200,
        "cuda_free_bytes": 800,
        "cuda_total_bytes": 1000,
    }
    rows = []
    cases = []
    warmup_traces = {}
    first_users = config["warmup_requests_per_scheme"] - 1
    warmup_fits = first_users * 2048 <= pool_tokens
    for scheme in SCHEMES:
        warmup_traces[scheme] = []
        for index, stage in zip(
            config["warmup_request_indices"],
            warmup_stages(first_users),
            strict=True,
        ):
            miss = (
                index == users
                and scheme != "hbm"
                and (scheme == "dense_prefetch" or not warmup_fits or parameters.get("legacy"))
            )
            warmup_traces[scheme].append(
                {
                    **{key: value for key, value in requests[index].items() if key != "input_ids"},
                    "stage": stage,
                    "prefix_cache_hit": index == users and (scheme != "hbm" or warmup_fits),
                    "candidate_persistence": "gpu_transient",
                    "retained_length": 2048,
                    "host_to_device_bytes": 1152 if miss else 0,
                    "device_to_host_bytes": 0,
                    "prefetched_records": 0,
                    "recalled_records": 1 if miss else 0,
                }
            )
        cases.append(
            {
                "scheme": scheme,
                "requests": users * 2,
                "warmup_requests": config["warmup_requests_per_scheme"],
                "warmup_request_ids": config["warmup_request_indices"],
                "started_empty": True,
                "torch_peak_allocated_bytes": 100,
                "torch_peak_reserved_bytes": 200,
            }
        )
        (tmp_path / "numerical" / scheme).mkdir(parents=True)
        for workload_request in requests:
            index = workload_request["request_id"]
            relative = f"numerical/{scheme}/{index:06d}.pt"
            torch.save(
                {
                    "request_id": index,
                    "input_sha256": workload_request["input_sha256"],
                    "hidden": torch.full((2, 3), index, dtype=torch.bfloat16),
                    "logits": torch.full((1, 5), index, dtype=torch.float32),
                },
                tmp_path / relative,
            )
            hbm = scheme == "hbm"
            resident_users = pool_tokens // 2048 if hbm else users
            prefix_hit = index >= users and resident_users >= users
            metrics = {key: value for key, value in workload_request.items() if key != "input_ids"}
            metrics.update(
                scheme=scheme,
                run_id="fixture",
                round_index=index // users,
                prefix_cache_hit=prefix_hit,
                prefix_hit_tier=("hbm" if hbm else "dram") if prefix_hit else "miss",
                stable_prefix_tokens=2048,
                candidate_suffix_tokens=2,
                resource_mode="fixed_pools",
                hbm_budget_bytes=None,
                dram_budget_bytes=None,
                cached_users=min(index + 1, users, resident_users),
                evicted_users=[],
                latency_ms=10,
                admission_ms=1,
                prefix_ms=2,
                extend_ms=5,
                cleanup_ms=2,
                cache_hbm_bytes=100,
                cache_dram_bytes=0 if hbm else 200,
                shared_cache_hbm_bytes=10,
                shared_cache_dram_bytes=0 if hbm else 20,
                cache_diagnostics={
                    "host_to_device_bytes": 0 if hbm else 256,
                    "device_to_host_bytes": 0,
                    "selection_records": 4 if index % 2 == 0 else 12,
                    "resident_selection_records": 4,
                },
                diagnostics_scope="candidate_only",
                workload_sha256=workload_id,
                memory_before=dict(memory),
                memory_after=dict(memory),
                numerical={"hidden": {"exact": True}, "logits": {"exact": True}},
                output_file=relative,
                output_sha256=digest(tmp_path / relative),
            )
            rows.append(metrics)
    (tmp_path / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(tmp_path / "memory.json", [memory])
    write_json(
        tmp_path / "metadata.json",
        {
            "schema": SCHEMA,
            "run_id": "fixture",
            "status": "accepted",
            "config": config,
            "cases": cases,
            "warmup_traces": warmup_traces,
            "workload_sha256": workload_id,
            "source_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest(),
            "model_dimensions": {"hidden": 3, "vocabulary": 5},
            "hardware": {"name": "fixture", "torch": "fixture", "cuda": "fixture"},
        },
    )
    return tmp_path


def replace_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_independent_check_omits_request_timings(saved_run):
    from experiments.deepseek_v32_motivation.src.measure import CHECK_SCHEMA
    from experiments.nosa_motivation.src.validation import without_performance

    metadata = json.loads((saved_run / "metadata.json").read_text())
    metadata.update(schema=CHECK_SCHEMA, mode="check")
    write_json(saved_run / "metadata.json", metadata)
    rows = [
        without_performance(json.loads(line))
        for line in (saved_run / "measurements.jsonl").read_text().splitlines()
    ]
    replace_rows(saved_run / "measurements.jsonl", rows)
    _, rows, audit = audit_run(saved_run)
    assert audit["all_candidate_hidden_and_logits_exact"]
    assert all("latency_ms" not in row for row in rows)
    with pytest.raises(ValueError, match="check runs do not publish"):
        write_report(saved_run, saved_run / "check_report")


def test_clean_bench_uses_external_check_without_loading_outputs(
    saved_run, tmp_path_factory, monkeypatch
):
    from evaluation.validation import write_receipt
    from experiments.deepseek_v32_motivation.src.measure import BENCH_SCHEMA, RECEIPT_KIND
    from experiments.nosa_motivation.src.validation import base_identity

    check_dir = tmp_path_factory.mktemp("deepseek_check")
    shutil.copytree(saved_run, check_dir, dirs_exist_ok=True)
    metadata = json.loads((saved_run / "metadata.json").read_text())
    metadata.update(
        schema=BENCH_SCHEMA,
        mode="bench",
        execution_environment={},
        checkpoint={"identity_boundary": "CPU fixture"},
        precision_settings={},
    )
    identity = {
        "base": base_identity(metadata, saved_run),
        "methods": {name: {"cpu_fixture_only": True} for name in SCHEMES},
    }
    receipt = check_dir / "receipt.json"
    write_receipt(
        receipt,
        kind=RECEIPT_KIND,
        identity=identity,
        checks={"passed": True},
        artifacts={"checked_rows": check_dir / "measurements.jsonl"},
    )
    metadata["validation_identity"] = identity
    metadata["correctness_receipt"] = {
        "path": str(receipt),
        "sha256": digest(receipt),
        "kind": RECEIPT_KIND,
        "identity": identity,
    }
    write_json(saved_run / "metadata.json", metadata)
    rows = [
        {
            key: value
            for key, value in json.loads(line).items()
            if key not in {"numerical", "output_file", "output_sha256"}
        }
        for line in (saved_run / "measurements.jsonl").read_text().splitlines()
    ]
    replace_rows(saved_run / "measurements.jsonl", rows)
    shutil.rmtree(saved_run / "numerical")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("bench loaded tensors"))
    _, _, audit = audit_run(saved_run)
    assert not audit["all_candidate_hidden_and_logits_exact"]
    assert audit["independent_correctness_receipt"]["checks"]["passed"]
    write_report(saved_run, saved_run / "bench_report")
    metadata["config"]["candidate_tokens"] += 1
    write_json(saved_run / "metadata.json", metadata)
    with pytest.raises(ValueError, match="identity differs"):
        audit_run(saved_run)


def test_report_rechecks_all_outputs_and_uses_weighted_hit_ratio(saved_run):
    metadata, rows, audit = audit_run(saved_run)
    assert audit["checked_requests"] == 16
    assert audit["compared_offload_requests"] == 12
    assert audit["checked_warmup_requests"] == 12
    assert audit["host_recall_warmed_schemes"] == ["echo", "serial_sparse", "dense_prefetch"]
    summary = summarize(rows)
    assert len(summary) == 8
    assert summary[0]["candidate_consumer_union_resident_ratio"] == 0.5
    hbm_revisit = summary[1]
    assert hbm_revisit["requests"] == 2 and hbm_revisit["prefix_hits"] == 0
    assert summary[3]["prefix_hits"] == 2
    write_report(saved_run, saved_run / "report")
    report = (saved_run / "report/results.md").read_text()
    assert "只覆盖 candidate forward" in report
    assert "被淘汰后的请求仍计为复访" in report
    assert "allocated" in report and "reserved" in report
    assert "共 3 次预热请求" in report and "recalled_records>0" in report
    assert metadata["config"]["byte_subbudgets"] is None


@pytest.mark.parametrize(
    "saved_run,warmup_count,recalled_schemes",
    [
        ({"users": 4, "pool_tokens": 8192}, 20, ["dense_prefetch"]),
        ({"users": 8, "pool_tokens": 8192}, 24, list(SCHEMES[1:])),
        ({"users": 2, "pool_tokens": 2048, "legacy": True}, 12, list(SCHEMES[1:])),
    ],
    indirect=["saved_run"],
)
def test_report_audits_actual_capacity_dependent_and_legacy_warmup(
    saved_run, warmup_count, recalled_schemes
):
    metadata, _, audit = audit_run(saved_run)
    assert audit["warmup_policy"] == metadata["config"]["warmup_policy"]
    assert audit["checked_warmup_requests"] == warmup_count
    assert audit["host_recall_warmed_schemes"] == recalled_schemes
    write_report(saved_run, saved_run / "report")
    report = (saved_run / "report/results.md").read_text()
    assert f"共 {warmup_count // len(SCHEMES)} 次预热请求" in report
    assert f"最后一次预热中，{', '.join(recalled_schemes)} 同时满足" in report


def test_report_rejects_incomplete_matrix(saved_run):
    path = saved_run / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    replace_rows(path, rows[:-1])
    with pytest.raises(ValueError, match="incomplete measurement"):
        audit_run(saved_run)


@pytest.mark.parametrize("defect", ["old_count", "wrong_user", "prefetch_without_recall"])
def test_report_rejects_unverified_host_miss_warmup(saved_run, defect):
    path = saved_run / "metadata.json"
    metadata = json.loads(path.read_text())
    if defect == "old_count":
        metadata["cases"][1]["warmup_requests"] = 2
    elif defect == "wrong_user":
        metadata["warmup_traces"]["echo"][1]["user_id"] = 0
    else:
        metadata["warmup_traces"]["echo"][2]["recalled_records"] = 0
    write_json(path, metadata)
    with pytest.raises(ValueError, match="warmup|initialization"):
        audit_run(saved_run)


def test_report_rejects_corrupt_saved_output(saved_run):
    path = saved_run / "numerical/echo/000000.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="output bytes changed"):
        audit_run(saved_run)


def test_report_checks_logits_even_if_recorded_status_says_exact(saved_run):
    path = saved_run / "numerical/echo/000000.pt"
    payload = torch.load(path, weights_only=True)
    payload["logits"][0, 2] = 3
    torch.save(payload, path)
    row_path = saved_run / "measurements.jsonl"
    rows = [json.loads(line) for line in row_path.read_text().splitlines()]
    rows[4]["output_sha256"] = digest(path)
    replace_rows(row_path, rows)
    with pytest.raises(AssertionError):
        audit_run(saved_run)


def test_report_rejects_modified_source_or_peak(saved_run):
    path = saved_run / "source/example.py"
    original = path.read_bytes()
    path.write_text("different implementation\n")
    with pytest.raises(ValueError, match="source changed"):
        audit_run(saved_run)
    path.write_bytes(original)
    metadata = json.loads((saved_run / "metadata.json").read_text())
    metadata["cases"][0]["torch_peak_reserved_bytes"] += 1
    write_json(saved_run / "metadata.json", metadata)
    with pytest.raises(ValueError, match="memory peak differs"):
        audit_run(saved_run)
