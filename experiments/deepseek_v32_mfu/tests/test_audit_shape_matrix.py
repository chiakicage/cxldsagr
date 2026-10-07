"""Reject corrupted measurement evidence before matrix publication."""

import copy
import json

import pytest

from experiments.deepseek_v32_mfu.src import audit_shape_matrix as audit


def panel_fixture():
    rows = []
    for index, (lane, start, end) in enumerate(
        [("Compute", 0, 3), ("Compute + IO", 2, 5), ("IO", 4, 8), ("GPU control", 9, 10)]
    ):
        rows.append(
            {
                "lane": lane,
                "kind": "kernel",
                "name": f"node_{index}",
                "raw_start_ns": start * 1_000_000,
                "raw_end_ns": end * 1_000_000,
                "start_ns": start * 1_000_000,
                "end_ns": end * 1_000_000,
                "process": 7,
                "device_id": 0,
                "stream": index,
                "correlation": index,
                "bytes": None,
                "graph_id": 3,
                "graph_node_id": index,
            }
        )
    return {
        "rows": rows,
        "window": {
            "start_ns": 0,
            "end_ns": 10_000_000,
            "window_ms": 10.0,
            "compute_union_ms": 3.0,
            "fused_union_ms": 3.0,
            "io_union_ms": 4.0,
            "compute_io_union_ms": 8.0,
            "gpu_idle_ms": 1.0,
            "control_only_ms": 1.0,
            "gap_ms": 2.0,
            "gap_percent": 20.0,
            "pure_io_only_ms": 3.0,
            "non_io_window_ms": 7.0,
            "gap_no_io_percent": 200 / 7,
            "unresolved_gather_count": 0,
            "gate_certifiable": True,
        },
    }


def native_fixture(panel):
    return {
        "gpu": [
            {
                **row,
                "start": row["raw_start_ns"],
                "end": row["raw_end_ns"],
                "device": row["device_id"],
            }
            for row in panel["rows"]
        ]
    }


def test_fused_work_keeps_overlap_in_denominator():
    panel = panel_fixture()
    result = audit.window_inventory(native_fixture(panel), panel)
    assert result["native_gpu_intervals"] == 4
    assert result["non_io_window_ms"] == 7.0
    assert result["gap_no_io_percent"] == pytest.approx(200 / 7)


@pytest.mark.parametrize("field", ["name", "correlation", "graph_node_id", "bytes"])
def test_native_identity_tampering_is_rejected(field):
    panel = panel_fixture()
    native = native_fixture(panel)
    panel["rows"][0][field] = "changed" if field == "name" else 999
    with pytest.raises(ValueError, match="native GPU interval"):
        audit.window_inventory(native, panel)


def test_overlapping_omitted_or_duplicate_kernel_is_rejected():
    panel = panel_fixture()
    native = native_fixture(panel)
    duplicate = copy.deepcopy(native["gpu"][0])
    duplicate["graph_node_id"] = 88
    native["gpu"].append(duplicate)
    # The missing activity lies wholly inside another compute interval, so all
    # union totals still match. Native multiset comparison must catch omission.
    with pytest.raises(ValueError, match="native GPU interval"):
        audit.window_inventory(native, panel)


def test_gap_arithmetic_tampering_is_rejected():
    panel = panel_fixture()
    panel["window"]["pure_io_only_ms"] = 4.0
    with pytest.raises(ValueError, match="pure_io_only_ms"):
        audit.window_arithmetic(panel)


def test_input_mutation_after_hash_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"accepted": true}')
    evidence = audit.Evidence()
    evidence.read(path)
    path.write_text('{"accepted":false}')
    with pytest.raises(ValueError, match="Audit input changed"):
        evidence.verify()


def test_changed_archived_or_current_source_is_rejected(tmp_path, monkeypatch):
    root, run = tmp_path / "repo", tmp_path / "run"
    root.mkdir()
    (run / "source").mkdir(parents=True)
    (root / "operator.py").write_text("valid\n")
    (run / "source" / "operator.py").write_text("valid\n")
    digest = audit.Evidence().digest(root / "operator.py")
    manifest = {"operator.py": digest}
    (run / "sources.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, "ROOT", root)
    assert (
        audit.source_audit(run, {"source_sha256": manifest}, audit.Evidence())["execution_files"]
        == 1
    )
    (root / "operator.py").write_text("changed\n")
    with pytest.raises(ValueError, match="Current execution source changed"):
        audit.source_audit(run, {"source_sha256": manifest}, audit.Evidence())


def test_current_hardware_timeout_can_differ_only_by_the_exact_line(tmp_path, monkeypatch):
    root, run = tmp_path / "repo", tmp_path / "run"
    name = audit.shape_matrix_sources.HARDWARE_SOURCE
    archived, current = run / "source" / name, root / name
    archived.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    archived.write_bytes(audit.shape_matrix_sources._PROBE_LINES[20])
    current.write_bytes(audit.shape_matrix_sources._PROBE_LINES[120])
    manifest = {name: audit.Evidence().digest(archived)}
    (run / "sources.json").write_text(json.dumps(manifest))
    result = {"source_sha256": manifest, "hardware": {"hardware_source_sha256": manifest[name]}}
    monkeypatch.setattr(audit, "ROOT", root)
    evidence = audit.Evidence()
    snapshot = audit.source_audit(run, result, evidence)["snapshot"]
    assert snapshot["hardware_probe"]["nvidia_smi_timeout_seconds"] == 20
    assert snapshot["current_hardware_probe"]["nvidia_smi_timeout_seconds"] == 120
    assert str(archived) in evidence.hashes and str(current) in evidence.hashes
    current.write_bytes(current.read_bytes() + b"# unrelated change\n")
    with pytest.raises(ValueError, match="differs beyond"):
        audit.source_audit(run, result, audit.Evidence())


def timing_fixture():
    result = {
        "correctness": {},
        "num_layers": 3,
        "repeats": 3,
        "prefill_repeats": 3,
        "measurements": {
            method: {
                "prefill_samples_ms": [3.0, 1.0, 2.0],
                "prefill_median_ms": 2.0,
                "extend_samples_ms": [0.3, 0.1, 0.2],
                "extend_median_ms": 0.2,
            }
            for method in audit.METHODS
        },
    }
    for method, row in result["measurements"].items():
        for phase, final_key in (
            ("prefill", "prefix_cache_per_layer"),
            ("extend", "extend_cache_per_layer"),
        ):
            samples = []
            for sample in range(3):
                prefetched = sample if method == "echo" else 0
                recalled = 10 - prefetched if method != "hbm" else 0
                written = 128 if method != "hbm" else 0
                layers = [
                    {
                        "prefetched_records": prefetched,
                        "recalled_records": recalled,
                        "evicted_records": 0,
                        "prefetch_capacity_failures": 0,
                        "host_to_device_bytes": (prefetched + recalled) * 1152,
                        "device_to_host_bytes": written * 1152,
                        "host_written_records": written,
                        "record_bytes": 1152,
                    }
                    for _ in range(3)
                ]
                samples.append(layers)
            row[phase + "_cache_samples"] = samples
            row[final_key] = copy.deepcopy(samples[-1])
    return result


def test_timing_median_and_nonfinite_sample_are_rejected():
    result = timing_fixture()
    assert len(audit.timing_audit(result)) == 8
    result["measurements"]["hbm"]["extend_median_ms"] = 0.3
    with pytest.raises(ValueError, match="Timing median"):
        audit.timing_audit(result)
    result = timing_fixture()
    result["measurements"]["echo"]["prefill_samples_ms"][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid timing sample"):
        audit.timing_audit(result)


def test_tensor_reread_rejects_wrong_shape_and_nonfinite():
    import torch

    with pytest.raises(ValueError, match="shape differs"):
        audit.tensor_record(torch.zeros(128, 7168), "echo/hidden", 256)
    value = torch.zeros(1, 129280)
    value[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.tensor_record(value, "echo/logits", 128)
    assert not torch.cuda.is_initialized()


def test_prefix_identity_uses_tokens_and_excludes_candidate(tmp_path):
    request = {
        "stable_prefix_tokens": 3,
        "candidate_suffix_tokens": 1,
        "input_ids": [11, 22, 33, 44],
        "history_sha256": "declared-text-hash",
        "history_token_span": [1, 3],
        "candidate_token_span": [3, 4],
    }
    result = {"prefix_tokens": 3, "extend_tokens": 1}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    original = audit.request_prefix(tmp_path, result, audit.Evidence())
    request["input_ids"][-1] = 99
    path.write_text(json.dumps(request))
    changed_candidate = audit.request_prefix(tmp_path, result, audit.Evidence())
    assert original == changed_candidate
    request["input_ids"][0] = 99
    path.write_text(json.dumps(request))
    changed_instruction = audit.request_prefix(tmp_path, result, audit.Evidence())
    assert original["prefix_token_ids_sha256"] != changed_instruction["prefix_token_ids_sha256"]


def test_rejects_history_identity_change_across_extend_sizes():
    rows = [
        {
            "prefix_tokens": 4096,
            "extend_tokens": extend,
            "request_prefix": {"prefix_token_ids_sha256": "same-history"},
        }
        for extend in (128, 256)
    ]
    identities = audit.history_identity_audit(rows)
    assert identities[0]["same_prefix_token_ids_sha256"] is True
    assert identities[0]["extend_tokens"] == [128, 256]
    rows[1]["request_prefix"]["prefix_token_ids_sha256"] = "changed-history"
    with pytest.raises(ValueError, match="History prefix token IDs differ"):
        audit.history_identity_audit(rows)


def test_manifest_receipt_hash_binds_file_not_signed_payload(tmp_path):
    path = tmp_path / "receipt.json"
    payload = {"checks": {"passed": True}}
    payload_sha = audit.identity_digest(payload)
    path.write_text(json.dumps({**payload, "receipt_sha256": payload_sha}, indent=2) + "\n")
    file_sha = audit.Evidence().digest(path)
    assert file_sha != payload_sha
    audit.manifest_receipt_binding({"validation_receipt_sha256": file_sha}, path, audit.Evidence())
    with pytest.raises(ValueError, match="Manifest check receipt SHA differs"):
        audit.manifest_receipt_binding(
            {"validation_receipt_sha256": payload_sha}, path, audit.Evidence()
        )
    path.write_text(json.dumps({**payload, "receipt_sha256": payload_sha}))
    with pytest.raises(ValueError, match="Manifest check receipt SHA differs"):
        audit.manifest_receipt_binding(
            {"validation_receipt_sha256": file_sha}, path, audit.Evidence()
        )


def test_each_timing_sample_requires_matching_counter_and_byte_evidence():
    result = timing_fixture()
    rows = audit.timing_audit(result)
    echo = next(row for row in rows if (row["method"], row["phase"]) == ("echo", "extend"))
    assert [sample[0]["prefetched_records"] for sample in echo["cache_samples"]] == [0, 1, 2]
    result["measurements"]["echo"]["extend_cache_samples"][0][0]["host_to_device_bytes"] += 1152
    with pytest.raises(ValueError, match="traffic differs"):
        audit.timing_audit(result)
    result = timing_fixture()
    result["measurements"]["hbm"]["prefill_cache_samples"].pop()
    with pytest.raises(ValueError, match="sample coverage"):
        audit.timing_audit(result)


def prefetch_fixture(tmp_path):
    import torch

    from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as transitions
    from experiments.deepseek_v32_mfu.tests.test_prefetch_transition_audit import layer_fixture

    layers = [copy.deepcopy(layer_fixture()) for _ in range(3)]
    for index, layer in enumerate(layers):
        layer["layer"] = index
    raw = {
        "schema_version": 1,
        "method": "echo",
        "residency": "cold",
        "single_session": True,
        "layers": layers,
    }
    compact = transitions.compact_evidence(raw)
    proof = transitions.validate_execution(compact)
    first = layers[0]
    scope = {
        key: proof["scope"][key]
        for key in (
            "method",
            "residency",
            "single_session",
            "num_layers",
            "H",
            "A",
            "slots",
            "max_prefetch",
        )
    }
    final = {
        "length": first["H"] + first["A"],
        "layers": [layer["final_cache_state"] for layer in layers],
    }
    result = {
        "prefix_tokens": first["H"],
        "extend_tokens": first["A"],
        "slots": first["slots"],
        "extend_residency": "cold",
        "extend_graph_checks": {
            "echo": {
                "bounded_prefetch": {},
                "baseline_cache": final,
                "changed_baseline_cache": final,
            }
        },
    }
    receipt = {"artifacts": {}}
    for label in audit.PREFETCH_LABELS:
        evidence_name = f"echo_{label}_prefetch_evidence.pt"
        path = tmp_path / evidence_name
        torch.save(compact, path)
        accepted = {
            "schema": "cold-echo-stage-acceptance-v1",
            "passed": True,
            "scope": scope,
            "proof": proof,
            "evidence_file": evidence_name,
            "evidence_sha256": audit.Evidence().digest(path),
            "final_cache_state_sha256": audit.identity_digest(final),
            "indices": [transitions.tensor_identity(layer["indices"]) for layer in layers],
            "scores": [transitions.tensor_identity(layer["scores"]) for layer in layers],
            "initial_hints": [
                transitions.tensor_identity(layer["initial_hint"]) for layer in layers
            ],
        }
        result["extend_graph_checks"]["echo"]["bounded_prefetch"][label] = accepted
    rebind_prefetch(tmp_path, result, receipt)
    return result, receipt


def rebind_prefetch(directory, result, receipt):
    for label, accepted in result["extend_graph_checks"]["echo"]["bounded_prefetch"].items():
        accepted["evidence_sha256"] = audit.Evidence().digest(directory / accepted["evidence_file"])
        name = f"echo_{label}_prefetch_receipt.json"
        (directory / name).write_text(json.dumps(accepted))
        for filename in (name, accepted["evidence_file"]):
            path = directory / filename
            receipt["artifacts"][filename] = {
                "path": filename,
                "sha256": audit.Evidence().digest(path),
                "bytes": path.stat().st_size,
            }


def test_prefetch_compact_reread_binds_all_six_executions_and_rejects_corrupt_transition(tmp_path):
    import torch

    result, receipt = prefetch_fixture(tmp_path)
    verified = audit.prefetch_audit(tmp_path, result, receipt, audit.Evidence())
    assert verified["passed"] and verified["compact_evidence_files"] == 6
    assert "raw scores and KV payloads are not retained" in verified["boundary"]
    path = tmp_path / "echo_replay_0_prefetch_evidence.pt"
    compact = torch.load(path, weights_only=True)
    compact["layers"][0]["stages"]["after_recall"]["clock"] += 1
    torch.save(compact, path)
    rebind_prefetch(tmp_path, result, receipt)
    with pytest.raises(ValueError, match="clock transitions"):
        audit.prefetch_audit(tmp_path, result, receipt, audit.Evidence())


def test_prefetch_omitted_score_identity_and_receipt_are_not_replaceable(tmp_path):
    result, receipt = prefetch_fixture(tmp_path)
    result["extend_graph_checks"]["echo"]["bounded_prefetch"]["default_graph"]["scores"][0][
        "sha256"
    ] = "0" * 64
    rebind_prefetch(tmp_path, result, receipt)
    with pytest.raises(ValueError, match="index/score/hint identity"):
        audit.prefetch_audit(tmp_path, result, receipt, audit.Evidence())
