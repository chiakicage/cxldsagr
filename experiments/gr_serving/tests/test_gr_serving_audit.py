"""Independent acceptance gates, exercised with small CPU-only fixtures."""

import csv
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.gr_serving.src import audit


def small_contract():
    return audit.RunContract(("nosa",), (3,), 6, 2, 2, 1, 49, 100, 10)


def run_parameters(**overrides):
    return {
        "run_id": "unit",
        "models": ["nosa"],
        "users": [1, 8, 32, 64, 128, 256, 512],
        "requests": 32,
        "max_revisits": 8,
        "history_tokens": 4096,
        "candidate_tokens": 128,
        "seed": 42,
        "hbm_budget_bytes": 4 * 2**30,
        "dram_budget_bytes": 16 * 2**30,
        "hbm_budget_gib": 4.0,
        "dram_budget_gib": 16.0,
        "warmup": 2,
        "chunk_size": 1024,
        "atol": 0.0,
        "rtol": 0.0,
        "device": "cuda:0",
        **overrides,
    }


def metadata_fixture(**overrides):
    params = run_parameters(**overrides)
    contract = audit.RunContract.from_parameters(params)
    return {
        "status": "accepted",
        "schema_version": 1,
        "run_id": "unit",
        "source_sha256": "d" * 64,
        "measured_requests": contract.measured_requests,
        "started_unix": 1.0,
        "completed_unix": 2.0,
        "parameters": params,
        "hardware": {
            "device": "cuda:0",
            "compute_capability": [9, 0],
            "total_memory_bytes": 100 * 2**30,
        },
        "models": {
            "nosa": {
                "layers": 32,
                "dtype": "torch.bfloat16",
                "device": "cuda:0",
                "prefix_chunk_size": params["chunk_size"],
                "max_seq_len": params["history_tokens"] + params["candidate_tokens"],
                "output": "all candidate normalized hidden states; no LM head or decode",
            }
        },
        "cases": [
            {
                "model": model,
                "scheme": scheme,
                "num_users": users,
                "requests": contract.request_counts[users],
                "all_hidden_exact": True,
                "max_abs": 0.0,
                "duration_seconds": 1.0,
                "gpu_peak_allocated_bytes": 1000,
                "gpu_peak_reserved_bytes": 2000,
                "session_reservation": {"hbm": 10, "dram": 0 if scheme == "hbm" else 5},
            }
            for model, scheme, users in contract.cases
        ],
    }


def fixture(directory, *, contract=None, trace=None, weights=None):
    contract = small_contract() if contract is None else contract
    trace = [0, 1, 0, 2, 1, 0] if trace is None else trace
    users = contract.populations[0]
    model = contract.models[0]
    weights = {uid: 1.0 for uid in range(users)} if weights is None else weights
    history, candidate_length = contract.history_tokens, contract.candidate_tokens
    rows, visits, previous = [], Counter(), {}
    for rid, uid in enumerate(trace):
        prefix = [uid + 10] + [20] * (history - 1)
        candidate = [rid + 30] * candidate_length
        ids = prefix + candidate
        rows.append(
            {
                "model": model,
                "request_id": rid,
                "user_id": uid,
                "user_heat_weight": weights[uid],
                "visit_index": visits[uid],
                "visit_number": visits[uid] + 1,
                "is_revisit": visits[uid] > 0,
                "timestamp": float(rid),
                "stable_prefix_tokens": history,
                "candidate_suffix_tokens": candidate_length,
                "total_input_tokens": len(ids),
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "prefix_sha256": audit.token_hash(prefix),
                "candidate_sha256": audit.token_hash(candidate),
                "input_sha256": audit.token_hash(ids),
                "previous_request_id": previous.get(uid),
                "revisit_interval_s": None if uid not in previous else rid - previous[uid],
            }
        )
        visits[uid] += 1
        previous[uid] = rid
    config = {
        "model": model,
        "num_users": users,
        "requests": contract.request_cap,
        "history_tokens": history,
        "candidate_tokens": candidate_length,
        "seed": contract.seed,
        "sampling": contract.sampling,
        "heat_dataset": contract.heat_dataset,
        "heat_field": contract.heat_field,
        "max_revisits": contract.max_revisits,
        "context_limit": history + candidate_length if history + candidate_length > 32768 else None,
    }
    identity = {
        "config": config,
        "heat_sha256": None if contract.sampling == "sequential" else "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "requests": [{name: row[name] for name in audit.IDENTITY_FIELDS} for row in rows],
    }
    manifest = {
        "schema_version": 1,
        **identity,
        "workload_sha256": audit.hashlib.sha256(audit.canonical(identity)).hexdigest(),
        "access_trace_sha256": audit.hashlib.sha256(
            audit.canonical(
                [{name: row[name] for name in audit.ACCESS_TRACE_FIELDS} for row in rows]
            )
        ).hexdigest(),
        "heat": {
            "source": "explicit_synthetic_ids",
            "synthetic": True,
            "selected_users": users,
            "selected_weight_sum": float(users),
            "user_identity": "synthetic IDs 0..N-1",
            "weight_semantics": (
                "equal placeholders required by GR; no heat curve or probabilistic sampling"
            ),
        }
        if contract.sampling == "sequential"
        else {
            "sha256": "a" * 64,
            "dataset": contract.heat_dataset,
            "field": contract.heat_field,
            "selected_users": users,
            "seed": contract.seed,
            "selected_weight_sum": sum(weights.values()),
        },
        "schedule": {
            "seed": contract.seed,
            "qps": 1,
            "arrival": "constant",
            "sampling": contract.sampling,
            "start_timestamp": 0.0,
            "max_revisits": contract.max_revisits,
        },
        "context": {
            "format_default_tokens": 32768,
            "effective_limit_tokens": max(32768, history + candidate_length),
            "override_explicit": history + candidate_length > 32768,
            "interpretation": "generation boundary only; does not validate model quality",
        },
        "observed": {
            "requests": len(trace),
            "unique_users": len(visits),
            "first_visits": len(visits),
            "revisits": len(trace) - len(visits),
            "max_visits": max(visits.values(), default=0),
            "max_revisits": max((count - 1 for count in visits.values()), default=0),
        },
        "users": [
            {
                "user_id": uid,
                "weight": weights[uid],
                "probability": None
                if contract.sampling == "sequential"
                else weights[uid] / sum(weights.values()),
                "visits": visits[uid],
                "prefix_sha256": audit.token_hash([uid + 10] + [20] * (history - 1))
                if visits[uid]
                else None,
            }
            for uid in range(users)
        ],
    }
    write_fixture(directory, manifest, rows)
    return manifest, rows


def write_fixture(directory, manifest, rows):
    (directory / "workload.json").write_text(json.dumps(manifest))
    (directory / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def measurements(manifest, requests):
    # Hand-calculated two-entry LRU trace; deliberately no call to the oracle.
    hits = [False, False, True, False, False, False]
    victims = [[], [], [], [1], [0], [2]]
    rows = []
    for rid, request in enumerate(requests):
        count = min(rid + 1, 2)
        rows.append(
            {
                **request,
                "scheme": "serial_sparse",
                "num_users": 3,
                "run_id": "unit",
                "workload_sha256": manifest["workload_sha256"],
                "history_tokens": 2,
                "candidate_tokens": 1,
                "seed": 49,
                "prefix_cache_hit": hits[rid],
                "prefix_hit_tier": "dram" if hits[rid] else "miss",
                "evicted_users": victims[rid],
                "cached_users": count,
                "latency_ms": 10.0,
                "admission_ms": 1.0,
                "prefix_ms": 1.0,
                "extend_ms": 7.0,
                "cleanup_ms": 1.0,
                "correctness_exact": True,
                "correctness_max_abs": 0.0,
                "hbm_budget_bytes": 100,
                "dram_budget_bytes": 10,
                "reserved_hbm_bytes": count * 10,
                "reserved_dram_bytes": count * 5,
                "cache_hbm_bytes": count * 8,
                "request_cache_hbm_bytes": count * 9,
                "cache_dram_bytes": count * 5,
                "request_cache_dram_bytes": count * 5,
            }
        )
    return rows


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit-unit-")
        self.path = Path(self.temp.name)
        self.manifest, self.requests = fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_workload_accepts_exact_tokens_and_visits(self):
        requests, manifest = audit.audit_workload(self.path, "nosa", 3, small_contract())
        self.assertEqual(len(requests), 6)
        self.assertNotIn("input_ids", requests[0])
        self.assertEqual(manifest["observed"]["revisits"], 3)

    def test_workload_rejects_token_tampering(self):
        self.requests[2]["input_ids"][-1] += 1
        write_fixture(self.path, self.manifest, self.requests)
        with self.assertRaisesRegex(audit.AuditError, "hash"):
            audit.audit_workload(self.path, "nosa", 3, small_contract())

    def test_workload_rejects_manifest_tampering(self):
        self.manifest["requests"][1]["visit_index"] = 4
        write_fixture(self.path, self.manifest, self.requests)
        with self.assertRaisesRegex(audit.AuditError, "signed workload"):
            audit.audit_workload(self.path, "nosa", 3, small_contract())

    def test_workload_rejects_false_first_visit(self):
        self.requests[2]["is_revisit"] = False
        write_fixture(self.path, self.manifest, self.requests)
        with self.assertRaisesRegex(audit.AuditError, "revisit label"):
            audit.audit_workload(self.path, "nosa", 3, small_contract())

    def test_workload_rejects_gapped_visits(self):
        self.requests[2]["visit_index"] = 2
        write_fixture(self.path, self.manifest, self.requests)
        with self.assertRaisesRegex(audit.AuditError, "contiguous visit"):
            audit.audit_workload(self.path, "nosa", 3, small_contract())

    def test_lru_reuse_distance_and_dram_constraint(self):
        result = list(
            audit.lru_oracle(self.requests, {"hbm": 10, "dram": 5}, {"hbm": 100, "dram": 10})
        )
        self.assertEqual([x["capacity"] for x in result], [2] * 6)
        self.assertEqual([x["hit"] for x in result], [False, False, True, False, False, False])
        self.assertEqual([x["evicted"] for x in result], [[], [], [], [1], [0], [2]])

    def test_lru_zero_dram_does_not_limit_capacity(self):
        result = list(
            audit.lru_oracle(self.requests, {"hbm": 20, "dram": 0}, {"hbm": 100, "dram": 0})
        )
        self.assertEqual([x["capacity"] for x in result], [5] * 6)
        self.assertEqual([x["hit"] for x in result], [False, False, True, False, True, True])
        self.assertFalse(any(x["evicted"] for x in result))

    def case_fixture(self):
        requests, manifest = audit.audit_workload(self.path, "nosa", 3, small_contract())
        return requests, manifest, measurements(manifest, requests)

    def check_case(self, requests, manifest, rows):
        case = {"session_reservation": {"hbm": 10, "dram": 5}, "duration_seconds": 0.1}
        return audit.audit_case_rows(
            ("nosa", "serial_sparse", 3),
            rows,
            case,
            requests,
            manifest,
            "unit",
            caps={"hbm": 100, "dram": 10},
        )

    def test_case_acceptance_retains_evicted_revisits(self):
        requests, manifest, rows = self.case_fixture()
        result = self.check_case(requests, manifest, rows)
        self.assertEqual(
            (result["hits"], result["evictions"], result["revisits"], result["revisit_misses"]),
            (1, 3, 3, 2),
        )

    def test_case_rejects_wrong_lru_victim(self):
        requests, manifest, rows = self.case_fixture()
        rows[3]["evicted_users"] = [0]
        with self.assertRaisesRegex(audit.AuditError, "LRU victim"):
            self.check_case(requests, manifest, rows)

    def test_case_rejects_cache_miss_revisit_label(self):
        requests, manifest, rows = self.case_fixture()
        rows[4]["is_revisit"] = False
        with self.assertRaisesRegex(audit.AuditError, "request field is_revisit"):
            self.check_case(requests, manifest, rows)

    def test_case_rejects_boundary_over_budget(self):
        requests, manifest, rows = self.case_fixture()
        rows[4]["request_cache_hbm_bytes"] = 101
        with self.assertRaisesRegex(audit.AuditError, "allocation bound"):
            self.check_case(requests, manifest, rows)

    def test_case_rejects_undercounted_reservation(self):
        requests, manifest, rows = self.case_fixture()
        rows[4]["reserved_hbm_bytes"] = 19
        with self.assertRaisesRegex(audit.AuditError, "reserved capacity"):
            self.check_case(requests, manifest, rows)

    def test_case_rejects_latency_omitted_phase(self):
        requests, manifest, rows = self.case_fixture()
        rows[4]["latency_ms"] = 9
        with self.assertRaisesRegex(audit.AuditError, "phase sum"):
            self.check_case(requests, manifest, rows)

    def correctness_fixture(self):
        row = {
            "model": "nosa",
            "scheme": "overlap",
            "num_users": 1,
            "request_id": 0,
            "correctness_exact": True,
            "correctness_max_abs": 0.0,
        }
        comparison = {
            "model": "nosa",
            "scheme": "overlap",
            "num_users": 1,
            "request_id": 0,
            "shape": [1024, 4096],
            "dtype": "torch.bfloat16",
            "exact": True,
            "max_abs": 0.0,
            "relative_l2": 0.0,
            "atol": 0.0,
            "rtol": 0.0,
        }
        return {audit.key(row, True): row}, comparison

    def test_correctness_rejects_missing_record(self):
        rows, _record = self.correctness_fixture()
        with self.assertRaisesRegex(audit.AuditError, "bijection"):
            audit.audit_correctness([], rows, 1024)

    def test_correctness_rejects_duplicate_record(self):
        rows, record = self.correctness_fixture()
        with self.assertRaisesRegex(audit.AuditError, "duplicate"):
            audit.audit_correctness([record, record], rows, 1024)

    def test_correctness_rejects_last_hidden_only(self):
        rows, record = self.correctness_fixture()
        record["shape"] = [1, 4096]
        with self.assertRaisesRegex(audit.AuditError, "complete hidden shape"):
            audit.audit_correctness([record], rows, 1024)

    def test_correctness_rejects_nonzero_error_and_nonfinite(self):
        rows, record = self.correctness_fixture()
        self.assertEqual(audit.audit_correctness([record], rows, 1024), 1)
        for invalid in (0.01, float("nan"), float("inf")):
            modified = {**record, "relative_l2": invalid}
            with self.assertRaises(audit.AuditError):
                audit.audit_correctness([modified], rows, 1024)

    def test_inclusive_percentiles_and_empty_scope(self):
        self.assertEqual(audit.latency_stats([])["mean_ms"], None)
        self.assertEqual(audit.latency_stats([10])["p95_ms"], 10)
        self.assertEqual(audit.latency_stats([0, 100])["p95_ms"], 95)
        self.assertEqual(audit.latency_stats([0, 100])["p99_ms"], 99)

    def test_json_rejects_duplicates_nonfinite_and_truncation(self):
        for invalid in ('{"x":1,"x":2}', '{"x": NaN}', '{"x": Infinity}'):
            with self.assertRaises(audit.AuditError):
                audit.parse_json(invalid)
        path = self.path / "broken.jsonl"
        path.write_text('{"x":1}')
        with self.assertRaisesRegex(audit.AuditError, "incomplete"):
            list(audit.read_jsonl(path))

    def test_unaccepted_status_fails_before_tensor_loading_and_writes_nothing(self):
        (self.path / "metadata.json").write_text('{"status":"running"}')
        output = self.path / "audit.json"
        with patch.object(
            audit, "audit_references", side_effect=AssertionError("must not load references")
        ):
            status = audit.main([str(self.path), "--json", str(output)])
        self.assertEqual(status, 1)
        self.assertFalse(output.exists())

    def test_cpu_reference_shapes_finite_values_and_file_set(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU tensor check requires repository .venv")
        folder = self.path / "reference/nosa/1"
        folder.mkdir(parents=True)
        tensors = [
            torch.ones((2, 3), dtype=torch.bfloat16),
            torch.zeros((2, 3), dtype=torch.bfloat16),
        ]
        for rid, tensor in enumerate(tensors):
            torch.save(tensor, folder / f"{rid:06d}.pt")
        contract = audit.RunContract(("nosa",), (1,), 2, 1, 2, 2, 42, 100, 10)
        with patch.object(audit, "WIDTHS", {"nosa": 3}):
            result = audit.audit_references(self.path, "cpu", contract)
            self.assertEqual(result["finite_elements_checked"], 12)
            self.assertEqual(result["files"], 2)
            tensors[0][0, 0] = float("nan")
            torch.save(tensors[0], folder / "000000.pt")
            with self.assertRaisesRegex(audit.AuditError, "nonfinite reference"):
                audit.audit_references(self.path, "cpu", contract)
            torch.save(torch.ones((1, 3), dtype=torch.bfloat16), folder / "000000.pt")
            with self.assertRaisesRegex(audit.AuditError, "reference shape"):
                audit.audit_references(self.path, "cpu", contract)
            (folder / "000001.pt").unlink()
            with self.assertRaisesRegex(audit.AuditError, "reference file set"):
                audit.audit_references(self.path, "cpu", contract)

    def analysis_fixture(self):
        _requests, _manifest, rows = self.case_fixture()
        path = self.path / "analysis"
        path.mkdir()
        items = []
        for scope, chosen_ids in (
            ("all", [0, 1, 2, 3, 4, 5]),
            ("first_visit", [0, 1, 3]),
            ("revisit", [2, 4, 5]),
        ):
            chosen = [rows[index] for index in chosen_ids]
            count = len(chosen)

            def constant_stats(value, count=count):
                return {"count": count, **{name: value for name in audit.STATS if name != "count"}}

            item = {
                name: rows[0][name]
                for name in (
                    "run_id",
                    "model",
                    "scheme",
                    "num_users",
                    "workload_sha256",
                    "history_tokens",
                    "candidate_tokens",
                    "seed",
                    "hbm_budget_bytes",
                    "dram_budget_bytes",
                )
            }
            hits = 0 if scope == "first_visit" else 1
            item.update(
                scope=scope,
                **constant_stats(10.0),
                unique_users=2 if scope == "revisit" else 3,
                prefix_hits=hits,
                prefix_misses=count - hits,
                prefix_hit_rate=hits / count,
                prefix_hit_tiers={"miss": count - hits, **({"dram": hits} if hits else {})},
                evictions={"all": 3, "first_visit": 1, "revisit": 2}[scope],
                prefix_ms=constant_stats(1.0),
                extend_ms=constant_stats(7.0),
                cleanup_ms=constant_stats(1.0),
                peak_cache_hbm_bytes=16,
                peak_cache_dram_bytes=10,
            )
            items.append(item)
        (path / "summary.json").write_text(json.dumps({"schema_version": 1, "groups": items}))

        def save_csv(filename, values):
            with (path / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                for row in values:
                    writer.writerow(
                        {
                            name: json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list))
                            else value
                            for name, value in row.items()
                        }
                    )

        save_csv("summary.csv", items)
        save_csv(
            "per_request.csv", [{**row, "prefix_hit": row["prefix_cache_hit"]} for row in rows]
        )
        for filename in ("summary.svg", "per_request.svg"):
            (path / filename).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        return rows, items

    def test_analysis_accepts_paired_csv_and_all_scopes(self):
        rows, _items = self.analysis_fixture()
        result = audit.audit_analysis(
            self.path, {("nosa", "serial_sparse", 3): rows}, audit.indexed(rows, True)
        )
        self.assertEqual(result["summary_groups"], 3)
        self.assertEqual(result["per_request_csv_rows"], 6)

    def test_analysis_rejects_revisit_miss_exclusion(self):
        rows, items = self.analysis_fixture()
        items[-1]["count"] = 1
        (self.path / "analysis/summary.json").write_text(json.dumps({"groups": items}))
        with self.assertRaisesRegex(audit.AuditError, "count"):
            audit.audit_analysis(
                self.path, {("nosa", "serial_sparse", 3): rows}, audit.indexed(rows, True)
            )

    def test_analysis_rejects_truncated_per_request_csv(self):
        rows, _items = self.analysis_fixture()
        path = self.path / "analysis/per_request.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(audit.AuditError, "CSV length"):
            audit.audit_analysis(
                self.path, {("nosa", "serial_sparse", 3): rows}, audit.indexed(rows, True)
            )


@pytest.mark.parametrize("history", [4096, 16384, 65536])
def test_dynamic_metadata_geometry_matrix_budget_and_source(history):
    meta = metadata_fixture(history_tokens=history)
    contract, cases = audit.audit_metadata(meta, "unit", None)
    assert len(cases) == 28
    assert contract.request_counts == {1: 9, 8: 32, 32: 32, 64: 32, 128: 32, 256: 32, 512: 32}
    assert contract.measured_requests == 804
    assert contract.history_tokens == history
    assert contract.candidate_tokens == 128
    assert contract.caps == {"hbm": 4 * 2**30, "dram": 16 * 2**30}
    with pytest.raises(audit.AuditError, match="source identity"):
        audit.audit_metadata(meta, "unit", "e" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"users": [1, 1]},
        {"users": [0]},
        {"users": [True]},
        {"models": ["unknown"]},
        {"models": ["nosa", "nosa"]},
        {"requests": None},
        {"requests": 0},
        {"requests": True},
        {"max_revisits": -1},
        {"max_revisits": True},
    ],
)
def test_dynamic_contract_rejects_invalid_inputs(changes):
    with pytest.raises(audit.AuditError):
        audit.RunContract.from_parameters(run_parameters(**changes))


def test_uncapped_contract_uses_exact_total_without_forcing_user_coverage():
    contract = audit.RunContract.from_parameters(run_parameters(max_revisits=None))
    assert set(contract.request_counts.values()) == {32}
    assert list(audit.scheduler_replay({0: 1.0}, 32, None, 42)) == [0] * 32
    assert list(audit.scheduler_replay({0: 999.0, 1: 1.0}, 5, None, 42)) == [0] * 5


def test_uncapped_workload_authenticates_actual_users_and_revisits(tmp_path):
    contract = replace(
        small_contract(), populations=(8,), request_cap=20, max_revisits=None, seed=42
    )
    weights = {uid: 1.0 if uid else 99999.0 for uid in range(8)}
    manifest, rows = fixture(tmp_path, contract=contract, trace=[0] * 20, weights=weights)
    manifest["observed"]["returning_users"] = 1
    for row in manifest["users"]:
        row["revisits"] = max(0, row["visits"] - 1)
    write_fixture(tmp_path, manifest, rows)
    audit.audit_workload(tmp_path, "nosa", 8, contract)
    manifest["observed"]["returning_users"] = 8
    write_fixture(tmp_path, manifest, rows)
    with pytest.raises(audit.AuditError, match="visit counts"):
        audit.audit_workload(tmp_path, "nosa", 8, contract)


def test_metadata_cannot_redefine_grid_by_omitting_a_case():
    meta = metadata_fixture()
    meta["cases"].pop()
    with pytest.raises(audit.AuditError, match="complete model/scheme/population matrix"):
        audit.audit_metadata(meta, None, None)


def test_metadata_rejects_disagreement_with_backend_geometry():
    meta = metadata_fixture(history_tokens=65536)
    meta["models"]["nosa"]["max_seq_len"] = 65536
    with pytest.raises(audit.AuditError, match="sequence capacity"):
        audit.audit_metadata(meta, None, None)


def test_bounded_sampler_keeps_hot_revisits_without_forcing_cold_first_visits():
    assert list(audit.scheduler_replay({0: 999.0, 1: 1.0}, 5, 8, 42)) == [0] * 5
    assert list(audit.scheduler_replay({0: 1.0}, 32, 8, 42)) == [0] * 9
    assert list(audit.scheduler_replay({0: 1e16, 1: 1.0, 2: 1.0}, 7, 0, 42)) == [0, 1, 2]


def test_bounded_sampler_uses_sorted_ids_and_right_cdf_boundary():
    with patch.object(audit.random.Random, "random", return_value=0.5):
        assert list(audit.scheduler_replay({9: 1.0, 2: 1.0}, 1, 0, 42)) == [9]


def test_recomputed_signature_does_not_hide_an_incorrect_heat_schedule(tmp_path):
    # This is a self-consistent renamed-user trace: tokens, visit counts and
    # signatures all agree, but it disagrees with the seeded heat draws.
    fixture(tmp_path, trace=[1, 0, 1, 2, 0, 1])
    with pytest.raises(audit.AuditError, match="independent seeded heat schedule"):
        audit.audit_workload(tmp_path, "nosa", 3, small_contract())


@pytest.mark.parametrize("field", ["schedule", "observed"])
def test_cap_metadata_cannot_disagree_with_the_trace(tmp_path, field):
    manifest, requests = fixture(tmp_path)
    manifest[field]["max_revisits"] += 1
    write_fixture(tmp_path, manifest, requests)
    with pytest.raises(audit.AuditError, match="trace schedule|visit counts"):
        audit.audit_workload(tmp_path, "nosa", 3, small_contract())


def test_workload_rejects_extra_visits_after_population_exhaustion(tmp_path):
    contract = replace(small_contract(), populations=(1,), request_cap=32, max_revisits=8)
    fixture(tmp_path, contract=contract, trace=[0] * 10)
    with pytest.raises(audit.AuditError, match="identities count"):
        audit.audit_workload(tmp_path, "nosa", 1, contract)


def test_workload_allows_unvisited_cold_users_but_not_shorter_than_requested_trace(tmp_path):
    contract = replace(small_contract(), populations=(8,), request_cap=3, max_revisits=8, seed=42)
    weights = {uid: 1.0 if uid else 999.0 for uid in range(8)}
    manifest, _ = fixture(tmp_path, contract=contract, trace=[0, 0, 0], weights=weights)
    requests, _ = audit.audit_workload(tmp_path, "nosa", 8, contract)
    assert len(requests) == 3
    assert manifest["users"][1]["visits"] == 0
    fixture(tmp_path, contract=contract, trace=[0, 0], weights=weights)
    with pytest.raises(audit.AuditError, match="identities count"):
        audit.audit_workload(tmp_path, "nosa", 8, contract)


@pytest.mark.parametrize("history", [4096, 16384, 65536])
def test_signed_token_geometry_and_explicit_context_override(tmp_path, history):
    contract = replace(
        small_contract(),
        populations=(1,),
        request_cap=2,
        max_revisits=1,
        history_tokens=history,
        candidate_tokens=2,
    )
    manifest, _ = fixture(tmp_path, contract=contract, trace=[0, 0])
    requests, _ = audit.audit_workload(tmp_path, "nosa", 1, contract)
    assert len(requests) == 2
    assert manifest["context"]["override_explicit"] is (history == 65536)


def test_empty_revisit_trace_needs_explicit_allowance_and_null_summary(tmp_path):
    from experiments.gr_serving.src.report import write_report

    contract = replace(small_contract(), populations=(8,), request_cap=2, max_revisits=8, seed=42)
    fixture(tmp_path, contract=contract, trace=[5, 0])
    with pytest.raises(audit.AuditError, match="no revisits without explicit allowance"):
        audit.audit_workload(tmp_path, "nosa", 8, contract)
    contract = replace(contract, allow_empty_revisits=True)
    requests, manifest = audit.audit_workload(tmp_path, "nosa", 8, contract)
    rows = measurements(manifest, requests)
    for row in rows:
        row.update(num_users=8, seed=42)
    write_report(rows, tmp_path / "analysis")
    groups = {("nosa", "serial_sparse", 8): rows}
    audit.audit_analysis(tmp_path, groups, audit.indexed(rows, True))
    path = tmp_path / "analysis/summary.json"
    summary = json.loads(path.read_text())
    revisit = next(group for group in summary["groups"] if group["scope"] == "revisit")
    assert revisit["count"] == 0
    assert all(revisit[name] is None for name in audit.STATS if name != "count")
    revisit["mean_ms"] = 0.0
    path.write_text(json.dumps(summary))
    with pytest.raises(audit.AuditError, match="mean_ms"):
        audit.audit_analysis(tmp_path, groups, audit.indexed(rows, True))


def test_optional_case_observed_counts_are_verified(tmp_path):
    fixture(tmp_path)
    requests, manifest = audit.audit_workload(tmp_path, "nosa", 3, small_contract())
    rows = measurements(manifest, requests)
    case = {
        "session_reservation": {"hbm": 10, "dram": 5},
        "duration_seconds": 0.1,
        "observed_users": 3,
        "revisits": 3,
        "max_revisits_observed": 2,
    }
    audit.audit_case_rows(
        ("nosa", "serial_sparse", 3),
        rows,
        case,
        requests,
        manifest,
        "unit",
        {"hbm": 100, "dram": 10},
    )
    case["max_revisits_observed"] = 3
    with pytest.raises(audit.AuditError, match="case max_revisits_observed"):
        audit.audit_case_rows(
            ("nosa", "serial_sparse", 3),
            rows,
            case,
            requests,
            manifest,
            "unit",
            {"hbm": 100, "dram": 10},
        )


def test_empty_revisit_allowance_is_a_strict_boolean():
    params = run_parameters(allow_empty_revisits=True)
    assert audit.RunContract.from_parameters(params).allow_empty_revisits is True
    params["allow_empty_revisits"] = 1
    with pytest.raises(audit.AuditError, match="must be boolean"):
        audit.RunContract.from_parameters(params)


@pytest.mark.parametrize("field", [None, "pv_share"])
def test_industrial_contract_preserves_1024_user_4096_request_scope(field):
    meta = metadata_fixture(
        users=[1024],
        requests=4096,
        max_revisits=None,
        history_tokens=65536,
        heat_dataset="industrial_10M",
        heat_field=field,
        hbm_budget_bytes=64 * 2**30,
        hbm_budget_gib=64.0,
        dram_budget_bytes=1024 * 2**30,
        dram_budget_gib=1024.0,
    )
    contract, cases = audit.audit_metadata(meta, "unit", None)
    assert contract.heat_dataset == "industrial_10M"
    assert contract.heat_field == "pv_share"
    assert contract.request_counts == {1024: 4096}
    assert contract.measured_requests == 16384
    assert len(cases) == 4
    assert contract.caps == {"hbm": 64 * 2**30, "dram": 1024 * 2**30}


def archived_trace_fixture(directory):
    contract = replace(
        small_contract(), max_revisits=None, heat_dataset="industrial_10M", heat_field="pv_share"
    )
    workload_dir, archive_dir = directory / "workload", directory / "archive"
    workload_dir.mkdir()
    archive_dir.mkdir()
    manifest, requests = fixture(workload_dir, contract=contract)
    (archive_dir / "requests_3.csv").write_text(
        "request_id,user_id,heat_rank,visit_index,is_revisit,previous_request_id,"
        "reuse_distance_users,synthetic_timestamp\n"
        "0,0,1,0,0,,,0.0\n1,1,2,0,0,,,1.0\n2,0,1,1,1,0,1,2.0\n"
        "3,2,3,0,0,,,3.0\n4,1,2,1,1,1,2,4.0\n5,0,1,2,1,2,2,5.0\n"
    )
    (archive_dir / "users_3.csv").write_text(
        "pool_users,user_id,heat_rank,probability,expected_visits,visits,revisits,"
        "first_request_id,last_request_id\n"
        "3,0,1,0.3333333333333333,2.0,3,2,0,5\n"
        "3,1,2,0.3333333333333333,2.0,2,1,1,4\n"
        "3,2,3,0.3333333333333333,2.0,1,0,3,3\n"
    )
    archive = {
        "dataset": "industrial_10M",
        "field": "pv_share",
        "users": [3],
        "requests_per_population": 6,
        "population_seed": 49,
        "sampling_seed": 49,
        "schedule": manifest["schedule"],
        "populations": [manifest["heat"]],
        "artifact_sha256": {
            name: audit.sha256(archive_dir / name) for name in ("requests_3.csv", "users_3.csv")
        },
    }
    (archive_dir / "manifest.json").write_text(json.dumps(archive))
    return contract, requests, manifest, archive_dir


def test_archived_access_trace_checks_all_requests_and_all_users(tmp_path):
    contract, requests, manifest, archive = archived_trace_fixture(tmp_path)
    audit.audit_workload(tmp_path / "workload", "nosa", 3, contract)
    result = audit.audit_expected_trace(archive, 3, contract, requests, manifest)
    assert result["requests"] == 6
    assert set(result["files_sha256"]) == {"manifest.json", "requests_3.csv", "users_3.csv"}


@pytest.mark.parametrize(
    ("filename", "before", "after", "message"),
    [
        ("requests_3.csv", "2,0,1,1,1,0,1,2.0", "2,1,1,1,1,0,1,2.0", "request CSV row"),
        ("requests_3.csv", "2,0,1,1,1,0,1,2.0", "2,0,1,0,0,0,1,2.0", "request CSV row"),
        ("requests_3.csv", "2,0,1,1,1,0,1,2.0", "2,0,1,1,1,1,1,2.0", "request CSV row"),
        ("requests_3.csv", "2,0,1,1,1,0,1,2.0", "2,0,1,1,1,0,0,2.0", "request CSV row"),
        ("requests_3.csv", "2,0,1,1,1,0,1,2.0", "2,0,1,1,1,0,1,2.5", "request CSV row"),
        ("requests_3.csv", "5,0,1,2,1,2,2,5.0\n", "", "request CSV row"),
        ("requests_3.csv", "5,0,1,2,1,2,2,5.0\n", "5,0,1,2,1,2,2,5.0\nextra\n", "excess rows"),
        ("users_3.csv", "0.3333333333333333", "0.2", "user CSV row"),
        ("users_3.csv", "2.0,3,2,0,5", "2.0,2,1,0,5", "user CSV row"),
    ],
)
def test_resigned_archive_cannot_hide_access_or_population_changes(
    tmp_path, filename, before, after, message
):
    contract, requests, manifest, archive_dir = archived_trace_fixture(tmp_path)
    path = archive_dir / filename
    path.write_text(path.read_text().replace(before, after))
    archive = audit.read_json(archive_dir / "manifest.json")
    archive["artifact_sha256"][filename] = audit.sha256(path)
    (archive_dir / "manifest.json").write_text(json.dumps(archive))
    with pytest.raises(audit.AuditError, match=message):
        audit.audit_expected_trace(archive_dir, 3, contract, requests, manifest)


def test_archived_trace_rejects_changed_digest_or_capped_contract(tmp_path):
    contract, requests, manifest, archive = archived_trace_fixture(tmp_path)
    with pytest.raises(audit.AuditError, match="requires no revisit cap"):
        audit.audit_expected_trace(
            archive, 3, replace(contract, max_revisits=2), requests, manifest
        )
    path = archive / "requests_3.csv"
    path.write_text(path.read_text() + "extra\n")
    with pytest.raises(audit.AuditError, match="artifact hash"):
        audit.audit_expected_trace(archive, 3, contract, requests, manifest)


def test_industrial_workload_rejects_an_unsigned_or_incorrect_access_identity(tmp_path):
    contract, requests, manifest, _ = archived_trace_fixture(tmp_path)
    for digest in (None, "0" * 64):
        manifest["access_trace_sha256"] = digest
        write_fixture(tmp_path / "workload", manifest, requests)
        with pytest.raises(audit.AuditError, match="access trace identity"):
            audit.audit_workload(tmp_path / "workload", "nosa", 3, contract)


def test_industrial_audit_requires_explicit_source_dataset_before_source_or_tensor_checks(tmp_path):
    meta = metadata_fixture(heat_dataset="industrial_10M", heat_field="pv_share")
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with (
        patch.object(audit, "audit_sources", side_effect=AssertionError("must not reach sources")),
        pytest.raises(audit.AuditError, match="require --expected-trace-directory"),
    ):
        audit.audit(tmp_path, tmp_path)


def test_legacy_beauty_manifest_without_explicit_heat_field_still_audits(tmp_path):
    manifest, requests = fixture(tmp_path)
    del manifest["config"]["heat_field"]
    identity = {
        name: manifest[name] for name in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    manifest["workload_sha256"] = audit.hashlib.sha256(audit.canonical(identity)).hexdigest()
    write_fixture(tmp_path, manifest, requests)
    audit.audit_workload(tmp_path, "nosa", 3, small_contract())


def test_saved_access_csv_must_match_metadata_and_archived_bytes(tmp_path):
    path = tmp_path / "access_trace.csv"
    path.write_text("request_id,user_id\n0,631\n")
    digest = audit.sha256(path)
    trace = {"nosa/1024": {"files_sha256": {"requests_1024.csv": digest}}}
    assert (
        audit.audit_access_trace_snapshot(tmp_path, {"access_trace_sha256": digest}, trace)
        == digest
    )
    with pytest.raises(audit.AuditError, match="metadata identity"):
        audit.audit_access_trace_snapshot(tmp_path, {"access_trace_sha256": "a" * 64}, trace)
    trace["nosa/1024"]["files_sha256"]["requests_1024.csv"] = "b" * 64
    with pytest.raises(audit.AuditError, match="archived identity"):
        audit.audit_access_trace_snapshot(tmp_path, {"access_trace_sha256": digest}, trace)


def sequential_contract():
    return replace(
        small_contract(),
        populations=(16,),
        request_cap=32,
        max_revisits=None,
        sampling="sequential",
        heat_dataset=None,
        heat_field=None,
        hbm_bytes=4 * 2**30,
        dram_bytes=64 * 2**30,
    )


def test_sequential_contract_has_no_heat_or_cap_and_preserves_requested_scope():
    params = run_parameters(
        models=["nosa", "deepseek_v32"],
        users=[16],
        requests=32,
        sampling="sequential",
        max_revisits=None,
        history_tokens=65536,
        candidate_tokens=128,
        hbm_budget_bytes=4 * 2**30,
        hbm_budget_gib=4.0,
        dram_budget_bytes=64 * 2**30,
        dram_budget_gib=64.0,
    )
    contract = audit.RunContract.from_parameters(params)
    assert contract.sampling == "sequential"
    assert contract.heat_dataset is None
    assert contract.heat_field is None
    assert contract.request_counts == {16: 32}
    assert contract.measured_requests == 256
    assert len(contract.cases) == 8
    assert contract.caps == {"hbm": 4 * 2**30, "dram": 64 * 2**30}


@pytest.mark.parametrize(
    "changes",
    [
        {"sampling": "unknown"},
        {"sampling": None},
        {"sampling": "sequential", "max_revisits": 1},
        {"sampling": "sequential", "max_revisits": None, "access_trace": "heat.csv"},
    ],
)
def test_sequential_parameters_reject_wrong_schedule_contract(changes):
    with pytest.raises(audit.AuditError):
        audit.RunContract.from_parameters(run_parameters(**changes))


def test_sequential_workload_has_exactly_sixteen_first_and_sixteen_revisits(tmp_path):
    contract = sequential_contract()
    fixture(tmp_path, contract=contract, trace=list(range(16)) * 2)
    with patch.object(audit, "scheduler_replay", side_effect=AssertionError("no heat replay")):
        requests, manifest = audit.audit_workload(tmp_path, "nosa", 16, contract)
    assert [row["user_id"] for row in requests] == list(range(16)) * 2
    assert [row["visit_index"] for row in requests] == [0] * 16 + [1] * 16
    assert manifest["observed"]["first_visits"] == 16
    assert manifest["observed"]["revisits"] == 16
    assert all(row["probability"] is None for row in manifest["users"])
    # An eight-session cache misses on every second-round request; these remain revisits.
    oracle = list(
        audit.lru_oracle(
            requests,
            {"hbm": 2**29, "dram": 2**30},
            contract.caps,
        )
    )
    assert [row["capacity"] for row in oracle] == [8] * 32
    assert not any(row["hit"] for row in oracle)
    assert all(row["is_revisit"] for row in requests[16:])


@pytest.mark.parametrize(
    "trace",
    [
        list(range(1, 16)) + [0] + list(range(16)),
        list(range(16)) + list(reversed(range(16))),
        list(range(16)) * 2 + [0],
        list(range(16)) + list(range(15)),
    ],
)
def test_sequential_rejects_resigned_wrong_or_incomplete_order(tmp_path, trace):
    contract = sequential_contract()
    fixture(tmp_path, contract=contract, trace=trace)
    with pytest.raises(audit.AuditError, match="sequential schedule|identities count"):
        audit.audit_workload(tmp_path, "nosa", 16, contract)


@pytest.mark.parametrize("change", ["weight", "probability", "visit_index", "metadata"])
def test_sequential_rejects_heat_semantics_or_wrong_visit_counter(tmp_path, change):
    contract = sequential_contract()
    manifest, requests = fixture(tmp_path, contract=contract, trace=list(range(16)) * 2)
    if change == "weight":
        manifest["users"][0]["weight"] = 1.5
    elif change == "probability":
        manifest["users"][0]["probability"] = 1 / 16
    elif change == "visit_index":
        requests[16]["visit_index"] = 0
    else:
        manifest["heat"]["path"] = "GR/analysis/heat_curves.csv"
    write_fixture(tmp_path, manifest, requests)
    with pytest.raises(audit.AuditError, match="weight|probability|sequential"):
        audit.audit_workload(tmp_path, "nosa", 16, contract)


def test_legacy_weighted_manifest_without_sampling_still_audits(tmp_path):
    manifest, requests = fixture(tmp_path)
    del manifest["config"]["sampling"]
    identity = {
        name: manifest[name] for name in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    manifest["workload_sha256"] = audit.hashlib.sha256(audit.canonical(identity)).hexdigest()
    write_fixture(tmp_path, manifest, requests)
    audit.audit_workload(tmp_path, "nosa", 3, small_contract())


def test_lru_oracle_subtracts_shared_reservation_and_obeys_host_page_quota():
    requests = [{"user_id": user} for user in [0, 1, 2, 0]]
    reservation = {"hbm": 10, "dram": 4}
    caps = {"hbm": 100, "dram": 100}
    rows = list(
        audit.lru_oracle(
            requests,
            reservation,
            caps,
            shared={"hbm": 50, "dram": 64},
            host_pages=4,
            session_pages=2,
        )
    )
    assert [row["capacity"] for row in rows] == [2] * 4
    assert [row["hit"] for row in rows] == [False] * 4
    assert [row["evicted"] for row in rows] == [[], [], [0], [1]]
    with pytest.raises(audit.AuditError, match="no whole session"):
        list(
            audit.lru_oracle(
                requests,
                reservation,
                caps,
                shared={"hbm": 100, "dram": 64},
                host_pages=4,
                session_pages=2,
            )
        )


@pytest.fixture
def shared_host_case(tmp_path):
    def build(scheme):
        contract = replace(small_contract(), models=("deepseek_v32",))
        manifest, requests = fixture(tmp_path, contract=contract)
        rows = measurements(manifest, requests)
        pooled = scheme in ("echo", "serial_sparse")
        shared_host = 120 if pooled else 0
        session_host = 4 if pooled else (50 if scheme == "dense_prefetch" else 0)
        case = {
            "session_reservation": {"hbm": 10, "dram": session_host},
            "shared_reservation": {"hbm": 80, "dram": shared_host + 40},
            "host_page_capacity": 12 if pooled else 0,
            "session_host_pages": 4 if pooled else 0,
            "duration_seconds": 0.1,
            "cache_resource_plan": {
                "workspace_cpu_indexer_bytes": 8,
                "workspace_cpu_scalar_bytes": 8,
                "workspace_cpu_metrics_bytes": 24,
                "workspace_cpu_bytes": 40,
            },
        }
        # The existing hand-computed trace retains at most two users. HBM
        # admission fixes that capacity here; DRAM and host pages do not bind.
        for row in rows:
            count = row["cached_users"]
            fixed_host = shared_host + session_host * count
            row.update(
                scheme=scheme,
                prefix_hit_tier=("hbm" if scheme == "hbm" else "dram")
                if row["prefix_cache_hit"]
                else "miss",
                dram_budget_bytes=1000,
                reserved_hbm_bytes=80 + count * 10,
                reserved_dram_bytes=fixed_host + 40,
                cache_hbm_bytes=60 + count * 8,
                request_cache_hbm_bytes=60 + count * 9,
                cache_dram_bytes=fixed_host,
                request_cache_dram_bytes=fixed_host,
                shared_reserved_hbm_bytes=80,
                shared_reserved_dram_bytes=shared_host + 40,
                session_reserved_hbm_bytes=count * 10,
                session_reserved_dram_bytes=count * session_host,
                shared_cache_hbm_bytes=60,
                shared_cache_dram_bytes=shared_host,
                cache_host_pages=case["session_host_pages"] * count,
                host_page_capacity=case["host_page_capacity"],
            )
        args = (
            ("deepseek_v32", scheme, 3),
            rows,
            case,
            requests,
            manifest,
            "unit",
            {"hbm": 100, "dram": 1000},
        )
        return args

    return build


@pytest.mark.parametrize("scheme", ["hbm", "echo", "serial_sparse", "dense_prefetch"])
def test_host_capacity_excludes_only_validated_shared_cpu_workspace(shared_host_case, scheme):
    args = shared_host_case(scheme)
    result = audit.audit_case_rows(*args)
    assert result["requests"] == 6
    assert result["admitted_session_capacity"] == 2
    assert result["peaks"]["reserved_dram_bytes"] - result["peaks"]["cache_dram_bytes"] == 40


@pytest.mark.parametrize("scheme", ["echo", "serial_sparse", "dense_prefetch"])
@pytest.mark.parametrize(
    ("field", "change", "error"),
    [
        ("cache_dram_bytes", -1, "fixed host backing capacity"),
        ("shared_cache_dram_bytes", 1, "fixed shared host storage"),
        ("request_cache_dram_bytes", 1, "pre-cleanup fixed host backing capacity"),
    ],
)
def test_host_capacity_rejects_missing_or_misattributed_storage(
    shared_host_case, scheme, field, change, error
):
    args = shared_host_case(scheme)
    args[1][-1][field] += change
    with pytest.raises(audit.AuditError, match=error):
        audit.audit_case_rows(*args)


def test_host_capacity_rejects_missing_shared_backing_with_unchanged_total(shared_host_case):
    args = shared_host_case("echo")
    args[1][-1]["shared_cache_dram_bytes"] -= 1
    with pytest.raises(audit.AuditError, match="fixed shared host storage"):
        audit.audit_case_rows(*args)


def test_host_capacity_rejects_missing_session_page_table(shared_host_case):
    args = shared_host_case("serial_sparse")
    row = args[1][-1]
    row["cache_dram_bytes"] -= args[2]["session_reservation"]["dram"]
    row["request_cache_dram_bytes"] = row["cache_dram_bytes"]
    with pytest.raises(audit.AuditError, match="fixed host backing capacity"):
        audit.audit_case_rows(*args)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("workspace_cpu_bytes", 41, "CPU workspace sum"),
        ("workspace_cpu_metrics_bytes", 23, "workspace_cpu_metrics_bytes"),
        ("workspace_cpu_scalar_bytes", None, "workspace_cpu_scalar_bytes"),
    ],
)
def test_host_capacity_rejects_unverified_workspace_allowance(
    shared_host_case, field, value, error
):
    args = shared_host_case("echo")
    plan = args[2]["cache_resource_plan"]
    if value is None:
        del plan[field]
    else:
        plan[field] = value
    with pytest.raises(audit.AuditError, match=error):
        audit.audit_case_rows(*args)


def test_host_capacity_does_not_assume_workspace_from_an_unexplained_gap(shared_host_case):
    args = shared_host_case("dense_prefetch")
    del args[2]["cache_resource_plan"]
    with pytest.raises(audit.AuditError, match="fixed shared host storage"):
        audit.audit_case_rows(*args)


def test_host_capacity_rejects_unreserved_workspace(shared_host_case):
    args = shared_host_case("hbm")
    args[2]["shared_reservation"]["dram"] = 39
    with pytest.raises(audit.AuditError, match="unreserved CPU workspace"):
        audit.audit_case_rows(*args)
