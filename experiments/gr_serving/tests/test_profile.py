"""CPU checks for saved-request provenance and post-run interval validation."""

import copy
import hashlib
import importlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def profile_module():
    """Load the installed source location or the isolated pre-publication draft."""
    name = "experiments.gr_serving.src.profile"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    path = Path(__file__).resolve().parents[1] / "src" / "profile.py"
    spec = importlib.util.spec_from_file_location("gr_serving_profile_draft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Fixed independently from the profiler's constants: these are the experiment
# geometry and saved workload schema, not a reduced model/checkpoint execution.
PREFIX = 16384
QUERIES = 1024
IDENTITY_KEYS = (
    "request_id",
    "user_id",
    "visit_index",
    "visit_number",
    "is_revisit",
    "timestamp",
    "stable_prefix_tokens",
    "candidate_suffix_tokens",
    "prefix_sha256",
    "candidate_sha256",
    "input_sha256",
)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _refresh_manifest(rows, manifest, *, prefix=PREFIX):
    """Construct a self-consistent fixture independently of profile helpers."""
    for row in rows:
        ids = row["input_ids"]
        row["prefix_sha256"] = _digest(ids[:prefix])
        row["candidate_sha256"] = _digest(ids[prefix:])
        row["input_sha256"] = _digest(ids)
    manifest["requests"] = [{key: row[key] for key in IDENTITY_KEYS} for row in rows]
    counts = Counter(row["user_id"] for row in rows)
    manifest["observed"] = {
        "requests": len(rows),
        "unique_users": len(counts),
        "first_visits": len(counts),
        "revisits": sum(count - 1 for count in counts.values()),
        "max_visits": max(counts.values(), default=0),
        "max_revisits": max((count - 1 for count in counts.values()), default=0),
    }
    manifest["workload_sha256"] = _digest(
        {key: manifest[key] for key in ("config", "heat_sha256", "tokenizer_sha256", "requests")}
    )


def _write_workload(directory, rows, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workload.json").write_text(json.dumps(manifest))
    (directory / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def saved_workload(tmp_path):
    # The third revisit is another visit by user 5: selection must follow request
    # order, without restricting the result to three different users.
    visits = Counter()
    rows = []
    for request_id, user in enumerate((5, 2, 5, 7, 2, 5, 7, 0)):
        visit = visits[user]
        rows.append(
            {
                "request_id": request_id,
                "user_id": user,
                "visit_index": visit,
                "visit_number": visit + 1,
                "is_revisit": visit > 0,
                "timestamp": float(request_id),
                "stable_prefix_tokens": PREFIX,
                "candidate_suffix_tokens": QUERIES,
                "input_ids": [17 + user] * PREFIX + [1000 + request_id] * QUERIES,
            }
        )
        visits[user] += 1
    manifest = {
        "schema_version": 1,
        "config": {
            "model": "nosa",
            "num_users": 8,
            "requests": len(rows),
            "history_tokens": PREFIX,
            "candidate_tokens": QUERIES,
            "seed": 42,
            "heat_dataset": "beauty",
        },
        "heat_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
    }
    _refresh_manifest(rows, manifest)
    directory = tmp_path / "workload"
    _write_workload(directory, rows, manifest)
    return directory, rows, manifest


def _write_formal_run(
    directory, workload_sha256, requests, *, num_users=8, prefix=PREFIX, queries=QUERIES
):
    directory.mkdir()
    schemes = ("hbm", "serial_sparse", "dense_prefetch", "overlap")
    metadata = {
        "status": "accepted",
        "run_id": "formal-latency",
        "parameters": {
            "models": ["nosa"],
            "users": [num_users],
            "history_tokens": prefix,
            "candidate_tokens": queries,
            "chunk_size": QUERIES,
        },
        "cases": [
            {"model": "nosa", "num_users": num_users, "scheme": scheme, "requests": requests}
            for scheme in schemes
        ],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    measurements = [
        {
            "model": "nosa",
            "num_users": num_users,
            "scheme": scheme,
            "request_id": request,
            "workload_sha256": workload_sha256,
        }
        for scheme in schemes
        for request in range(requests)
    ]
    (directory / "measurements.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in measurements)
    )
    return metadata


def test_saved_workload_selects_first_three_revisits_with_exact_identities(
    profile_module, saved_workload
):
    directory, rows, manifest = saved_workload
    loaded, selected = profile_module.load_saved_workload(directory)
    assert loaded == manifest
    assert [row["request_id"] for row in selected] == [2, 4, 5]
    assert [row["user_id"] for row in selected] == [5, 2, 5]
    assert [row["visit_index"] for row in selected] == [1, 1, 2]
    assert selected == [rows[index] for index in (2, 4, 5)]
    assert all(row["is_revisit"] for row in selected)


def test_resigned_64_user_workload_and_formal_measurements_accept_explicit_population(
    profile_module, saved_workload, tmp_path
):
    directory, rows, manifest = saved_workload
    original_digest = manifest["workload_sha256"]
    manifest["config"]["num_users"] = 64
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    assert manifest["workload_sha256"] != original_digest
    loaded, selected = profile_module.load_saved_workload(directory, num_users=64)
    assert loaded == manifest
    assert selected == [rows[index] for index in (2, 4, 5)]
    formal = tmp_path / "formal64"
    metadata = _write_formal_run(formal, manifest["workload_sha256"], len(rows), num_users=64)
    assert (
        profile_module.validate_latency_run(
            formal, "separate-profile", loaded["workload_sha256"], num_users=64
        )
        == metadata
    )
    with pytest.raises(ValueError):
        profile_module.load_saved_workload(directory)
    with pytest.raises(ValueError):
        profile_module.validate_latency_run(formal, "separate-profile", loaded["workload_sha256"])


def test_sequential_sixteen_user_64k_workload_without_heat_selects_second_round(
    profile_module, tmp_path
):
    prefix, queries = 65536, 128
    rows = []
    for request_id in range(32):
        user, visit = request_id % 16, request_id // 16
        rows.append(
            {
                "request_id": request_id,
                "user_id": user,
                "visit_index": visit,
                "visit_number": visit + 1,
                "is_revisit": visit > 0,
                "timestamp": float(request_id),
                "stable_prefix_tokens": prefix,
                "candidate_suffix_tokens": queries,
                "input_ids": [17 + user] * prefix + [1000 + request_id] * queries,
            }
        )
    manifest = {
        "schema_version": 1,
        "config": {
            "model": "nosa",
            "num_users": 16,
            "requests": 32,
            "history_tokens": prefix,
            "candidate_tokens": queries,
            "seed": 42,
            "sampling": "sequential",
            "heat_dataset": None,
            "heat_field": None,
            "max_revisits": None,
        },
        "heat_sha256": None,
        "tokenizer_sha256": "b" * 64,
    }
    _refresh_manifest(rows, manifest, prefix=prefix)
    directory = tmp_path / "sequential"
    _write_workload(directory, rows, manifest)
    loaded, selected = profile_module.load_saved_workload(
        directory, num_users=16, prefix=prefix, queries=queries
    )
    assert loaded == manifest
    assert selected == [rows[index] for index in (16, 17, 18)]
    assert [row["user_id"] for row in selected] == [0, 1, 2]
    assert [row["visit_index"] for row in selected] == [1, 1, 1]
    assert loaded["observed"]["first_visits"] == loaded["observed"]["revisits"] == 16
    formal = tmp_path / "formal16"
    metadata = _write_formal_run(
        formal, manifest["workload_sha256"], 32, num_users=16, prefix=prefix, queries=queries
    )
    assert (
        profile_module.validate_latency_run(
            formal,
            "separate-profile",
            loaded["workload_sha256"],
            num_users=16,
            prefix=prefix,
            queries=queries,
            request_count=32,
        )
        == metadata
    )


def test_industrial_4096_request_trace_streams_and_authenticates_after_selected_samples(
    profile_module, tmp_path, monkeypatch
):
    # Small token payloads isolate CPU provenance checking. The complete 4096-row
    # trace and uncapped visit counts are real here; no model execution is mocked.
    prefix, queries, counts, rows = 4, 2, Counter(), []
    for request_id in range(4096):
        user = 0 if request_id % 2 else (request_id // 2) % 1024
        visit = counts[user]
        rows.append(
            {
                "request_id": request_id,
                "user_id": user,
                "visit_index": visit,
                "visit_number": visit + 1,
                "is_revisit": visit > 0,
                "timestamp": float(request_id),
                "stable_prefix_tokens": prefix,
                "candidate_suffix_tokens": queries,
                "input_ids": [17 + user] * prefix + [2000 + request_id] * queries,
            }
        )
        counts[user] += 1
    manifest = {
        "schema_version": 1,
        "config": {
            "model": "nosa",
            "num_users": 1024,
            "requests": 4096,
            "history_tokens": prefix,
            "candidate_tokens": queries,
            "seed": 42,
            "heat_dataset": "industrial_10M",
            "heat_field": "pv_share",
            "max_revisits": None,
        },
        "heat_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
    }
    _refresh_manifest(rows, manifest, prefix=prefix)
    directory = tmp_path / "industrial"
    _write_workload(directory, rows, manifest)
    read_text = Path.read_text

    def no_bulk_token_read(path, *args, **kwargs):
        if path.name == "requests.jsonl":
            raise AssertionError("Token requests must be streamed, not bulk-loaded")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_bulk_token_read)
    loaded, selected = profile_module.load_saved_workload(
        directory, num_users=1024, prefix=prefix, queries=queries
    )
    assert loaded == manifest
    assert selected == [rows[index] for index in (1, 3, 5)]
    assert loaded["observed"]["requests"] == 4096
    assert loaded["observed"]["unique_users"] == 1024
    assert loaded["observed"]["max_revisits"] > 2

    # A late corrupt row must still fail, even after all three samples were found.
    rows[-1]["input_ids"][-1] += 1
    _write_workload(directory, rows, manifest)
    with pytest.raises(ValueError, match="Request 4095: candidate_sha256"):
        profile_module.load_saved_workload(
            directory, num_users=1024, prefix=prefix, queries=queries
        )


def test_formal_profile_binding_rejects_duplicate_request_ids(profile_module, tmp_path):
    directory = tmp_path / "formal"
    _write_formal_run(directory, "a" * 64, 4096, num_users=1024)
    path = directory / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["request_id"] -= 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="request IDs"):
        profile_module.validate_latency_run(
            directory, "profile", "a" * 64, num_users=1024, request_count=4096
        )


def test_formal_profile_binding_requires_full_saved_trace(profile_module, tmp_path):
    directory = tmp_path / "formal"
    _write_formal_run(directory, "a" * 64, 16, num_users=1024)
    with pytest.raises(ValueError, match="complete same saved request trace"):
        profile_module.validate_latency_run(
            directory, "profile", "a" * 64, num_users=1024, request_count=4096
        )


@pytest.mark.parametrize(
    "mutation",
    ["token", "input_hash", "prefix_hash", "candidate_hash", "manifest_hash", "manifest_identity"],
)
def test_saved_workload_rejects_tampered_hashes_or_content(
    profile_module, saved_workload, mutation
):
    directory, rows, manifest = saved_workload
    if mutation == "token":
        rows[2]["input_ids"][-1] += 1
    elif mutation.endswith("_hash") and mutation != "manifest_hash":
        field = {
            "input_hash": "input_sha256",
            "prefix_hash": "prefix_sha256",
            "candidate_hash": "candidate_sha256",
        }[mutation]
        rows[2][field] = "0" * 64
    elif mutation == "manifest_hash":
        manifest["workload_sha256"] = "0" * 64
    else:
        manifest["requests"][2]["timestamp"] += 1
    _write_workload(directory, rows, manifest)
    with pytest.raises(ValueError, match="match saved token IDs|authenticate"):
        profile_module.load_saved_workload(directory)


@pytest.mark.parametrize(
    "field,value", [("visit_index", 0), ("visit_number", 1), ("is_revisit", False)]
)
def test_rehashed_workload_cannot_change_revisit_semantics(
    profile_module, saved_workload, field, value
):
    directory, rows, manifest = saved_workload
    rows[2][field] = value
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    with pytest.raises(ValueError, match="visit identity"):
        profile_module.load_saved_workload(directory)


def test_coordinated_workload_edit_is_rejected_against_formal_measurements(
    profile_module, saved_workload, tmp_path
):
    directory, rows, manifest = saved_workload
    source_digest = manifest["workload_sha256"]
    formal = tmp_path / "formal"
    metadata = _write_formal_run(formal, source_digest, len(rows))
    assert (
        profile_module.validate_latency_run(formal, "separate-profile", source_digest) == metadata
    )

    # Editing both files and recomputing all hashes is internally consistent;
    # only binding to the accepted formal run detects a changed workload.
    rows[2]["input_ids"][-1] += 7
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    modified, selected = profile_module.load_saved_workload(directory)
    assert selected[0]["input_ids"][-1] == rows[2]["input_ids"][-1]
    assert modified["workload_sha256"] != source_digest
    with pytest.raises(ValueError, match="formal measured workload hash"):
        profile_module.validate_latency_run(formal, "separate-profile", modified["workload_sha256"])


@pytest.fixture
def interval_profile():
    labels = ["request_000002_layer31", "request_000004_layer31", "request_000005_layer31"]
    profile = {
        "schema_version": 3,
        "run_id": "separate-profile",
        "clock": "device_globaltimer_ns",
        "math_coverage": "softmax_only",
        "fetch_coverage": "per_page_copy_envelopes",
        "stripe_coverage": "nonempty_stripe_copy_windows",
        "byte_accounting": "logical_unique_kv_payload",
        "fetch_stripes": 8,
        "cases": {},
        "records": [],
    }
    # One selected full historical page. Real byte/slot identities are used;
    # overlapping stripe timestamps test interval unions, not duration sums.
    for label, math_start in zip(labels, (150, 140, 201), strict=True):
        profile["cases"][label] = {
            "prefix": PREFIX,
            "queries": QUERIES,
            "kv_heads": 2,
            "expected_prefix_bytes": 32768,
            "expected_fetch_rows": [{"row": 0, "bytes": 32768}],
        }
        page = {"row": 0, "start_ns": 100, "end_ns": 1100, "bytes": 32768, "kind": 1}
        serial = {
            "case": label,
            "mode": "serialized",
            "sample": 0,
            "recorded_transfer_bytes": 32768,
            "intervals": [dict(page)],
            "stripe_intervals": [],
        }
        overlap = {
            "case": label,
            "mode": "overlap",
            "sample": 0,
            "recorded_transfer_bytes": 32768,
            "intervals": [
                dict(page),
                {"row": 512, "start_ns": math_start, "end_ns": 1100, "bytes": 0, "kind": 2},
            ],
            "stripe_intervals": [
                {"row": 16896 + stripe, "start_ns": 100, "end_ns": 1100, "bytes": 4096, "kind": 3}
                for stripe in range(8)
            ],
        }
        profile["records"].extend((serial, overlap))
    return profile, labels


def test_valid_below_threshold_sample_is_preserved_and_fails_all_sample_claim(
    profile_module, interval_profile
):
    profile, labels = interval_profile
    original = copy.deepcopy(profile)
    result = profile_module.analyze_profile(profile, labels)
    assert profile == original
    assert result["status"] == "accepted"
    assert len(result["records"]) == 6
    overlap = [row for row in result["records"] if row["mode"] == "overlap"]
    assert [row["case"] for row in overlap] == labels
    assert [row["work_metrics"]["fetch_math_overlap_fraction"] for row in overlap] == pytest.approx(
        [0.95, 0.96, 0.899]
    )
    assert [
        row["stripe_metrics"]["fetch_stripe_math_overlap_fraction"] for row in overlap
    ] == pytest.approx([0.95, 0.96, 0.899])
    assert [row["both_overlap_ratios_at_least_90_percent"] for row in overlap] == [
        True,
        True,
        False,
    ]
    assert result["overlap_claim"]["samples"] == 3
    assert result["overlap_claim"]["all_samples_both_ratios_at_least_90_percent"] is False
    assert result["overlap_claim"]["below_threshold_samples_remain_valid"] is True
    for row in result["records"]:
        assert row["work_metrics"]["copied_bytes"] == 32768
        if row["mode"] == "serialized":
            assert row["stripe_metrics"] is None
            assert row["work_metrics"]["fetch_math_overlap_fraction"] is None
            assert row["both_overlap_ratios_at_least_90_percent"] is None


def test_exact_ninety_percent_passes_inclusive_threshold(profile_module, interval_profile):
    profile, labels = interval_profile
    for row in profile["records"]:
        if row["mode"] == "overlap":
            row["intervals"][1]["start_ns"] = 200
    result = profile_module.analyze_profile(profile, labels)
    assert result["overlap_claim"]["all_samples_both_ratios_at_least_90_percent"] is True
    assert all(
        row["work_metrics"]["fetch_math_overlap_fraction"] == 0.9
        for row in result["records"]
        if row["mode"] == "overlap"
    )


def test_valid_envelope_gap_does_not_hide_zero_stripe_overlap(profile_module, interval_profile):
    profile, labels = interval_profile
    raw = profile["records"][1]
    raw["intervals"][1].update(start_ns=140, end_ns=1060)
    for index, stripe in enumerate(raw["stripe_intervals"]):
        stripe.update(start_ns=100 if index < 4 else 1060, end_ns=140 if index < 4 else 1100)
    result = profile_module.analyze_profile(profile, labels)
    row = result["records"][1]
    assert result["status"] == "accepted"
    assert row["work_metrics"]["fetch_math_overlap_fraction"] == 0.92
    assert row["stripe_metrics"]["fetch_stripe_math_overlap_fraction"] == 0
    assert row["stripe_metrics"]["page_envelope_only_union_us"] == pytest.approx(0.92)
    assert row["both_overlap_ratios_at_least_90_percent"] is False


@pytest.mark.parametrize("endpoint,delta", [("start_ns", -1), ("end_ns", 1)])
def test_profile_rejects_corrupt_page_envelope(profile_module, interval_profile, endpoint, delta):
    profile, labels = interval_profile
    profile["records"][1]["intervals"][0][endpoint] += delta
    with pytest.raises(ValueError, match="Page envelope"):
        profile_module.analyze_profile(profile, labels)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_sample"])
def test_profile_requires_exact_record_identity_coverage(
    profile_module, interval_profile, mutation
):
    profile, labels = interval_profile
    if mutation == "missing":
        profile["records"].pop()
    elif mutation == "duplicate":
        profile["records"][-1] = copy.deepcopy(profile["records"][-2])
    else:
        profile["records"][-1]["sample"] = 1
    with pytest.raises(ValueError, match="three cases"):
        profile_module.analyze_profile(profile, labels)


@pytest.mark.parametrize(
    "mutation", ["wrong_page", "missing_stripe", "duplicate_stripe", "float_counter"]
)
def test_profile_rejects_corrupt_native_records(profile_module, interval_profile, mutation):
    profile, labels = interval_profile
    record = profile["records"][1]
    if mutation == "wrong_page":
        # Same logical byte size, different selected head: counters alone cannot
        # authenticate that the captured selection was the one actually copied.
        record["intervals"][0]["row"] = 1
    elif mutation == "missing_stripe":
        record["stripe_intervals"].pop()
    elif mutation == "duplicate_stripe":
        record["stripe_intervals"].append(dict(record["stripe_intervals"][0]))
    else:
        record["recorded_transfer_bytes"] = 32768.0
    with pytest.raises(ValueError):
        profile_module.analyze_profile(profile, labels)


def test_selection_validation_preserves_valid_masked_padding(profile_module):
    import torch

    ids = torch.full((QUERIES, 2, 64), -1, dtype=torch.int32)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, :, 0] = 0
    valid[:, :, 0] = True
    before = ids.clone(), valid.clone()
    profile_module.validate_selection(ids, valid)
    assert torch.equal(ids, before[0])
    assert torch.equal(valid, before[1])


@pytest.mark.parametrize("mutation", ["duplicate", "noncausal", "empty", "negative"])
def test_selection_validation_rejects_invalid_consumed_selection(profile_module, mutation):
    import torch

    ids = torch.full((QUERIES, 2, 64), -1, dtype=torch.int32)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, :, 0] = 0
    valid[:, :, 0] = True
    if mutation == "duplicate":
        ids[0, 0, 1] = 0
        valid[0, 0, 1] = True
    elif mutation == "noncausal":
        ids[0, 0, 0] = 257
    elif mutation == "empty":
        valid[0, 0, 0] = False
    else:
        ids[0, 0, 0] = -1
    with pytest.raises(ValueError, match="empty, duplicate, or noncausal"):
        profile_module.validate_selection(ids, valid)


@pytest.fixture
def accepted_output(profile_module, interval_profile, tmp_path):
    """A small CPU artifact set with real selection/trace tensor serialization."""
    import torch

    profile, labels = interval_profile
    renamed = {label: label.removesuffix("_layer31") for label in labels}
    profile["cases"] = {renamed[label]: case for label, case in profile["cases"].items()}
    for record in profile["records"]:
        record["case"] = renamed[record["case"]]
    labels = list(renamed.values())
    output = tmp_path / "accepted"
    output.mkdir()
    metadata = {
        "status": "accepted",
        "run_id": profile["run_id"],
        "requests": [{"request_id": request} for request in (2, 4, 5)],
        "samples": [],
        "reference_identities": {},
    }
    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    evidence = output / "formal_inputs"
    evidence.mkdir()
    for request in (2, 4, 5):
        path = evidence / f"hbm_{request:06d}.pt"
        torch.save(hidden, path)
        metadata["reference_identities"][str(request)] = {
            "source": f"formal/reference/nosa/8/{request:06d}.pt",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    # Head 0 consumes historical block 0; head 1 consumes the suffix's first
    # block. The exact CPU historical union is consequently page row 0 only.
    ids = torch.full((QUERIES, 2, 64), -1, dtype=torch.int32)
    ids[:, 0, 0] = 0
    ids[:, 1, 0] = 256
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[:, :, 0] = True
    for record in profile["records"]:
        directory = output / "samples" / record["case"] / record["mode"]
        directory.mkdir(parents=True)
        record["query_tile_size"] = 128
        record["expected_tile_transfer_bytes"] = [32768] + [0] * 7
        record["recorded_tile_transfer_bytes"] = [32768] + [0] * 7
        torch.save({"ids": ids, "valid_mask": valid}, directory / "selection.pt")
        trace = torch.zeros((20992, 4), dtype=torch.int64)
        for row in record["intervals"] + record["stripe_intervals"]:
            trace[row["row"]] = torch.tensor(
                [row[name] for name in ("start_ns", "end_ns", "bytes", "kind")]
            )
        torch.save(trace, directory / "native_trace.pt")
        torch.save(hidden, directory / "unprofiled_hidden.pt")
        torch.save(hidden, directory / "profiled_hidden.pt")
        check = {
            "shape": [2, 4],
            "dtype": "torch.bfloat16",
            "exact": True,
            "max_abs": 0.0,
            "relative_l2": 0.0,
            "atol": 0.0,
            "rtol": 0.0,
        }
        metadata["samples"].append(
            {
                "case": record["case"],
                "mode": record["mode"],
                "sample": record["sample"],
                "request": {"request_id": int(record["case"].removeprefix("request_"))},
                "unprofiled_control_vs_formal_hbm": dict(check),
                "profiled_vs_unprofiled_control": dict(check),
                "profiled_vs_formal_hbm": dict(check),
                "selection_sha256": hashlib.sha256(
                    (directory / "selection.pt").read_bytes()
                ).hexdigest(),
                "raw_trace_sha256": hashlib.sha256(
                    (directory / "native_trace.pt").read_bytes()
                ).hexdigest(),
            }
        )
    (output / "metadata.json").write_text(json.dumps(metadata))
    _write_profile_analysis(profile_module, output, profile, labels)
    return output, profile, labels, metadata


def _write_profile_analysis(profile_module, output, profile, labels):
    (output / "work_intervals.json").write_text(json.dumps(profile))
    analysis = profile_module.analyze_profile(profile, labels)
    analysis["work_intervals_sha256"] = hashlib.sha256(
        (output / "work_intervals.json").read_bytes()
    ).hexdigest()
    (output / "analysis.json").write_text(json.dumps(analysis))


def test_publication_verifier_accepts_consistent_cpu_artifact_set(profile_module, accepted_output):
    output, profile, _, metadata = accepted_output
    assert profile_module.verify_accepted_output(output, profile["run_id"]) == metadata


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_publication_verifier_requires_complete_sample_attestations(
    profile_module, accepted_output, mutation
):
    output, profile, _, metadata = accepted_output
    if mutation == "duplicate":
        metadata["samples"][-1] = copy.deepcopy(metadata["samples"][0])
    else:
        metadata["samples"].pop()
    (output / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        profile_module.verify_accepted_output(output, profile["run_id"])


def test_publication_verifier_binds_intervals_to_raw_trace(profile_module, accepted_output):
    output, profile, labels, _ = accepted_output
    # This preserves every metric and record identity but no longer describes
    # the actual serialized device-globaltimer evidence saved in native_trace.pt.
    for record in profile["records"]:
        for row in record["intervals"] + record["stripe_intervals"]:
            row["start_ns"] += 17
            row["end_ns"] += 17
    _write_profile_analysis(profile_module, output, profile, labels)
    with pytest.raises(ValueError):
        profile_module.verify_accepted_output(output, profile["run_id"])


def test_publication_verifier_binds_expected_union_to_saved_selection(
    profile_module, accepted_output
):
    output, profile, labels, _ = accepted_output
    # Replace the chosen KV head while retaining its exact logical byte count.
    # Expected rows and trace JSON agree with one another, but the saved consumed
    # selection and raw native tensor still identify historical page row 0.
    for case in profile["cases"].values():
        case["expected_fetch_rows"][0]["row"] = 1
    for record in profile["records"]:
        record["intervals"][0]["row"] = 1
        for stripe in record["stripe_intervals"]:
            stripe["row"] += 8
    _write_profile_analysis(profile_module, output, profile, labels)
    with pytest.raises(ValueError):
        profile_module.verify_accepted_output(output, profile["run_id"])


@pytest.mark.parametrize("prefix", [4096, 16384, 65536])
def test_saved_geometry_is_explicit_and_independent_of_prefix_chunk(
    profile_module, saved_workload, tmp_path, prefix
):
    import torch

    directory, rows, manifest = saved_workload
    queries = 128
    manifest["config"].update(history_tokens=prefix, candidate_tokens=queries)
    for row in rows:
        row["input_ids"] = [17 + row["user_id"]] * prefix + [1000 + row["request_id"]] * queries
        row["stable_prefix_tokens"] = prefix
        row["candidate_suffix_tokens"] = queries
        row["prefix_sha256"] = _digest(row["input_ids"][:prefix])
        row["candidate_sha256"] = _digest(row["input_ids"][prefix:])
        row["input_sha256"] = _digest(row["input_ids"])
    manifest["requests"] = [{key: row[key] for key in IDENTITY_KEYS} for row in rows]
    manifest["workload_sha256"] = _digest(
        {key: manifest[key] for key in ("config", "heat_sha256", "tokenizer_sha256", "requests")}
    )
    _write_workload(directory, rows, manifest)
    loaded, selected = profile_module.load_saved_workload(directory, prefix=prefix, queries=queries)
    assert loaded == manifest
    assert [row["request_id"] for row in selected] == [2, 4, 5]
    formal = tmp_path / "formal_dynamic"
    _write_formal_run(formal, manifest["workload_sha256"], len(rows))
    metadata = json.loads((formal / "metadata.json").read_text())
    metadata["parameters"].update(history_tokens=prefix, candidate_tokens=queries, chunk_size=1024)
    (formal / "metadata.json").write_text(json.dumps(metadata))
    profile_module.validate_latency_run(
        formal, "profile", manifest["workload_sha256"], prefix=prefix, queries=queries
    )
    ids = torch.zeros((queries, 2, 64), dtype=torch.int32)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[:, :, 0] = True
    profile_module.validate_selection(ids, valid, prefix=prefix, queries=queries)


@pytest.fixture
def capped_single_user_workload(saved_workload):
    directory, original, manifest = saved_workload
    rows = []
    for request_id in range(9):
        row = copy.deepcopy(original[0])
        row.update(
            request_id=request_id,
            visit_index=request_id,
            visit_number=request_id + 1,
            is_revisit=request_id > 0,
            timestamp=float(request_id),
            input_ids=original[0]["input_ids"][:PREFIX] + [1000 + request_id] * QUERIES,
        )
        rows.append(row)
    manifest["config"].update(num_users=1, requests=32, max_revisits=8)
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    return directory, rows, manifest


@pytest.mark.parametrize("requested,actual", [(32, 9), (4, 4)])
def test_capped_single_user_selects_first_three_revisits(
    profile_module, capped_single_user_workload, tmp_path, requested, actual
):
    directory, rows, manifest = capped_single_user_workload
    rows = rows[:actual]
    manifest["config"]["requests"] = requested
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    loaded, selected = profile_module.load_saved_workload(directory, num_users=1)
    assert loaded["observed"]["requests"] == actual
    assert [row["request_id"] for row in selected] == [1, 2, 3]
    formal = tmp_path / "formal_capped"
    _write_formal_run(formal, manifest["workload_sha256"], actual, num_users=1)
    profile_module.validate_latency_run(formal, "profile", manifest["workload_sha256"], num_users=1)


@pytest.mark.parametrize("cap", [-1, True, 1.5])
def test_saved_revisit_cap_must_be_a_nonnegative_integer(
    profile_module, capped_single_user_workload, cap
):
    directory, rows, manifest = capped_single_user_workload
    manifest["config"]["max_revisits"] = cap
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    with pytest.raises(ValueError, match="max_revisits"):
        profile_module.load_saved_workload(directory, num_users=1)


def test_saved_trace_must_obey_each_users_revisit_cap(profile_module, capped_single_user_workload):
    directory, rows, manifest = capped_single_user_workload
    rows = rows[:8]
    manifest["config"].update(num_users=2, requests=8, max_revisits=3)
    _refresh_manifest(rows, manifest)
    _write_workload(directory, rows, manifest)
    with pytest.raises(ValueError, match="exceeds the saved revisit cap"):
        profile_module.load_saved_workload(directory, num_users=2)


@pytest.mark.parametrize(
    "field", ["requests", "unique_users", "first_visits", "revisits", "max_visits", "max_revisits"]
)
def test_saved_observed_counts_are_verified(profile_module, capped_single_user_workload, field):
    directory, rows, manifest = capped_single_user_workload
    manifest["observed"][field] += 1
    _write_workload(directory, rows, manifest)
    with pytest.raises(ValueError, match=f"observed {field}"):
        profile_module.load_saved_workload(directory, num_users=1)


def test_uncapped_legacy_observed_counts_remain_supported(profile_module, saved_workload):
    directory, rows, manifest = saved_workload
    manifest["observed"].pop("max_visits")
    manifest["observed"].pop("max_revisits")
    _write_workload(directory, rows, manifest)
    _, selected = profile_module.load_saved_workload(directory)
    assert [row["request_id"] for row in selected] == [2, 4, 5]


def test_snapshot_skips_synthetic_module_files(profile_module, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from experiments.gr_serving.src import measure

    root, output = tmp_path / "repo", tmp_path / "snapshot"
    root.mkdir()
    output.mkdir()
    for name in ("pyproject.toml", "uv.lock", "profile.py", "dependency.py"):
        (root / name).write_text(f"fixture {name}\n")
    (root / "directory.py").mkdir()
    outside = tmp_path / "external.py"
    outside.write_text("outside repository\n")
    monkeypatch.setattr(profile_module, "__file__", str(root / "profile.py"))
    monkeypatch.setattr(
        measure, "source_snapshot", lambda path: (path / "source_manifest.json").write_text("{}")
    )
    for name, value in {
        "actual": root / "dependency.py",
        "synthetic": root / "_classes.py",
        "nonfile": object(),
        "directory": root / "directory.py",
        "external": outside,
    }.items():
        monkeypatch.setitem(
            sys.modules, f"snapshot_fixture_{name}", SimpleNamespace(__file__=value)
        )
    profile_module._snapshot_sources(output, root)
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert set(manifest) == {"pyproject.toml", "uv.lock", "profile.py", "dependency.py"}
    assert (output / "source/dependency.py").read_bytes() == (root / "dependency.py").read_bytes()


@pytest.fixture
def formal_source_fixture(tmp_path):
    root, formal = tmp_path / "repo", tmp_path / "formal"
    root.mkdir()
    formal.mkdir()
    names = (
        "experiments/gr_serving/src/profile.py",
        "experiments/gr_serving/tests/test_profile.py",
        "experiments/gr_serving/src/measure.py",
        "experiments/gr_serving/src/audit.py",
        "experiments/gr_serving/scripts/profile.sh",
        "models/nosa/attention.py",
        "cache/prefix_pool.py",
    )
    manifest = {}
    for name in names:
        current, saved = root / name, formal / "source" / name
        current.parent.mkdir(parents=True, exist_ok=True)
        saved.parent.mkdir(parents=True, exist_ok=True)
        current.write_text(f"formal source {name}\n")
        saved.write_bytes(current.read_bytes())
        manifest[name] = hashlib.sha256(current.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (formal / "source_manifest.json").write_text(json.dumps(manifest))
    (formal / "metadata.json").write_text(json.dumps({"source_sha256": digest}))
    return root, formal, manifest, digest


def test_formal_sources_allow_only_verified_diagnostic_changes(
    profile_module, formal_source_fixture
):
    root, formal, manifest, digest = formal_source_fixture
    allowed = [
        "experiments/gr_serving/src/profile.py",
        "experiments/gr_serving/tests/test_profile.py",
    ]
    for name in allowed:
        (root / name).write_text("corrected post-run diagnostic\n")
    result = profile_module._verify_formal_sources(formal, root)
    assert result["source_sha256"] == digest
    assert {row["path"] for row in result["allowed_diagnostic_source_changes"]} == set(allowed)
    for row in result["allowed_diagnostic_source_changes"]:
        assert row["formal_sha256"] == manifest[row["path"]]
        assert (
            row["profile_sha256"] == hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
        )
        assert row["formal_source_copy_verified"] is True
    assert "Neither is executed by the formal latency measurement" in result["reason"]


@pytest.mark.parametrize(
    "name",
    [
        "experiments/gr_serving/src/measure.py",
        "experiments/gr_serving/src/audit.py",
        "experiments/gr_serving/scripts/profile.sh",
        "models/nosa/attention.py",
        "cache/prefix_pool.py",
    ],
)
def test_formal_sources_reject_all_other_drift(profile_module, formal_source_fixture, name):
    root, formal, _, _ = formal_source_fixture
    (root / name).write_text("not allowed\n")
    with pytest.raises(RuntimeError, match="Formal sources changed"):
        profile_module._verify_formal_sources(formal, root)


def test_diagnostic_drift_requires_authentic_formal_copy(profile_module, formal_source_fixture):
    root, formal, _, _ = formal_source_fixture
    name = "experiments/gr_serving/src/profile.py"
    (root / name).write_text("corrected diagnostic\n")
    (formal / "source" / name).write_text("corrupted formal copy\n")
    with pytest.raises(RuntimeError, match="Saved formal diagnostic source differs"):
        profile_module._verify_formal_sources(formal, root)
