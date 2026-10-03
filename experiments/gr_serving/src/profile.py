"""Post-run NOSA layer-31 work intervals for three saved GR revisit requests.

This instrumented diagnostic does not measure serving latency or LRU occupancy.
Use scripts/profile.sh to publish only numerically and semantically valid runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PREFIX, QUERIES, LAYER, USERS, SAMPLES = 16384, 1024, 31, 8, 3
MODES = {"serial_sparse": "serialized", "overlap": "overlap"}
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


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_json(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_saved_workload(directory, *, num_users=USERS, prefix=PREFIX, queries=QUERIES):
    """Stream-authenticate the complete trace; retain only its first three revisits."""
    directory = Path(directory)
    manifest = json.loads((directory / "workload.json").read_text())
    config = manifest["config"]
    requested = config["requests"]
    max_revisits = config.get("max_revisits")
    if type(requested) is not int or requested <= 0:
        raise ValueError("Saved request upper bound must be a positive integer")
    if max_revisits is not None and (type(max_revisits) is not int or max_revisits < 0):
        raise ValueError("Saved max_revisits must be a nonnegative integer or None")
    if type(num_users) is not int or num_users <= 0:
        raise ValueError("Saved population must be a positive integer")
    expected_actual = (
        requested if max_revisits is None else min(requested, num_users * (max_revisits + 1))
    )
    if manifest.get("schema_version") != 1 or any(
        config.get(key) != value
        for key, value in {
            "model": "nosa",
            "num_users": num_users,
            "history_tokens": prefix,
            "candidate_tokens": queries,
        }.items()
    ):
        raise ValueError(
            "Profile requires the chosen saved NOSA population, matching prefix/candidate workload"
        )
    identities, counts, prefixes, candidates = [], Counter(), {}, {}
    selected = []
    with (directory / "requests.jsonl").open() as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            user, ids = row["user_id"], row["input_ids"]
            if (
                not isinstance(ids, list)
                or len(ids) != prefix + queries
                or any(type(value) is not int or value < 0 for value in ids)
            ):
                raise ValueError(f"Request {index}: malformed saved token IDs")
            if (
                type(row["request_id"]) is not int
                or row["request_id"] != index
                or type(row["visit_index"]) is not int
                or row["visit_index"] != counts[user]
                or type(row["visit_number"]) is not int
                or row["visit_number"] != counts[user] + 1
                or type(row["is_revisit"]) is not bool
                or row["is_revisit"] != (counts[user] > 0)
                or row["stable_prefix_tokens"] != prefix
                or row["candidate_suffix_tokens"] != queries
            ):
                raise ValueError(f"Request {index}: invalid visit identity or token boundary")
            for name, tokens in (
                ("prefix_sha256", ids[:prefix]),
                ("candidate_sha256", ids[prefix:]),
                ("input_sha256", ids),
            ):
                if row[name] != _hash_json(tokens):
                    raise ValueError(f"Request {index}: {name} does not match saved token IDs")
            if prefixes.setdefault(user, row["prefix_sha256"]) != row["prefix_sha256"]:
                raise ValueError(f"Request {index}: user history changed across visits")
            if candidates.get(user) == row["candidate_sha256"]:
                raise ValueError(f"Request {index}: consecutive candidate suffixes are identical")
            candidates[user] = row["candidate_sha256"]
            counts[user] += 1
            if max_revisits is not None and counts[user] > max_revisits + 1:
                raise ValueError(f"Request {index}: user exceeds the saved revisit cap")
            identities.append({key: row[key] for key in IDENTITY_KEYS})
            if row["is_revisit"] and len(selected) < SAMPLES:
                selected.append(row)
    if len(identities) != expected_actual:
        raise ValueError("Saved request trace length differs from its declared request count")
    if len(counts) > num_users:
        raise ValueError("Saved trace contains more users than its declared population")
    expected_observed = {
        "requests": len(identities),
        "unique_users": len(counts),
        "first_visits": len(counts),
        "revisits": sum(count - 1 for count in counts.values()),
        "max_visits": max(counts.values(), default=0),
        "max_revisits": max((count - 1 for count in counts.values()), default=0),
    }
    observed = manifest["observed"]
    for key, expected in expected_observed.items():
        # Uncapped older manifests predate the two maximum-count summaries.
        if max_revisits is None and key in ("max_visits", "max_revisits") and key not in observed:
            continue
        if type(observed.get(key)) is not int or observed[key] != expected:
            raise ValueError(f"Saved observed {key} differs from the actual trace")
    identity = {
        "config": config,
        "heat_sha256": manifest["heat_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "requests": identities,
    }
    if identities != manifest["requests"] or _hash_json(identity) != manifest["workload_sha256"]:
        raise ValueError("Saved workload manifest does not authenticate the request trace")
    if len(selected) != SAMPLES:
        raise ValueError("Saved trace must contain at least three revisit requests")
    return manifest, selected


def validate_latency_run(
    directory,
    run_id,
    workload_sha256,
    *,
    num_users=USERS,
    prefix=PREFIX,
    queries=QUERIES,
    request_count=None,
):
    """Bind a valid saved workload to the accepted formal measurements."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("status") != "accepted":
        raise ValueError("Wait for the formal latency run to finish and be accepted")
    if metadata["run_id"] == run_id:
        raise ValueError("Profile and latency runs must have different run IDs")
    parameters = metadata["parameters"]
    if (
        "nosa" not in parameters["models"]
        or num_users not in parameters["users"]
        or parameters["history_tokens"] != prefix
        or parameters["candidate_tokens"] != queries
        or parameters["chunk_size"] <= 0
    ):
        raise ValueError(
            "Formal latency geometry must include the chosen NOSA population, matching prefix/candidate geometry"
        )
    cases = [c for c in metadata["cases"] if c["model"] == "nosa" and c["num_users"] == num_users]
    if Counter(c["scheme"] for c in cases) != Counter(
        {"hbm": 1, "serial_sparse": 1, "dense_prefetch": 1, "overlap": 1}
    ):
        raise ValueError("Formal NOSA population comparison is incomplete")
    case_counts = {case["requests"] for case in cases}
    if (
        len(case_counts) != 1
        or any(type(count) is not int or count <= 0 for count in case_counts)
        or (request_count is not None and case_counts != {request_count})
    ):
        raise ValueError("Formal cases must measure the complete same saved request trace")
    schemes = {case["scheme"] for case in cases}
    seen = Counter()
    with (directory / "measurements.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["model"] == "nosa" and row["num_users"] == num_users:
                if row["workload_sha256"] != workload_sha256:
                    raise ValueError("Saved workload differs from formal measured workload hash")
                scheme = row["scheme"]
                if (
                    scheme not in schemes
                    or type(row["request_id"]) is not int
                    or row["request_id"] != seen[scheme]
                ):
                    raise ValueError("Formal request IDs must cover each complete trace in order")
                seen[scheme] += 1
    if any(seen[c["scheme"]] != c["requests"] for c in cases):
        raise ValueError("Formal workload measurement rows are incomplete")
    return metadata


def validate_selection(ids, valid, *, prefix=PREFIX, queries=QUERIES):
    """Check consumed selections before deriving expected logical copy rows."""
    import torch

    if (
        ids.device.type != "cpu"
        or valid.device.type != "cpu"
        or ids.shape != (queries, 2, 64)
        or valid.shape != ids.shape
        or ids.dtype not in (torch.int32, torch.int64)
        or valid.dtype != torch.bool
    ):
        raise ValueError("Expected CPU selection IDs/mask of shape [queries,2,64]")
    for query in range(queries):
        for head in range(2):
            selected = ids[query, head][valid[query, head]]
            if (
                not selected.numel()
                or selected.unique().numel() != selected.numel()
                or bool((selected < 0).any())
                or bool((selected > (prefix + query) // 64).any())
            ):
                raise ValueError("Consumed selection has empty, duplicate, or noncausal blocks")


def analyze_profile(profile, case_labels):
    """Validate raw work, then retain all samples irrespective of the 90% claim."""
    from experiments.nosa_offload_overlap.src.analyze import (
        stripe_interval_metrics,
        work_interval_metrics,
        work_profile_metadata,
    )

    semantics = work_profile_metadata(profile)
    if profile["schema_version"] != 3:
        raise ValueError("Serving diagnostic requires native work schema 3")
    labels = set(case_labels)
    expected = {(case, mode, 0) for case in labels for mode in MODES.values()}
    records = profile["records"]
    actual = [(r["case"], r["mode"], r["sample"]) for r in records]
    if len(labels) != SAMPLES or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected three cases with one serialized and overlap record each")
    if set(profile["cases"]) != labels:
        raise ValueError("Unexpected or missing profile case geometry")
    analyzed, claims = [], []
    for record in records:
        if type(record["recorded_transfer_bytes"]) is not int:
            raise ValueError("Native transfer counters must be integers")
        case = profile["cases"][record["case"]]
        page = work_interval_metrics(case, record)
        stripe = stripe_interval_metrics(case, record, profile["fetch_stripes"])
        if page is None:
            raise ValueError("Serial control must retain its actual page-copy rows")
        if record["mode"] == "serialized" and page["softmax_interval_count"]:
            raise ValueError("Serial native profile does not instrument attention softmax work")
        claim = None
        if record["mode"] == "overlap":
            claim = (
                page["fetch_math_overlap_fraction"] >= 0.9
                and stripe["fetch_stripe_math_overlap_fraction"] >= 0.9
            )
            claims.append(claim)
        analyzed.append(
            {
                "case": record["case"],
                "mode": record["mode"],
                "sample": record["sample"],
                "work_metrics": page,
                "stripe_metrics": stripe,
                "both_overlap_ratios_at_least_90_percent": claim,
            }
        )
    return {
        "schema_version": 3,
        "run_id": profile["run_id"],
        **semantics,
        "status": "accepted",
        "records": analyzed,
        "overlap_claim": {
            "threshold": 0.9,
            "samples": len(claims),
            "all_samples_both_ratios_at_least_90_percent": all(claims),
            "below_threshold_samples_remain_valid": True,
        },
        "definition": (
            "Independent page-envelope and nonempty stripe-copy unions intersect actual "
            "consumer softmax intervals on device globaltimer. Exact stripe min/max define "
            "each page envelope. Softmax is partial attention coverage; these ratios are "
            "not wire traffic or full-attention latency hiding. Serial records have page "
            "copy windows only; unmeasured math overlap is null, not inferred zero."
        ),
        "scope": "Layer 31, first three revisits of the chosen saved NOSA population, one sample each",
        "latency_source": "Separate uninstrumented formal run only",
    }


def _snapshot_sources(output, root):
    from experiments.gr_serving.src.measure import source_snapshot

    source_snapshot(output)
    manifest = json.loads((output / "source_manifest.json").read_text())
    paths = {root / "pyproject.toml", root / "uv.lock", Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not isinstance(name, (str, os.PathLike)):
            continue
        try:
            path = Path(name).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            # Some module proxies advertise synthetic names such as _classes.py.
            continue
        if (
            path.is_file()
            and path.is_relative_to(root)
            and path.suffix == ".py"
            and path.relative_to(root).parts[0] != ".venv"
        ):
            paths.add(path)
    for path in sorted(paths):
        relative = path.relative_to(root)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = _sha256(path)
    _write(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _verify_formal_sources(directory, root):
    manifest = json.loads((directory / "source_manifest.json").read_text())
    formal = json.loads((directory / "metadata.json").read_text())
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != formal["source_sha256"]:
        raise ValueError("Formal source manifest digest differs from accepted metadata")
    diagnostic_only = {
        "experiments/gr_serving/src/profile.py",
        "experiments/gr_serving/tests/test_profile.py",
    }
    changed, allowed = [], []
    for name, expected in manifest.items():
        if name in diagnostic_only:
            saved = directory / "source" / name
            if not saved.is_file() or _sha256(saved) != expected:
                raise RuntimeError(f"Saved formal diagnostic source differs from manifest: {name}")
        current = root / name
        if not current.is_file():
            changed.append(name)
            continue
        actual = _sha256(current)
        if actual == expected:
            continue
        if name not in diagnostic_only:
            changed.append(name)
            continue
        allowed.append(
            {
                "path": name,
                "formal_sha256": expected,
                "profile_sha256": actual,
                "formal_source_copy_verified": True,
            }
        )
    if changed:
        raise RuntimeError(f"Formal sources changed before/during profiling: {changed}")
    return {
        "source_sha256": digest,
        "allowed_diagnostic_source_changes": allowed,
        "reason": (
            "Only the post-run profile entry point and its CPU tests may differ. Neither "
            "is executed by the formal latency measurement. Saved formal copies are verified "
            "against its manifest; current diagnostic copies have their own profile snapshot. "
            "All other formal sources, including performance/runtime code, must be unchanged."
        ),
    }


def _checkpoint_identity(directory):
    return {
        "path": str(directory.resolve()),
        "metadata_sha256": {p.name: _sha256(p) for p in sorted(directory.glob("*.json"))},
        "weight_files": [
            {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(directory.glob("*.safetensors"))
        ],
        "weight_identity_boundary": "sizes and mtimes; weight-file contents are not hashed",
    }


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _profile_request(
    backend, request, case_label, output, expected_hbm, native_profile, *, prefix, queries
):
    import torch

    from experiments.gr_serving.src.measure import numerical_comparison
    from experiments.nosa_offload_overlap.src.measure import (
        expected_fetch_rows,
        prefix_transfer_bytes,
    )

    mode = MODES[backend.scheme]
    sample_dir = output / "samples" / case_label / mode
    sample_dir.mkdir(parents=True, exist_ok=False)
    ids = torch.tensor(request["input_ids"], dtype=torch.long, device=backend.device)
    session = backend.create_session(prefix + queries)
    original_attention = backend.model.main_attention
    held, calls, workspace = {}, [], None
    try:
        if session.length != 0:
            raise AssertionError("Each diagnostic prefix must start with an empty session")
        backend.prefill(session, ids[:prefix])
        backend.synchronize()
        if session.length != prefix:
            raise AssertionError("Independent sparse prefix construction did not complete")
        control_gpu = backend.extend(session, ids[prefix:])
        backend.synchronize()
        control = control_gpu.detach().cpu()
        del control_gpu
        control_check = numerical_comparison(control, expected_hbm, atol=0.0, rtol=0.0)
        backend.truncate(session, prefix)
        workspace = session.attention_workspace
        if workspace.query_tile_size != 128 or workspace.profile_work_intervals:
            raise AssertionError("Unexpected workspace configuration before profiling")
        memory_before = {
            "session": backend.session_bytes(session),
            "workspace_capacity_bytes": workspace.capacity_bytes,
            "trace_storage_bytes": _tensor_bytes(workspace._trace_storage),
        }
        torch.cuda.reset_peak_memory_stats(backend.device)

        def attention(q, selection, access, context):
            calls.append(context.layer_idx)
            if context.layer_idx != LAYER:
                return original_attention(q, selection, access, context)
            if (
                context.query_start != prefix
                or context.query_length != queries
                or access is not session
                or q.shape != (queries, 32, 128)
                or q.dtype != torch.bfloat16
                or selection.block_size != 64
                or selection.valid_mask is None
                or held
            ):
                raise AssertionError("Target attention call differs from the profile contract")
            held["ids"] = selection.block_ids.detach()
            held["valid"] = selection.valid_mask.detach()
            workspace.profile_work_intervals = True
            torch.cuda.nvtx.range_push(f"nosa_overlap/{case_label}/{mode}/sample_0")
            try:
                return original_attention(q, selection, access, context)
            finally:
                try:
                    workspace.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
                    workspace.profile_work_intervals = False

        backend.model.main_attention = attention
        try:
            profiled_gpu = backend.extend(session, ids[prefix:])
            backend.synchronize()
        finally:
            backend.model.main_attention = original_attention
        if calls != list(range(32)) or session.length != prefix + queries:
            raise AssertionError("Profile must execute all 32 layers with layer 31 last")

        # Export before any subsequent append or use of this shared workspace.
        selected_ids, valid = held["ids"].cpu(), held["valid"].cpu()
        validate_selection(selected_ids, valid, prefix=prefix, queries=queries)
        rows = expected_fetch_rows(selected_ids, valid, prefix, head_dim=128, element_size=2)
        expected_bytes, expected_tiles = prefix_transfer_bytes(
            selected_ids, valid, prefix, 128, 2, tile_size=128
        )
        recorded_bytes = int(workspace.last_transfer_bytes.item())
        recorded_tiles = workspace.last_tile_transfer_bytes.cpu().tolist()
        if (
            expected_bytes != sum(row["bytes"] for row in rows)
            or recorded_bytes != expected_bytes
            or recorded_tiles != expected_tiles
        ):
            raise AssertionError("Native transfers differ from actual selection's unique union")
        record = {
            "case": case_label,
            "mode": mode,
            "sample": 0,
            "recorded_transfer_bytes": recorded_bytes,
            "recorded_tile_transfer_bytes": recorded_tiles,
            "expected_tile_transfer_bytes": expected_tiles,
            "query_tile_size": 128,
            "intervals": workspace.work_intervals(),
            "stripe_intervals": workspace.stripe_work_intervals(),
        }
        case = {
            "prefix": prefix,
            "queries": queries,
            "kv_heads": 2,
            "expected_prefix_bytes": expected_bytes,
            "expected_fetch_rows": rows,
        }
        # Save observed evidence before numerical gates, retained only in staging on failure.
        raw_trace = workspace.last_work_intervals.detach().cpu()
        torch.save({"ids": selected_ids, "valid_mask": valid}, sample_dir / "selection.pt")
        torch.save(raw_trace, sample_dir / "native_trace.pt")
        _write(sample_dir / "work_record.json", {"case": case, "record": record})
        actual = profiled_gpu.detach().cpu()
        del profiled_gpu
        torch.save(control, sample_dir / "unprofiled_hidden.pt")
        torch.save(actual, sample_dir / "profiled_hidden.pt")
        profile_check = numerical_comparison(actual, control, atol=0.0, rtol=0.0)
        hbm_check = numerical_comparison(actual, expected_hbm, atol=0.0, rtol=0.0)
        memory_after = {
            "session": backend.session_bytes(session),
            "workspace_capacity_bytes": workspace.capacity_bytes,
            "trace_storage_bytes": _tensor_bytes(workspace._trace_storage),
        }
        added_trace = memory_after["trace_storage_bytes"] - memory_before["trace_storage_bytes"]
        if (
            added_trace <= 0
            or (
                memory_after["workspace_capacity_bytes"] - memory_before["workspace_capacity_bytes"]
            )
            != added_trace
        ):
            raise AssertionError("Unexpected profile workspace growth beyond trace storage")
        details = {
            "case": case_label,
            "scheme": backend.scheme,
            "mode": mode,
            "sample": 0,
            "request": {key: request[key] for key in IDENTITY_KEYS},
            "diagnostic_session": "new independent sparse prefix; not an observed formal LRU hit",
            "unprofiled_control_vs_formal_hbm": control_check,
            "profiled_vs_unprofiled_control": profile_check,
            "profiled_vs_formal_hbm": hbm_check,
            "selection_shape": list(selected_ids.shape),
            "selection_dtype": str(selected_ids.dtype),
            "selection_sha256": _sha256(sample_dir / "selection.pt"),
            "raw_trace_sha256": _sha256(sample_dir / "native_trace.pt"),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "profile_added_hbm_trace_storage_bytes": added_trace,
            "retained_selection_hbm_storage_bytes": sum(
                t.untyped_storage().nbytes() for t in held.values()
            ),
            "selection_memory": "existing tensors retained longer; no GPU clone",
            "retained_cpu_tensor_bytes": sum(
                _tensor_bytes(t)
                for t in (control, actual, expected_hbm, selected_ids, valid, raw_trace)
            ),
            "profiled_candidate_process_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                backend.device
            ),
            "profiled_candidate_process_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                backend.device
            ),
            "process_peak_boundary": "includes model/live tensors; reset after control/truncate",
            "profile_storage_budget_boundary": (
                "Diagnostic storage is additional to formal cache reservations. This one-session "
                "diagnostic has no LRU admission and makes no formal budget-occupancy claim."
            ),
        }
        _write(sample_dir / "checks.json", details)
        if case_label in native_profile["cases"] and native_profile["cases"][case_label] != case:
            # Never replace a serial case's expected union with another observed union.
            mismatch = case_label + "__overlap_selection_mismatch"
            native_profile["cases"][mismatch] = case
            native_profile["records"].append({**record, "case": mismatch})
            _write(output / "work_intervals.json", native_profile)
            raise AssertionError("Independent serial/overlap selection unions differ")
        native_profile["cases"][case_label] = case
        native_profile["records"].append(record)
        _write(output / "work_intervals.json", native_profile)
        return details, selected_ids, valid
    finally:
        backend.model.main_attention = original_attention
        if workspace is not None:
            workspace.profile_work_intervals = False
        backend.release_session(session)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--run-id", required=True)
    result.add_argument("--latency-data", type=Path, help="accepted formal run data directory")
    result.add_argument(
        "--output-dir", type=Path, required=True, help="fresh staging outside experiments"
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--num-users", type=int, help="saved population; default 8 if present, otherwise smallest"
    )
    result.add_argument(
        "--verify-only", action="store_true", help="CPU-only accepted-output verification"
    )
    return result


def verify_accepted_output(output, run_id):
    metadata = json.loads((output / "metadata.json").read_text())
    analysis = json.loads((output / "analysis.json").read_text())
    profile = json.loads((output / "work_intervals.json").read_text())
    if (
        metadata.get("status") != "accepted"
        or analysis.get("status") != "accepted"
        or any(item["run_id"] != run_id for item in (metadata, analysis, profile))
        or len(metadata["samples"]) != SAMPLES * len(MODES)
    ):
        raise ValueError("Only complete accepted profiles may be published")
    labels = [f"request_{row['request_id']:06d}" for row in metadata["requests"]]
    keys = {(label, mode, 0) for label in labels for mode in MODES.values()}
    attestations = [(s["case"], s["mode"], s["sample"]) for s in metadata["samples"]]
    if len(set(attestations)) != len(attestations) or set(attestations) != keys:
        raise ValueError("Numerical sample attestations must cover every unique work record")
    recomputed = analyze_profile(profile, labels)
    recomputed["work_intervals_sha256"] = _sha256(output / "work_intervals.json")
    if analysis != recomputed:
        raise ValueError("Saved analysis differs from independent reanalysis of raw intervals")
    import torch

    from experiments.gr_serving.src.measure import numerical_comparison
    from experiments.nosa_offload_overlap.src.measure import (
        expected_fetch_rows,
        prefix_transfer_bytes,
    )

    records = {(r["case"], r["mode"], r["sample"]): r for r in profile["records"]}
    selections = {}
    for sample in metadata["samples"]:
        for name in (
            "unprofiled_control_vs_formal_hbm",
            "profiled_vs_unprofiled_control",
            "profiled_vs_formal_hbm",
        ):
            if not sample[name]["exact"] or sample[name]["max_abs"] != 0:
                raise ValueError("All candidate hidden values must match both controls exactly")
        directory = output / "samples" / sample["case"] / sample["mode"]
        if (
            _sha256(directory / "selection.pt") != sample["selection_sha256"]
            or _sha256(directory / "native_trace.pt") != sample["raw_trace_sha256"]
        ):
            raise ValueError("Saved selection or raw native trace has changed")
        selection = torch.load(directory / "selection.pt", map_location="cpu", weights_only=True)
        ids, valid = selection["ids"], selection["valid_mask"]
        prior = selections.setdefault(sample["case"], (ids, valid))
        if not torch.equal(prior[0], ids) or not torch.equal(prior[1], valid):
            raise ValueError("Saved independently constructed modes consumed different selections")
        record = records[(sample["case"], sample["mode"], sample["sample"])]
        case = profile["cases"][sample["case"]]
        prefix, queries = case["prefix"], case["queries"]
        if "measurement" in metadata and (prefix, queries) != (
            metadata["measurement"]["prefix_tokens"],
            metadata["measurement"]["candidate_tokens"],
        ):
            raise ValueError("Sample geometry differs from the saved measurement contract")
        validate_selection(ids, valid, prefix=prefix, queries=queries)
        expected_bytes, expected_tiles = prefix_transfer_bytes(
            ids, valid, prefix, 128, 2, tile_size=128
        )
        if (
            case["kv_heads"] != 2
            or case["expected_fetch_rows"] != expected_fetch_rows(ids, valid, prefix, 128, 2)
            or case["expected_prefix_bytes"] != expected_bytes
            or record["recorded_transfer_bytes"] != expected_bytes
            or record["recorded_tile_transfer_bytes"] != expected_tiles
            or record["expected_tile_transfer_bytes"] != expected_tiles
            or record["query_tile_size"] != 128
        ):
            raise ValueError("Saved consumed selections do not authenticate transfer provenance")
        raw = torch.load(directory / "native_trace.pt", map_location="cpu", weights_only=True)
        capacity = ((prefix + 63) // 64) * 2 * (1 + profile["fetch_stripes"])
        capacity += ((queries + 7) // 8) * 2 * 64
        if raw.dtype != torch.int64 or raw.shape != (capacity, 4):
            raise ValueError("Invalid saved native trace geometry or dtype")
        raw_rows = [
            {"row": i, "start_ns": row[0], "end_ns": row[1], "bytes": row[2], "kind": row[3]}
            for i, row in enumerate(raw.tolist())
            if row[3] != 0
        ]
        if (
            any(row["kind"] not in (1, 2, 3) for row in raw_rows)
            or [row for row in raw_rows if row["kind"] in (1, 2)] != record["intervals"]
            or [row for row in raw_rows if row["kind"] == 3] != record["stripe_intervals"]
        ):
            raise ValueError("Extracted work intervals differ from saved raw native trace")
        request_id = sample["request"]["request_id"]
        if sample["case"] != f"request_{request_id:06d}":
            raise ValueError("Sample request identity differs from its case")
        reference = output / "formal_inputs" / f"hbm_{request_id:06d}.pt"
        if _sha256(reference) != metadata["reference_identities"][str(request_id)]["sha256"]:
            raise ValueError("Saved formal HBM reference identity changed")
        hbm = torch.load(reference, map_location="cpu", weights_only=True)
        control = torch.load(
            directory / "unprofiled_hidden.pt", map_location="cpu", weights_only=True
        )
        actual = torch.load(directory / "profiled_hidden.pt", map_location="cpu", weights_only=True)
        for name, left, right in (
            ("unprofiled_control_vs_formal_hbm", control, hbm),
            ("profiled_vs_unprofiled_control", actual, control),
            ("profiled_vs_formal_hbm", actual, hbm),
        ):
            if numerical_comparison(left, right, atol=0.0, rtol=0.0) != sample[name]:
                raise ValueError("Saved complete hidden tensors differ from numerical attestations")
    return metadata


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Run ID must contain only letters, digits, underscores or hyphens")
    if args.verify_only:
        verify_accepted_output(args.output_dir, args.run_id)
        print(f"Verified accepted profile: {args.run_id}")
        return
    if args.latency_data is None:
        raise ValueError("--latency-data is required for GPU profiling")
    # Lazy imports keep --help and --verify-only free of CUDA initialization.
    import torch

    from experiments.gr_serving.src.measure import ROOT, verify_source_snapshot
    from experiments.nosa_offload_overlap.src.measure import native_build_metadata, new_work_profile
    from models.nosa.serving import NosaServingBackend

    latency, output = args.latency_data.resolve(), args.output_dir.resolve()
    if output.is_relative_to(ROOT / "experiments"):
        raise ValueError(
            "Collect outside experiments; scripts/profile.sh publishes after acceptance"
        )
    initial_metadata = json.loads((latency / "metadata.json").read_text())
    parameters = initial_metadata["parameters"]
    prefix, queries, chunk_size = (
        parameters[key] for key in ("history_tokens", "candidate_tokens", "chunk_size")
    )
    if any(type(value) is not int or value <= 0 for value in (prefix, queries, chunk_size)):
        raise ValueError("Formal prefix, candidate, and chunk sizes must be positive integers")
    populations = parameters["users"]
    num_users = (
        args.num_users
        if args.num_users is not None
        else (8 if 8 in populations else min(populations))
    )
    if num_users <= 0 or num_users not in populations:
        raise ValueError("--num-users must identify a population saved in the formal run")
    workload_dir = latency / "workloads/nosa" / str(num_users)
    manifest, requests = load_saved_workload(
        workload_dir, num_users=num_users, prefix=prefix, queries=queries
    )
    formal = validate_latency_run(
        latency,
        args.run_id,
        manifest["workload_sha256"],
        request_count=len(manifest["requests"]),
        num_users=num_users,
        prefix=prefix,
        queries=queries,
    )
    formal_source = _verify_formal_sources(latency, ROOT)
    references = {
        row["request_id"]: latency
        / "reference/nosa"
        / str(num_users)
        / f"{row['request_id']:06d}.pt"
        for row in requests
    }
    if any(not path.is_file() for path in references.values()):
        raise FileNotFoundError("Three accepted formal HBM hidden references are required")
    device = torch.device(args.device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("Post-run diagnostic requires the same SM90/Hopper GPU")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if str(props.uuid) != formal["hardware"]["gpu_uuid"]:
        raise ValueError("Profile must use the formal latency run's physical GPU")
    os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
    if os.environ["CXLDSAGR_SM90_BACKEND"] != "native":
        raise ValueError("Native SM90 backend is required")
    native_build = native_build_metadata()
    profile = new_work_profile(args.run_id, native_build)
    checkpoint = Path(formal["parameters"]["nosa_path"])
    output.mkdir(parents=True, exist_ok=False)
    source_id = _snapshot_sources(output, ROOT)
    evidence = output / "formal_inputs"
    evidence.mkdir()
    for name, source in {
        "latency_metadata.json": latency / "metadata.json",
        "latency_source_manifest.json": latency / "source_manifest.json",
        "workload.json": workload_dir / "workload.json",
    }.items():
        shutil.copyfile(source, evidence / name)
    (evidence / "selected_requests.jsonl").write_text(
        "".join(_json(row) + "\n" for row in requests)
    )
    reference_identities = {}
    for request_id, source in references.items():
        target = evidence / f"hbm_{request_id:06d}.pt"
        shutil.copyfile(source, target)
        reference_identities[str(request_id)] = {"source": str(source), "sha256": _sha256(target)}
    checkpoint_before = _checkpoint_identity(checkpoint)
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "started_unix": time.time(),
        "latency_run_id": formal["run_id"],
        "latency_data": str(latency),
        "formal_source_sha256": formal_source["source_sha256"],
        "formal_source_comparison": formal_source,
        "source_sha256": source_id,
        "workload_sha256": manifest["workload_sha256"],
        "num_users": num_users,
        "workload_config": manifest["config"],
        "workload_observed": manifest["observed"],
        "heat_sha256": manifest["heat_sha256"],
        "access_trace_sha256": manifest.get("access_trace_sha256"),
        "requests_file_sha256": _sha256(workload_dir / "requests.jsonl"),
        "reference_identities": reference_identities,
        "native_build": native_build,
        "checkpoint": checkpoint_before,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host": platform.node(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "flashinfer-python",
                    "apache-tvm-ffi",
                    "safetensors",
                    "triton",
                )
            },
        },
        "measurement": {
            "layer": LAYER,
            "prefix_tokens": prefix,
            "candidate_tokens": queries,
            "prefix_chunk_size": chunk_size,
            "profiled_samples_per_request_and_mode": 1,
            "unprofiled_controls_per_request_and_mode": 1,
            "profile_warmup": 0,
            "selection": "actual main-attention block IDs and validity at layer 31",
            "precision": "BF16; exact all-hidden comparison against control and formal HBM",
            "formal_cache_caps_bytes": {
                tier: formal["parameters"][f"{tier}_budget_bytes"] for tier in ("hbm", "dram")
            },
            "budget_scope": "reference only; separate profile storage and no LRU admission",
            "nsys": os.environ.get("GR_SERVING_NSYS_CAPTURE") == "1",
            "nsys_validation": "supplemental raw capture; existing SQLite analyzer requires third resident mode",
        },
        "requests": [{key: row[key] for key in IDENTITY_KEYS} for row in requests],
        "samples": [],
    }
    _write(output / "metadata.json", metadata)
    backend = NosaServingBackend.from_pretrained(
        checkpoint,
        scheme="serial_sparse",
        device=device,
        chunk_size=chunk_size,
        max_seq_len=prefix + queries,
    )
    metadata["model"] = backend.describe()
    selected_by_case, case_labels = {}, []
    try:
        for scheme, mode in MODES.items():
            backend = NosaServingBackend(backend.model, scheme, chunk_size=chunk_size)
            for request in requests:
                request_id = request["request_id"]
                label = f"request_{request_id:06d}"
                if scheme == "serial_sparse":
                    case_labels.append(label)
                print(
                    _json({"event": "profile_request", "scheme": scheme, "request_id": request_id}),
                    flush=True,
                )
                hbm = torch.load(
                    evidence / f"hbm_{request_id:06d}.pt", map_location="cpu", weights_only=True
                )
                details, ids, valid = _profile_request(
                    backend, request, label, output, hbm, profile, prefix=prefix, queries=queries
                )
                if label in selected_by_case:
                    prior_ids, prior_valid = selected_by_case[label]
                    if not torch.equal(ids, prior_ids) or not torch.equal(valid, prior_valid):
                        _write(
                            output / "selection_mismatch.json",
                            {
                                "case": label,
                                "detail": "Actual IDs/masks differ; both mode files retained",
                            },
                        )
                        raise AssertionError(
                            "Independent serial/overlap consumed selections differ"
                        )
                else:
                    selected_by_case[label] = (ids, valid)
                metadata["samples"].append(details)
                _write(output / "metadata.json", metadata)
                print(
                    _json(
                        {
                            "event": "sample_complete",
                            "case": label,
                            "mode": mode,
                            "bytes": profile["cases"][label]["expected_prefix_bytes"],
                        }
                    ),
                    flush=True,
                )
        analysis = analyze_profile(profile, case_labels)
        analysis["work_intervals_sha256"] = _sha256(output / "work_intervals.json")
        _write(output / "analysis.json", analysis)
        verify_source_snapshot(output)
        if _verify_formal_sources(latency, ROOT) != formal_source:
            raise RuntimeError("Formal/diagnostic source identities changed during profiling")
        if checkpoint_before != _checkpoint_identity(checkpoint):
            raise RuntimeError("Checkpoint metadata or file identities changed during profiling")
        metadata.update(
            status="accepted", completed_unix=time.time(), overlap_claim=analysis["overlap_claim"]
        )
        _write(output / "metadata.json", metadata)
        verify_accepted_output(output, args.run_id)
        print(
            _json({"event": "accepted", "run_id": args.run_id, **analysis["overlap_claim"]}),
            flush=True,
        )
    finally:
        backend.synchronize()


if __name__ == "__main__":
    main()
