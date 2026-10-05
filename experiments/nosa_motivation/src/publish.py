"""Publish clean NOSA timing with independently audited diagnostic evidence.

The check remains the numerical reference. Every wall-latency or wall-MFU
denominator below comes from the explicitly selected clean benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matched_profile(bench, profile):
    for field in ("config", "workload_sha256", "checkpoint", "precision_settings"):
        require(bench[field] == profile[field], f"Benchmark/profile {field} differs")
    require(bench["mode"] == "bench", "Wall metrics require a clean benchmark")
    require(profile["status"] == "diagnostic_valid", "Profile has not passed its audit")


def _build_publication(bench_dir, profile_dir, traces, destination, attention_dir=None):
    from experiments.cache_management.src.report import memory_rows
    from experiments.nosa_motivation.src.matrix_comparison import (
        audit_matrix_reference,
        compare_request_apis,
    )
    from experiments.nosa_motivation.src.profile_audit import audit_profile
    from experiments.nosa_motivation.src.report import (
        async_speedup_gate,
        audit_run,
        write_csv,
        write_report,
    )

    bench_dir, profile_dir, traces, destination = map(
        Path, (bench_dir, profile_dir, traces, destination)
    )
    if destination.exists():
        raise FileExistsError(destination)
    bench, rows, bench_audit = audit_run(bench_dir)
    profile = read(profile_dir / "metadata.json")
    matched_profile(bench, profile)
    receipt = read(bench["correctness_receipt"]["path"])
    # audit_run has already authenticated the exact receipt bytes. The profile
    # auditor independently follows the same benchmark-to-check binding.
    gates = audit_profile(profile_dir, traces, bench_dir)
    keyed = {(row["method"], row["request_id"]): row for row in rows}
    raw_path = profile_dir / profile["matrix_reference"]["file"]
    raw = read(raw_path)
    matrix = audit_matrix_reference(raw, bench["model_config"], bench["config"])
    comparisons, overlap = [], []
    for record in profile["records"]:
        row = keyed[record["method"], record["request_id"]]
        require(
            row["prefix_cache_hit"] == record["instrumented_runner_metrics"]["prefix_cache_hit"],
            "Benchmark and diagnostic history residency differs",
        )
        if record["mode"] == "timeline":
            comparisons.append(
                {
                    "sample": record["sample"],
                    **compare_request_apis(
                        row, bench["model_config"], bench["config"], record["api_analysis"], raw
                    ),
                }
            )
        elif record["method"] == "async_sparse":
            for layer in record["layers"]:
                applies = layer["page_and_stripe_90pct"] is not None
                overlap.append(
                    {
                        "request_id": record["request_id"],
                        "sample": record["sample"],
                        "layer": layer["layer"],
                        "miss_pages": len(layer["geometry"]["expected_fetch_rows"]),
                        "page_ratio": layer["page_metrics"]["fetch_math_overlap_fraction"]
                        if applies
                        else None,
                        "stripe_ratio": layer["stripe_metrics"][
                            "fetch_stripe_math_overlap_fraction"
                        ]
                        if applies
                        else None,
                        "page_and_stripe_90pct": layer["page_and_stripe_90pct"],
                    }
                )
    require(comparisons and overlap, "Publication requires timeline and asynchronous overlap")
    independent = None
    if attention_dir is not None:
        from experiments.nosa_motivation.src.attention_reference_audit import (
            audit_attention_timings,
        )
        from experiments.nosa_motivation.src.profile_attention_reference import (
            audit_reference,
            combine_reused_history,
            combined_comparison,
        )

        attention_dir = Path(attention_dir)
        attention_audit = audit_reference(attention_dir, bench_dir, profile_dir, traces)
        metadata = read(attention_dir / "metadata.json")
        summaries, combined, candidates = {}, [], []
        for index in (0, bench["config"]["num_users"]):
            entry = metadata["captures"][str(index)]
            manifest_path = attention_dir / entry["directory"] / "manifest.json"
            require(digest(manifest_path) == entry["manifest_sha256"], "Capture manifest changed")
            manifest = read(manifest_path)
            manifest["manifest_sha256"] = entry["manifest_sha256"]
            timings = read(attention_dir / entry["benchmark_file"])
            summaries[index] = audit_attention_timings(manifest, timings)
            row = keyed["hbm", index]
            attention = (
                combine_reused_history(summaries[0], summaries[index])
                if index and not row["prefix_cache_hit"]
                else summaries[index]["captured_workload"]
            )
            combined.append(
                combined_comparison(
                    row,
                    matrix["candidate_only" if row["prefix_cache_hit"] else "full_request"],
                    attention,
                    model_config=bench["model_config"],
                    config=bench["config"],
                )
            )
            candidate_row = {
                **row,
                "latency_ms": row["extend_ms"],
                "effective_work": {
                    **row["effective_work"],
                    "request_flops": row["effective_work"]["candidate_flops"],
                },
            }
            candidate = combined_comparison(
                candidate_row,
                matrix["candidate_only"],
                summaries[index]["candidate_only"],
                model_config=bench["model_config"],
                config=bench["config"],
            )
            candidate["accepted_candidate_wall_ms"] = candidate.pop("accepted_runner_wall_ms")
            candidate["accepted_candidate_mfu_pct"] = candidate.pop("accepted_runner_mfu_pct")
            candidate["useful_candidate_matrix_flops"] = candidate.pop("full_useful_matrix_flops")
            candidate["denominator_boundary"] = (
                "Clean benchmark extend_ms; excludes admission, prefix construction and separate "
                "runner cleanup_ms. Backend-internal discard and lease drain remain inside extend_ms."
            )
            candidates.append(candidate)
        independent = {
            "run_id": metadata["run_id"],
            "audit": attention_audit,
            "metadata_sha256": digest(attention_dir / "metadata.json"),
            "comparisons": combined,
            "candidate_comparisons": candidates,
        }
    write_report(bench_dir, destination)
    denominator = {
        "run_id": bench["run_id"],
        "source_sha256": bench["source_sha256"],
        "metadata_sha256": digest(bench_dir / "metadata.json"),
        "measurements_sha256": digest(bench_dir / "measurements.jsonl"),
        "boundary": bench["measurement_boundary"],
    }
    write_csv(destination / "memory.csv", memory_rows("nosa", bench, rows))
    write_csv(destination / "async_overlap.csv", overlap)
    write(destination / "profile_review.json", gates)
    write(
        destination / "api_comparison.json",
        {
            "denominator": denominator,
            "profile_run_id": profile["run_id"],
            "request_comparisons": comparisons,
            "independent_attention": independent,
        },
    )
    write(
        destination / "profile_summary.json",
        {
            "run_id": profile["run_id"],
            "numerical_reference_run_id": profile["reference_run_id"],
            "denominator": denominator,
            "records": [
                {k: r[k] for k in ("method", "request_id", "sample", "analysis", "api_analysis")}
                for r in profile["records"]
                if r["mode"] == "timeline"
            ],
        },
    )
    acceptance = {
        "measurement_integrity": "passed",
        "profile_integrity": "passed",
        "denominator": denominator,
        "async_end_to_end_speedup": async_speedup_gate(rows),
        "async_every_sample_page_and_stripe_90pct": gates[
            "async_every_sample_page_and_stripe_90pct"
        ],
        "raw_matrix_api_mfu_proximity": "requires interpretation of separately reported API compositions",
        "compute_io_dominance": "requires interpretation of separate complete timelines",
        "independent_attention_integrity": "passed" if independent else "not measured",
    }
    write(destination / "acceptance.json", acceptance)
    summary = read(destination / "summary.json")
    lines = [
        f"# NOSA motivation: `{bench['run_id']}`",
        "",
        bench["model_boundary"],
        "",
        bench["measurement_boundary"],
        "",
        "| Method | Visit | Requests | History hits | Request mean ms | Candidate mean ms | Candidate MFU % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {r['method']} | {r['visit_kind']} | {r['requests']} | {r['prefix_hits']} | "
        f"{r['latency_mean_ms']:.6f} | {r['extend_mean_ms']:.6f} | {r['candidate_wall_mfu_pct']:.6f} |"
        for r in summary
    )
    lines += [
        "",
        (
            f"Diagnostic source: `{profile['run_id']}`. Numerical reference: "
            f"`{profile['reference_run_id']}`. Every runner wall denominator in "
            f"api_comparison.json comes from clean bench `{bench['run_id']}`."
        ),
        "",
        (
            "Async versus synchronous sparse whole-trace latency gate: "
            f"**{acceptance['async_end_to_end_speedup']['status']}**. "
            "Every applicable layer/sample page-and-stripe 90% overlap gate: "
            f"**{acceptance['async_every_sample_page_and_stripe_90pct']['status']}**."
        ),
        "",
        (
            "[API comparisons](api_comparison.json) retain complete API spans and activity "
            "unions separately. Independent matrix/attention medians form composition "
            "estimates, not measured complete requests. Timing validity does not imply "
            "a speedup or a passed efficiency gate."
        ),
        "",
        (
            "[Memory](memory.csv) separates allocator peaks, after-request device samples, "
            "cache charges and cache reservations. Charges include graph private reservations; "
            "they are not a tensor-payload sum. These observations do not validate "
            "physical capacity at a filled NH quota."
        ),
        "",
        f"Benchmark source: `{bench['source_sha256']}`. Workload: `{bench['workload_sha256']}`.",
        "",
    ]
    (destination / "results.md").write_text("\n".join(lines))
    source = destination / "source"
    source.mkdir()
    (source / "publish.py").write_bytes(Path(__file__).read_bytes())
    write(destination / "command.json", {"argv": sys.orig_argv, "cwd": str(Path.cwd())})
    from evaluation.provenance import snapshot_report_helpers

    snapshot_report_helpers(destination)
    write(
        destination / "publication.json",
        {
            "schema": "nosa-motivation-independent-publication-v1",
            "measurement_run_id": bench["run_id"],
            "profile_run_id": profile["run_id"],
            "source_sha256": bench["source_sha256"],
            "generator_sha256": digest(__file__),
            "generator_snapshot": "source/publish.py",
            "command_sha256": digest(destination / "command.json"),
            "numerical_receipt_sha256": digest(bench["correctness_receipt"]["path"]),
            "numerical_receipt_schema": receipt["schema"],
            "benchmark_audit": bench_audit,
            "profile_metadata_sha256": digest(profile_dir / "metadata.json"),
            "assets": [
                {"path": p.name, "sha256": digest(p)}
                for p in sorted(destination.iterdir())
                if p.is_file()
            ],
        },
    )
    return acceptance


def publish(bench_dir, profile_dir, traces, destination, attention_dir=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        result = _build_publication(
            bench_dir, profile_dir, traces, staging / "publication", attention_dir
        )
        (staging / "publication").rename(destination)
    except BaseException as error:
        try:
            shutil.rmtree(staging)
        except BaseException as cleanup_error:  # noqa: BLE001 -- preserve publication and cleanup.
            raise BaseExceptionGroup(
                "publication and staging cleanup failed", [error, cleanup_error]
            ) from None
        raise
    else:
        staging.rmdir()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--profile-data", required=True, type=Path)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--attention-data", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = publish(
        args.bench, args.profile_data, args.profile_dir, args.output_dir, args.attention_data
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
