"""Prepare or install a hash-guarded NOSA subset of three replaced mixed GR runs.

The default is read-only. Preparation writes only outside the repository. Apply
requires a published replacement ledger; it never reruns or reaccepts old models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from types import ModuleType

EXPERIMENT = Path("experiments/gr_serving")
SPECS = {
    "h4k": ("67f91c34ead43d4056500661caa1c67e8ef8a50bc25327e081a0561bbab43060", 804, 201, "02"),
    "h16k": ("45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c", 804, 201, "01"),
    "h64k": ("45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c", 168, 42, "01"),
}
SCHEMA = "echo-gr-nosa-subset-v1"
REPORT_FILES = {
    "audit.json",
    "cache_and_memory.csv",
    "metadata.json",
    "per_request.csv",
    "per_request.png",
    "per_request.svg",
    "population_coverage.csv",
    "profile_analysis.json",
    "profile_metadata.json",
    "provenance.json",
    "source_manifest.json",
    "summary.csv",
    "summary.json",
    "summary.png",
    "summary_readable.png",
    "summary_readable.svg",
    "summary.svg",
}
READABLE_REQUESTS = {"per_request_readable.png", "per_request_readable.svg"}
ANALYSIS_FILES = {
    "summary.csv",
    "summary.json",
    "summary.svg",
    "per_request.csv",
    "per_request.svg",
}
NOTE = (
    "Retained NOSA subset of the original mixed run, not a new measurement. "
    "Original numerical and latency evidence is unchanged for NOSA. Original "
    "cache reservations and boundary samples do not establish complete hard-budget "
    "acceptance, and do not validate the later NOSA shared-workspace implementation."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def safe_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe relative path")
    path = root / relative
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), f"symlink is outside the cleanup contract: {parent}")
        if parent == root:
            break
    require(path.resolve().is_relative_to(root.resolve()), "path escapes repository")
    return path


def inventory(root, relative):
    directory = safe_path(root, relative)
    require(directory.is_dir(), f"missing directory: {relative}")
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"symlink in preserved evidence: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def select_lines(blob, *, log=False):
    """Keep original NOSA JSONL bytes, rejecting ambiguous or unknown records."""
    selected = []
    for line in blob.splitlines(keepends=True):
        require(bool(line.strip()), "blank JSONL record")
        record = json.loads(line)
        if log and record.get("event") == "accepted" and "model" not in record:
            continue  # The full-matrix acceptance is not subset acceptance.
        require(record.get("model") in {"deepseek_v32", "nosa"}, "unknown model in JSONL")
        if record["model"] == "nosa":
            selected.append(line)
    return b"".join(selected)


def select_csv(blob):
    """CSV tables are regenerated; raw measurements remain byte-identical lines."""
    source = csv.DictReader(io.StringIO(blob.decode()))
    rows = list(source)
    require(rows and "model" in rows[0], "CSV has no model column")
    require({row["model"] for row in rows} == {"deepseek_v32", "nosa"}, "unexpected CSV models")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=source.fieldnames)
    writer.writeheader()
    writer.writerows(row for row in rows if row["model"] == "nosa")
    return output.getvalue().encode()


def metadata_subset(original, original_sha):
    value = deepcopy(original)
    require(value["status"] == "accepted", "original run is not accepted")
    require(set(value["parameters"]["models"]) == {"deepseek_v32", "nosa"}, "not a mixed run")
    value["parameters"]["models"] = ["nosa"]
    value["models"] = {"nosa": value["models"]["nosa"]}
    value["cases"] = [row for row in value["cases"] if row["model"] == "nosa"]
    value["measured_requests"] = sum(row["requests"] for row in value["cases"])
    value["status"] = "retained_subset"
    value["subset_retention"] = {
        "schema": SCHEMA,
        "original_metadata_sha256": original_sha,
        "original_run_id": original["run_id"],
        "original_status": original["status"],
        "retained_models": ["nosa"],
        "removed_models": ["deepseek_v32"],
        "note": NOTE,
    }
    return value


def load_original_report(data, manifest):
    name = "experiments/gr_serving/src/report.py"
    path = data / "source" / name
    require(digest(path) == manifest[name], "original report source changed")
    # Do not use SourceFileLoader: even a read-only audit must not create .pyc
    # files inside the historical source snapshot.
    module = ModuleType("_retained_original_report")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def render_subset(summary, samples, output, label):
    """One NOSA model, categorical populations, explicit gaps for empty scopes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": SCHEMA})
    colors = {
        "hbm": "#0072b2",
        "serial_sparse": "#d55e00",
        "dense_prefetch": "#cc79a7",
        "overlap": "#009e73",
    }
    users = sorted({row["num_users"] for row in samples})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, scope in zip(axes, ("all", "revisit"), strict=True):
        for scheme, color in colors.items():
            rows = sorted(
                (
                    row
                    for row in summary["groups"]
                    if row["scope"] == scope and row["scheme"] == scheme
                ),
                key=lambda row: row["num_users"],
            )
            for metric, style in (("mean_ms", "-"), ("p95_ms", "--")):
                axis.plot(
                    range(len(users)),
                    [row[metric] if row[metric] is not None else float("nan") for row in rows],
                    style,
                    color=color,
                    marker="o",
                    label=scheme if metric == "mean_ms" else None,
                )
            for index, row in enumerate(rows):
                if scheme == "hbm" and row["count"] == 0:
                    axis.axvspan(index - 0.35, index + 0.35, color="#eeeeee")
                    axis.text(
                        index,
                        0.95,
                        "n=0",
                        ha="center",
                        va="top",
                        transform=axis.get_xaxis_transform(),
                    )
        axis.set(
            xticks=range(len(users)),
            xticklabels=users,
            xlabel="Configured population (categorical)",
            ylabel="Request latency (ms)",
            title="All requests" if scope == "all" else "Revisits (including rebuilds)",
            ylim=(0, None),
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"NOSA · {label} + 128 · retained original run subset\nSolid: mean; dashed: p95. Not a new measurement.",
        fontsize=11,
    )
    for name in ("summary", "summary_readable"):
        fig.savefig(output / f"{name}.svg", metadata={"Date": None})
        fig.savefig(output / f"{name}.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(users), 1, figsize=(11, 2.1 * len(users)), layout="constrained", squeeze=False
    )
    for axis, population in zip(axes[:, 0], users, strict=True):
        for scheme, color in colors.items():
            rows = sorted(
                (
                    row
                    for row in samples
                    if row["num_users"] == population and row["scheme"] == scheme
                ),
                key=lambda row: row["request_id"],
            )
            axis.plot(
                [row["request_id"] for row in rows],
                [row["latency_ms"] for row in rows],
                color=color,
                linewidth=1,
                label=scheme,
            )
            for revisit in (False, True):
                chosen = [row for row in rows if (row["visit_index"] > 0) == revisit]
                axis.scatter(
                    [row["request_id"] for row in chosen],
                    [row["latency_ms"] for row in chosen],
                    edgecolors=color,
                    facecolors=color if revisit else "white",
                    s=12,
                )
        ids = sorted({row["request_id"] for row in samples if row["num_users"] == population})
        axis.set(
            xticks=ids[:: max(1, len(ids) // 10)],
            xlabel="Request ID",
            ylabel="Latency (ms)",
            title=f"NOSA · configured users={population}",
            ylim=(0, None),
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.suptitle(
        f"{label} original NOSA requests · hollow: first visit; filled: revisit", fontsize=11
    )
    for name in ("per_request", "per_request_readable"):
        fig.savefig(output / f"{name}.svg", metadata={"Date": None})
        fig.savefig(output / f"{name}.png", dpi=140)
    plt.close(fig)


def inspect_run(repo, label):
    expected_source, expected_rows, expected_refs, profile_suffix = SPECS[label]
    run = f"gr_serving_h200_20261002_{label}_01"
    data_rel = EXPERIMENT / "output/data" / run
    report_rel = EXPERIMENT / "report" / label
    log_rel = EXPERIMENT / "output/log" / run
    profile_run = f"gr_serving_h200_20261002_{label}_profile_{profile_suffix}"
    profile_rel = EXPERIMENT / "output/data" / profile_run
    roots = [data_rel, report_rel, log_rel, profile_rel]
    before = {name: value for root in roots for name, value in inventory(repo, root).items()}
    data = repo / data_rel
    meta, audit, manifest = (
        read(data / name) for name in ("metadata.json", "audit.json", "source_manifest.json")
    )
    require(meta["run_id"] == run and audit["run_id"] == run, "run identity differs")
    require(meta["source_sha256"] == expected_source, "unexpected original implementation")
    require(audit["status"] == "accepted", "original audit is not accepted")
    require(
        hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        == expected_source,
        "source aggregate differs",
    )
    for name, expected in manifest.items():
        require(
            digest(safe_path(data / "source", name)) == expected, f"original source changed: {name}"
        )
    for name, expected in audit["evidence_sha256"].items():
        require(digest(safe_path(data, name)) == expected, f"old audited evidence changed: {name}")
    references = {
        str(Path(name).relative_to(data_rel)): value
        for name, value in before.items()
        if Path(name).is_relative_to(data_rel / "reference")
    }
    reference_digest = hashlib.sha256(
        json.dumps(references, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    require(
        reference_digest == audit["references"]["file_hash_manifest_sha256"],
        "original CPU-audited reference inventory changed",
    )
    selected = select_lines((data / "measurements.jsonl").read_bytes())
    correctness = select_lines((data / "correctness.jsonl").read_bytes())
    rows = [json.loads(line) for line in selected.splitlines()]
    require(
        len(rows) == expected_rows == len(correctness.splitlines()),
        "retained request counts differ",
    )
    require(
        len([name for name in references if name.startswith("reference/nosa/")]) == expected_refs,
        "retained references incomplete",
    )
    original_report = load_original_report(data, manifest)
    summary = original_report.summarize(rows)
    old_summary = read(data / "analysis/summary.json")
    require(
        summary
        == {
            **old_summary,
            "groups": [row for row in old_summary["groups"] if row["model"] == "nosa"],
        },
        "recomputed NOSA groups changed",
    )
    require(len(summary["groups"]) == 84, "retained summary groups incomplete")
    expected_empty = 16 if label == "h64k" else 0
    require(
        sum(row["count"] == 0 for row in summary["groups"]) == expected_empty,
        "empty revisit group coverage changed",
    )
    profile = repo / profile_rel
    profile_meta = read(profile / "metadata.json")
    require(
        profile_meta["status"] == "accepted" and profile_meta["latency_run_id"] == run,
        "profile identity differs",
    )
    require(
        (profile / "formal_inputs/latency_metadata.json").read_bytes()
        == (data / "metadata.json").read_bytes(),
        "profile has different original formal metadata",
    )
    require(
        (profile / "formal_inputs/latency_source_manifest.json").read_bytes()
        == (data / "source_manifest.json").read_bytes(),
        "profile formal source differs",
    )
    # Unknown extra files must be reviewed, never silently retained as mixed results.
    require(
        {p.name for p in data.iterdir()}
        == {
            "analysis",
            "audit.json",
            "correctness.jsonl",
            "measurements.jsonl",
            "metadata.json",
            "reference",
            "render_report.py",
            "source",
            "source_manifest.json",
            "workloads",
        },
        "unexpected old run files",
    )
    require(
        {p.name for p in (repo / log_rel).iterdir()}
        == {"measure.stdout.log", "measure.stderr.log"},
        "unexpected log files",
    )
    require(
        not (repo / log_rel / "measure.stderr.log").read_bytes(),
        "nonempty mixed stderr needs manual classification",
    )
    expected_report = REPORT_FILES | (READABLE_REQUESTS if label == "h64k" else set())
    require(
        {p.name for p in (repo / report_rel).iterdir()} == expected_report,
        "unexpected report files require scoped review",
    )
    require(
        {p.name for p in (data / "analysis").iterdir()} == ANALYSIS_FILES,
        "unexpected analysis files require scoped review",
    )
    return {
        "run_id": run,
        "label": label,
        "data": data_rel,
        "report": report_rel,
        "log": log_rel,
        "profile": profile_rel,
        "roots": roots,
        "before": before,
        "meta": meta,
        "audit": audit,
        "manifest": manifest,
        "summary": summary,
        "rows": rows,
        "selected": selected,
        "correctness": correctness,
        "original_report": original_report,
        "reference_files": expected_refs,
    }


def preview(repo):
    results = [inspect_run(repo, label) for label in SPECS]
    return results, {
        "schema": SCHEMA,
        "mode": "dry_run",
        "repository_unchanged": True,
        "runs": [
            {
                "run_id": item["run_id"],
                "retained_measurements": len(item["rows"]),
                "retained_cases": 28,
                "retained_summary_groups": 84,
                "retained_references": item["reference_files"],
                "checked_original_files": len(item["before"]),
            }
            for item in results
        ],
        "preserved": [
            "all original source snapshots/manifests",
            "NOSA raw rows, workloads and references",
            "original NOSA profile metadata, samples and sources",
            "all independent DeepSeek nonmatrix/MFU reports and runs",
        ],
        "pending": [
            "publish accepted replacement measurements and reviewed default",
            "review prepared plots and README edits",
            "apply a freshly verified external preparation",
        ],
    }


def prepare(repo, destination):
    destination = destination.resolve()
    require(
        not destination.is_relative_to(repo.resolve()), "preparation must be outside repository"
    )
    require(not destination.exists(), "preparation destination already exists")
    items, result = preview(repo)
    destination.mkdir(parents=True)
    payload = destination / "files"
    before, roots, removed, audits = {}, [], [], []
    for item in items:
        before.update(item["before"])
        roots.extend(map(str, item["roots"]))
        data_rel, report_rel = item["data"], item["report"]
        data, report = payload / data_rel, payload / report_rel
        data.mkdir(parents=True)
        report.mkdir(parents=True)
        meta = metadata_subset(item["meta"], before[str(data_rel / "metadata.json")])
        for name, blob in {
            "metadata.json": json_bytes(meta),
            "measurements.jsonl": item["selected"],
            "correctness.jsonl": item["correctness"],
        }.items():
            (data / name).write_bytes(blob)
        item["original_report"].write_report(item["rows"], data / "analysis")
        render_subset(item["summary"], item["rows"], report, item["label"])
        for name in ("summary.svg", "per_request.svg"):
            shutil.copyfile(report / name, data / "analysis" / name)
        for name in ("summary.json", "summary.csv", "per_request.csv"):
            shutil.copyfile(data / "analysis" / name, report / name)
        for name in (
            "source_manifest.json",
            "profile_metadata.json",
            "profile_analysis.json",
            "population_coverage.csv",
        ):
            shutil.copyfile(repo / report_rel / name, report / name)
        shutil.copyfile(data / "metadata.json", report / "metadata.json")
        (report / "cache_and_memory.csv").write_bytes(
            select_csv((repo / report_rel / "cache_and_memory.csv").read_bytes())
        )
        profile_note = {
            "schema": SCHEMA,
            "note": NOTE,
            "profile_identity_unchanged": True,
            "profile_run_id": item["profile"].name,
            "profile_data_directory": str(item["profile"]),
            "original_profile_metadata_sha256": before[str(item["profile"] / "metadata.json")],
            "original_formal_metadata_sha256": before[str(data_rel / "metadata.json")],
            "retained_formal_metadata_sha256": digest(data / "metadata.json"),
            "formal_source_manifest_sha256": before[str(data_rel / "source_manifest.json")],
            "original_formal_run_id": item["run_id"],
            "boundary": "Only this copied formal metadata is derived. Original profile metadata, source identity, inputs, actual intervals and output tensors are byte-identical; this is not profile reacceptance.",
        }
        formal = payload / item["profile"] / "formal_inputs"
        formal.mkdir(parents=True)
        shutil.copyfile(data / "metadata.json", formal / "latency_metadata.json")
        (formal / "subset_retention.json").write_bytes(json_bytes(profile_note))
        subset_audit = {
            "schema": SCHEMA,
            "status": "retained_subset",
            "run_id": item["run_id"],
            "note": NOTE,
            "original_audit_sha256": before[str(data_rel / "audit.json")],
            "original_source_sha256": item["meta"]["source_sha256"],
            "original_numerical_limits": item["audit"]["limits"],
            "retained_models": ["nosa"],
            "removed_models": ["deepseek_v32"],
            "measured_requests": len(item["rows"]),
            "correctness_records": len(item["rows"]),
            "cases": 28,
            "summary_groups": 84,
            "references": item["reference_files"],
            "exact_non_hbm_records": len(item["rows"]) * 3 // 4,
            "case_audit_inherited_from_original": [
                row for row in item["audit"]["case_audit"] if row["model"] == "nosa"
            ],
            "checks": [
                "all original audit evidence hashes verified",
                "complete original CPU-audited reference hash inventory verified",
                "all original source file hashes and aggregate verified",
                "NOSA JSONL bytes retained without reserialization",
                "summary recomputed with original frozen report source; all retained groups equal",
                "complete 28 cases and 84 summary groups retained; empty revisit groups remain null",
                "profile formal metadata originally identical to latency metadata",
            ],
            "profile_metadata_derivation": profile_note,
            "tool_sha256": digest(Path(__file__)),
            "derived_evidence_sha256": {
                str(path.relative_to(data)): digest(path)
                for path in sorted(data.rglob("*"))
                if path.is_file()
            },
            "retained_raw_files_sha256": {
                str(Path(name).relative_to(data_rel)): value
                for name, value in item["before"].items()
                if Path(name).is_relative_to(data_rel / "reference/nosa")
                or Path(name).is_relative_to(data_rel / "workloads/nosa")
            },
            "old_files_sha256": {
                name: value
                for name, value in item["before"].items()
                if "/source/" not in name and "/reference/" not in name and "/samples/" not in name
            },
        }
        for directory in (data, report):
            (directory / "audit.json").write_bytes(json_bytes(subset_audit))
        (report / "provenance.json").write_bytes(
            json_bytes(
                {
                    "schema": SCHEMA,
                    "run_id": item["run_id"],
                    "status": "retained_subset",
                    "note": NOTE,
                    "source_sha256": item["meta"]["source_sha256"],
                    "original_provenance_sha256": before[str(report_rel / "provenance.json")],
                    "derivation_tool_sha256": digest(Path(__file__)),
                    "source_directory": str(data_rel),
                    "profile_metadata_derivation": profile_note,
                    "derived_files_sha256": {
                        p.name: digest(p)
                        for p in sorted(report.iterdir())
                        if p.name != "provenance.json"
                    },
                    "visual_inspection": "Preparation does not assert visual acceptance. The final echo-cache-publication-v1 ledger separately records review of these hashed images before installation.",
                }
            )
        )
        log = payload / item["log"]
        log.mkdir(parents=True)
        (log / "measure.stdout.log").write_bytes(
            select_lines((repo / item["log"] / "measure.stdout.log").read_bytes(), log=True)
            + json.dumps(
                {
                    "event": "derived_subset_retained",
                    "schema": SCHEMA,
                    "run_id": item["run_id"],
                    "models": ["nosa"],
                    "cases": 28,
                    "measured_requests": len(item["rows"]),
                    "not_a_new_measurement": True,
                }
            ).encode()
            + b"\n"
        )
        for subtree in ("reference/deepseek_v32", "workloads/deepseek_v32"):
            removed.extend(
                name for name in item["before"] if Path(name).is_relative_to(data_rel / subtree)
            )
        # Remove obsolete mixed renderings, including any no-longer-present aliases.
        removed.extend(
            name
            for name in item["before"]
            if Path(name).is_relative_to(report_rel) and not (payload / name).exists()
        )
        audits.append(str(data_rel / "audit.json"))
    writes = {
        str(path.relative_to(payload)): digest(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file()
    }
    plan = {
        **result,
        "mode": "prepared",
        "repository": str(repo.resolve()),
        "tool_sha256": digest(Path(__file__)),
        "roots": roots,
        "before": before,
        "writes": writes,
        "remove": sorted(set(removed)),
        "subset_audits": audits,
    }
    require(not set(plan["remove"]) & writes.keys(), "write/delete overlap")
    (destination / "plan.json").write_bytes(json_bytes(plan))
    verify_prepared(repo, destination)
    return {
        key: value for key, value in plan.items() if key not in {"before", "writes", "remove"}
    } | {
        "prepared_directory": str(destination),
        "write_files": len(writes),
        "remove_files": len(removed),
    }


def allowed_path(name):
    path = Path(name)
    for label, (_, _, _, profile_suffix) in SPECS.items():
        run = f"gr_serving_h200_20261002_{label}_01"
        if any(
            path.is_relative_to(EXPERIMENT / category / leaf)
            for category, leaf in (("report", label), ("output/data", run), ("output/log", run))
        ):
            return True
        profile = (
            EXPERIMENT
            / "output/data"
            / f"gr_serving_h200_20261002_{label}_profile_{profile_suffix}"
            / "formal_inputs"
        )
        if path in (profile / "latency_metadata.json", profile / "subset_retention.json"):
            return True
    return False


def allowed_operation(name, *, remove=False):
    path = Path(name)
    for label, (_, _, _, profile_suffix) in SPECS.items():
        run = f"gr_serving_h200_20261002_{label}_01"
        data = EXPERIMENT / "output/data" / run
        if remove:
            if any(
                path.is_relative_to(data / subtree)
                for subtree in ("reference/deepseek_v32", "workloads/deepseek_v32")
            ):
                return True
            continue
        permitted = {
            data / name
            for name in ("metadata.json", "audit.json", "measurements.jsonl", "correctness.jsonl")
        }
        permitted |= {data / "analysis" / name for name in ANALYSIS_FILES}
        permitted |= {
            EXPERIMENT / "report" / label / name for name in REPORT_FILES | READABLE_REQUESTS
        }
        permitted.add(EXPERIMENT / "output/log" / run / "measure.stdout.log")
        profile = (
            EXPERIMENT
            / "output/data"
            / f"gr_serving_h200_20261002_{label}_profile_{profile_suffix}"
            / "formal_inputs"
        )
        permitted |= {profile / "latency_metadata.json", profile / "subset_retention.json"}
        if path in permitted:
            return True
    return False


def verify_prepared(repo, directory):
    plan = read(directory / "plan.json")
    require(plan["schema"] == SCHEMA and plan["mode"] == "prepared", "invalid preparation")
    require(plan["repository"] == str(repo.resolve()), "prepared for another repository")
    require(
        plan["tool_sha256"] == digest(Path(__file__)), "preparation tool changed; prepare again"
    )
    actual = {
        name: value for root in plan["roots"] for name, value in inventory(repo, root).items()
    }
    require(actual == plan["before"], "original products changed since preparation")
    payload = directory / "files"
    files = {str(p.relative_to(payload)): digest(p) for p in payload.rglob("*") if p.is_file()}
    require(files == plan["writes"], "prepared replacement files changed")
    for name in (*plan["writes"], *plan["remove"]):
        require(
            allowed_operation(name, remove=name in plan["remove"]),
            f"operation outside allowed files: {name}",
        )
        safe_path(repo, name)
        require(
            "/source/" not in name and not name.endswith("/render_report.py"),
            "cannot modify historical source",
        )
        require(
            "/reference/nosa/" not in name and "/workloads/nosa/" not in name,
            "cannot modify NOSA raw inputs",
        )
        if name.endswith("/source_manifest.json"):
            require(
                plan["writes"][name] == plan["before"][name],
                "historical source manifest must remain unchanged",
            )
    return plan


def verify_publication(repo, path):
    ledger = read(path)
    require(
        ledger.get("schema") == "echo-cache-publication-v1" and ledger.get("status") == "published",
        "replacement publication is not complete",
    )
    require(ledger.get("retired_model") == "deepseek_v32", "retirement model differs")
    require(
        set(ledger.get("retired_runs", []))
        == {f"gr_serving_h200_20261002_{label}_01" for label in SPECS},
        "retirement run scope differs",
    )
    require(
        ledger.get("prepared_visual_review") == "passed", "prepared plots have not been reviewed"
    )
    require(
        set(ledger.get("accepted_gates", []))
        == {
            "layers3_numeric",
            "memory",
            "chunk_screen",
            "formal_traces",
            "diagnostics",
            "default_review",
        },
        "replacement acceptance is incomplete",
    )
    require(bool(ledger.get("published_files_sha256")), "publication evidence is empty")
    for name, expected in ledger["published_files_sha256"].items():
        require(digest(safe_path(repo, name)) == expected, f"published replacement changed: {name}")
    return ledger


def apply_prepared(repo, directory, publication):
    plan = verify_prepared(repo, directory)
    ledger = verify_publication(repo, publication)
    # A rollback copy exists only during this operation, outside experiments.
    backup = directory / "rollback"
    backup.mkdir(exist_ok=False)
    changed, created_temporary = [], set()
    try:
        for name in (*plan["writes"], *plan["remove"]):
            target = safe_path(repo, name)
            if target.exists():
                old = backup / name
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, old)
            changed.append(name)
            if name in plan["writes"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".subset-tmp")
                require(not temporary.exists(), f"temporary path already exists: {temporary}")
                created_temporary.add(temporary)
                shutil.copyfile(directory / "files" / name, temporary)
                os.replace(temporary, target)
            else:
                target.unlink()
        for name, expected in plan["writes"].items():
            require(digest(repo / name) == expected, f"installed file changed: {name}")
        require(all(not (repo / name).exists() for name in plan["remove"]), "retired files remain")
        untouched = plan["before"].keys() - plan["writes"].keys() - set(plan["remove"])
        require(
            all(digest(repo / name) == plan["before"][name] for name in untouched),
            "retained source or NOSA evidence changed",
        )
    except BaseException:
        for name in reversed(changed):
            target, old = repo / name, backup / name
            if old.exists():
                shutil.copy2(old, target)
            else:
                target.unlink(missing_ok=True)
        for temporary in created_temporary:
            temporary.unlink(missing_ok=True)
        raise
    for name in plan["remove"]:
        parent = (repo / name).parent
        while parent != repo and allowed_path(str(parent.relative_to(repo))):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    shutil.rmtree(backup)
    receipt = {
        "schema": SCHEMA,
        "status": "applied",
        "publication_sha256": digest(publication),
        "publication": ledger,
        "plan_sha256": digest(directory / "plan.json"),
        "installed_files_sha256": plan["writes"],
        "removed_files": plan["remove"],
        "retained_files_verified": len(untouched),
        "note": NOTE,
    }
    (directory / "receipt.json").write_bytes(json_bytes(receipt))
    return {"status": "applied", "receipt": str(directory / "receipt.json"), "note": NOTE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare", type=Path, help="new external staging directory; originals unchanged"
    )
    modes.add_argument(
        "--verify", type=Path, help="verify a prepared directory; originals unchanged"
    )
    modes.add_argument(
        "--apply", type=Path, help="install prepared files after replacement publication"
    )
    parser.add_argument(
        "--publication", type=Path, help="published replacement ledger, required for --apply"
    )
    args = parser.parse_args(argv)
    repo = args.repository.resolve()
    if args.apply:
        if args.publication is None:
            parser.error("--apply requires --publication")
        result = apply_prepared(repo, args.apply.resolve(), args.publication.resolve())
    elif args.prepare:
        result = prepare(repo, args.prepare)
    elif args.verify:
        result = {"status": "verified", "writes": len(verify_prepared(repo, args.verify)["writes"])}
    else:
        _, result = preview(repo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
