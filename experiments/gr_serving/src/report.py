"""Validate paired serving measurements and summarize every request and revisit."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

GROUP_FIELDS = ("run_id", "model", "num_users", "workload_sha256", "scheme")
BUDGET_FIELDS = ("hbm_budget_bytes", "dram_budget_bytes")
IDENTITY_FIELDS = ("request_id", "user_id", "visit_index", "input_sha256")
PHASE_METRICS = ("prefix_ms", "extend_ms", "cleanup_ms")


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be finite and nonnegative")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def percentile(values: list[float], probability: float) -> float | None:
    """Linear interpolation at (n - 1) * p; descriptive quantiles, not confidence bounds."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: list[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean_ms": math.fsum(values) / len(values) if values else None,
        "median_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def validate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reject incomplete traces, unequal budgets, or mismatched scheme inputs.

    Each run/model/population/workload is a paired case. Within it all schemes
    must report the same complete request identities and the same allowed cache
    budgets. Actual allocation may differ. A revisit can legitimately be a miss.
    """
    normalized = []
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if "prefix_cache_hit" in row:
            if "prefix_hit" in row and row["prefix_hit"] != row["prefix_cache_hit"]:
                raise ValueError("prefix_cache_hit disagrees with prefix_hit")
            row["prefix_hit"] = row["prefix_cache_hit"]
        missing = {*GROUP_FIELDS, *BUDGET_FIELDS, *IDENTITY_FIELDS, "latency_ms"} - row.keys()
        if missing:
            raise ValueError(f"measurement is missing fields: {sorted(missing)}")
        for name in ("run_id", "model", "workload_sha256", "scheme", "input_sha256"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError(f"{name} must be a nonempty string")
        for name in (*BUDGET_FIELDS, "request_id", "user_id", "visit_index"):
            _integer(name, row[name])
        _integer("num_users", row["num_users"], 1)
        _number("latency_ms", row["latency_ms"])
        for name in PHASE_METRICS:
            if name in row and row[name] is not None:
                _number(name, row[name])
        if row.get("phase", "measured") != "measured":
            raise ValueError("warmup or validation requests cannot enter measured results")
        revisit = row["visit_index"] > 0
        if "is_revisit" in row and row["is_revisit"] is not revisit:
            raise ValueError("is_revisit must equal visit_index > 0, independently of cache hits")
        if "prefix_hit" in row and not isinstance(row["prefix_hit"], bool):
            raise ValueError("prefix_hit must be a boolean")
        if "prefix_hit_tier" in row and row["prefix_hit_tier"] not in ("hbm", "dram", "miss"):
            raise ValueError("prefix_hit_tier must be hbm, dram, or miss")
        if (
            "prefix_hit" in row
            and "prefix_hit_tier" in row
            and row["prefix_hit"] != (row["prefix_hit_tier"] != "miss")
        ):
            raise ValueError("prefix_hit disagrees with prefix_hit_tier")
        if not revisit and (row.get("prefix_hit") or row.get("prefix_hit_tier") in ("hbm", "dram")):
            raise ValueError("first visits cannot hit a whole-user session after an empty reset")
        row["is_revisit"] = revisit
        row["visit_number"] = row["visit_index"] + 1
        for name in ("cache_hbm_bytes", "cache_dram_bytes", "transfer_bytes"):
            if name in row and row[name] is not None:
                _integer(name, row[name])
        for actual, allowed in (
            ("cache_hbm_bytes", "hbm_budget_bytes"),
            ("cache_dram_bytes", "dram_budget_bytes"),
        ):
            if row.get(actual) is not None and row[actual] > row[allowed]:
                raise ValueError(f"{actual} exceeds {allowed}")
        if "evicted_users" in row:
            if isinstance(row["evicted_users"], list):
                for user in row["evicted_users"]:
                    _integer("evicted user ID", user)
            else:
                _integer("evicted_users", row["evicted_users"])
        groups[tuple(row[name] for name in GROUP_FIELDS)].append(row)
        normalized.append(row)
    if not normalized:
        raise ValueError("no measured requests")
    paired: dict[tuple, tuple] = {}
    # Also catch a scheme silently changing workload while retaining case coordinates.
    case_workloads: dict[tuple, str] = {}
    for key, samples in groups.items():
        samples.sort(key=lambda row: row["request_id"])
        if [row["request_id"] for row in samples] != list(range(len(samples))):
            raise ValueError(f"{key}: request IDs must form a complete 0-based trace")
        if any("sequential_loops_schema" in row for row in samples):
            users = samples[0]["num_users"]
            rounds = _integer("rounds", samples[0].get("rounds"), 2)
            if len(samples) != users * rounds:
                raise ValueError(f"{key}: sequential trace must cover every complete round")
            for row in samples:
                expected = {
                    "sequential_loops_schema": 1,
                    "rounds": rounds,
                    "round_index": row["request_id"] // users,
                    "round_user_index": row["request_id"] % users,
                    "user_id": row["request_id"] % users,
                    "visit_index": row["request_id"] // users,
                }
                if any(row.get(name) != value for name, value in expected.items()):
                    raise ValueError(f"{key}: inconsistent sequential round coverage")
        visits: Counter[int] = Counter()
        for row in samples:
            if row["visit_index"] != visits[row["user_id"]]:
                raise ValueError(f"{key}: visit_index does not match request order")
            visits[row["user_id"]] += 1
        if len(visits) > samples[0]["num_users"]:
            raise ValueError(f"{key}: observed users exceed configured population")
        budgets = {tuple(row[name] for name in BUDGET_FIELDS) for row in samples}
        if len(budgets) != 1:
            raise ValueError(f"{key}: allowed cache budgets changed inside a trace")
        signature = (
            next(iter(budgets)),
            tuple(tuple(row[name] for name in IDENTITY_FIELDS) for row in samples),
        )
        if paired.setdefault(key[:-1], signature) != signature:
            raise ValueError(f"{key[:-1]}: schemes used different budgets or request inputs")
        first = samples[0]
        coordinates = tuple(
            first.get(name)
            for name in (
                "run_id",
                "model",
                "num_users",
                "history_tokens",
                "candidate_tokens",
                "seed",
            )
        )
        if case_workloads.setdefault(coordinates, key[-2]) != key[-2]:
            raise ValueError(f"{coordinates}: schemes changed the workload hash")
    return sorted(
        normalized, key=lambda row: (*[row[name] for name in GROUP_FIELDS], row["request_id"])
    )


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    samples = validate_rows(rows)
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        groups[tuple(row[name] for name in GROUP_FIELDS)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        for scope in ("all", "first_visit", "revisit"):
            selected = [
                row for row in group if scope == "all" or row["is_revisit"] == (scope == "revisit")
            ]
            result: dict[str, Any] = {
                **dict(zip(GROUP_FIELDS, key, strict=True)),
                **{name: group[0][name] for name in BUDGET_FIELDS},
                "scope": scope,
                "unique_users": len({row["user_id"] for row in selected}),
                **_stats([float(row["latency_ms"]) for row in selected]),
            }
            for name in ("history_tokens", "candidate_tokens", "seed"):
                if name in group[0]:
                    result[name] = group[0][name]
            if all("prefix_hit" in row or "prefix_hit_tier" in row for row in selected):
                hits = sum(
                    row.get("prefix_hit", row.get("prefix_hit_tier") != "miss") for row in selected
                )
                result["prefix_hits"] = hits
                result["prefix_misses"] = len(selected) - hits
                result["prefix_hit_rate"] = hits / len(selected) if selected else None
            if all("prefix_hit_tier" in row for row in selected):
                result["prefix_hit_tiers"] = dict(
                    Counter(row["prefix_hit_tier"] for row in selected)
                )
            if all("evicted_users" in row for row in selected):
                result["evictions"] = sum(
                    len(row["evicted_users"])
                    if isinstance(row["evicted_users"], list)
                    else row["evicted_users"]
                    for row in selected
                )
            for name in PHASE_METRICS:
                if selected and all(row.get(name) is not None for row in selected):
                    result[name] = _stats([float(row[name]) for row in selected])
            for name in ("cache_hbm_bytes", "cache_dram_bytes"):
                if selected and all(row.get(name) is not None for row in selected):
                    result[f"peak_{name}"] = max(row[name] for row in selected)
            if selected and all(row.get("transfer_bytes") is not None for row in selected):
                result["transfer_bytes"] = sum(row["transfer_bytes"] for row in selected)
            summaries.append(result)
    round_summaries = []
    for key, group in sorted(groups.items()):
        if "sequential_loops_schema" in group[0]:
            for index in range(group[0]["rounds"]):
                selected = [row for row in group if row["round_index"] == index]
                round_summaries.append(
                    {
                        **dict(zip(GROUP_FIELDS, key, strict=True)),
                        "round_index": index,
                        "prefix_hits": sum(row["prefix_hit"] for row in selected),
                        **_stats([float(row["latency_ms"]) for row in selected]),
                    }
                )
    return {
        "schema_version": 2 if round_summaries else 1,
        "definitions": {
            "latency_ms": "synchronized wall time per request, including prefix miss, suffix execution, and cleanup",
            "revisit": "visit_index > 0; a revisited user may have been evicted and miss again",
            "quantiles": "linear interpolation at (n - 1) * p over measured requests",
            "empty_scope": "count is zero and latency statistics are null",
            "budgets": "equal allowed HBM/DRAM cache caps; actual allocated bytes may differ",
            "phase_times": "reported separately when measured; total latency is authoritative",
            "rounds": "sequential complete passes over users 0..U-1; round zero contains first visits",
        },
        "groups": summaries,
        "rounds": round_summaries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _latency_svg(rows: list[dict[str, Any]]) -> str:
    """Standalone plot without adding a plotting dependency to the serving runtime."""
    cases: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cases[tuple(row[name] for name in GROUP_FIELDS[:-1])].append(row)
    width, panel = 1100, 340
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{len(cases) * panel + 40}" viewBox="0 0 {width} {len(cases) * panel + 40}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}</style>",
        '<text x="75" y="23">Per-request serving latency; hollow marker = first visit, filled = revisit</text>',
    ]
    for index, (key, group) in enumerate(sorted(cases.items())):
        top, left, plot_width, plot_height = 40 + index * panel, 75, 980, 220
        max_latency = max(float(row["latency_ms"]) for row in group) or 1
        max_request = max(row["request_id"] for row in group) or 1
        schemes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            schemes[row["scheme"]].append(row)
        title = f"{key[1]} · {key[2]} users · {key[0]} · workload {key[3][:12]}"
        parts.append(
            f'<text x="{left}" y="{top + 17}" font-weight="bold">{html.escape(title)}</text>'
        )
        origin = top + 45
        for tick in range(5):
            y = origin + plot_height * (1 - tick / 4)
            parts.append(f'<path d="M {left} {y} H {left + plot_width}" stroke="#ddd"/>')
            parts.append(
                f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{max_latency * tick / 4:.3g}</text>'
            )
        parts.append(f'<text x="{left - 35}" y="{origin - 9}">ms</text>')
        for tick in range(5):
            x = left + plot_width * tick / 4
            parts.append(
                f'<text x="{x}" y="{origin + plot_height + 19}" text-anchor="middle">{max_request * tick / 4:.0f}</text>'
            )
        parts.append(
            f'<text x="{left + plot_width / 2}" y="{origin + plot_height + 37}" text-anchor="middle">request ID</text>'
        )
        for scheme_index, (scheme, selected) in enumerate(sorted(schemes.items())):
            color = colors[scheme_index % len(colors)]
            points = []
            for row in sorted(selected, key=lambda item: item["request_id"]):
                x = left + plot_width * row["request_id"] / max_request
                y = origin + plot_height * (1 - float(row["latency_ms"]) / max_latency)
                points.append(f"{x:.2f},{y:.2f}")
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1" opacity="0.55"/>'
            )
            for point, row in zip(
                points, sorted(selected, key=lambda item: item["request_id"]), strict=True
            ):
                x, y = point.split(",")
                fill = color if row["is_revisit"] else "white"
                label = f"{scheme}: request {row['request_id']}, user {row['user_id']}, visit {row['visit_number']}, {row['latency_ms']:.3f} ms"
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="2.5" fill="{fill}" stroke="{color}"><title>{html.escape(label)}</title></circle>'
                )
            legend_x = left + scheme_index * 230
            parts.append(
                f'<path d="M {legend_x} {top + 327} h 18" stroke="{color}" stroke-width="3"/>'
            )
            parts.append(f'<text x="{legend_x + 24}" y="{top + 331}">{html.escape(scheme)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _comparison_svg(groups: list[dict[str, Any]]) -> str:
    """Compare mean and tail latency as the configured user population grows."""
    cases: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if group["scope"] not in ("all", "revisit"):
            continue
        key = tuple(
            group.get(name)
            for name in (
                "run_id",
                "model",
                "history_tokens",
                "candidate_tokens",
                "seed",
                "hbm_budget_bytes",
                "dram_budget_bytes",
            )
        )
        cases[key].append(group)
    width, height = 1240, 390
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{len(cases) * height + 40}" viewBox="0 0 {width} {len(cases) * height + 40}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}</style>",
        '<text x="70" y="23">Serving latency versus user population; mean = solid, p95 = dashed; all requests retain cache misses</text>',
    ]
    for case_index, (key, case) in enumerate(sorted(cases.items(), key=lambda item: str(item[0]))):
        top = 40 + case_index * height
        title = f"{key[1]} · {key[0]} · prefix {key[2]}, candidate {key[3]} · seed {key[4]}"
        parts.append(f'<text x="70" y="{top + 17}" font-weight="bold">{html.escape(title)}</text>')
        schemes = sorted({row["scheme"] for row in case})
        populations = sorted({row["num_users"] for row in case})
        low, high = min(populations), max(populations)
        maximum = max((row[metric] or 0) for row in case for metric in ("mean_ms", "p95_ms")) or 1
        plot_width, plot_height = 505, 235
        for column, scope in enumerate(("all", "revisit")):
            left, origin = 70 + column * 615, top + 63
            label = "All requests" if scope == "all" else "Revisits only (exclude first visit)"
            parts.append(f'<text x="{left}" y="{origin - 18}">{label}</text>')
            parts.append(f'<text x="{left - 35}" y="{origin - 4}">ms</text>')
            for tick in range(5):
                y = origin + plot_height * (1 - tick / 4)
                parts.append(f'<path d="M {left} {y} H {left + plot_width}" stroke="#ddd"/>')
                parts.append(
                    f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{maximum * tick / 4:.3g}</text>'
                )
            for users in populations:
                x = left + plot_width * ((users - low) / (high - low) if high != low else 0.5)
                parts.append(
                    f'<text x="{x}" y="{origin + plot_height + 20}" text-anchor="middle">{users}</text>'
                )
            parts.append(
                f'<text x="{left + plot_width / 2}" y="{origin + plot_height + 40}" text-anchor="middle">configured users</text>'
            )
            for scheme_index, scheme in enumerate(schemes):
                selected = sorted(
                    (row for row in case if row["scope"] == scope and row["scheme"] == scheme),
                    key=lambda row: row["num_users"],
                )
                color = colors[scheme_index % len(colors)]
                for metric in ("mean_ms", "p95_ms"):
                    points = []
                    for row in selected:
                        if row[metric] is None:
                            continue
                        x = left + plot_width * (
                            (row["num_users"] - low) / (high - low) if high != low else 0.5
                        )
                        y = origin + plot_height * (1 - row[metric] / maximum)
                        points.append((x, y, row))
                    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
                    dash = ' stroke-dasharray="6 4"' if metric == "p95_ms" else ""
                    parts.append(
                        f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
                    )
                    for x, y, row in points:
                        fill = "white" if metric == "p95_ms" else color
                        label = f"{scheme}, {row['num_users']} users, {scope}, n={row['count']}, {metric}={row[metric]:.3f}"
                        parts.append(
                            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{fill}" stroke="{color}"><title>{html.escape(label)}</title></circle>'
                        )
        for scheme_index, scheme in enumerate(schemes):
            x = 70 + scheme_index * 280
            color = colors[scheme_index % len(colors)]
            parts.append(f'<path d="M {x} {top + 366} h 20" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{x + 26}" y="{top + 370}">{html.escape(scheme)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_report(rows: Iterable[Mapping[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    samples = validate_rows(rows)
    summary = summarize(samples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(output / "summary.csv", summary["groups"])
    if summary["rounds"]:
        _write_csv(output / "rounds.csv", summary["rounds"])
    _write_csv(output / "per_request.csv", samples)
    (output / "per_request.svg").write_text(_latency_svg(samples))
    (output / "summary.svg").write_text(_comparison_svg(summary["groups"]))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    try:
        for path in args.input:
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        summary = write_report(rows, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(args.output_dir), "groups": len(summary["groups"])}))


if __name__ == "__main__":
    main()
