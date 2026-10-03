import json
from copy import deepcopy
from xml.etree import ElementTree

import pytest

from experiments.gr_serving.src.report import percentile, summarize, validate_rows, write_report


def measured_rows():
    rows = []
    for scheme in ("hbm", "serial_sparse"):
        for request_id, (uid, visit, latency) in enumerate(
            [(0, 0, 10.0), (1, 0, 20.0), (0, 1, 3.0), (1, 1, 30.0)]
        ):
            rows.append(
                {
                    "run_id": "test_run",
                    "model": "nosa",
                    "scheme": scheme,
                    "num_users": 2,
                    "workload_sha256": "workload",
                    "request_id": request_id,
                    "user_id": uid,
                    "visit_index": visit,
                    "input_sha256": f"input_{request_id}",
                    "latency_ms": latency,
                    "hbm_budget_bytes": 1000,
                    "dram_budget_bytes": 2000,
                    # The last request is a revisit after eviction and must be
                    # retained in repeat-only latency despite missing cache.
                    "prefix_hit": request_id == 2,
                    "evicted_users": [1] if request_id == 2 else [],
                    "cache_hbm_bytes": 1000,
                    "cache_dram_bytes": 1500 if scheme != "hbm" else 0,
                    "prefix_ms": latency - 2,
                    "extend_ms": 1.5,
                    "cleanup_ms": 0.5,
                }
            )
    return rows


def test_revisits_include_evicted_user_and_quantiles_are_request_weighted():
    summary = summarize(measured_rows())
    repeat = next(
        row for row in summary["groups"] if row["scheme"] == "hbm" and row["scope"] == "revisit"
    )
    assert repeat["count"] == 2
    assert repeat["median_ms"] == 16.5
    assert repeat["p95_ms"] == pytest.approx(28.65)
    assert repeat["p99_ms"] == pytest.approx(29.73)
    assert repeat["prefix_hits"] == 1
    assert repeat["prefix_misses"] == 1
    assert repeat["evictions"] == 1
    assert repeat["extend_ms"]["median_ms"] == 1.5
    assert repeat["peak_cache_dram_bytes"] == 0


def test_no_revisits_produce_null_latency_not_zero():
    rows = [row for row in measured_rows() if row["request_id"] < 2]
    summary = summarize(rows)
    repeat = [row for row in summary["groups"] if row["scope"] == "revisit"]
    assert all(row["count"] == 0 for row in repeat)
    assert all(row["median_ms"] is None and row["p99_ms"] is None for row in repeat)
    assert percentile([1], 0.95) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hbm_budget_bytes", 1001, "budgets changed"),
        ("input_sha256", "different", "different budgets or request inputs"),
        (
            "workload_sha256",
            "different",
            "complete 0-based trace|different budgets or request inputs",
        ),
        ("is_revisit", False, "independently of cache hits"),
        ("visit_index", 4, "visit_index does not match"),
        ("phase", "warmup", "warmup or validation"),
        ("cache_hbm_bytes", 1001, "exceeds"),
        ("latency_ms", float("nan"), "finite"),
    ],
)
def test_measurement_integrity_failures(field, value, message):
    rows = measured_rows()
    rows[-1][field] = value
    with pytest.raises(ValueError, match=message):
        validate_rows(rows)


def test_all_scheme_budgets_and_hashes_must_match():
    for field, value in (("hbm_budget_bytes", 1100), ("workload_sha256", "other")):
        rows = measured_rows()
        for row in rows:
            if row["scheme"] == "serial_sparse":
                row[field] = value
        with pytest.raises(ValueError, match="different budgets|changed the workload hash"):
            validate_rows(rows)


def test_runner_prefix_cache_hit_alias():
    rows = measured_rows()
    for row in rows:
        row["prefix_cache_hit"] = row.pop("prefix_hit")
    summary = summarize(rows)
    repeats = [row for row in summary["groups"] if row["scope"] == "revisit"]
    assert all(row["prefix_hits"] == 1 and row["prefix_misses"] == 1 for row in repeats)


def test_missing_request_and_visit_trace_fail():
    rows = measured_rows()
    with pytest.raises(ValueError, match="complete 0-based trace"):
        validate_rows(rows[:2] + rows[3:])
    rows[0]["prefix_hit"] = True
    with pytest.raises(ValueError, match="first visits cannot hit"):
        validate_rows(rows)


def test_report_artifacts_preserve_rows_and_render_valid_svg(tmp_path):
    rows = measured_rows()
    original = deepcopy(rows)
    summary = write_report(reversed(rows), tmp_path)
    assert rows == original
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert len((tmp_path / "per_request.csv").read_text().splitlines()) == 9
    assert "revisit" in (tmp_path / "summary.csv").read_text()
    root = ElementTree.fromstring((tmp_path / "per_request.svg").read_text())
    assert len(root.findall("{http://www.w3.org/2000/svg}circle")) == len(rows)
    comparison = ElementTree.fromstring((tmp_path / "summary.svg").read_text())
    assert len(comparison.findall("{http://www.w3.org/2000/svg}circle")) == 8
    assert "Revisits only" in (tmp_path / "summary.svg").read_text()
