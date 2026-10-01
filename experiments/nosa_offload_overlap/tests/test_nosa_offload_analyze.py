import sqlite3

import pytest

from experiments.nosa_offload_overlap.src.analyze import (
    extract_sqlite,
    interval_metrics,
    stripe_interval_metrics,
    work_interval_metrics,
    work_profile_metadata,
)


def test_interval_union_excludes_double_counted_concurrency():
    result = interval_metrics([(0, 10000), (5000, 15000)], [(8000, 20000)])
    assert result["fetch_union_us"] == 15
    assert result["attention_union_us"] == 12
    assert result["overlap_us"] == 7
    assert result["fetch_hidden_fraction"] == 7 / 15
    assert interval_metrics([], [(0, 1000)])["fetch_hidden_fraction"] is None


def test_touching_intervals_do_not_overlap():
    assert interval_metrics([(0, 1000)], [(1000, 2000)])["overlap_us"] == 0
    with pytest.raises(ValueError, match="end >= start"):
        interval_metrics([(2, 1)], [])


def make_trace(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER,
            correlationId INTEGER, streamId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, correlationId INTEGER,
            globalTid INTEGER);
    """)
    connection.executemany(
        "INSERT INTO StringIds VALUES (?,?)",
        [
            (1, "void nosa_attention::attention_kernel"),
            (2, "void nosa_offload::nosa_sparse_fetch_kernel"),
            (3, "nosa_first_use_plan_kernel"),
        ],
    )
    correlation = 0
    for index, mode in enumerate(("resident", "serialized", "overlap")):
        start = index * 10000
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
            (start, start + 10000, f"nosa_overlap/layer_00/{mode}/sample_0", 42),
        )
        activities = [(2000, 6000, 1, 1)]
        if mode == "serialized":
            activities = [(1000, 4000, 2, 1), (4000, 7000, 1, 1)]
        if mode == "overlap":
            activities = [(1000, 5000, 2, 2), (3000, 7000, 1, 1)]
        for begin, end, name, stream in activities:
            correlation += 1
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
                (start + begin, start + end, name, correlation, stream),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                (start + 100, start + 200, correlation, 42),
            )
        # A launch on a different host thread must not be attributed just
        # because its GPU activity falls inside the measured wall-clock range.
        correlation += 1
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
            (start + 1000, start + 9000, 3, correlation, 3),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
            (start + 50, start + 60, correlation, 99),
        )
    connection.commit()
    connection.close()


def test_sqlite_uses_actual_gpu_intersections_and_launch_attribution(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_trace(path)
    records = {item["mode"]: item for item in extract_sqlite(path)}
    assert records["resident"]["fetch_hidden_fraction"] is None
    assert records["serialized"]["overlap_us"] == 0
    assert records["overlap"]["overlap_us"] == 2
    assert records["overlap"]["fetch_hidden_fraction"] == 0.5
    assert records["overlap"]["attribution"] == {"launch_correlation": 2}
    assert records["overlap"]["fetch_streams"] == [2]
    assert records["overlap"]["attention_streams"] == [1]
    assert not records["overlap"]["other_kernel_counts"]


def test_incomplete_three_mode_profile_is_rejected(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_trace(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM NVTX_EVENTS WHERE text LIKE '%/overlap/%'")
    with pytest.raises(ValueError, match="Incomplete or unequal"):
        extract_sqlite(path)


def test_mismatched_sample_ids_are_rejected(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_trace(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE NVTX_EVENTS SET text = REPLACE(text, 'sample_0', 'sample_1') "
            "WHERE text LIKE '%/overlap/%'"
        )
    with pytest.raises(ValueError, match="Mismatched measured sample IDs"):
        extract_sqlite(path)


def test_serialized_control_with_concurrent_fetch_is_rejected(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_trace(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end = 15000 "
            "WHERE start = 11000 AND demangledName = 2"
        )
    with pytest.raises(ValueError, match="Serialized control contains overlapping"):
        extract_sqlite(path)


def fused_work_profile():
    case = {
        "prefix": 64,
        "queries": 8,
        "kv_heads": 2,
        "expected_prefix_bytes": 65536,
        "expected_fetch_rows": [{"row": 0, "bytes": 32768}, {"row": 1, "bytes": 32768}],
    }
    return {
        "schema_version": 1,
        "run_id": "fixture",
        "clock": "device_globaltimer_ns",
        "math_coverage": "softmax_only",
        "cases": {"layer_00": case},
        "records": [
            {
                "case": "layer_00",
                "mode": "serialized",
                "sample": 0,
                "recorded_transfer_bytes": 65536,
                "intervals": [],
            },
            {
                "case": "layer_00",
                "mode": "overlap",
                "sample": 0,
                "recorded_transfer_bytes": 65536,
                "intervals": [
                    {"row": 0, "start_ns": 50100, "end_ns": 53100, "bytes": 32768, "kind": 1},
                    {"row": 1, "start_ns": 51000, "end_ns": 54000, "bytes": 32768, "kind": 1},
                    {"row": 2, "start_ns": 52000, "end_ns": 55000, "bytes": 0, "kind": 2},
                ],
            },
        ],
    }


def make_fused_trace(path):
    make_trace(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO StringIds VALUES (4, 'void nosa_offload_fused::fused_main')"
        )
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName = 4, end = 27000 "
            "WHERE start >= 20000 AND demangledName = 2"
        )
        connection.execute(
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start = 27000, end = 28000 "
            "WHERE start >= 20000 AND demangledName = 1"
        )


def test_fused_queue_compaction_is_timed_preparation_without_host_fetch(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_fused_trace(path)
    name = "void nosa_offload_fused::compact_fetch_queue"
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO StringIds VALUES (?, ?)", (5, name))
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
            (20200, 21000, 5, 100, 1),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
            (20050, 20100, 100, 42),
        )
    rows = {row["mode"]: row for row in extract_sqlite(path, work_profile=fused_work_profile())}
    candidate = rows["overlap"]
    assert candidate["fused_main_count"] == 1
    assert candidate["other_kernel_counts"] == {name: 1}
    assert candidate["fetch_kernel_sum_us"] == 0
    assert candidate["attention_kernel_sum_us"] == 1
    assert candidate["gpu_span_us"] == 7.8
    assert candidate["attribution"] == {"launch_correlation": 3}


@pytest.mark.parametrize("schema", [1, 2])
def test_fused_overlap_uses_globaltimer_math_not_entire_kernel(tmp_path, schema):
    path = tmp_path / "timeline.sqlite"
    make_fused_trace(path)
    profile = fused_work_profile()
    if schema == 2:
        profile.update(
            schema_version=2,
            fetch_coverage="per_page_copy_windows",
            byte_accounting="logical_unique_kv_payload",
        )
    rows = {row["mode"]: row for row in extract_sqlite(path, work_profile=profile)}
    candidate = rows["overlap"]
    assert candidate["fused_main_count"] == 1
    assert candidate["fused_main_us"] == 6
    assert candidate["kernel_intervals"] is None
    assert "fetch_hidden_fraction" not in candidate
    work = candidate["work_metrics"]
    assert work["fetch_work_union_us"] == 3.9
    assert work["softmax_union_us"] == 3
    assert work["fetch_math_overlap_us"] == 2
    assert work["fetch_math_overlap_fraction"] == pytest.approx(2 / 3.9)
    assert work["math_coverage"] == "softmax_only"
    assert work_profile_metadata(profile)["fetch_coverage"] == "per_page_copy_windows"
    assert rows["serialized"]["kernel_intervals"]["overlap_us"] == 0
    assert rows["serialized"]["work_metrics"] is None
    # Globaltimer timestamps deliberately lie outside every Nsight host range;
    # only durations and intersections within their own clock domain are used.
    with pytest.raises(ValueError, match="intra-kernel work intervals"):
        extract_sqlite(path)


def test_batch_shared_start_unions_windows_without_merging_page_provenance(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_fused_trace(path)
    profile = fused_work_profile()
    profile.update(
        schema_version=2,
        fetch_coverage="batch_copy_windows",
        byte_accounting="logical_unique_kv_payload",
    )
    entries = profile["records"][1]["intervals"]
    entries[1]["start_ns"] = entries[0]["start_ns"]
    rows = {row["mode"]: row for row in extract_sqlite(path, work_profile=profile)}
    work = rows["overlap"]["work_metrics"]
    assert work["fetch_work_union_us"] == 3.9
    assert work["fetch_math_overlap_us"] == 2
    assert work["fetch_math_overlap_fraction"] == pytest.approx(2 / 3.9)
    assert work["fetch_interval_count"] == 2
    assert work["copied_bytes"] == 65536
    assert work_profile_metadata(profile)["fetch_coverage"] == "batch_copy_windows"
    # Equal timestamps must not collapse distinct page identities or payload.
    entries.append(dict(entries[0]))
    with pytest.raises(ValueError, match="duplicate"):
        extract_sqlite(path, work_profile=profile)


@pytest.mark.parametrize("field", ["fetch_coverage", "byte_accounting"])
def test_batch_schema_requires_explicit_window_and_logical_byte_semantics(tmp_path, field):
    path = tmp_path / "timeline.sqlite"
    make_fused_trace(path)
    profile = fused_work_profile()
    profile.update(
        schema_version=2,
        fetch_coverage="batch_copy_windows",
        byte_accounting="logical_unique_kv_payload",
    )
    profile.pop(field)
    with pytest.raises(ValueError, match="fetch coverage or byte accounting"):
        extract_sqlite(path, work_profile=profile)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "bytes", "counter", "math", "overflow", "timestamp"]
)
def test_globaltimer_rejects_incomplete_or_invalid_work(mutation):
    profile = fused_work_profile()
    record = profile["records"][1]
    entries = record["intervals"]
    if mutation == "duplicate":
        entries.append(dict(entries[0]))
    elif mutation == "missing":
        entries.pop(0)
    elif mutation == "bytes":
        entries[0]["bytes"] += 16
    elif mutation == "counter":
        record["recorded_transfer_bytes"] += 16
    elif mutation == "math":
        entries.pop()
    elif mutation == "overflow":
        entries[-1]["row"] = 130
    elif mutation == "timestamp":
        entries[0]["end_ns"] = entries[0]["start_ns"]
    with pytest.raises(ValueError):
        work_interval_metrics(profile["cases"]["layer_00"], record)


def test_fused_main_count_and_phase_sample_coverage_are_required(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_fused_trace(path)
    profile = fused_work_profile()
    profile["records"].pop(0)
    with pytest.raises(ValueError, match="samples do not match"):
        extract_sqlite(path, work_profile=profile)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT start, end, demangledName, correlationId, streamId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName = 4"
        )
    with pytest.raises(ValueError, match="exactly one fused main"):
        extract_sqlite(path, work_profile=fused_work_profile())


def stripe_work_profile(fetch_stripes=8):
    profile = fused_work_profile()
    profile.update(
        schema_version=3,
        fetch_coverage="per_page_copy_envelopes",
        stripe_coverage="nonempty_stripe_copy_windows",
        byte_accounting="logical_unique_kv_payload",
        fetch_stripes=fetch_stripes,
    )
    for record in profile["records"]:
        record["stripe_intervals"] = [
            {
                **page,
                "kind": 3,
                "row": 130 + page["row"] * fetch_stripes + stripe,
                "bytes": page["bytes"] // fetch_stripes,
            }
            for page in record["intervals"]
            if page["kind"] == 1
            for stripe in range(fetch_stripes)
        ]
    return profile


def test_schema3_retains_page_metrics_and_independently_measures_stripes(tmp_path):
    path = tmp_path / "timeline.sqlite"
    make_fused_trace(path)
    profile = stripe_work_profile()
    rows = {row["mode"]: row for row in extract_sqlite(path, work_profile=profile)}
    page = rows["overlap"]["work_metrics"]
    stripe = rows["overlap"]["stripe_metrics"]
    assert page["fetch_work_union_us"] == stripe["fetch_stripe_union_us"] == 3.9
    assert stripe["fetch_stripe_math_overlap_fraction"] == pytest.approx(2 / 3.9)
    assert stripe["copied_bytes"] == page["copied_bytes"] == 65536
    assert stripe["stripe_interval_count"] == 16
    assert stripe["fetch_interval_count"] == 2
    assert stripe["page_envelope_only_union_us"] == 0
    assert stripe["page_envelope_only_math_overlap_us"] == 0
    assert stripe["page_minus_stripe_math_overlap_fraction"] == 0
    assert rows["resident"]["stripe_metrics"] is None
    assert rows["serialized"]["stripe_metrics"] is None
    assert work_profile_metadata(profile)["fetch_stripes"] == 8


@pytest.mark.parametrize("tail", [1, 7, 8, 9, 63])
def test_stripes_partition_partial_historical_pages_and_ceil_query_slots(tail):
    profile = stripe_work_profile()
    case, record = profile["cases"]["layer_00"], profile["records"][1]
    case.update(
        prefix=64 + tail,
        queries=9,
        expected_prefix_bytes=tail * 512,
        expected_fetch_rows=[{"row": 3, "bytes": tail * 512}],
    )
    # Four page slots, two 8-query groups, two KV heads, 64 math rows/group/head.
    stripe_base = 4 + 2 * 2 * 64
    record.update(
        recorded_transfer_bytes=tail * 512,
        intervals=[
            {"row": 3, "start_ns": 100, "end_ns": 200, "kind": 1, "bytes": tail * 512},
            {"row": 4, "start_ns": 150, "end_ns": 250, "kind": 2, "bytes": 0},
        ],
        stripe_intervals=[
            {
                "row": stripe_base + 3 * 8 + stripe,
                "start_ns": 100,
                "end_ns": 200,
                "kind": 3,
                "bytes": len(range(tail)[stripe * 8 : (stripe + 1) * 8]) * 512,
            }
            for stripe in range((tail + 7) // 8)
        ],
    )
    metrics = stripe_interval_metrics(case, record, 8)
    assert metrics["copied_bytes"] == tail * 512
    assert metrics["stripe_interval_count"] == (tail + 7) // 8
    assert metrics["fetch_stripe_math_overlap_fraction"] == 0.5
    if tail <= 56:
        # An empty historical stripe is not evidence, even with a plausible row.
        record["stripe_intervals"].append(
            {"row": stripe_base + 3 * 8 + 7, "start_ns": 100, "end_ns": 200, "kind": 3, "bytes": 0}
        )
        with pytest.raises(ValueError, match="selected nonempty stripe bytes"):
            stripe_interval_metrics(case, record, 8)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "Missing selected"),
        ("duplicate", "duplicate"),
        ("bytes", "stripe bytes"),
        ("wrong_row", "stripe bytes"),
        ("kind", "Invalid"),
        ("timestamp", "Invalid"),
        ("boolean", "Invalid"),
        ("envelope_start", "Page envelope"),
        ("envelope_end", "Page envelope"),
    ],
)
def test_stripe_provenance_cannot_be_forged(mutation, match):
    profile = stripe_work_profile()
    case, record = profile["cases"]["layer_00"], profile["records"][1]
    entries = record["stripe_intervals"]
    if mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries.append(dict(entries[0]))
    elif mutation == "bytes":
        entries[0]["bytes"] += 16
    elif mutation == "wrong_row":
        entries[0]["row"] = 129
    elif mutation == "kind":
        entries[0]["kind"] = 1
    elif mutation == "timestamp":
        entries[0]["end_ns"] = entries[0]["start_ns"]
    elif mutation == "boolean":
        entries[0]["start_ns"] = True
    elif mutation == "envelope_start":
        record["intervals"][0]["start_ns"] -= 1
    elif mutation == "envelope_end":
        record["intervals"][0]["end_ns"] += 1
    with pytest.raises(ValueError, match=match):
        stripe_interval_metrics(case, record, 8)


@pytest.mark.parametrize("fetch_stripes", [None, True, 0, -1, 3, 65, 8.0, "8"])
def test_schema3_rejects_invalid_stripe_geometry(fetch_stripes):
    profile = stripe_work_profile()
    profile["fetch_stripes"] = fetch_stripes
    with pytest.raises(ValueError, match="positive integer divisor"):
        work_profile_metadata(profile)


def test_stripes_require_explicit_coverage_and_do_not_appear_in_serialized_control():
    profile = stripe_work_profile()
    profile["stripe_coverage"] = "page_envelope"
    with pytest.raises(ValueError, match="stripe coverage"):
        work_profile_metadata(profile)
    serial = profile["records"][0]
    serial["stripe_intervals"] = profile["records"][1]["stripe_intervals"]
    with pytest.raises(ValueError, match="Only fused overlap"):
        stripe_interval_metrics(profile["cases"]["layer_00"], serial, 8)
    serial.pop("stripe_intervals")
    with pytest.raises(ValueError, match="explicit stripe_intervals"):
        stripe_interval_metrics(profile["cases"]["layer_00"], serial, 8)


@pytest.mark.parametrize("math_in_gap", [True, False])
def test_page_fraction_is_not_a_one_sided_bound_on_stripe_fraction(math_in_gap):
    profile = stripe_work_profile(fetch_stripes=2)
    case, record = profile["cases"]["layer_00"], profile["records"][1]
    for page in record["intervals"][:2]:
        page.update(start_ns=100, end_ns=1100)
    for index, stripe in enumerate(record["stripe_intervals"]):
        stripe.update(
            start_ns=100 if index % 2 == 0 else 1060, end_ns=140 if index % 2 == 0 else 1100
        )
    record["intervals"][2].update(
        start_ns=140 if math_in_gap else 100, end_ns=1060 if math_in_gap else 140
    )
    page = work_interval_metrics(case, record)
    stripe = stripe_interval_metrics(case, record, 2)
    assert stripe["fetch_stripe_union_us"] == 0.08
    assert stripe["page_envelope_only_union_us"] == 0.92
    if math_in_gap:
        assert page["fetch_math_overlap_fraction"] == 0.92
        assert stripe["fetch_stripe_math_overlap_fraction"] == 0
        assert stripe["page_envelope_only_math_overlap_us"] == 0.92
        assert stripe["page_minus_stripe_math_overlap_fraction"] > 0
    else:
        assert page["fetch_math_overlap_fraction"] == 0.04
        assert stripe["fetch_stripe_math_overlap_fraction"] == 0.5
        assert stripe["page_envelope_only_math_overlap_us"] == 0
        assert stripe["page_minus_stripe_math_overlap_fraction"] < 0


def test_stripe_gaps_filled_by_other_pages_do_not_change_global_union():
    profile = stripe_work_profile(fetch_stripes=2)
    case, record = profile["cases"]["layer_00"], profile["records"][1]
    for page, (start, end) in zip(
        record["intervals"], [(100, 1000), (300, 700), (300, 700)], strict=True
    ):
        page.update(start_ns=start, end_ns=end)
    for stripe, (start, end) in zip(
        record["stripe_intervals"], [(100, 300), (700, 1000), (300, 500), (500, 700)], strict=True
    ):
        stripe.update(start_ns=start, end_ns=end)
    metrics = stripe_interval_metrics(case, record, 2)
    assert metrics["fetch_stripe_union_us"] == 0.9
    assert metrics["fetch_stripe_math_overlap_us"] == 0.4
    assert metrics["page_envelope_only_union_us"] == 0
    assert metrics["page_envelope_only_math_overlap_us"] == 0
    assert metrics["page_minus_stripe_math_overlap_fraction"] == 0
