"""Measure GPU fetch/attention interval intersections from an Nsight Systems export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

RANGE_PREFIX = "nosa_overlap/"
FETCH_PATTERN = r"nosa_sparse_fetch_kernel|nosa_offload_fused::serialized_fetch"
FUSED_PATTERN = r"nosa_offload_fused::fused_main"
ATTENTION_PATTERN = (
    r"nosa_attention::|nosa_fa3::|nosa_attention_fa3::|PrefillWithKVCacheKernel"
    r"|sort_work_by_union_size|_nosa_block_attention|nosa_offload_fa3_base::"
)
BATCH_FETCH_DEFINITION = (
    "Each selected (block, KV head) has one row for its identity and logical K+V bytes. "
    "A copy batch may contain one or more pages. Its rows share a timestamp recorded after "
    "the participating fetch threads finish scanning, followed by a participating fetch-thread "
    "barrier before host copying begins; this is not an individual vector's load-start time. "
    "Each row ends after that page's stores, per-thread fences and participating fetch-thread "
    "barrier, before its "
    "own byte-counter atomics and ready-flag release publication. Later page windows can "
    "include earlier pages' counter/publication work. The union measures host-copy windows, "
    "not PCIe/CXL wire occupancy."
)
PAGE_FETCH_DEFINITION = (
    "Each selected (block, KV head) has one copy window from before its host loads through "
    "per-thread store fences and the following CTA barrier, ending before byte-counter "
    "atomics and ready-flag release publication. This is not PCIe/CXL wire occupancy."
)
FETCH_THREAD_PAGE_DEFINITION = (
    "Each selected (block, KV head) has one row for its identity, logical K+V bytes and "
    "host-copy window. Its timestamp starts after all participating fetch threads finish "
    "page-selection/task-claim and decoding metadata work, followed by a participating "
    "fetch-thread barrier before host copying begins. "
    "It ends after that page's stores, per-thread fences and participating fetch-thread "
    "barrier, before byte-counter atomics and ready-flag release publication. The window "
    "excludes page-selection/task-claim metadata work and consumer-ready polling; it is not individual vector "
    "load latency or PCIe/CXL wire occupancy."
)
PAGE_ENVELOPE_DEFINITION = (
    "Each selected (block, KV head) has one row for its unique logical K+V bytes and "
    "the envelope from the earliest nonempty stripe start to the latest nonempty stripe "
    "end. An envelope can contain gaps between stripe copy windows; its union alone "
    "does not establish copy-window occupancy or an overlap threshold. Page envelopes "
    "are retained as a separate metric and checked against every nonempty stripe."
)
STRIPE_COPY_DEFINITION = (
    "Each selected page is partitioned into fetch_stripes equal token ranges within "
    "its 64-token block. Each nonempty historical range has one kind-3 row, starting "
    "after task claim and metadata decoding, before the participating fetch-thread "
    "barrier and host loads, and ending after stores, per-thread fences and the "
    "participating fetch-thread barrier, before counters and ready publication. "
    "Empty historical ranges have no row or payload. The union counts concurrent "
    "stripe windows once and excludes gaps covered only by page envelopes; it is "
    "not PCIe/CXL wire occupancy or individual vector load latency."
)


def work_profile_metadata(profile):
    """Validate timestamp semantics separately from the unchanged numeric row layout."""
    version = profile.get("schema_version")
    if (
        type(version) is not int
        or version not in (1, 2, 3)
        or profile.get("clock") != "device_globaltimer_ns"
        or profile.get("math_coverage") != "softmax_only"
    ):
        raise ValueError("Unsupported work interval clock, schema or math coverage")
    coverage = profile.get("fetch_coverage") if version >= 2 else "per_page_copy_windows"
    allowed_coverage = (
        ("per_page_copy_envelopes",)
        if version == 3
        else ("batch_copy_windows", "per_page_copy_windows")
    )
    if version >= 2 and (
        coverage not in allowed_coverage
        or profile.get("byte_accounting") != "logical_unique_kv_payload"
    ):
        raise ValueError("Unsupported work interval fetch coverage or byte accounting")
    metadata = {
        "work_interval_schema_version": version,
        "fetch_coverage": coverage,
        "byte_accounting": "logical_unique_kv_payload",
        "fetch_definition": PAGE_ENVELOPE_DEFINITION
        if version == 3
        else PAGE_FETCH_DEFINITION
        if version == 1
        else BATCH_FETCH_DEFINITION
        if coverage == "batch_copy_windows"
        else FETCH_THREAD_PAGE_DEFINITION,
    }
    if version == 3:
        validate_fetch_stripes(profile.get("fetch_stripes"))
        if profile.get("stripe_coverage") != "nonempty_stripe_copy_windows":
            raise ValueError("Unsupported work interval stripe coverage")
        metadata.update(
            fetch_stripes=profile["fetch_stripes"],
            stripe_coverage=profile["stripe_coverage"],
            stripe_definition=STRIPE_COPY_DEFINITION,
        )
    return metadata


def validate_fetch_stripes(value):
    if type(value) is not int or value <= 0 or 64 % value:
        raise ValueError("fetch_stripes must be a positive integer divisor of 64")


def merged_intervals(intervals):
    """Union half-open intervals, so simultaneous kernels are never double counted."""
    result = []
    for start, end in sorted(intervals):
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise ValueError("Intervals must contain integer nanoseconds with end >= start")
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def interval_metrics(fetch, attention):
    fetch, attention = merged_intervals(fetch), merged_intervals(attention)
    i = j = overlap = 0
    while i < len(fetch) and j < len(attention):
        overlap += max(0, min(fetch[i][1], attention[j][1]) - max(fetch[i][0], attention[j][0]))
        if fetch[i][1] <= attention[j][1]:
            i += 1
        else:
            j += 1
    fetch_ns = sum(end - start for start, end in fetch)
    attention_ns = sum(end - start for start, end in attention)
    return {
        "fetch_union_us": fetch_ns / 1000,
        "attention_union_us": attention_ns / 1000,
        "overlap_us": overlap / 1000,
        "fetch_without_attention_us": (fetch_ns - overlap) / 1000,
        "fetch_hidden_fraction": overlap / fetch_ns if fetch_ns else None,
        "attention_during_fetch_fraction": overlap / attention_ns if attention_ns else None,
    }


def _columns(connection, table):
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def work_interval_metrics(case, record):
    """Validate exact block-copy provenance and intersect one device clock domain.

    Batch rows share a start and keep per-page completion times and logical bytes.
    Their union counts each batch copy window once; page durations cannot be
    interpreted as independent load latencies. Each row ends before that page's
    counters/release, but can include earlier pages' publication in its batch.
    Softmax intervals cover a subset of attention computation, so the intersection
    is a lower bound on host-copy-window overlap with attention math, not wire
    occupancy. Neither waiting CTAs nor the whole fused lifetime count as math.
    """
    prefix, queries, heads = (case[key] for key in ("prefix", "queries", "kv_heads"))
    if any(type(value) is not int or value <= 0 for value in (prefix, queries, heads)):
        raise ValueError("Invalid work-trace geometry")
    fetch_rows = ((prefix + 63) // 64) * heads
    capacity = fetch_rows + ((queries + 7) // 8) * heads * 64
    expected = {item["row"]: item["bytes"] for item in case["expected_fetch_rows"]}
    if len(expected) != len(case["expected_fetch_rows"]) or any(
        type(row) is not int
        or not 0 <= row < fetch_rows
        or type(size) is not int
        or size != min(64, prefix - (row // heads) * 64) * 128 * 2 * 2
        for row, size in expected.items()
    ):
        raise ValueError("Invalid expected unique host block copies")
    expected_bytes = sum(expected.values())
    if (
        expected_bytes != case["expected_prefix_bytes"]
        or record["recorded_transfer_bytes"] != expected_bytes
    ):
        raise ValueError("Work trace transfer counter differs from exact CPU union")
    entries = record["intervals"]
    if not entries:
        if record["mode"] == "overlap":
            raise ValueError("Fused overlap requires actual fetch and softmax work intervals")
        return None
    fetch, math, copied, seen = [], [], {}, set()
    for entry in entries:
        row, start, end, size, kind = (
            entry[key] for key in ("row", "start_ns", "end_ns", "bytes", "kind")
        )
        if (
            any(type(value) is not int for value in (row, start, end, size, kind))
            or not 0 <= row < capacity
            or row in seen
            or not 0 < start < end
        ):
            raise ValueError("Invalid, duplicate or overflowing globaltimer work interval")
        seen.add(row)
        if kind == 1:
            if row not in expected or size != expected[row]:
                raise ValueError("Host-copy interval does not match selected block bytes")
            copied[row] = size
            fetch.append((start, end))
        elif kind == 2:
            if row < fetch_rows or size != 0:
                raise ValueError("Invalid softmax interval slot or byte count")
            math.append((start, end))
        else:
            raise ValueError("Unknown work interval kind")
    if copied != expected:
        raise ValueError("Missing selected host-copy intervals")
    if record["mode"] == "overlap" and not math:
        raise ValueError("Fused overlap requires instrumented softmax computation")
    metrics = interval_metrics(fetch, math)
    return {
        "clock": "device_globaltimer_ns",
        "math_coverage": "softmax_only" if math else "none",
        "fetch_work_union_us": metrics["fetch_union_us"],
        "softmax_union_us": metrics["attention_union_us"],
        "fetch_math_overlap_us": metrics["overlap_us"],
        "fetch_without_softmax_us": metrics["fetch_without_attention_us"],
        "fetch_math_overlap_fraction": metrics["fetch_hidden_fraction"] if math else None,
        "instrumented_span_us": (
            max(end for _, end in fetch + math) - min(start for start, _ in fetch + math)
        )
        / 1000,
        "copied_bytes": expected_bytes,
        "fetch_interval_count": len(fetch),
        "softmax_interval_count": len(math),
    }


def stripe_interval_metrics(case, record, fetch_stripes):
    """Independently validate every nonempty stripe and its exact page envelope.

    Row identity, logical payload and min/max timestamps are checked before the
    stripe union is intersected with softmax. This does not infer active copying
    from a page's first start and last completion, or from the fused lifetime.
    """
    validate_fetch_stripes(fetch_stripes)
    page_metrics = work_interval_metrics(case, record)
    entries = record.get("stripe_intervals")
    if entries is None:
        raise ValueError("Schema 3 requires explicit stripe_intervals for every offload sample")
    if not isinstance(entries, list):
        raise TypeError("stripe_intervals must be a list")
    if record["mode"] != "overlap":
        if record["mode"] != "serialized" or entries:
            raise ValueError("Only fused overlap may claim stripe intervals")
        return None
    prefix, queries, heads = (case[key] for key in ("prefix", "queries", "kv_heads"))
    fetch_slots = ((prefix + 63) // 64) * heads
    stripe_base = fetch_slots + ((queries + 7) // 8) * heads * 64
    stripe_tokens = 64 // fetch_stripes
    expected = {}
    for page in case["expected_fetch_rows"]:
        slot = page["row"]
        historical_tokens = min(64, prefix - (slot // heads) * 64)
        for stripe in range(fetch_stripes):
            tokens = min(stripe_tokens, historical_tokens - stripe * stripe_tokens)
            if tokens > 0:
                expected[stripe_base + slot * fetch_stripes + stripe] = (
                    slot,
                    tokens * 128 * 2 * 2,
                )
    if not expected:
        raise ValueError("Fused stripe evidence requires selected historical pages")
    seen, stripes, by_page = set(), [], {}
    for entry in entries:
        row, start, end, size, kind = (
            entry[key] for key in ("row", "start_ns", "end_ns", "bytes", "kind")
        )
        if (
            any(type(value) is not int for value in (row, start, end, size, kind))
            or row in seen
            or not 0 < start < end
            or kind != 3
        ):
            raise ValueError("Invalid or duplicate globaltimer stripe interval")
        if row not in expected or size != expected[row][1]:
            raise ValueError("Stripe interval does not match selected nonempty stripe bytes")
        seen.add(row)
        stripes.append((start, end))
        by_page.setdefault(expected[row][0], []).append((start, end, size))
    if seen != expected.keys():
        raise ValueError("Missing selected nonempty stripe intervals")
    for page in record["intervals"]:
        if page["kind"] != 1:
            continue
        children = by_page[page["row"]]
        if (
            page["start_ns"] != min(start for start, _, _ in children)
            or page["end_ns"] != max(end for _, end, _ in children)
            or page["bytes"] != sum(size for _, _, size in children)
        ):
            raise ValueError("Page envelope differs from nonempty stripe min/max or byte sum")
    math = [(r["start_ns"], r["end_ns"]) for r in record["intervals"] if r["kind"] == 2]
    metrics = interval_metrics(stripes, math)
    return {
        "clock": "device_globaltimer_ns",
        "math_coverage": "softmax_only",
        "fetch_stripe_union_us": metrics["fetch_union_us"],
        "softmax_union_us": metrics["attention_union_us"],
        "fetch_stripe_math_overlap_us": metrics["overlap_us"],
        "fetch_stripe_without_softmax_us": metrics["fetch_without_attention_us"],
        "fetch_stripe_math_overlap_fraction": metrics["fetch_hidden_fraction"],
        "page_envelope_only_union_us": page_metrics["fetch_work_union_us"]
        - metrics["fetch_union_us"],
        "page_envelope_only_math_overlap_us": page_metrics["fetch_math_overlap_us"]
        - metrics["overlap_us"],
        "page_minus_stripe_math_overlap_fraction": page_metrics["fetch_math_overlap_fraction"]
        - metrics["fetch_hidden_fraction"],
        "copied_bytes": sum(size for _, size in expected.values()),
        "stripe_interval_count": len(entries),
        "fetch_interval_count": len(by_page),
        "softmax_interval_count": len(math),
    }


def extract_sqlite(
    path,
    *,
    fetch_pattern=FETCH_PATTERN,
    attention_pattern=ATTENTION_PATTERN,
    fused_pattern=FUSED_PATTERN,
    work_profile=None,
):
    """Attribute kernels using launch correlation and measured NVTX host ranges.

    Each harness range waits for GPU completion before closing. A CPU-range
    containment fallback is explicit for exports without launch correlation.
    It does not infer overlap from stream IDs or from host submission ranges.
    """
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"}
        if missing := required - tables:
            raise ValueError(f"Missing Nsight tables: {sorted(missing)}")
        strings = {row[0]: row[1] for row in connection.execute("SELECT id,value FROM StringIds")}
        ranges = []
        for raw in connection.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL"):
            row = dict(raw)
            label = row.get("text") or strings.get(row.get("textId"), "")
            if label.startswith(RANGE_PREFIX):
                parts = label.split("/")
                if len(parts) != 4 or not parts[-1].startswith("sample_"):
                    continue
                ranges.append(
                    {
                        "label": label,
                        "case": parts[1],
                        "mode": parts[2],
                        "sample": int(parts[3].removeprefix("sample_")),
                        "start": row["start"],
                        "end": row["end"],
                        "global_tid": row.get("globalTid"),
                    }
                )
        if not ranges:
            raise ValueError("No completed nosa_overlap measured NVTX ranges in trace")
        if len({item["label"] for item in ranges}) != len(ranges):
            raise ValueError("Measured NVTX labels must be unique")
        launches = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            if not {"correlationId", "start", "end", "globalTid"} <= _columns(connection, table):
                continue
            for row in connection.execute(
                f'SELECT correlationId,start,end,globalTid FROM "{table}"'
            ):
                launches.setdefault(row["correlationId"], []).append(dict(row))
        kernels = []
        for raw in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            row = dict(raw)
            name_id = row.get("demangledName", row.get("shortName", row.get("mangledName")))
            row["name"] = strings.get(name_id, str(name_id))
            kernels.append(row)
        fetch_re, attention_re = re.compile(fetch_pattern), re.compile(attention_pattern)
        fused_re = re.compile(fused_pattern)
        work_records = {}
        if work_profile is not None:
            work_profile_metadata(work_profile)
            work_records = {
                (row["case"], row["mode"], row["sample"]): row for row in work_profile["records"]
            }
            expected_keys = {
                (row["case"], row["mode"], row["sample"])
                for row in ranges
                if row["mode"] != "resident"
            }
            if (
                len(work_records) != len(work_profile["records"])
                or work_records.keys() != expected_keys
            ):
                raise ValueError("Work interval samples do not match measured Nsight ranges")
        records = []
        for scope in sorted(ranges, key=lambda item: item["start"]):
            selected = []
            attribution = Counter()
            for kernel in kernels:
                calls = launches.get(kernel.get("correlationId"), [])
                matched = any(
                    scope["start"] <= call["start"] < scope["end"]
                    and (scope["global_tid"] is None or call["globalTid"] == scope["global_tid"])
                    for call in calls
                )
                contained = scope["start"] <= kernel["start"] <= kernel["end"] <= scope["end"]
                if matched:
                    if not contained:
                        raise ValueError("Measured NVTX range closed before its GPU work completed")
                    attribution["launch_correlation"] += 1
                elif not calls and contained:
                    attribution["completed_range_containment"] += 1
                else:
                    continue
                selected.append(kernel)
            if not selected:
                raise ValueError(f"No GPU kernels attributed to {scope['label']}")
            fused = [kernel for kernel in selected if fused_re.search(kernel["name"])]
            fetch = [
                kernel
                for kernel in selected
                if kernel not in fused and fetch_re.search(kernel["name"])
            ]
            attention = [
                kernel
                for kernel in selected
                if kernel not in fused and attention_re.search(kernel["name"])
            ]
            if fused and (scope["mode"] != "overlap" or len(fused) != 1 or fetch):
                raise ValueError(
                    "Fused overlap requires exactly one fused main and no separate host fetch"
                )
            if work_profile is not None and scope["mode"] == "overlap" and len(fused) != 1:
                raise ValueError("Expected exactly one fused main kernel per overlap sample")
            if fused and work_profile is None:
                raise ValueError(
                    "Fused main needs intra-kernel work intervals; its lifetime is not overlap"
                )
            if not attention or (scope["mode"] != "resident" and not fetch and not fused):
                raise ValueError(f"Missing expected fetch/attention kernels in {scope['label']}")
            if any(kernel in attention for kernel in fetch):
                raise ValueError("Fetch and attention classification patterns overlap")
            other = [
                kernel
                for kernel in selected
                if kernel not in fetch and kernel not in attention and kernel not in fused
            ]
            intervals = lambda items: [(item["start"], item["end"]) for item in items]
            metrics = interval_metrics(intervals(fetch), intervals(attention))
            if scope["mode"] == "serialized" and metrics["overlap_us"] != 0:
                raise ValueError(
                    f"Serialized control contains overlapping fetch/attention: {scope['label']}"
                )
            if scope["mode"] == "resident" and fetch:
                raise ValueError(f"Resident control contains host fetch: {scope['label']}")
            record = {
                **{key: scope[key] for key in ("case", "mode", "sample", "label")},
                **metrics,
                "fetch_kernel_sum_us": sum(item["end"] - item["start"] for item in fetch) / 1000,
                "attention_kernel_sum_us": sum(item["end"] - item["start"] for item in attention)
                / 1000,
                "gpu_span_us": (
                    max(item["end"] for item in selected) - min(item["start"] for item in selected)
                )
                / 1000,
                "nvtx_wall_us": (scope["end"] - scope["start"]) / 1000,
                "fetch_streams": sorted({item["streamId"] for item in fetch}),
                "attention_streams": sorted({item["streamId"] for item in attention}),
                "kernel_counts": dict(Counter(item["name"] for item in selected)),
                "other_kernel_counts": dict(Counter(item["name"] for item in other)),
                "attribution": dict(attribution),
            }
            if work_profile is not None:
                # CUPTI and globaltimer timestamps are separate clock domains.
                # Only intervals within the same domain may be intersected.
                for key in metrics:
                    record.pop(key)
                record["kernel_intervals"] = None if fused else metrics
                record["fused_main_count"] = len(fused)
                record["fused_main_us"] = sum(k["end"] - k["start"] for k in fused) / 1000
                record["work_metrics"] = None
                if scope["mode"] != "resident":
                    key = scope["case"], scope["mode"], scope["sample"]
                    record["work_metrics"] = work_interval_metrics(
                        work_profile["cases"][scope["case"]], work_records[key]
                    )
                if work_profile["schema_version"] == 3:
                    record["stripe_metrics"] = (
                        None
                        if scope["mode"] == "resident"
                        else stripe_interval_metrics(
                            work_profile["cases"][scope["case"]],
                            work_records[key],
                            work_profile["fetch_stripes"],
                        )
                    )
                record["profile_method"] = (
                    "globaltimer_fetch_softmax" if fused else "nsys_separate_kernel_intervals"
                )
            records.append(record)
        cases = {record["case"] for record in records}
        for case in cases:
            counts = Counter(record["mode"] for record in records if record["case"] == case)
            if (
                set(counts) != {"resident", "serialized", "overlap"}
                or len(set(counts.values())) != 1
            ):
                raise ValueError(
                    f"Incomplete or unequal three-mode profile: {case}: {dict(counts)}"
                )
            sample_ids = {
                mode: {
                    record["sample"]
                    for record in records
                    if record["case"] == case and record["mode"] == mode
                }
                for mode in counts
            }
            if any(ids != sample_ids["resident"] for ids in sample_ids.values()):
                raise ValueError(f"Mismatched measured sample IDs: {case}: {sample_ids}")
        return records
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fetch-pattern", default=FETCH_PATTERN)
    parser.add_argument("--attention-pattern", default=ATTENTION_PATTERN)
    parser.add_argument("--fused-pattern", default=FUSED_PATTERN)
    parser.add_argument("--work-intervals", type=Path)
    args = parser.parse_args(argv)
    work_profile = json.loads(args.work_intervals.read_text()) if args.work_intervals else None
    if work_profile is not None and work_profile["run_id"] != args.run_id:
        parser.error("Work interval run ID differs from the requested profile run")
    records = extract_sqlite(
        args.sqlite,
        fetch_pattern=args.fetch_pattern,
        attention_pattern=args.attention_pattern,
        fused_pattern=args.fused_pattern,
        work_profile=work_profile,
    )
    result = {
        "schema_version": max(2, work_profile["schema_version"]) if work_profile is not None else 1,
        "run_id": args.run_id,
        "sqlite_sha256": hashlib.file_digest(args.sqlite.open("rb"), "sha256").hexdigest(),
        "fetch_pattern": args.fetch_pattern,
        "attention_pattern": args.attention_pattern,
        "definition": "Intersection of the unions of actual GPU fetch and attention kernel execution intervals; fraction divides intersection by fetch union. Includes FA3 preparation and repair, and fetch-kernel first-use scanning even for tiles with zero transferred bytes; excludes planner, memcpy and launch gaps. It does not measure PCIe packets or a 50 GB/s rate limit.",
        "records": records,
    }
    if work_profile is not None:
        result.update(
            **work_profile_metadata(work_profile),
            work_intervals_sha256=hashlib.file_digest(
                args.work_intervals.open("rb"), "sha256"
            ).hexdigest(),
            fused_pattern=args.fused_pattern,
            definition="Fused calls: intersect unions of instrumented copy windows and consumer softmax-update intervals on the device globaltimer clock, using the separately stated fetch definition. Softmax-only coverage gives a lower bound on copy-window overlap with attention math, not complete attention hidden time or PCIe/CXL wire occupancy. Byte counts are unique logical K+V payload and do not establish physical host-read traffic. Separate-kernel controls use Nsight intervals; absolute timestamps from the two clocks are never mixed.",
        )
        if work_profile["schema_version"] == 3:
            result["definition"] = (
                "Fused calls: independently intersect the unions of page envelopes and "
                "nonempty stripe copy windows with consumer softmax-update intervals on "
                "device globaltimer. Page envelopes are validated against stripe min/max; "
                "their union may include gaps absent from the stripe union, so envelopes "
                "alone cannot establish an overlap threshold. Softmax-only coverage is a "
                "lower bound on each window union's overlap with attention math, not "
                "complete attention hidden time or wire occupancy. Byte counts are unique "
                "logical K+V payload, not physical host-read traffic. Separate-kernel "
                "controls use Nsight intervals; the two absolute clocks are never mixed."
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"Validated {len(records)} complete measured ranges: {args.output}")


if __name__ == "__main__":
    main()
