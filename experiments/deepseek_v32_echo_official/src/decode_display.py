"""Display-only lanes; published activity classifications remain unchanged."""

RECALL_CATEGORY = "ECHO recall (IO unknown)"
RECALL_DISPLAY = {
    "source_category": RECALL_CATEGORY,
    "lane": "H2D",
    "category": "H2D",
    "hatch": None,
    "boundary": (
        "Orange H2D spans show the complete official recall custom kernel by function. "
        "The kernel may update mappings and copy GPU staging (D2D) or mapped host records "
        "(H2D), and may transfer zero records. Actual mapped-host bytes and any pure-copy "
        "subspan are unknown. Original classifications, timestamps and busy/idle unions "
        "are preserved; true widths are not expanded."
    ),
}

FUSED_DISPLAY = {
    "schema": "q1-fused-capability-hatch-v1",
    "kernel_name_contains": "sm90_fp8_paged_mqa_logits_fused_v2<",
    "hatch": "////",
    "boundary": (
        "Hatching marks the complete fused score/prefetch kernel on both sides, "
        "including executions with verified zero transfers. It denotes mapped-host "
        "IO capability, not measured transfer bytes or a pure-copy interval. "
        "Original potential_io and actual_io evidence remain unchanged."
    ),
}


def activity_style(row):
    if row["kind"] == "kernel" and row["category"] == RECALL_CATEGORY:
        return "H2D", "H2D", False
    if row["kind"] == "kernel" and FUSED_DISPLAY["kernel_name_contains"] in row.get("name", ""):
        return row["lane"], row["category"], True
    return row["lane"], row["category"], row["potential_io"]
