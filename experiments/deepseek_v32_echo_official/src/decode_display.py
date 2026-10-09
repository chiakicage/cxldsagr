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


def activity_style(row):
    if row["kind"] == "kernel" and row["category"] == RECALL_CATEGORY:
        return "H2D", "H2D", False
    return row["lane"], row["category"], row["potential_io"]
