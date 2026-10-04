"""Compile-time layout checks for typed norm copies."""


def check_layout(label, actual, canonical):
    """Keep every thread/value coordinate equal to the Float32 constructor."""
    actual_text, canonical_text = str(actual), str(canonical)
    if actual_text != canonical_text:
        raise ValueError(f"Norm {label} layout changed: {actual_text} != {canonical_text}")
