"""Layout adaptation required by the official FlashMLA SM90 sparse prefill API."""

DECODE_SPLITS = 16
DECODE_SELECTION_COUNT = 2048


def padded_selection_count(selected):
    """Pad selection capacity for FlashMLA's pair of 64-token tiles.

    The adapter appends invalid IDs, preserving every original selected slot.
    This helper is shared with measured executed-FLOPs accounting; it does not
    select or reproduce an upstream kernel configuration.
    """
    return (selected + 127) // 128 * 128
