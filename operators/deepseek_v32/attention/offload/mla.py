"""Sparse MLA after ECHO prefetch and exact recall into an HBM token pool.

ECHO fuses prefetch with indexer scoring. Attention itself reuses the device-only
MLA kernel once all selected records for this query batch have been recalled.
The model/cache layer owns exact selection, recall, physical remapping, batch
splitting, allocation, and transaction lifetime; none of those are implicit here.
"""

from operators.deepseek_v32.attention.device_only.mla import sparse_mla


def sparse_mla_from_pool(q, records, physical_indices, scale, value_dim=512):
    """Consume recalled HBM records using physical pool IDs.

    Every non-padding ID must already name its selected record in ``records``.
    Causality is established on logical IDs before the caller remaps them.
    The caller keeps the selected slots resident until stream completion.
    Returns the same ``[Q, H, value_dim]`` result as device-only sparse MLA.
    """
    return sparse_mla(q, records, physical_indices, scale, value_dim)
