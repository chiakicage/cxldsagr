"""NOSA resident block sparse adapter for the shared main-attention contract."""


class NosaSparseAttention:
    """Consume logical blocks and resident K/V/CIS without expanding selected KV.

    ``auto`` uses Triton on CUDA and the mathematical reference on CPU. An
    unavailable or unsupported CUDA backend fails explicitly. Offloaded cache
    access belongs to a future fetch/compute operator, not this resident adapter.
    """

    def __init__(self, *, backend="auto"):
        if backend not in ("auto", "reference", "triton"):
            raise ValueError("backend must be auto, reference or triton")
        self.backend = backend

    def __call__(self, q, selection, cache_access, context):
        from operators.sm90.nosa_attention import (
            nosa_block_sparse_attention,
            reference_nosa_block_sparse_attention,
        )

        if selection is None:
            raise ValueError("NOSA sparse attention requires a block selection")
        if context.query_length != q.shape[0]:
            raise ValueError("query_length must match the number of Q tokens")
        layer_view = getattr(cache_access, "layer_view", None)
        if not callable(layer_view):
            raise NotImplementedError("NOSA sparse attention requires resident cache layer views")
        records = layer_view(context.layer_idx)
        state = context.auxiliary_state
        cis = getattr(state, "cis_scores", None)
        if cis is None:
            cis = records.get("cis_scores")
        if cis is None:
            raise ValueError("NOSA sparse attention requires per-token CIS scores")
        backend = self.backend
        if backend == "auto":
            backend = "triton" if q.is_cuda else "reference"
        attention = (
            nosa_block_sparse_attention
            if backend == "triton"
            else reference_nosa_block_sparse_attention
        )
        return attention(
            q,
            records["keys"],
            records["values"],
            selection,
            query_start=context.query_start,
            cis_bias=cis,
        )
