"""NOSA resident and offloaded adapters for the shared main-attention contract."""


class NosaSparseAttention:
    """Consume logical blocks and resident K/V/CIS without expanding selected KV.

    ``auto`` uses Triton on CUDA and the mathematical reference on CPU. An
    unavailable or unsupported CUDA backend fails explicitly. Offloaded cache
    access delegates fused sparse fetching and attention to the SM90 operator.
    """

    def __init__(self, *, backend="auto"):
        if backend not in ("auto", "reference", "triton"):
            raise ValueError("backend must be auto, reference or triton")
        self.backend = backend

    def __call__(self, q, selection, cache_access, context):
        from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
        from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention

        if selection is None:
            raise ValueError("NOSA sparse attention requires a block selection")
        if context.query_length != q.shape[0]:
            raise ValueError("query_length must match the number of Q tokens")
        from models.nosa.offload_cache import NosaOffloadCache

        if isinstance(cache_access, NosaOffloadCache):
            if context.auxiliary_state is not None:
                raise ValueError("Offloaded NOSA attention does not accept external CIS state")
            records = cache_access.offload_layer_view(context.layer_idx)
            if context.query_start != records["query_start"]:
                raise ValueError("Offloaded attention must consume the pending append")
            if not q.is_cuda:
                # CPU-only reference checks do not represent GPU overlap.
                import torch

                return reference_nosa_block_sparse_attention(
                    q,
                    torch.cat(
                        (records["host_keys"][: context.query_start], records["suffix_keys"])
                    ),
                    torch.cat(
                        (records["host_values"][: context.query_start], records["suffix_values"])
                    ),
                    selection,
                    query_start=context.query_start,
                    cis_bias=records["cis_scores"],
                )
            if self.backend == "reference":
                raise ValueError("CUDA offloaded NOSA attention requires the native backend")
            return cache_access.attention_workspace.run(
                q,
                selection,
                records["host_keys"],
                records["host_values"],
                records["suffix_keys"],
                records["suffix_values"],
                records["cis_scores"],
                context.query_start,
            )
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
