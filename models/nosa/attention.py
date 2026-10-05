"""NOSA resident and offloaded adapters for the shared main-attention contract."""

from collections.abc import Callable

import torch


class DenseMainAttention:
    """Bridge resident NHD K/V to the existing ``attention(q, k, v)`` seam."""

    def __init__(self, attention: Callable):
        self.attention = attention

    def __call__(self, q, selection, cache_access, context):
        if selection is not None:
            raise NotImplementedError("Dense main attention does not implement block selection")
        layer_view = getattr(cache_access, "layer_view", None)
        if not callable(layer_view):
            raise NotImplementedError("Dense main attention requires resident cache layer views")
        records = layer_view(context.layer_idx)
        return self.attention(q, records["keys"], records["values"])


class ResidentLayerView:
    """Ephemeral cache access for a forward without a persistent KV session."""

    def __init__(self, layer_idx: int, **records: torch.Tensor):
        self.layer_idx = layer_idx
        self.records = records
        self.state = None
        self.length = 0
        self.max_seq_len = next(iter(records.values())).shape[0]

    def _check_layer(self, layer_idx):
        if layer_idx != self.layer_idx:
            raise IndexError("This resident view belongs to a different layer")

    def layer_view(self, layer_idx):
        self._check_layer(layer_idx)
        return self.records

    def get_layer_state(self, layer_idx):
        self._check_layer(layer_idx)
        return self.state

    def set_layer_state(self, layer_idx, state):
        self._check_layer(layer_idx)
        self.state = state


class NosaSparseAttention:
    """Consume logical blocks and resident K/V/CIS without expanding selected KV.

    ``auto`` uses Triton on CUDA and the mathematical reference on CPU. An
    unavailable or unsupported CUDA backend fails explicitly. Offloaded cache
    access delegates fused sparse fetching and attention to the SM90 operator.
    """

    def __init__(self, *, backend="auto", workspace=None):
        if backend not in ("auto", "reference", "triton"):
            raise ValueError("backend must be auto, reference or triton")
        self.backend = backend
        self.workspace = workspace

    def __call__(self, q, selection, cache_access, context):
        from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
        from operators.nosa.attention.reference.torch import reference_nosa_block_sparse_attention

        if selection is None:
            raise ValueError("NOSA sparse attention requires a block selection")
        if context.query_length != q.shape[0]:
            raise ValueError("query_length must match the number of Q tokens")
        from models.nosa.cache.offload import NosaOffloadCache

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
            workspace = cache_access.attention_workspace
            output = workspace.run(
                q,
                selection,
                records["host_keys"],
                records["host_values"],
                records["suffix_keys"],
                records["suffix_values"],
                records["cis_scores"],
                context.query_start,
            )
            if cache_access._transfer_metrics is not None:
                cache_access._transfer_metrics.add_sparse_fetch(workspace.last_transfer_bytes)
            return output
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
        options = {}
        resources = getattr(cache_access, "_execution_resources", None)
        if q.is_cuda and resources is not None:
            resources.check_execution(cache_access, len(q))
            if self.workspace is not resources.attention_workspace or self.workspace is None:
                raise RuntimeError(
                    "Borrowed resident cache requires its reserved attention workspace"
                )
        if q.is_cuda and self.workspace is not None:
            if backend == "reference":
                raise NotImplementedError("Reserved CUDA workspace requires native NOSA attention")
            options["workspace"] = self.workspace
        return attention(
            q,
            records["keys"],
            records["values"],
            selection,
            query_start=context.query_start,
            cis_bias=cis,
            **options,
        )


class NosaDensePrefetchAttention:
    def __call__(self, q, selection, cache_access, context):
        from models.nosa.cache.dense_prefetch import NosaDensePrefetchCache

        if not isinstance(cache_access, NosaDensePrefetchCache):
            raise TypeError("Dense prefetch attention requires its matching cache")
        records = cache_access.dense_layer_view(context.layer_idx)
        options = {}
        if q.is_cuda:
            from operators.nosa.attention.device_only.api import nosa_block_sparse_attention

            attention = nosa_block_sparse_attention
            resources = cache_access._execution_resources
            if resources is not None:
                resources.check_execution(cache_access, len(q))
                options["workspace"] = resources.attention_workspace
        else:
            from operators.nosa.attention.reference.torch import (
                reference_nosa_block_sparse_attention,
            )

            attention = reference_nosa_block_sparse_attention
        return attention(
            q,
            records["keys"],
            records["values"],
            selection,
            query_start=context.query_start,
            cis_bias=records["cis_scores"],
            **options,
        )


class NosaFixedAttention:
    """Use the same logical selections with resident, dense, or sparse IO."""

    def __call__(self, q, selection, cache_access, context):
        resources = cache_access._execution_resources
        resources.check_execution(cache_access, len(q))
        if resources.scheme == "hbm":
            return NosaSparseAttention(workspace=resources.attention_workspace)(
                q, selection, cache_access, context
            )
        if resources.scheme == "dense_prefetch":
            records = cache_access.dense_layer_view(context.layer_idx)
        elif not q.is_cuda:
            records = resources.fetch_reference(cache_access, context.layer_idx, selection)
        else:
            records = cache_access.offload_layer_view(context.layer_idx)
            workspace = resources.fetch(cache_access)
            workspace.keys = resources.keys[context.layer_idx, : workspace.max_seq_len]
            workspace.values = resources.values[context.layer_idx, : workspace.max_seq_len]
            output = workspace.run(
                q,
                selection,
                records["host_keys"],
                records["host_values"],
                records["suffix_keys"],
                records["suffix_values"],
                records["cis_scores"],
                context.query_start,
                cache_tags=resources.tags[context.layer_idx],
                cache_owner=cache_access.cache_owner,
            )
            cache_access._transfer_metrics.add_sparse_fetch(workspace.last_transfer_bytes)
            return output
        if q.is_cuda:
            from operators.nosa.attention.device_only.api import nosa_block_sparse_attention

            attention = nosa_block_sparse_attention
            options = {"workspace": resources.attention_workspace}
        else:
            from operators.nosa.attention.reference.torch import (
                reference_nosa_block_sparse_attention,
            )

            attention = reference_nosa_block_sparse_attention
            options = {}
        return attention(
            q,
            records["keys"],
            records["values"],
            selection,
            query_start=context.query_start,
            cis_bias=records["cis_scores"],
            **options,
        )
