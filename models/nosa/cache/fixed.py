"""Fixed-history NOSA caches with transient candidates and finite HBM slots.

The supported fixed workload uses page-aligned history and history chunks.
Each logical historical page maps directly to its offset in the per-layer
pool. A page tag identifies its resident session; a conflicting session
replaces that page. This is a direct-mapped policy, not a token-LRU policy.
"""

from models.nosa.cache.offload import NosaOffloadCache
from models.nosa.cache.resident import NosaKVCache, _NosaTransferMetrics


class _FixedHistory:
    def _init_fixed_history(self, history_capacity):
        if type(history_capacity) is not int or history_capacity <= 0 or history_capacity % 64:
            raise ValueError("Fixed NOSA history must be a positive multiple of 64 tokens")
        self.history_capacity = history_capacity
        self.transient_candidate = False

    def begin_step(self, token_count):
        if not self.transient_candidate and self.length + token_count > self.history_capacity:
            raise ValueError("Persistent append exceeds retained history capacity")
        if self.transient_candidate and self.length != self.history_capacity:
            raise ValueError("Candidates require a complete retained history")
        return super().begin_step(token_count)

    def commit_step(self):
        if self.transient_candidate:
            self.validate_commit()
            # The attention output has already been submitted. Roll back only
            # the pending cursors; history and its derived records are immutable.
            self.abort_step()
            return
        start, end = self.length, self._pending_end
        super().commit_step()
        if isinstance(self, NosaFixedOffloadCache):
            self._execution_resources.commit_history(self, start, end)


class NosaFixedResidentCache(_FixedHistory, NosaKVCache):
    def __init__(
        self, config, history_capacity, *, execution_resources, execution_capacity=None, **kwargs
    ):
        self._init_fixed_history(history_capacity)
        if execution_capacity is None:
            execution_capacity = (
                history_capacity + execution_resources.plan.metadata["max_candidate_tokens"]
            )
        super().__init__(
            config,
            execution_capacity,
            execution_resources=execution_resources,
            **kwargs,
        )


class NosaFixedOffloadCache(_FixedHistory, NosaOffloadCache):
    """Only retained history has host storage; candidates are device-only.

    The backend owns per-layer historical K/V slots and candidate tail capacity.
    Session-owned suffix references keep current projection outputs alive until
    the attention and asynchronous host writes have completed.
    """

    def __init__(
        self, config, history_capacity, *, execution_resources, execution_capacity=None, **kwargs
    ):
        self._init_fixed_history(history_capacity)
        if execution_capacity is None:
            execution_capacity = (
                history_capacity + execution_resources.plan.metadata["max_candidate_tokens"]
            )
        super().__init__(
            config,
            execution_capacity,
            execution_resources=execution_resources,
            host_capacity=history_capacity,
            **kwargs,
        )
        self.cache_owner = execution_resources.next_cache_owner()
        self._prefetched_layers = set()
        if execution_resources.scheme == "dense_prefetch":
            self._transfer_metrics = _NosaTransferMetrics(
                execution_resources, device_copy_counter=True
            )

    def begin_step(self, token_count):
        result = super().begin_step(token_count)
        self._prefetched_layers.clear()
        if self._execution_resources.scheme == "dense_prefetch":
            try:
                self._execution_resources.prefetch(self, 0)
                self._prefetched_layers.add(0)
            except BaseException as error:
                try:
                    self.abort_step()
                except BaseException as rollback_error:  # noqa: BLE001 -- retain both failures.
                    raise BaseExceptionGroup(
                        "NOSA cache begin and rollback both failed", [error, rollback_error]
                    ) from None
                raise
        return result

    def _write_shared_layer(self, layer_idx, records):
        if self.transient_candidate:
            # No candidate host allocation, copy, or persistent publication.
            self._suffixes[layer_idx] = dict(records)
            self._written_layers.add(layer_idx)
        else:
            super()._write_shared_layer(layer_idx, records)

    def write_layer(self, layer_idx, **records):
        super().write_layer(layer_idx, **records)
        resources = self._execution_resources
        if resources.scheme == "dense_prefetch":
            if layer_idx not in self._prefetched_layers:
                raise RuntimeError("Dense prefetch requires increasing layer order")
            resources.wait_prefetch(self, layer_idx)
            resources.invalidate_suffix(layer_idx, self.length, self._pending_end)
            resources.keys[layer_idx, self.length : self._pending_end].copy_(records["keys"])
            resources.values[layer_idx, self.length : self._pending_end].copy_(records["values"])
            if layer_idx + 1 < self.spec.num_layers:
                resources.prefetch(self, layer_idx + 1)
                self._prefetched_layers.add(layer_idx + 1)

    def dense_layer_view(self, layer_idx):
        resources = self._execution_resources
        resources.check_execution(self)
        end = self.visible_length(layer_idx)
        resources.wait_prefetch(self, layer_idx)
        return {
            "keys": resources.keys[layer_idx, :end],
            "values": resources.values[layer_idx, :end],
            "cis_scores": self.cis_scores[layer_idx, :end],
        }
