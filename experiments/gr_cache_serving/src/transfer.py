"""Payload counters for successful ECHO copies, without changing cache decisions.

GPU counter snapshots are enqueued at the original copy boundary. Their cost is
inside service time; reduction and CPU readback occur outside that timer after
the runner has already synchronized. Counts are payload, not PCIe bus traffic.
"""

from contextlib import ExitStack


class TransferMeter:
    def __init__(self, runner):
        import torch

        self.torch = torch
        self.runner = runner
        self.mode = runner.provenance["mode"]
        self.layers = runner.provenance["num_layers"]
        self.stack = ExitStack()
        self.begin_request()

    def _override(self, instance, name, wrap):
        present = name in vars(instance)
        original = getattr(instance, name)
        instance_value = vars(instance).get(name)
        setattr(instance, name, wrap(original))

        def restore():
            if present:
                setattr(instance, name, instance_value)
            else:
                delattr(instance, name)

        self.stack.callback(restore)

    def __enter__(self):
        try:
            return self._install()
        except BaseException:
            self.stack.close()
            raise

    def _install(self):
        pool = self.runner.runner.token_to_kv_pool

        def forward_wrapper(original):
            def forward(ids, locations):
                result = original(ids, locations)
                if self.mode != "resident":
                    self.d2h += (len(ids) - len(locations)) * self.layers * 1152
                if self.mode == "dense_prefetch":
                    self.dense_h2d += self.runner.dense_controller.last_batch_stats[
                        "h2d_payload_bytes"
                    ]
                return result

            return forward

        self._override(self.runner, "_forward", forward_wrapper)
        if self.mode in ("sparse_sync", "echo_gr_adapted"):

            def recall_wrapper(original):
                def recall(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.pending.append(("recall", pool.recall_counter.clone().reshape(())))
                    return result

                return recall

            self._override(pool, "recall_miss_tokens_extend_cuda_graph", recall_wrapper)
        if self.mode == "echo_gr_adapted":

            def prefetch_wrapper(original):
                def prefetch(locations):
                    result = original(locations)
                    # Counter atomicAdd may exceed the task cap; nonzero output
                    # allocations count actual completed prefetch records instead.
                    self.pending.append(("prefetch", (locations > 0).sum()))
                    return result

                return prefetch

            for allocator in pool.device_pool_allocator:
                self._override(allocator, "post_alloc", prefetch_wrapper)
        return self

    def begin_request(self):
        self.pending = []
        self.dense_h2d = self.d2h = 0

    def finish_request(self):
        counts = {"recall": 0, "prefetch": 0}
        if self.pending:
            values = (
                self.torch.stack([value.to(self.torch.int64) for _, value in self.pending])
                .cpu()
                .tolist()
            )
            for (kind, _), value in zip(self.pending, values, strict=True):
                counts[kind] += value
        return {
            "h2d_mla_payload_bytes": (counts["recall"] + counts["prefetch"]) * 1152
            + self.dense_h2d,
            "h2d_recall_payload_bytes": counts["recall"] * 1152,
            "h2d_prefetch_payload_bytes": counts["prefetch"] * 1152,
            "h2d_dense_payload_bytes": self.dense_h2d,
            "d2h_mla_payload_bytes": self.d2h,
        }

    def __exit__(self, *exc):
        self.stack.close()
        self.pending = []
