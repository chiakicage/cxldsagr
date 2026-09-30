"""CPU-only dense transport contracts; not GPU numerical or overlap validation."""

from __future__ import annotations

import gc
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from models.deepseek_v32.echo_dense import (
    MLA_RECORD_BYTES,
    DensePrefetchController,
    _override_extend,
    dense_allocation_plan,
    plan_dense_batch,
    scoped_dense_prefetch,
)


class Vector(list):
    def __ge__(self, other):
        return Vector(x >= other for x in self)

    def __lt__(self, other):
        return Vector(x < other for x in self)

    def __eq__(self, other):
        return Vector(x == other for x in self)

    def __and__(self, other):
        return Vector(a and b for a, b in zip(self, other, strict=True))

    def __or__(self, other):
        return Vector(a or b for a, b in zip(self, other, strict=True))

    def all(self):
        return all(self)


class Mapping(list):
    def __getitem__(self, indices):
        return Vector(list.__getitem__(self, index) for index in indices)


class Buffer:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, key):
        return self

    def copy_(self, source, non_blocking=False):
        self.calls.append(("copy", self.name, source.name, non_blocking))

    def reshape(self, *args):
        return self

    def view(self, *args):
        return self


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_controller(prefix=2):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))

    class Event:
        count = 0

        def __init__(self):
            self.name = f"event{Event.count}"
            Event.count += 1

        def record(self, stream):
            calls.append(("record", self.name, stream.name))

        def synchronize(self):
            calls.append(("sync", self.name))

    controller = object.__new__(DensePrefetchController)
    controller.device = "fake_cuda"
    compute = Stream("compute")
    controller.torch = SimpleNamespace(
        cuda=SimpleNamespace(
            stream=lambda stream: nullcontext(),
            Event=Event,
            current_stream=lambda device: compute,
            synchronize=lambda device: calls.append(("device_sync", device)),
        ),
        index_select=lambda host, dim, indices, out: calls.append(("gather", host, out.name)),
        _assert_async=require,
        cat=lambda values, dim: (values, dim),
    )
    controller._copy_stream = Stream("copy")
    controller._staging_ready = [None, None]
    controller._buffer_released = [None, None]
    controller._last_compute_event = None
    controller._host_staging = [Buffer(f"stage{i}", calls) for i in range(2)]
    controller._device_buffers = [Buffer(f"scratch{i}", calls) for i in range(2)]
    controller._host_layers = ["host0", "host1", "host2"]
    controller._prefix_ids_cpu = [9, 3][:prefix]
    controller._plan = plan_dense_batch(
        [9, 3, 7],
        prefix_tokens=prefix,
        max_context_tokens=3,
        host_capacity_tokens=16,
        num_layers=3,
    )
    controller.num_layers = 3
    controller._next_layer = 0
    controller._current_layer = None
    controller._attention_seen = False
    controller._copy_events = [None] * 3
    controller._active_batch = None
    controller._closed = False
    controller._failed = False
    controller._prepare_batch = lambda batch: calls.append(("prepare",))
    return controller, calls


class DensePlanTests(unittest.TestCase):
    def test_transfer_all_existing_prefix_not_newly_generated_suffix(self):
        plan = plan_dense_batch(
            [9, 3, 7, 6],
            prefix_tokens=3,
            max_context_tokens=4,
            host_capacity_tokens=16,
            num_layers=3,
        )
        self.assertEqual(plan.host_locations, (9, 3, 7, 6))
        self.assertEqual(plan.suffix_tokens, 1)
        self.assertEqual(plan.h2d_bytes, 3 * 3 * 1152)
        cold = plan_dense_batch(
            [9],
            prefix_tokens=0,
            max_context_tokens=4,
            host_capacity_tokens=16,
            num_layers=1,
        )
        self.assertEqual(cold.h2d_bytes, 0)

    def test_full_context_and_id_validation(self):
        base = {
            "host_locations": [1, 2, 3],
            "prefix_tokens": 2,
            "max_context_tokens": 3,
            "host_capacity_tokens": 16,
            "num_layers": 3,
        }
        for overrides in (
            {"max_context_tokens": 2},
            {"host_locations": [1, 1, 3]},
            {"host_locations": [0, 1, 2]},
            {"host_locations": [1, 2, 16]},
            {"host_locations": [1, 2, -1]},
            {"host_locations": [1, 2, True]},
            {"host_locations": []},
            {"prefix_tokens": 3},
            {"prefix_tokens": -1},
            {"prefix_tokens": True},
            {"num_layers": 4},
            {"num_layers": True},
            {"max_context_tokens": 0},
            {"host_capacity_tokens": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                plan_dense_batch(**(base | overrides))

    def test_two_scratch_buffers_and_original_pool_exclusion_explicit(self):
        values = dense_allocation_plan(65536 + 1024, 128 * 65536)
        expected = 2 * (65536 + 1024 + 64) * 1152
        self.assertEqual(values["scratch_buffers"], 2)
        self.assertEqual(values["scratch_hbm_bytes"], expected)
        self.assertEqual(values["pinned_staging_bytes"], expected)
        self.assertEqual(
            values["additional_persistent_hbm_bytes"],
            expected + (128 * 65536 + 1) * 4 + (65536 + 1024 + 1) * 4,
        )
        self.assertIn("original_write_pool", values["excluded_from_additional_bytes"])
        self.assertFalse(values["full_hbm_budget_verified"])
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                dense_allocation_plan(invalid, 16)


class DensePipelineTests(unittest.TestCase):
    def test_ping_pong_waits_for_both_dma_source_and_compute_destination(self):
        controller, calls = make_controller()
        with controller.forward_batch(object()):
            for layer in range(3):
                controller._before_layer(layer)
                controller._attention_seen = True
                controller._after_layer(layer)
        self.assertEqual(
            [call for call in calls if call[0] == "copy"],
            [
                ("copy", "scratch0", "stage0", True),
                ("copy", "scratch1", "stage1", True),
                ("copy", "scratch0", "stage0", True),
            ],
        )
        self.assertLess(
            calls.index(("record", "event1", "compute")), calls.index(("gather", "host1", "stage1"))
        )
        self.assertLess(calls.index(("sync", "event0")), calls.index(("gather", "host2", "stage0")))
        self.assertLess(
            calls.index(("wait", "copy", "event1")), calls.index(("record", "event4", "copy"))
        )
        for ready in ("event0", "event2", "event4"):
            self.assertIn(("wait", "compute", ready), calls)
        self.assertEqual(controller.last_batch_stats["h2d_payload_bytes"], 6 * 1152)
        self.assertEqual(controller.last_batch_stats["staged_prefix_tokens"], 6)
        self.assertEqual(controller.last_batch_stats["staged_prefix_tokens_per_layer"], [2, 2, 2])
        self.assertEqual(
            controller.last_batch_stats["h2d_payload_bytes_per_layer"], [2304, 2304, 2304]
        )
        self.assertEqual(controller.last_batch_stats["device_generated_suffix_bytes"], 3 * 1152)

    def test_cold_request_transfers_no_uninitialized_host_suffix(self):
        controller, calls = make_controller(prefix=0)
        with controller.forward_batch(object()):
            for layer in range(3):
                controller._before_layer(layer)
                controller._attention_seen = True
                controller._after_layer(layer)
        self.assertFalse(any(call[0] in ("copy", "gather") for call in calls))
        self.assertEqual(controller.last_batch_stats["h2d_payload_bytes"], 0)
        self.assertEqual(controller.last_batch_stats["staged_prefix_tokens_per_layer"], [0, 0, 0])
        self.assertEqual(
            controller.last_batch_stats["device_generated_suffix_bytes"], 9 * MLA_RECORD_BYTES
        )

    def test_failure_poisoning_and_incomplete_forward(self):
        for error_inside in (True, False):
            controller, calls = make_controller()
            with self.assertRaises(RuntimeError), controller.forward_batch(object()):
                if error_inside:
                    raise RuntimeError("compute failed")
            self.assertTrue(controller._failed)
            self.assertIsNone(controller._active_batch)
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                controller._ensure_alive()
            controller.close()
            self.assertIn(("device_sync", "fake_cuda"), calls)

    def test_wrong_layer_order_and_missing_attention_rejected(self):
        controller, _calls = make_controller()
        with self.assertRaisesRegex(RuntimeError, "order"), controller.forward_batch(object()):
            controller._before_layer(1)
        controller, _calls = make_controller()
        with (
            self.assertRaisesRegex(RuntimeError, "exactly one"),
            controller.forward_batch(object()),
        ):
            controller._before_layer(0)
            controller._after_layer(0)

    def test_close_drops_resources_after_sync_even_when_sync_fails(self):
        class Resource:
            pass

        for sync_fails in (False, True):
            with self.subTest(sync_fails=sync_fails):
                controller, calls = make_controller()
                references = {}
                for name in (
                    "runner",
                    "pool",
                    "backend",
                    "_mapping",
                    "_positions",
                    "_prefix_ids_cpu",
                    "_active_batch",
                    "_plan",
                    "_copy_stream",
                    "_last_compute_event",
                ):
                    resource = Resource()
                    setattr(controller, name, resource)
                    references[name] = weakref.ref(resource)
                for name in (
                    "layers",
                    "_host_layers",
                    "_device_buffers",
                    "_host_staging",
                    "_copy_events",
                    "_staging_ready",
                    "_buffer_released",
                ):
                    resource = Resource()
                    setattr(controller, name, (resource,))
                    references[name] = weakref.ref(resource)
                del resource
                provenance = {"mode": "test"}
                stats = {"h2d_payload_bytes": 0}
                accounting = {"full_hbm_budget_verified": False}
                controller.provenance = provenance
                controller.last_batch_stats = stats
                controller.memory_accounting = accounting

                def synchronize(
                    device, *, references=references, calls=calls, sync_fails=sync_fails
                ):
                    self.assertTrue(
                        all(reference() is not None for reference in references.values())
                    )
                    calls.append(("device_sync", device))
                    if sync_fails:
                        raise RuntimeError("synchronization failed")

                controller.torch.cuda.synchronize = synchronize
                if sync_fails:
                    with self.assertRaisesRegex(RuntimeError, "synchronization failed"):
                        controller.close()
                    self.assertTrue(controller._failed)
                else:
                    controller.close()
                gc.collect()
                for name, reference in references.items():
                    self.assertIsNone(reference(), name)
                self.assertIs(controller.provenance, provenance)
                self.assertIs(controller.last_batch_stats, stats)
                self.assertIs(controller.memory_accounting, accounting)
                with self.assertRaisesRegex(RuntimeError, "fresh process"):
                    controller._ensure_alive()
                controller.close()
                self.assertEqual(calls, [("device_sync", "fake_cuda")])

    def test_scope_restores_backend_and_hooks_when_caller_fails(self):
        class Backend:
            def forward_extend(self):
                return "original"

        class Layer:
            def __init__(self):
                self.hooks = []

            def register_forward_pre_hook(self, hook):
                self.hooks.append(hook)
                return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

            register_forward_hook = register_forward_pre_hook

        for close_early, sync_fails in ((False, False), (True, False), (False, True)):
            with self.subTest(close_early=close_early, sync_fails=sync_fails):
                backend = Backend()
                layers = [Layer(), Layer(), Layer()]
                controller, calls = make_controller()
                controller.layers = layers
                controller.backend = backend

                def synchronize(device, *, calls=calls, sync_fails=sync_fails):
                    calls.append(("device_sync", device))
                    if sync_fails:
                        raise RuntimeError("synchronization failed")

                controller.torch.cuda.synchronize = synchronize
                with patch(
                    "models.deepseek_v32.echo_dense.DensePrefetchController",
                    return_value=controller,
                ):
                    message = "synchronization failed" if sync_fails else "caller failed"
                    with (
                        self.assertRaisesRegex(RuntimeError, message),
                        scoped_dense_prefetch(
                            SimpleNamespace(attn_backend=backend), max_context_tokens=3
                        ),
                    ):
                        self.assertIn("forward_extend", vars(backend))
                        self.assertTrue(all(len(layer.hooks) == 2 for layer in layers))
                        if close_early:
                            controller.close()
                        raise RuntimeError("caller failed")
                self.assertNotIn("forward_extend", vars(backend))
                self.assertEqual(backend.forward_extend(), "original")
                self.assertTrue(all(not layer.hooks for layer in layers))
                self.assertEqual(calls, [("device_sync", "fake_cuda")])
        backend = Backend()
        override = lambda: "prior instance override"
        backend.forward_extend = override
        with _override_extend(backend, lambda: "temporary"):
            self.assertEqual(backend.forward_extend(), "temporary")
        self.assertIs(backend.forward_extend, override)

    def test_same_sparse_kernel_and_selection_with_no_recall(self):
        controller, calls = make_controller()
        batch = SimpleNamespace(out_cache_loc=object(), req_pool_indices_cpu=[0])
        controller._active_batch = batch
        controller._current_layer = 0
        controller.source = SimpleNamespace(NSA_FUSE_TOPK=True)
        controller.pool = SimpleNamespace(
            size=16,
            set_mla_kv_buffer=lambda *args, **kwargs: calls.append(("original_host_write",)),
            recall_miss_tokens_extend_cuda_graph=lambda *args: self.fail("sparse recall used"),
        )
        controller.backend = SimpleNamespace(
            forward_metadata=object(),
            _forward_flashmla_prefill=lambda **kwargs: kwargs,
        )
        controller._mapping = Mapping([-1] * 17)
        controller._mapping[0] = 0
        for dense, host in enumerate(controller._plan.host_locations, 1):
            controller._mapping[host] = dense
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            v_head_dim=512,
            head_dim=576,
            tp_q_head_num=128,
            scaling=0.1,
        )
        q, k, v, qr, kr = (Buffer(name, calls) for name in ("q", "k", "v", "qr", "kr"))
        result = controller._forward_extend(
            q, k, v, layer, batch, q_rope=qr, k_rope=kr, topk_indices=Vector([9, 7, -1, 0])
        )
        self.assertEqual(list(result["page_table_1"]), [1, 3, -1, 0])
        self.assertIs(result["kv_cache"], controller._device_buffers[0])
        self.assertEqual(result["sm_scale"], 0.1)
        self.assertEqual(calls[0], ("original_host_write",))
        self.assertEqual(
            calls[1:], [("copy", "scratch0", "k", False), ("copy", "scratch0", "kr", False)]
        )
        controller._attention_seen = False
        with self.assertRaisesRegex(RuntimeError, "outside the complete active request"):
            controller._forward_extend(
                q, k, v, layer, batch, q_rope=qr, k_rope=kr, topk_indices=Vector([8])
            )
        controller._attention_seen = True
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            controller._forward_extend(
                q, k, v, layer, batch, q_rope=qr, k_rope=kr, topk_indices=Vector([9])
            )

    def test_nonfused_topk_uses_original_logical_to_host_transform(self):
        controller, calls = make_controller()
        batch = SimpleNamespace(out_cache_loc=object(), req_pool_indices_cpu=[0])
        controller._active_batch = batch
        controller._current_layer = 0
        seen = {}

        def transform(**kwargs):
            seen.update(kwargs)
            return Vector([3, 7, -1])

        controller.source = SimpleNamespace(
            NSA_FUSE_TOPK=False, transform_index_page_table_prefill=transform
        )
        controller.pool = SimpleNamespace(size=16, set_mla_kv_buffer=lambda *args, **kwargs: None)
        controller.backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(page_table_1="original", nsa_extend_seq_lens_list=[1]),
            _forward_flashmla_prefill=lambda **kwargs: kwargs,
        )
        controller._mapping = Mapping([-1] * 17)
        for dense, host in enumerate(controller._plan.host_locations, 1):
            controller._mapping[host] = dense
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            v_head_dim=512,
            head_dim=576,
            tp_q_head_num=128,
            scaling=0.1,
        )
        q = Buffer("q", calls)
        selection = object()
        result = controller._forward_extend(
            q, q, q, layer, batch, q_rope=q, k_rope=q, topk_indices=selection
        )
        self.assertEqual(list(result["page_table_1"]), [2, 3, -1])
        self.assertEqual(
            seen,
            {
                "page_table": "original",
                "topk_indices": selection,
                "extend_lens_cpu": [1],
                "page_size": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
