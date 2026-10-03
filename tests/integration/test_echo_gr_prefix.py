"""Opt-in real-weight ECHO checks; no timings or experiment artifacts."""

import json
import os
import unittest
from contextlib import nullcontext
from pathlib import Path


@unittest.skipUnless(
    os.environ.get("SPARSEGR_RUN_ECHO_GPU_TESTS") == "1",
    "requires the isolated ECHO environment and explicit real-weight GPU opt-in",
)
class EchoGRPrefixTests(unittest.TestCase):
    @staticmethod
    def prefix_snapshot(runner, prefix):
        import torch

        pool = runner.runner.token_to_kv_pool
        pages = (prefix.locations // runner.runner.page_size).unique()
        layers = []
        for layer_id in range(runner.provenance["num_layers"]):
            if hasattr(pool, "device_pool"):
                kv = pool.kv_buffer[layer_id].view(torch.bfloat16)
            else:
                kv = pool.get_key_buffer(layer_id)
            records = kv.index_select(0, prefix.locations.to(kv.device)).clone()
            index = pool.get_index_k_with_scale_buffer(layer_id).index_select(0, pages).clone()
            layers.append((records, index))
        return layers

    def check_prefix_snapshot(self, runner, prefix, expected):
        import torch

        actual = self.prefix_snapshot(runner, prefix)
        for values, saved in zip(actual, expected, strict=True):
            for value, reference in zip(values, saved, strict=True):
                self.assertTrue(torch.equal(value, reference), "stable prefix cache was mutated")

    @staticmethod
    def device_available(runner):
        allocators = getattr(runner.runner.token_to_kv_pool, "device_pool_allocator", ())
        return tuple(int(allocator.available_size()) for allocator in allocators)

    def select_requests(self, requests):
        first = second = None
        users = {}
        for request in requests:
            users.setdefault(request["user_id"], request)
            if first is None:
                first = request
            elif second is None and request["user_id"] == first["user_id"]:
                second = request
            if second is not None and len(users) >= 3:
                break
        self.assertIsNotNone(second, "the GR input needs a revisit of its first user")
        self.manager_requests = tuple(users.values())[:3]
        return first, second

    def requests(self, model_path):
        trace = os.environ.get("SPARSEGR_ECHO_TEST_TRACE")
        if trace:
            with Path(trace).open() as handle:
                return self.select_requests(json.loads(line) for line in handle)
        from tokenizers import Tokenizer

        from GR.input_generator import INSTRUCTION, TextConfig, create_input_generator
        from GR.scheduling import ScheduleConfig
        from models.deepseek_v32 import request_format

        tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        instruction = len(
            tokenizer.encode(request_format.prefix(INSTRUCTION), add_special_tokens=False).ids
        )
        prefix_length, candidate_length = 4096, 1024
        generator = create_input_generator(
            model="deepseek_v32",
            heat_source="curve",
            curve_dataset="beauty",
            num_users=3,
            text_material="synthetic",
            tokenizer=tokenizer,
            text_config=TextConfig(
                user_lengths=(prefix_length - instruction,),
                user_probabilities=(1.0,),
                item_lengths=(candidate_length + instruction,),
                item_probabilities=(1.0,),
                max_input_tokens=prefix_length + candidate_length,
                history_cache_users=3,
            ),
            schedule_config=ScheduleConfig(seed=42, qps=1, sampling="weighted", arrival="poisson"),
        )
        return self.select_requests(generator.iter_generate(24))

    def check_interleaved_users(self, runner, first, second, reference_a, reference_b):
        from serving.echo_cache_manager import EchoCacheManager

        if len(self.manager_requests) < 3:
            self.fail("multi-user correctness requires three distinct users in the GR input")
        import torch

        _, request_b, request_c = self.manager_requests
        prefix_length = first["stable_prefix_tokens"]
        suffix_length = first["candidate_suffix_tokens"]
        for request in self.manager_requests:
            self.assertEqual(request["stable_prefix_tokens"], prefix_length)
            self.assertEqual(request["candidate_suffix_tokens"], suffix_length)
        page = runner.runner.page_size
        capacity = 2 * prefix_length + ((suffix_length + page - 1) // page + 1) * page
        manager = EchoCacheManager(
            runner,
            host_capacity_tokens=capacity,
            hbm_retained_users=None if runner.provenance["mode"] == "resident" else 1,
        )
        allocator = runner.runner.token_to_kv_pool_allocator
        available = allocator.available_size()
        device_available = self.device_available(runner)
        try:
            initial = manager.execute(first)
            self.assertFalse(initial.prefix_reused)
            torch.testing.assert_close(initial.hidden_states, reference_a, rtol=0.02, atol=0.02)
            other = manager.execute(request_b)
            self.assertFalse(other.prefix_reused)
            self.assertTrue(bool(torch.isfinite(other.hidden_states).all()))
            repeated = manager.execute(second)
            self.assertTrue(repeated.prefix_reused)
            torch.testing.assert_close(repeated.hidden_states, reference_b, rtol=0.02, atol=0.02)
            pressure = manager.execute(request_c)
            self.assertEqual(pressure.evicted_user_ids, (request_b["user_id"],))
            self.assertTrue(bool(torch.isfinite(pressure.hidden_states).all()))
            rebuilt = manager.execute(request_b)
            self.assertFalse(rebuilt.prefix_reused)
            self.assertEqual(rebuilt.evicted_user_ids, (first["user_id"],))
            torch.testing.assert_close(
                rebuilt.hidden_states, other.hidden_states, rtol=0.02, atol=0.02
            )
            self.assertEqual(manager.observed_frequency[first["user_id"]], 2)
            self.assertEqual(manager.retained_tokens, 2 * prefix_length)
        finally:
            manager.close()
        self.assertEqual(allocator.available_size(), available)
        self.assertEqual(self.device_available(runner), device_available)
        runner._ensure_alive()
        return {
            "inputs": [request_b["input_ids"], request_c["input_ids"]],
            "hidden": [other.hidden_states.cpu(), pressure.hidden_states.cpu()],
        }

    def test_real_gr_prefix_branches(self):
        import torch

        from experiments.gr_cache_serving.src.transfer import TransferMeter
        from models.deepseek_v32.echo_kernel import PREFETCH_PHASE_PATCH_ID
        from serving.echo_runner import open_echo_runner
        from tests.integration.echo_topk_control import scoped_logical_topk_order

        self.assertTrue(torch.cuda.is_available(), "explicit ECHO GPU check requires CUDA")
        root = Path(__file__).resolve().parents[2]
        model_path = Path(
            os.environ.get("SPARSEGR_ECHO_MODEL", "/mnt/nfs/share/models/DeepSeek-V3.2")
        )
        first, second = self.requests(model_path)
        prefix_length = first["stable_prefix_tokens"]
        candidate_length = first["candidate_suffix_tokens"]
        for request in (first, second):
            self.assertEqual(request["model"], "deepseek_v32")
            self.assertEqual(request["stable_prefix_tokens"], prefix_length)
            self.assertEqual(request["candidate_suffix_tokens"], candidate_length)
            self.assertEqual(len(request["input_ids"]), prefix_length + candidate_length)
        prefix_ids = first["input_ids"][:prefix_length]
        self.assertEqual(prefix_ids, second["input_ids"][:prefix_length])
        candidate_a = first["input_ids"][prefix_length:]
        candidate_b = second["input_ids"][prefix_length:]
        self.assertNotEqual(candidate_a, candidate_b)
        mode = os.environ.get("SPARSEGR_ECHO_MODE", "resident")
        num_layers = int(os.environ.get("SPARSEGR_ECHO_TEST_LAYERS", "3"))
        kernel_patch = os.environ.get("SPARSEGR_ECHO_TEST_KERNEL_PATCH", "phase_snapshot")
        self.assertIn(kernel_patch, ("phase_snapshot", "native"))
        topk_order = os.environ.get("SPARSEGR_ECHO_TEST_TOPK_ORDER", "logical")
        self.assertIn(topk_order, ("logical", "native"))
        device_tokens = max(8192, ((prefix_length + candidate_length + 63) // 64 + 1) * 64)
        budget = guard = host_guard = None
        fixed = os.environ.get("SPARSEGR_ECHO_TEST_FIXED_BUDGET") == "1"
        if fixed:
            from experiments.gr_cache_serving.src.host_memory import (
                HostCacheGuard,
                inspect_host_availability,
                plan_host_cache,
            )
            from experiments.gr_cache_serving.src.memory_guard import MemoryGuard
            from experiments.gr_cache_serving.src.preflight import audit_checkpoint
            from serving.echo_budget import plan_echo_budget

            self.assertEqual((num_layers, prefix_length, candidate_length), (5, 65536, 1024))
            host_plan = plan_host_cache(
                users=128, layers=5, prefix=65536, suffix=1024, budget_bytes=66 * 2**30
            )
            inspect_host_availability(host_plan)
            budget = plan_echo_budget(
                mode=mode,
                num_layers=5,
                users=128,
                prefix_tokens=65536,
                suffix_tokens=1024,
                checkpoint_stored_bytes=audit_checkpoint(model_path, 5)["selected_stored_bytes"],
            )
            self.assertEqual(budget.retained_users_capacity, 93 if mode == "resident" else 128)
            guard = MemoryGuard(torch, total_bytes=72 * 2**30, non_torch_reserve_bytes=2 * 2**30)
            host_guard = HostCacheGuard(torch, host_plan)
        with (
            guard if guard else nullcontext(),
            open_echo_runner(
                model_path=model_path,
                echo_path=root / "3rdparty/ECHO",
                num_layers=num_layers,
                mode=mode,
                max_total_tokens=budget.logical_pool_tokens if budget else 2 * device_tokens,
                device_cache_tokens=budget.device_cache_tokens if budget else device_tokens,
                mem_fraction_static=budget.torch_limit_bytes
                / torch.cuda.get_device_properties(0).total_memory
                if budget
                else 0.6,
                dense_context_tokens=budget.dense_context_tokens if budget else None,
                dense_prefetch_schedule=(
                    os.environ.get("SPARSEGR_ECHO_DENSE_SCHEDULE", "attention_window")
                    if mode == "dense_prefetch"
                    else None
                ),
                dense_prefetch_transport=(
                    os.environ.get("SPARSEGR_ECHO_DENSE_TRANSPORT", "gpu_direct")
                    if mode == "dense_prefetch"
                    else None
                ),
                prefill_chunk=1024,
                kernel_patch=(
                    PREFETCH_PHASE_PATCH_ID
                    if mode == "echo_gr_adapted" and kernel_patch == "phase_snapshot"
                    else None
                ),
            ) as runner,
            (
                scoped_logical_topk_order(runner.runner)
                if topk_order == "logical"
                else nullcontext({"id": "native", "calls": 0, "source_sha256": None})
            ) as control,
            TransferMeter(runner) as transfer_meter,
        ):
            if host_guard:
                host_guard.sample(runner)
            print(
                "ECHO kernel selection: "
                + json.dumps(runner.provenance["kernel_overlay"], sort_keys=True),
                flush=True,
            )
            self.assertFalse(hasattr(runner.runner.model, "lm_head"))
            self.assertEqual(len(runner.runner.model.model.layers), num_layers)
            if num_layers > 3:
                from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

                for layer in runner.runner.model.model.layers[3:]:
                    self.assertIsInstance(layer.mlp, DeepseekV2MoE)
                    self.assertEqual(layer.mlp.config.n_routed_experts, 256)
                    self.assertEqual(layer.mlp.top_k, 8)
            self.assertEqual(runner.runner.server_args.attention_backend, "nsa")
            if mode == "dense_prefetch":
                transport = os.environ.get("SPARSEGR_ECHO_DENSE_TRANSPORT", "gpu_direct")
                self.assertEqual(runner.dense_controller.provenance["transport"], transport)
                if transport == "gpu_direct":
                    self.assertEqual(runner.dense_controller._host_staging, ())
                    self.assertIsNone(runner.dense_controller._executor)
                    self.assertEqual(
                        runner.dense_controller.memory_accounting["pinned_staging_bytes"], 0
                    )

                def reject_sparse_recall(*args, **kwargs):
                    self.fail("dense prefetch must not fall back to sparse recall")

                runner.runner.token_to_kv_pool.recall_miss_tokens_extend_cuda_graph = (
                    reject_sparse_recall
                )
            available = runner.runner.token_to_kv_pool_allocator.available_size()
            device_available = self.device_available(runner)
            prefix = runner.prefill(prefix_ids)
            prefix_locations = prefix.locations.clone()
            prefix_snapshot = self.prefix_snapshot(runner, prefix)
            after_prefix = runner.runner.token_to_kv_pool_allocator.available_size()
            if mode != "resident":
                runner.evict_hbm(prefix)
                mapping = runner.runner.token_to_kv_pool.host_token_to_device
                self.assertTrue(bool((mapping[:, prefix.locations] == 2**31 - 1).all()))
            transfer_meter.begin_request()
            output_a = runner.extend(prefix, candidate_a)
            transfer = transfer_meter.finish_request()
            self.assertEqual(
                transfer["d2h_mla_payload_bytes"],
                0 if mode == "resident" else candidate_length * num_layers * 1152,
            )
            if mode == "resident":
                self.assertEqual(transfer["h2d_mla_payload_bytes"], 0)
            elif mode == "dense_prefetch":
                self.assertEqual(
                    transfer["h2d_mla_payload_bytes"], prefix_length * num_layers * 1152
                )
            else:
                self.assertGreater(transfer["h2d_mla_payload_bytes"], 0)
            if mode == "dense_prefetch":
                stats = runner.dense_controller.last_batch_stats
                self.assertEqual(stats["h2d_payload_bytes"], prefix_length * 576 * 2 * num_layers)
                self.assertEqual(stats["new_suffix_tokens"], candidate_length)
                self.assertEqual(
                    stats["schedule"],
                    os.environ.get("SPARSEGR_ECHO_DENSE_SCHEDULE", "attention_window"),
                )
            self.assertEqual(tuple(output_a.shape), (candidate_length, 7168))
            self.assertTrue(bool(torch.isfinite(output_a).all()))
            output_b = runner.extend(prefix, candidate_b)
            self.assertTrue(bool(torch.isfinite(output_b).all()))
            self.assertFalse(torch.equal(output_a, output_b))
            repeated_a = runner.extend(prefix, candidate_a)
            self.check_prefix_snapshot(runner, prefix, prefix_snapshot)
            torch.testing.assert_close(output_a, repeated_a, rtol=0.02, atol=0.02)
            torch.testing.assert_close(prefix.locations, prefix_locations, rtol=0, atol=0)
            self.assertEqual(
                runner.runner.token_to_kv_pool_allocator.available_size(), after_prefix
            )
            runner.release(prefix)
            self.assertEqual(runner.runner.token_to_kv_pool_allocator.available_size(), available)
            self.assertEqual(self.device_available(runner), device_available)
            with self.assertRaises(ValueError):
                runner.extend(prefix, candidate_a)
            rebuilt_prefix = runner.prefill(prefix_ids)
            rebuilt_a = runner.extend(rebuilt_prefix, candidate_a)
            torch.testing.assert_close(output_a, rebuilt_a, rtol=0.02, atol=0.02)
            runner.release(rebuilt_prefix)
            self.assertEqual(runner.runner.token_to_kv_pool_allocator.available_size(), available)
            other_users = self.check_interleaved_users(runner, first, second, output_a, output_b)
            if fixed:
                self.check_fixed_capacity(
                    runner, first, second, output_a, output_b, budget, guard, host_guard
                )
            if topk_order == "logical":
                self.assertGreater(control["calls"], 0)
            reference_path = os.environ.get("SPARSEGR_ECHO_REFERENCE")
            if reference_path:
                if mode == "resident":
                    with Path(reference_path).open("xb") as handle:
                        torch.save(
                            {
                                "input_a": first["input_ids"],
                                "input_b": second["input_ids"],
                                "checkpoint": runner.provenance["checkpoint_metadata_sha256"],
                                "echo_revision": runner.provenance["echo_revision"],
                                "num_layers": num_layers,
                                "echo_patch_sha256": runner.provenance["echo_patch_sha256"],
                                "hidden_a": output_a.cpu(),
                                "hidden_b": output_b.cpu(),
                                "other_users": other_users,
                                "topk_order_control": control["id"],
                                "topk_order_control_sha256": control["source_sha256"],
                            },
                            handle,
                        )
                else:
                    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
                    self.assertEqual(reference["input_a"], first["input_ids"])
                    self.assertEqual(reference["topk_order_control"], control["id"])
                    self.assertEqual(
                        reference["topk_order_control_sha256"], control["source_sha256"]
                    )
                    self.assertEqual(reference["input_b"], second["input_ids"])
                    self.assertEqual(
                        reference["checkpoint"], runner.provenance["checkpoint_metadata_sha256"]
                    )
                    self.assertEqual(reference["echo_revision"], runner.provenance["echo_revision"])
                    self.assertEqual(reference["num_layers"], num_layers)
                    self.assertEqual(
                        reference["echo_patch_sha256"], runner.provenance["echo_patch_sha256"]
                    )
                    torch.testing.assert_close(
                        output_a.cpu(), reference["hidden_a"], rtol=0.02, atol=0.02
                    )
                    torch.testing.assert_close(
                        output_b.cpu(), reference["hidden_b"], rtol=0.02, atol=0.02
                    )
                    self.assertEqual(other_users["inputs"], reference["other_users"]["inputs"])
                    for hidden, expected in zip(
                        other_users["hidden"], reference["other_users"]["hidden"], strict=True
                    ):
                        torch.testing.assert_close(hidden, expected, rtol=0.02, atol=0.02)

    def check_fixed_capacity(
        self, runner, first, second, reference_a, reference_b, budget, guard, host_guard
    ):
        """Exercise the real full-size pool, then rebuild two LRU-evicted histories."""
        import torch

        from serving.echo_cache_manager import EchoCacheManager

        capacity = budget.retained_users_capacity
        manager = EchoCacheManager(runner, host_capacity_tokens=budget.logical_pool_tokens)
        original_release = runner.release

        def checked_release(prefix):
            locations = prefix.locations.clone()
            original_release(prefix)
            pool = runner.runner.token_to_kv_pool
            if hasattr(pool, "device_pool"):
                self.assertTrue(bool((pool.host_token_to_device[:, locations] == 2**31 - 1).all()))

        runner.release = checked_release
        try:
            for uid in range(capacity):
                result = manager.execute(first | {"user_id": uid})
                self.assertFalse(result.prefix_reused)
                self.assertEqual(result.evicted_user_ids, ())
                self.assertEqual(manager.retained_tokens, (uid + 1) * 65536)
                self.assertEqual(
                    runner.runner.token_to_kv_pool_allocator.available_size(),
                    budget.logical_pool_tokens - manager.retained_tokens,
                )
                self.assertTrue(bool(torch.isfinite(result.hidden_states).all()))
                if uid in (0, capacity - 1):
                    torch.testing.assert_close(
                        result.hidden_states, reference_a, rtol=0.02, atol=0.02
                    )
                host_guard.sample(runner, manager)
                guard.check()
                del result
                if (uid + 1) % 16 == 0:
                    print(
                        f"Fixed-cache correctness: {uid + 1}/{capacity} histories filled",
                        flush=True,
                    )
            repeated = manager.execute(second | {"user_id": 0})
            self.assertTrue(repeated.prefix_reused)
            torch.testing.assert_close(repeated.hidden_states, reference_b, rtol=0.02, atol=0.02)
            del repeated
            pressure = manager.execute(first | {"user_id": capacity})
            self.assertFalse(pressure.prefix_reused)
            self.assertEqual(pressure.evicted_user_ids, (1,))
            del pressure
            rebuilt = manager.execute(first | {"user_id": 1})
            self.assertFalse(rebuilt.prefix_reused)
            self.assertEqual(rebuilt.evicted_user_ids, (2,))
            torch.testing.assert_close(rebuilt.hidden_states, reference_a, rtol=0.02, atol=0.02)
            self.assertEqual(manager.retained_tokens, capacity * 65536)
            host_guard.sample(runner, manager)
            guard.check()
            print("Fixed-cache Host observation: " + json.dumps(host_guard.metadata()), flush=True)
        finally:
            try:
                manager.close()
            finally:
                runner.release = original_release
        self.assertEqual(
            runner.runner.token_to_kv_pool_allocator.available_size(), budget.logical_pool_tokens
        )
        self.assertTrue(
            all(value == budget.device_cache_tokens for value in self.device_available(runner))
        )


if __name__ == "__main__":
    unittest.main()
