"""Dependency-free tests of ECHO selection, registration, and wrapper contracts."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.deepseek_v32 import echo_adapter as adapter


def fixture_config():
    return {
        "model_type": "deepseek_v32",
        "architectures": [adapter.ARCHITECTURE],
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "moe_layer_freq": 1,
        "num_experts_per_tok": 8,
        "n_group": 8,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "hidden_size": 7168,
        "vocab_size": 129280,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "num_attention_heads": 128,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "intermediate_size": 18432,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "max_position_embeddings": 163840,
        "rope_theta": 10000,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
        },
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }


class FakeModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeBase(FakeModule):
    def __init__(self, *args, **kwargs):
        raise AssertionError("the CausalLM constructor must never allocate an LM head")

    def load_weights(self, weights):
        return list(weights)


class FakeTensor:
    def __init__(self, value, shape=(1, 2)):
        self.value, self.shape = value, shape

    def __add__(self, other):
        return FakeTensor(self.value + other)

    def new_empty(self, shape):
        return FakeTensor(None, shape)

    def numel(self):
        return self.shape[0] * self.shape[1]


class FakeBackbone(FakeModule):
    def __init__(self, config, quant_config, prefix):
        self.norm = "original-final-norm"
        self.prefix = prefix

    def forward(self, input_ids, positions, batch):
        return self.norm(FakeTensor(4), 7)[0]


def fake_runtime():
    class HostPool:
        def recall_miss_tokens_extend_cuda_graph(self, *args):
            raise AssertionError("unpatched host pool")

    source = SimpleNamespace(
        DeepseekV32ForCausalLM=FakeBase,
        DeepseekV2Model=FakeBackbone,
        get_pp_group=lambda: SimpleNamespace(world_size=1),
        get_tensor_model_parallel_world_size=lambda: 1,
        get_global_server_args=lambda: SimpleNamespace(load_format="dummy"),
        LazyValue=lambda fn: fn,
        add_prefix=lambda suffix, prefix: f"{prefix}.{suffix}" if prefix else suffix,
    )
    return SimpleNamespace(
        torch=SimpleNamespace(nn=SimpleNamespace(Module=FakeModule), no_grad=lambda: lambda fn: fn),
        upstream=source,
        registry=SimpleNamespace(models={name: FakeBase for name in adapter.RUNTIME_ARCHITECTURES}),
        loader_base=object,
        logits_output=SimpleNamespace,
        safe_open=None,
        host_pool=SimpleNamespace(NSATokenToKVPoolHost=HostPool),
    )


class EchoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = fixture_config()
        self.specs = adapter._tensor_specs(self.config, 3)
        self.index = {"weight_map": {name: "first.safetensors" for name in self.specs}}
        self.index["weight_map"].update(
            {
                "model.layers.3.mlp.experts.0.gate_proj.weight": "never-open.safetensors",
                "model.norm.weight": "never-open.safetensors",
                "lm_head.weight": "never-open.safetensors",
            }
        )
        (self.root / "first.safetensors").write_bytes(b"header placeholder")
        self.write_metadata()
        self.environment_patch = patch.dict(os.environ, {}, clear=True)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)

    def write_metadata(self):
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(self.index), encoding="utf-8"
        )

    def test_selection_ignores_head_final_norm_and_unopened_later_shards(self):
        for layers, count in ((1, 28), (2, 55), (3, 82)):
            selected = adapter.inspect_checkpoint(self.root, layers)
            self.assertEqual(len(selected.selected_names), count)
            self.assertEqual(set(selected.shards), {"first.safetensors"})
            self.assertNotIn("lm_head.weight", selected.selected_names)
            self.assertNotIn("model.norm.weight", selected.selected_names)
            self.assertEqual(len(selected.metadata_fingerprint), 64)

    def test_missing_required_parameter_rejected(self):
        del self.index["weight_map"]["model.layers.2.self_attn.indexer.k_norm.bias"]
        self.write_metadata()
        with self.assertRaisesRegex(adapter.AdapterError, "missing required"):
            adapter.inspect_checkpoint(self.root, 3)

    def test_wrong_model_moe_layers_and_layout_rejected(self):
        for key, value in (
            ("model_type", "nosa"),
            ("index_head_dim", 64),
            ("kv_lora_rank", 256),
            ("hidden_act", "relu"),
        ):
            with self.subTest(key=key):
                config = dict(self.config, **{key: value})
                with self.assertRaises(adapter.AdapterError):
                    adapter.validate_checkpoint_config(config, 3)
        for layers in (0, 6, True):
            with self.subTest(layers=layers), self.assertRaises(adapter.AdapterError):
                adapter.validate_checkpoint_config(self.config, layers)

    def test_five_layers_include_every_expert_router_and_shared_expert(self):
        adapter.validate_checkpoint_config(self.config, 5)
        specs = adapter._tensor_specs(self.config, 5)
        self.assertEqual(len(specs), 3212)
        for layer in (3, 4):
            for expert in range(256):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                    self.assertEqual(specs[name + ".weight"].dtypes, ("F8_E4M3",))
                    self.assertIn(name + ".weight_scale_inv", specs)
            self.assertEqual(specs[f"model.layers.{layer}.mlp.gate.weight"].shape, (256, 7168))
            bias = specs[f"model.layers.{layer}.mlp.gate.e_score_correction_bias"]
            self.assertEqual((bias.shape, bias.dtypes), ((256,), ("F32",)))
            self.assertIn(f"model.layers.{layer}.mlp.shared_experts.down_proj.weight", specs)
            self.assertNotIn(f"model.layers.{layer}.mlp.down_proj.weight", specs)

    def test_moe_router_configuration_is_validated(self):
        for key, value in (
            ("topk_method", "greedy"),
            ("n_group", 7),
            ("topk_group", 9),
            ("n_routed_experts", 0),
            ("moe_intermediate_size", 0),
            ("num_experts_per_tok", 257),
            ("routed_scaling_factor", 0),
        ):
            with self.subTest(key=key), self.assertRaises(adapter.AdapterError):
                adapter.validate_checkpoint_config(self.config | {key: value}, 5)

    def test_missing_moe_expert_or_router_fails_before_loading(self):
        specs = adapter._tensor_specs(self.config, 5)
        self.index = {"weight_map": {name: "first.safetensors" for name in specs}}
        for name in (
            "model.layers.4.mlp.experts.255.down_proj.weight_scale_inv",
            "model.layers.3.mlp.gate.e_score_correction_bias",
        ):
            removed = self.index["weight_map"].pop(name)
            self.write_metadata()
            with self.assertRaisesRegex(adapter.AdapterError, "missing required"):
                adapter.inspect_checkpoint(self.root, 5)
            self.index["weight_map"][name] = removed

    def test_path_traversal_and_symlink_escape_rejected(self):
        for name in ("../outside.safetensors", "/tmp/outside.safetensors", "a\\b.safetensors"):
            with self.subTest(name=name), self.assertRaises(adapter.AdapterError):
                adapter._safe_shard_path(self.root, name)
        (self.root / "link.safetensors").symlink_to(self.root.parent / "outside.safetensors")
        with self.assertRaises(adapter.AdapterError):
            adapter._safe_shard_path(self.root, "link.safetensors")

    def test_fake_prefill_and_late_debug_layer_override_rejected(self):
        for key in ("FAKE_P_NODE", "DS_DEBUG_LAYERS"):
            for value in ("1", "0", ""):
                with self.subTest(key=key, value=value), self.assertRaises(adapter.AdapterError):
                    adapter.validate_runtime_environment({key: value})

    def fake_opener(self, selection, *, bad_shape=False, bad_dtype=False):
        owner = self

        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def keys(self):
                return list(owner.index["weight_map"])

            def get_slice(self, name):
                spec = selection.tensor_specs[name]
                return SimpleNamespace(
                    get_shape=lambda: [1] if bad_shape else list(spec.shape),
                    get_dtype=lambda: "INVALID" if bad_dtype else spec.dtypes[0],
                )

            def get_tensor(self, name):
                owner.materialized.append(name)
                return name

        def open_shard(path, **kwargs):
            self.assertEqual(Path(path).name, "first.safetensors")
            self.opened.append(path)
            return Handle()

        self.opened, self.materialized = [], []
        return open_shard

    def test_get_tensor_called_only_for_selected_names(self):
        selection = adapter.inspect_checkpoint(self.root, 1)
        loaded = dict(adapter.iter_selected_weights(selection, self.fake_opener(selection)))
        self.assertEqual(set(loaded), set(selection.selected_names))
        self.assertEqual(self.materialized, list(selection.selected_names))
        self.assertEqual(len(self.opened), 1)

    def test_bad_header_fails_before_payload_materialization(self):
        selection = adapter.inspect_checkpoint(self.root, 1)
        for option in ("bad_shape", "bad_dtype"):
            with self.subTest(option=option):
                opener = self.fake_opener(selection, **{option: True})
                with self.assertRaises(adapter.AdapterError):
                    list(adapter.iter_selected_weights(selection, opener))
                self.assertEqual(self.materialized, [])

    def test_checkpoint_changes_after_inspection_rejected(self):
        selection = adapter.inspect_checkpoint(self.root, 1)
        opener = self.fake_opener(selection)
        self.config["hidden_size"] += 1
        self.write_metadata()
        with self.assertRaisesRegex(adapter.AdapterError, "metadata changed"):
            list(adapter.iter_selected_weights(selection, opener))
        self.assertEqual(self.opened, [])

    def test_upstream_config_normalization_is_allowed_but_semantic_changes_are_not(self):
        selection = adapter.inspect_checkpoint(self.root, 3)
        config = copy.deepcopy(self.config)
        config.update(
            num_hidden_layers=3, model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"]
        )
        config["dtype"] = config.pop("torch_dtype")
        adapter._validate_runtime_config(config, selection)
        config["index_topk"] = 1024
        with self.assertRaisesRegex(adapter.AdapterError, "index_topk"):
            adapter._validate_runtime_config(config, selection)

    def test_wrapper_merges_residual_without_lm_head_or_final_norm(self):
        selection = adapter.inspect_checkpoint(self.root, 3)
        runtime = fake_runtime()
        runtime.upstream.get_global_server_args = lambda: SimpleNamespace(
            load_format=adapter._make_loader_class(runtime, selection)
        )
        model_class = adapter._make_model_class(runtime, selection)
        config = copy.deepcopy(self.config)
        config["num_hidden_layers"] = 3
        hf_config = SimpleNamespace(**config, to_dict=lambda: copy.deepcopy(config))
        model = model_class(hf_config, SimpleNamespace(get_name=lambda: "fp8"))
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
        output = model([1], [0], batch)
        self.assertEqual(output.hidden_states.value, 11)
        self.assertEqual(output.next_token_logits.shape, (1, 0))
        self.assertEqual(output.next_token_logits.numel(), 0)
        self.assertFalse(hasattr(model, "lm_head"))
        self.assertFalse(hasattr(model, "logits_processor"))
        self.assertEqual(model.model.norm(4), 4)
        self.assertEqual(model.load_weights([("real", "weight")]), [("real", "weight")])
        with self.assertRaisesRegex(adapter.AdapterError, "embeddings"):
            model([1], [0], batch, input_embeds=[0])
        batch.forward_mode = SimpleNamespace(is_extend=lambda: False)
        with self.assertRaisesRegex(adapter.AdapterError, "prefill/extend"):
            model([1], [0], batch)
        runtime.upstream.get_global_server_args = lambda: SimpleNamespace(load_format="dummy")
        with self.assertRaisesRegex(adapter.AdapterError, "real prefix-only weight loader"):
            model_class(hf_config, SimpleNamespace(get_name=lambda: "fp8"))

    def test_exact_loader_injected_quant_mapping_is_allowed_without_mutating_config(self):
        selection = adapter.inspect_checkpoint(self.root, 3)
        config = copy.deepcopy(self.config)
        config["num_hidden_layers"] = 3
        config["quantization_config"]["packed_modules_mapping"] = copy.deepcopy(
            adapter.PACKED_MODULES_MAPPING
        )
        before = copy.deepcopy(config)
        adapter._validate_runtime_config(config, selection)
        self.assertEqual(config, before)
        self.assertNotIn("packed_modules_mapping", selection.config["quantization_config"])

    def test_rope_type_alias_and_equal_numeric_casts_are_allowed(self):
        selection = adapter.inspect_checkpoint(self.root, 3)
        config = copy.deepcopy(self.config)
        config["num_hidden_layers"] = 3
        rope = config["rope_scaling"]
        rope["rope_type"] = "yarn"
        for key in ("beta_fast", "beta_slow", "factor"):
            rope[key] = float(rope[key])
        before = copy.deepcopy(config)
        adapter._validate_runtime_config(config, selection)
        self.assertEqual(config, before)
        self.assertNotIn("rope_type", selection.config["rope_scaling"])

    def test_rope_semantic_changes_are_rejected_with_expected_actual(self):
        selection = adapter.inspect_checkpoint(self.root, 3)
        for changed in (
            {"rope_type": "deepseek_yarn"},
            {"type": "linear"},
            {"factor": 20.0},
            {"original_max_position_embeddings": 8192},
            {"unexpected_option": True},
        ):
            with self.subTest(changed=changed):
                config = copy.deepcopy(self.config)
                config["num_hidden_layers"] = 3
                config["rope_scaling"]["rope_type"] = "yarn"
                config["rope_scaling"].update(changed)
                with self.assertRaisesRegex(adapter.AdapterError, "expected .*got"):
                    adapter._validate_runtime_config(config, selection)

    def test_changed_quant_mapping_or_other_quant_parameters_are_rejected(self):
        selection = adapter.inspect_checkpoint(self.root, 3)
        for changed in (
            {"packed_modules_mapping": {}},
            {
                "packed_modules_mapping": {
                    "fused_qkv_a_proj_with_mqa": ["kv_a_proj_with_mqa", "q_a_proj"]
                }
            },
            {"packed_modules_mapping": None},
            {"weight_block_size": [64, 128]},
            {"quant_method": "awq"},
            {"unexpected_quant_option": True},
        ):
            with self.subTest(changed=changed):
                config = copy.deepcopy(self.config)
                config["num_hidden_layers"] = 3
                config["quantization_config"]["packed_modules_mapping"] = copy.deepcopy(
                    adapter.PACKED_MODULES_MAPPING
                )
                config["quantization_config"].update(changed)
                with self.assertRaises(adapter.AdapterError):
                    adapter._validate_runtime_config(config, selection)

    def test_scope_restores_registry_and_sys_path_on_failure(self):
        runtime = fake_runtime()
        original_registry = dict(runtime.registry.models)
        original_recall = (
            runtime.host_pool.NSATokenToKVPoolHost.recall_miss_tokens_extend_cuda_graph
        )
        original_path = list(sys.path)
        with (
            patch.object(
                adapter,
                "verify_echo_checkout",
                return_value=(self.root, adapter.ECHO_REVISION, "patch"),
            ),
            patch.object(adapter, "_import_echo_runtime", return_value=runtime),
            self.assertRaisesRegex(RuntimeError, "caller failed"),
            adapter.scoped_echo_adapter(self.root, self.root, 3) as bindings,
        ):
            self.assertIs(runtime.registry.models[adapter.ARCHITECTURE], bindings.model_class)
            self.assertIs(runtime.registry.models["DeepseekV3ForCausalLM"], bindings.model_class)
            self.assertEqual(bindings.selection.num_layers, 3)
            self.assertTrue(bindings.model_instance_id)
            self.assertEqual(bindings.cache_adaptations, ("gr_extend_recall_free_slots_v1",))
            self.assertEqual(len(bindings.cache_adapter_sha256), 64)
            raise RuntimeError("caller failed")
        self.assertEqual(runtime.registry.models, original_registry)
        self.assertEqual(sys.path, original_path)
        self.assertIs(
            runtime.host_pool.NSATokenToKVPoolHost.recall_miss_tokens_extend_cuda_graph,
            original_recall,
        )


if __name__ == "__main__":
    unittest.main()
