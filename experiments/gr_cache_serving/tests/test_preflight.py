"""CPU-only coverage of metadata validation and payload accounting."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments.gr_cache_serving.src import preflight


def small_config() -> dict:
    return {
        "model_type": "deepseek_v32",
        "num_hidden_layers": 4,
        "first_k_dense_replace": 3,
        "hidden_size": 4,
        "vocab_size": 8,
        "q_lora_rank": 2,
        "kv_lora_rank": 2,
        "num_attention_heads": 2,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 2,
        "v_head_dim": 2,
        "intermediate_size": 8,
        "index_n_heads": 2,
        "index_head_dim": 128,
        "max_position_embeddings": 163840,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 2]},
    }


def write_safetensors(path: Path, tensors: dict, *, metadata: dict | None = None) -> int:
    header = {"__metadata__": metadata or {"format": "pt"}, **tensors}
    raw_header = json.dumps(header).encode("utf-8")
    payload_size = max((value["data_offsets"][1] for value in tensors.values()), default=0)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        handle.write(bytes(payload_size))
    return len(raw_header)


def write_checkpoint(root: Path, layers: int = 3) -> tuple[dict, dict]:
    config = small_config()
    specs = preflight.expected_tensors(config, layers)
    tensors, offset = {}, 0
    for name, spec in specs.items():
        if name.endswith("weight_scale_inv") or "norm." in name or "layernorm." in name:
            dtype = "F32"
        elif "F8_E4M3" in spec["dtypes"]:
            dtype = "F8_E4M3"
        else:
            dtype = "BF16"
        numel = 1
        for size in spec["shape"]:
            numel *= size
        size = numel * preflight.DTYPE_BYTES[dtype]
        tensors[name] = {
            "dtype": dtype,
            "shape": spec["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    write_safetensors(root / "model.safetensors", tensors)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    index = {
        "metadata": {"total_size": offset},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return config, tensors


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_header_only_reads_with_structured_metadata(self):
        path = self.root / "weights.safetensors"
        tensors = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        header_bytes = write_safetensors(path, tensors)
        with (
            path.open("rb") as handle,
            patch.object(handle, "read", wraps=handle.read) as reads,
            patch.object(Path, "open", return_value=handle),
        ):
            header = preflight.read_safetensors_header(path)
            self.assertEqual([call.args for call in reads.call_args_list], [(8,), (header_bytes,)])
        self.assertEqual(header["payload_bytes_read"], 0)
        self.assertEqual(header["metadata"], {"format": "pt"})
        self.assertEqual(header["tensors"]["x"]["stored_bytes"], 8)

    def test_invalid_or_truncated_header_length(self):
        path = self.root / "bad.safetensors"
        for raw in (
            b"short",
            struct.pack("<Q", 100) + b"{}",
            struct.pack("<Q", preflight.MAX_HEADER_BYTES + 1),
        ):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(preflight.AuditError):
                    preflight.read_safetensors_header(path)

    def test_header_duplicate_keys_rejected(self):
        path = self.root / "bad.safetensors"
        raw = b'{"x":{},"x":{}}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(preflight.AuditError, "duplicate"):
            preflight.read_safetensors_header(path)

    def test_invalid_tensor_metadata(self):
        path = self.root / "bad.safetensors"
        invalid_entries = [
            {"dtype": "BF16", "shape": [2], "data_offsets": [0, 3]},
            {"dtype": "BF16", "shape": [-1], "data_offsets": [0, 0]},
            {"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]},
            {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 1]},
            {"dtype": "BF16", "shape": [1], "data_offsets": [-2, 0]},
        ]
        for entry in invalid_entries:
            with self.subTest(entry=entry):
                write_safetensors(path, {"x": entry})
                with self.assertRaises(preflight.AuditError):
                    preflight.read_safetensors_header(path)

    def test_overlapping_offsets_rejected(self):
        path = self.root / "bad.safetensors"
        tensor = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        write_safetensors(path, {"x": tensor, "y": tensor})
        with self.assertRaisesRegex(preflight.AuditError, "overlapping"):
            preflight.read_safetensors_header(path)

    def test_payload_truncation_rejected_without_reading_payload(self):
        path = self.root / "bad.safetensors"
        header_bytes = write_safetensors(
            path, {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        )
        with path.open("r+b") as handle:
            handle.truncate(8 + header_bytes + 7)
        with self.assertRaisesRegex(preflight.AuditError, "offsets"):
            preflight.read_safetensors_header(path)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.tensors = write_checkpoint(self.root)

    def test_audit_counts_full_embedding_and_retains_f32_parameters(self):
        result = preflight.audit_checkpoint(self.root, 1)
        self.assertEqual(result["selected_tensor_count"], 28)
        self.assertEqual(len(result["shards"]), 1)
        self.assertEqual(result["payload_bytes_read"], 0)
        for name in ("config.json", "model.safetensors.index.json"):
            self.assertEqual(
                result["metadata_sha256"][name],
                hashlib.sha256((self.root / name).read_bytes()).hexdigest(),
            )
        tensors = result["tensors"]
        self.assertEqual(tensors["model.embed_tokens.weight"]["stored_bytes"], 64)
        self.assertEqual(
            tensors["model.layers.0.input_layernorm.weight"]["dequantized_bf16_bytes"], 16
        )
        self.assertEqual(
            tensors["model.layers.0.self_attn.q_a_proj.weight"]["dequantized_bf16_bytes"], 16
        )
        self.assertEqual(
            tensors["model.layers.0.self_attn.q_a_proj.weight_scale_inv"]["dequantized_bf16_bytes"],
            0,
        )
        self.assertIn("model.layers.0.self_attn.indexer.k_norm.bias", tensors)
        self.assertFalse(any(name.startswith("model.layers.1.") for name in tensors))
        self.assertEqual(
            result["selected_stored_bytes"], sum(t["stored_bytes"] for t in tensors.values())
        )

    def test_all_three_layers_selected(self):
        result = preflight.audit_checkpoint(self.root, 3)
        self.assertEqual(result["selected_tensor_count"], 82)
        self.assertIn("model.layers.2.mlp.down_proj.weight", result["tensors"])

    def test_missing_scale_or_indexer_parameter_fails(self):
        path = self.root / "model.safetensors.index.json"
        original = json.loads(path.read_text())
        for name in (
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
            "model.layers.0.self_attn.indexer.k_norm.bias",
        ):
            with self.subTest(name=name):
                index = json.loads(json.dumps(original))
                del index["weight_map"][name]
                path.write_text(json.dumps(index))
                with self.assertRaisesRegex(preflight.AuditError, "missing required"):
                    preflight.audit_checkpoint(self.root, 1)

    def test_index_and_header_mismatch_fails(self):
        name = "model.layers.0.self_attn.q_a_proj.weight"
        self.tensors[name]["shape"] = [1, 8]
        write_safetensors(self.root / "model.safetensors", self.tensors)
        with self.assertRaisesRegex(preflight.AuditError, "unexpected shape/dtype"):
            preflight.audit_checkpoint(self.root, 1)

    def test_shard_traversal_and_symlink_escape_rejected(self):
        for name in ("../escape.safetensors", "/tmp/escape.safetensors", "a\\b.safetensors"):
            with self.subTest(name=name), self.assertRaises(preflight.AuditError):
                preflight._shard_path(self.root, name)
        (self.root / "link.safetensors").symlink_to(self.root.parent / "escape.safetensors")
        with self.assertRaisesRegex(preflight.AuditError, "escapes"):
            preflight._shard_path(self.root, "link.safetensors")

    def test_unsupported_layer_selection(self):
        for layers in (0, 4, -1, True):
            with self.subTest(layers=layers), self.assertRaises(preflight.AuditError):
                preflight.expected_tensors(self.config, layers)
        self.config["first_k_dense_replace"] = 1
        with self.assertRaisesRegex(preflight.AuditError, "moe_intermediate_size"):
            preflight.expected_tensors(self.config, 2)

    def test_moe_header_spec_agrees_with_runtime_adapter(self):
        from models.deepseek_v32.echo_adapter import _tensor_specs
        from models.deepseek_v32.tests.test_echo_adapter import fixture_config

        config = fixture_config()
        specs = preflight.expected_tensors(config, 5)
        runtime = _tensor_specs(config, 5)
        self.assertEqual(set(specs), set(runtime))
        for name, spec in specs.items():
            self.assertEqual(tuple(spec["shape"]), runtime[name].shape)
            self.assertTrue(set(runtime[name].dtypes) <= set(spec["dtypes"]))

    def test_capacity_uses_bf16_mla_and_explicit_index_payload(self):
        config = {
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "max_position_embeddings": 163840,
        }
        result = preflight.cache_capacity(config, 3, 65536, 1024, 128)
        self.assertEqual(result["mla_bytes_per_token_per_layer"], 1152)
        self.assertEqual(result["index_bytes_per_token_per_layer"], 132)
        self.assertEqual(result["all_users_prefix_mla_bytes"], 28991029248)
        self.assertEqual(result["all_users_prefix_index_bytes"], 3321888768)
        self.assertIsNone(result["metadata_bytes"])
        self.assertIsNone(result["workspace_bytes"])
        with self.assertRaisesRegex(preflight.AuditError, "max_position_embeddings"):
            preflight.cache_capacity(config, 3, 163840, 1, 128)
        config["index_head_dim"] = 256
        with self.assertRaisesRegex(preflight.AuditError, "index_head_dim=128"):
            preflight.cache_capacity(config, 3, 65536, 1024, 128)

    def test_cli_refuses_existing_output_before_audit(self):
        output = self.root / "existing.json"
        output.write_text("preserve this audit", encoding="utf-8")
        with (
            patch.object(preflight, "audit_checkpoint") as audit,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            preflight.main(["--model-path", str(self.root), "--output", str(output)])
        self.assertEqual(caught.exception.code, 2)
        audit.assert_not_called()
        self.assertEqual(output.read_text(), "preserve this audit")

    def test_echo_tracks_dirty_source_separately_from_revision(self):
        (self.root / ".git").mkdir()
        command_results = [
            {"status": "ok", "stdout": "fixture-revision"},
            {"status": "ok", "stdout": " M source.py"},
        ]
        with patch.object(preflight, "probe_command", side_effect=command_results):
            result = preflight.probe_echo(self.root)
        self.assertEqual(result["git_revision"]["stdout"], "fixture-revision")
        self.assertTrue(result["tracked_dirty"])
        self.assertEqual(result["compiled_status"], "unverified")

    def test_cli_records_feasibility_and_unverified_statuses(self):
        output = self.root / "output.json"
        error_probe = {"status": "error", "error": "fixture: no CUDA"}
        with patch.object(preflight, "probe_command", return_value=error_probe):
            code = preflight.main(
                [
                    "--model-path",
                    str(self.root),
                    "--layers",
                    "2",
                    "--echo-path",
                    str(self.root / "missing"),
                    "--output",
                    str(output),
                ]
            )
        report = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(report["kind"], "feasibility")
        self.assertFalse(report["is_benchmark"])
        self.assertEqual(report["compiled_status"], "unverified")
        self.assertEqual(report["numerical_status"], "unverified")
        self.assertEqual(report["hardware_probes"]["cuda_compiler"]["status"], "error")

    def test_cli_invalid_checkpoint_fails_with_json(self):
        with (
            patch.object(preflight, "probe_command", return_value={"status": "error"}),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = preflight.main(
                [
                    "--model-path",
                    str(self.root / "missing"),
                    "--echo-path",
                    str(self.root / "missing"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["checkpoint"]["status"], "failed")

    def test_probe_command_reports_missing_binary(self):
        result = preflight.probe_command([str(self.root / "missing-program")])
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
