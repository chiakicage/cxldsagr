"""Actual-input observation must preserve calls, tensor layouts and provenance."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from experiments.nosa_mfu.src import capture_inputs
from models.attention_contracts import AttentionContext, BlockSelection


@pytest.mark.parametrize("has_mask", [False, True])
def test_recorder_forwards_original_call_and_saves_independent_strided_tensors(tmp_path, has_mask):
    # The nonzero storage offset and row gaps mimic a Q view into fused QKV.
    q = torch.randn(4, 4608, dtype=torch.bfloat16)[1:, :4096].view(3, 32, 128)
    raw = {
        "keys": torch.randn(67, 2, 128, dtype=q.dtype),
        "values": torch.randn(67, 2, 128, dtype=q.dtype),
        "cis_scores": torch.randn(67, 2, dtype=q.dtype),
    }
    compressed = torch.randn(3, 2, 128, dtype=q.dtype)
    cache = SimpleNamespace(
        layer_view=lambda layer: raw,
        indexer_cache=SimpleNamespace(layer_view=lambda layer: {"compressed_keys": compressed}),
    )
    ids = torch.arange(64).expand(3, 2, 64).clone()
    mask = ids % 2 == 0 if has_mask else None
    selection = BlockSelection(ids, 64, mask)
    context = AttentionContext(0, 64, 3)
    output = q.clone()
    calls = []

    def attention(query, selected, access, execution):
        assert query is q and selected is selection and access is cache and execution is context
        calls.append(execution.layer_idx)
        return output

    originals = {
        "q": q,
        "k": raw["keys"],
        "v": raw["values"],
        "cis": raw["cis_scores"],
        "compressed_k": compressed,
        "ids": ids,
    }
    if has_mask:
        originals["valid_mask"] = mask
    before = {name: tensor.clone() for name, tensor in originals.items()}
    config = SimpleNamespace(num_attention_heads=32, num_key_value_heads=2, head_dim=128)
    recorder = capture_inputs.InputRecorder(attention, config, [0], tmp_path, prefix=64, queries=3)
    assert recorder(q, selection, cache, context) is output
    assert calls == [0]

    path = tmp_path / "layer_00.pt"
    saved = torch.load(path, weights_only=True)
    record = recorder.records[0]
    assert set(saved) == {"q", "k", "v", "cis", "compressed_k", "ids", "valid_mask"}
    assert record["query_start"] == 64 and record["queries"] == 3
    assert record["selection_had_valid_mask"] is has_mask
    assert record["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, original in originals.items():
        assert torch.equal(original, before[name])
        assert torch.equal(saved[name], original)
        assert saved[name].stride() == original.stride()
        assert saved[name].data_ptr() != original.data_ptr()
        provenance = record["tensors"][name]
        assert provenance["shape"] == list(original.shape)
        assert provenance["stride"] == list(original.stride())
        assert provenance["dtype"] == str(original.dtype)
        expected_hash = hashlib.sha256(
            original.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        assert provenance["sha256"] == expected_hash
    if not has_mask:
        assert saved["valid_mask"].dtype == torch.bool and saved["valid_mask"].all()
    assert record["original_storage_offsets"]["q"] == q.storage_offset() > 0
    assert saved["q"].storage_offset() == 0
    flat = saved["q"].as_strided((saved["q"].untyped_storage().nbytes() // 2,), (1,))
    assert not flat[4096:4608].count_nonzero()
    assert not flat[4608 + 4096 : 2 * 4608].count_nonzero()
    saved["q"].zero_()
    assert torch.equal(q, before["q"])


def test_unselected_layer_is_transparent_and_duplicate_visit_fails_before_forward(tmp_path):
    calls = []
    output = object()

    def attention(*args):
        calls.append(args)
        return output

    recorder = capture_inputs.InputRecorder(attention, None, [1], tmp_path, prefix=64, queries=3)
    inputs = (object(), object(), object(), AttentionContext(0, 64, 3))
    assert recorder(*inputs) is output
    assert calls == [inputs]
    assert recorder.visited == [0] and not recorder.records
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="exactly once"):
        recorder(*inputs)
    assert calls == [inputs]


@pytest.mark.parametrize("changed", [False, True])
def test_snapshot_includes_capture_script_and_rejects_changed_source(
    tmp_path, monkeypatch, changed
):
    root = tmp_path / "repo"
    name = "experiments/nosa_mfu/scripts/capture.sh"
    script = root / name
    script.parent.mkdir(parents=True)
    content = b"#!/usr/bin/env bash\nexit 0\n"
    script.write_bytes(content)
    expected = {name: hashlib.sha256(content).hexdigest()}

    def fingerprints(*extras):
        assert script in extras
        if changed:
            script.write_bytes(b"changed during capture\n")
        return expected

    monkeypatch.setattr(capture_inputs, "ROOT", root)
    monkeypatch.setattr(capture_inputs, "source_hashes", fingerprints)
    destination = tmp_path / "snapshot"
    if changed:
        with pytest.raises(RuntimeError, match="Source changed"):
            capture_inputs.snapshot_sources(destination)
        assert not (destination / name).exists()
    else:
        assert capture_inputs.snapshot_sources(destination) == expected
        assert (destination / name).read_bytes() == content
