"""CPU reread of private model acceptance, retained scores and graph accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.validation import identity_digest
from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_model as analysis
from experiments.deepseek_v32_mfu.src import prefetch_transition_audit as transitions
from experiments.deepseek_v32_mfu.src.extend_graph_validation import compare_cache_state

require = analysis.require
ARMS, TOKENS = analysis.ARMS, analysis.TOKENS
GPU_UUID = "a2226185-cb05-a411-80da-f365154128fe"


def exact(left, right, label):
    require(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.device.type == right.device.type == "cpu"
        and left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        ),
        "Saved tensor bits differ: " + label,
    )


def outputs(left, right):
    require(set(left) == set(right), "Saved output fields differ")
    for field in left:
        require(
            bool(torch.isfinite(left[field]).all()) and bool(torch.isfinite(right[field]).all()),
            "Nonfinite saved model output",
        )
        exact(left[field], right[field], "output " + field)


def offsets(left, right):
    require(len(left) == len(right) == 3, "Missing layer offsets")
    for index, (actual, expected) in enumerate(zip(left, right, strict=True)):
        require(actual.shape == expected.shape == (16,), "Incomplete offset vector")
        exact(actual, expected, "all offset bits in layer " + str(index))


def exact_radix_topk(scores, indices):
    """Sort all finite FP32 keys, preserving radix signed-zero and ID tie rules."""
    require(scores.shape == (1, 65537) and scores.dtype == torch.float32, "Wrong score ABI")
    require(indices.shape == (1, 2048) and indices.dtype == torch.int32, "Wrong top-k ABI")
    require(bool(torch.isfinite(scores).all()), "Nonfinite causal score")
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = torch.where((bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
    expected = torch.argsort(keys, dim=1, descending=True, stable=True)[:, :2048].to(torch.int32)
    exact(indices, expected, "complete radix top-k ordering")


def memory_audit(memory):
    require(set(memory) == set(ARMS), "Missing memory arm")
    rows = {}
    for arm in ARMS:
        row, graph = memory[arm], memory[arm]["graph"]
        allocation = graph["memory_at_allocation"]
        for mapping, fields in (
            (row, ("allocated", "reserved", "device_used")),
            (
                graph,
                (
                    "private_reserved_bytes",
                    "chosen_private_limit_bytes",
                    "static_allocated_bytes",
                    "reservation_bytes",
                ),
            ),
            (allocation, ("pytorch_allocated", "pytorch_reserved", "device_used", "device_total")),
        ):
            require(
                all(type(mapping[key]) is int and mapping[key] >= 0 for key in fields),
                "Invalid memory counter",
            )
        require(
            row["allocated"] <= row["reserved"] <= row["device_used"] <= allocation["device_total"]
            and allocation["pytorch_allocated"]
            <= allocation["pytorch_reserved"]
            <= allocation["device_used"]
            <= allocation["device_total"],
            "Memory counter hierarchy differs",
        )
        require(
            0 < graph["private_reserved_bytes"] <= graph["chosen_private_limit_bytes"]
            and graph["reservation_bytes"]
            == graph["chosen_private_limit_bytes"] + graph["static_allocated_bytes"],
            "Graph private reservation accounting differs",
        )
        rows[arm] = {
            "private_reserved_bytes": graph["private_reserved_bytes"],
            "chosen_private_limit_bytes": graph["chosen_private_limit_bytes"],
            "static_allocated_bytes": graph["static_allocated_bytes"],
            "reservation_bytes": graph["reservation_bytes"],
            "allocated": row["allocated"],
            "reserved": row["reserved"],
            "device_used": row["device_used"],
        }
    return {
        "arms": rows,
        "candidate_minus_baseline_private_reserved_bytes": rows["candidate"][
            "private_reserved_bytes"
        ]
        - rows["baseline"]["private_reserved_bytes"],
        "boundary": "Saved graph.describe counters and reservation arithmetic are verified. Raw allocator segments are not retained, so private segment membership is not independently reconstructed. Process snapshots include sequentially prepared models and are not isolated model footprints or continuous peaks.",
    }


def compare_records(left, right):
    outputs(left["output"], right["output"])
    offsets(left["offsets"], right["offsets"])
    require(len(left["observed"]) == len(right["observed"]) == 3, "Missing observed layers")
    for a, b in zip(left["observed"], right["observed"], strict=True):
        require(set(a) == set(b) == {"scores", "indices", "initial_hint"}, "Observed fields differ")
        for key in a:
            exact(a[key], b[key], "observed " + key)
    return compare_cache_state(
        left["cache"],
        right["cache"],
        actual_prefetch=left["proof"],
        expected_prefetch=right["proof"],
    )


def audit_saved_tensors(directory, check, receipt, evidence):
    payload = torch.load(directory / "outputs.pt", map_location="cpu", weights_only=True)
    require(set(payload) == {"prefix", "diagnostic", "clean"}, "Saved output inventory differs")
    for field in payload.values():
        require(set(field) == set(ARMS), "Saved model arms differ")
    exact(payload["prefix"]["baseline"], payload["prefix"]["candidate"], "independent prefixes")
    proof_rows = []
    for arm in ARMS:
        diagnostic, clean = payload["diagnostic"][arm], payload["clean"][arm]
        require(set(diagnostic) == set(clean) == set(map(str, TOKENS)), "Token coverage differs")
        for token in map(str, TOKENS):
            records = diagnostic[token]
            require(set(records) == {"eager", "graph", "comparison"}, "Execution coverage differs")
            for mode in ("eager", "graph"):
                record = records[mode]
                name = f"echo_{token}_{mode}_prefetch"
                metadata_path = directory / arm / (name + "_receipt.json")
                tensor_path = directory / arm / (name + "_evidence.pt")
                for path in (metadata_path, tensor_path):
                    require(
                        str(path.relative_to(directory)) in receipt["artifacts"], "Unbound proof"
                    )
                accepted = evidence.read(metadata_path)
                require(accepted == record["proof"], "Saved record lost proof binding")
                evidence.file(tensor_path, accepted["evidence_sha256"])
                compact = torch.load(tensor_path, map_location="cpu", weights_only=True)
                require(
                    transitions.validate_execution(compact) == accepted["proof"],
                    "State proof differs",
                )
                require(
                    accepted["final_cache_state_sha256"] == identity_digest(record["cache"]),
                    "Final cache identity differs",
                )
                require(
                    len(record["observed"]) == len(compact["layers"]) == 3, "Missing proof layer"
                )
                raw_layers = []
                for layer, observed in zip(compact["layers"], record["observed"], strict=True):
                    index = layer["layer"]
                    require("scores" not in layer, "Expected compact evidence")
                    require(
                        observed["initial_hint"].shape == (16,) and layer["hint_index"] == 1,
                        "Expected full offset snapshot and decode threshold index",
                    )
                    threshold = observed["initial_hint"][1:2]
                    for key, identity_key in (
                        ("scores", "scores"),
                        ("indices", "indices"),
                        ("initial_hint", "initial_hints"),
                    ):
                        require(
                            transitions.tensor_identity(
                                threshold if key == "initial_hint" else observed[key]
                            )
                            == accepted[identity_key][index],
                            "Retained raw tensor identity differs",
                        )
                    exact(observed["indices"], layer["indices"], "compact selection")
                    exact(threshold, layer["initial_hint"], "compact initial hint")
                    exact_radix_topk(observed["scores"], observed["indices"])
                    raw_layer = {**layer, "scores": observed["scores"]}
                    eligible, metadata = transitions._raw_eligibility(raw_layer)
                    exact(eligible, layer["eligible_mask"], "strict eligibility bitmap")
                    require(metadata == layer["eligibility"], "Raw eligibility metadata differs")
                    raw_layers.append(raw_layer)
                require(
                    transitions.validate_execution({**compact, "layers": raw_layers})
                    == accepted["proof"],
                    "Proof reconstructed with raw scores differs",
                )
                proof_rows.append({"arm": arm, "token": int(token), "mode": mode, "layers": 3})
            require(
                compare_records(records["eager"], records["graph"]) == records["comparison"],
                "Eager/graph comparison differs",
            )
            reference = records["eager"]
            outputs(clean[token]["output"], reference["output"])
            offsets(clean[token]["offsets"], reference["offsets"])
            for actual, expected in zip(
                clean[token]["cache"]["layers"], reference["cache"]["layers"], strict=True
            ):
                require(actual["map_invariants_passed"] is True, "Clean graph map check failed")
                for key in (
                    "length",
                    "written",
                    "indexer_visible_end",
                    "records",
                    "index_keys",
                    "index_scales",
                    "hint",
                    "clock",
                ):
                    require(
                        actual[key] == expected[key], "Clean graph cache identity differs: " + key
                    )
    for token in map(str, TOKENS):
        for mode in ("eager", "graph"):
            actual = compare_records(
                payload["diagnostic"]["baseline"][token][mode],
                payload["diagnostic"]["candidate"][token][mode],
            )
            require(
                actual == check["result"]["cross_arm"][token][mode], "Cross-arm comparison differs"
            )
        outputs(
            payload["clean"]["baseline"][token]["output"],
            payload["clean"]["candidate"][token]["output"],
        )
        offsets(
            payload["clean"]["baseline"][token]["offsets"],
            payload["clean"]["candidate"][token]["offsets"],
        )
    return {
        "executions": proof_rows,
        "raw_layer_scores_and_exact_topk_recomputed": 36,
        "compact_transition_proofs_recomputed": 12,
        "prefix_eager_diagnostic_clean_outputs_and_all_offsets_bitwise": True,
        "boundary": "All retained raw scores, strict eligibility bitmaps, radix score/ID top-k order, maps, priorities, free bits, clocks and counters are reread. Full staged/host/final KV payloads are omitted from compact evidence; their exact byte comparisons remain signed runtime evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new review output")
    torch.set_num_threads(2)
    evidence = analysis.Evidence()
    receipt_path = args.receipt.resolve()
    raw = evidence.read(receipt_path)
    identity = raw["identity"]
    analysis.audit_contract(identity)
    require(identity["source"]["gpu"]["uuid"] == GPU_UUID, "Unexpected physical GPU UUID")
    receipt = evidence.receipt(receipt_path, kind=analysis.KIND, identity=identity)
    check = analysis.audit_run(receipt_path.parent, "check", identity, receipt, evidence)
    require(receipt["checks"].get("passed") is True, "Failed numerical receipt")
    analysis.audit_component(identity["source"]["component"], identity["timing_runtime"], evidence)
    for relative, digest in identity["source"]["sources"].items():
        evidence.file(analysis.ROOT / analysis.safe_relative(relative), digest)
    analysis.verify_runtime_files(identity["timing_runtime"], evidence)
    tensors = audit_saved_tensors(receipt_path.parent, check, receipt, evidence)
    memory = memory_audit(check["memory"])
    source_paths = {
        Path(__file__).resolve(),
        Path(transitions.__file__).resolve(),
        *[analysis.ROOT / name for name in analysis.SOURCE_CLOSURE],
    }
    sources = {str(path): evidence.file(path) for path in sorted(source_paths)}
    evidence.verify()
    result = {
        "schema": "private-fused-prepare-model-independent-check-review-v1",
        "passed": True,
        "receipt_signature": receipt["receipt_sha256"],
        "identity_sha256": identity_digest(identity),
        "gpu": identity["source"]["gpu"],
        "memory": memory,
        "saved_tensors": tensors,
        "analysis_sources": sources,
        "files": evidence.files,
        "boundary": "Independent CPU artifact and saved-tensor audit. No GPU execution, timing, weight payload rehash or allocator segment reconstruction.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("passed", "receipt_signature", "gpu", "memory", "saved_tensors")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
