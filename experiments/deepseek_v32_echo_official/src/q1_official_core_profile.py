"""Observe the unchanged, receipt-bound official core through the frozen profile entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--receipt", type=Path, required=True)
    value.add_argument("--layer", type=int, choices=(0, 1, 2), required=True)
    value.add_argument("--policy", choices=("zero", "warm", "empty"), required=True)
    value.add_argument("--physical-device", type=int, required=True)
    return value


def main():
    args = parser().parse_args()
    # Keep --help CPU-only, and preserve the accepted component's actual entry.
    import torch

    from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_pinned_run as pinned
    from experiments.deepseek_v32_echo_official.src import q1_fused_prepare_run as component

    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(component.EXPERIMENT / "output/data"):
        raise ValueError("Profile data must remain in this experiment's output/data")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="pass-", dir=output_root))
    source = Path(__file__).resolve()
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    (destination / source.name).write_bytes(source_bytes)
    original_profile = component.profile
    original_invoke = component.Case.invoke

    def observe_profile(profile_args):
        latest = []

        def observe_invoke(case, variant):
            result = original_invoke(case, variant)
            # Store references only. No new CUDA work, synchronization or reads
            # occur inside the original ProfilerAPI/NVTX scope.
            latest[:] = [(case, result)]
            return result

        with patch.object(component.Case, "invoke", observe_invoke):
            result = original_profile(profile_args)
        # The original profile has synchronized and stopped the profiler.
        case, (_, state) = latest[0]
        attempts = int(state.counter.item())
        ids = state.host_ids[0].cpu().long()
        staged = ids[ids >= 0]
        count = len(staged)
        component.require(count == min(64, attempts), "Observed stage count disagrees with cap")
        component.require(staged.unique().numel() == count, "Observed staging IDs repeat")
        component.require(bool(ids[count:].eq(-1).all()), "Observed unused slots were written")
        reference = case.reference_scores[0, : case.history].cpu()
        initial_map = case.initial_h2d[: case.history].cpu()
        threshold = case.threshold.cpu()
        eligible = (reference > threshold[0]) & initial_map.eq(component.MISSING)
        component.require(bool((staged < case.history).all()), "Candidate appeared in staging")
        component.require(
            bool(eligible[staged].all()), "Observed staged ID is not an eligible miss"
        )
        component.require(count == min(64, int(eligible.sum())), "Observed success count changed")
        records = state.records[0, :count].cpu()
        expected = case.host[staged]
        component.exact(records, expected, "Observed copied record bytes")
        observation = {
            "layer": args.layer,
            "policy": args.policy,
            "history": case.history,
            "n": case.n,
            "threshold_fp32_bits": int(threshold.view(torch.int32)[0]),
            "initial_resident_history": int(initial_map.ne(component.MISSING).sum()),
            "eligible_history_misses": int(eligible.sum()),
            "attempted_records": attempts,
            "staged_records": count,
            "logical_host_copy_bytes": records.numel() * records.element_size(),
            "staged_host_ids": staged.tolist(),
            "staged_records_sha256": component.tensor_digest(records),
            "readback_boundary": "After original synchronization and cudaProfilerStop",
            "replay_boundary": "Application-visible completion only; internal NCU replay passes can have different legal reservation attempts and selected IDs",
            "observer_source": str(source),
            "observer_sha256": source_sha256,
        }
        component.write(destination / "observation.json", observation)
        result["observation"] = observation
        return result

    # Receipt input identity is a mapping over all three layers. Keep the same
    # set and put the selected layer first, as the frozen profile reads inputs[0].
    layers = [args.layer, *(layer for layer in range(3) if layer != args.layer)]
    command = [
        str(Path(pinned.__file__).resolve()),
        "--mode",
        "profile",
        "--variant",
        "baseline",
        "--policy",
        args.policy,
        "--warmups",
        "1",
        "--physical-device",
        str(args.physical_device),
        "--receipt",
        str(args.receipt.resolve()),
        "--output-dir",
        str(destination / "component"),
        "--inputs",
        *(str(component.raw.INPUT_ROOT / f"kernel_inputs_layer_{layer}.pt") for layer in layers),
    ]
    component.write(
        destination / "invocation.json",
        {
            "argv": command,
            "pid": os.getpid(),
            "observer_sha256": source_sha256,
            "frozen_profile_sha256": component.digest(Path(component.__file__)),
            "baseline_only": True,
        },
    )
    with patch.object(sys, "argv", command), patch.object(component, "profile", observe_profile):
        pinned.main()
    if source.read_bytes() != source_bytes:
        raise RuntimeError("Observer source changed during capture")
    print(json.dumps({"profile_pass": str(destination), "completed": True}))


if __name__ == "__main__":
    main()
