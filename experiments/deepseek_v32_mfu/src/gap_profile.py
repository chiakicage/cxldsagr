"""Independent node-level gap capture with only native model NVTX scopes.

This reuses the checked model and shared experiment lifecycle. It intentionally
does not generate eager operator MFU: the full operator profile owns that task.
--trace-warmups controls matched warmups inside each capture (default 1); other options
match profile_layers. Native graph nodes remain visible in every capture.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.deepseek_v32_mfu.src import profile_layers as common
from experiments.deepseek_v32_mfu.src.run_contract import METHODS
from models.deepseek_v32.nonmatrix import rms_norm


class ModelScopes:
    """Record plain stage labels; expand the graph ledger after capture ends."""

    def __init__(self, mode, phase):
        self.mode, self.phase, self.layer = mode, phase, "shared"
        self.records = []

    @contextmanager
    def __call__(self, stage):
        if re.fullmatch(r"layer_\d+", stage):
            previous, self.layer = self.layer, stage
            torch.cuda.nvtx.range_push(f"echo/{self.mode}/{self.phase}/{stage}")
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
                self.layer = previous
            return
        # These model scopes include the entire corresponding mathematical API.
        # Fused indexer/cache helpers are still classified by actual node name.
        stage = {"indexer": "indexer_qk", "indexer_prefetch": "indexer_fused"}.get(stage, stage)
        call_id = len(self.records)
        label = f"echo/{self.mode}/{self.phase}/{self.layer}/{stage}/call_{call_id}"
        self.records.append((call_id, label, self.layer, stage))
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def calls(self, graph_capture, full_graph_capture=None):
        calls = [
            {
                "call_id": call_id,
                "nvtx": label,
                "mode": self.mode,
                "phase": self.phase,
                "layer": layer,
                "stage": stage,
                "useful_flops": None,
                "precision": None,
                "flop_reason": "minimal model scope, no eager matrix ledger",
            }
            for call_id, label, layer, stage in self.records
        ]
        if graph_capture is not None:
            expanded = []
            for call in calls:
                expanded.extend(graph_capture.expand_replay(call, len(calls) + len(expanded)))
            calls.extend(expanded)
        if full_graph_capture is not None:
            expanded = []
            for call in calls:
                expanded.extend(full_graph_capture.expand_replay(call, len(calls) + len(expanded)))
            calls.extend(expanded)
        return calls


@contextmanager
def profiler_capture():
    runtime = torch.cuda.cudart()
    runtime.cudaProfilerStart()
    try:
        yield
    except BaseException as original:
        try:
            runtime.cudaProfilerStop()
        except BaseException as cleanup:  # noqa: BLE001 -- preserve original and cleanup failure.
            raise BaseExceptionGroup(
                "profile execution and stop both failed", [original, cleanup]
            ) from None
        raise
    else:
        runtime.cudaProfilerStop()


@torch.inference_mode()
def annotate(model, ids, mode, phase, *, prepare, trace_warmups):
    scopes = ModelScopes(mode, phase)
    block_outputs = []
    record_outputs = False
    bank = getattr(model, "_compute_graphs", None)
    if bank is not None:
        original = bank.forward_block

        def remember(layer, *args, **kwargs):
            output = original(layer, *args, **kwargs)
            if record_outputs and layer == model.num_layers - 1 and phase == "extend_annotated":
                block_outputs.append(output)
            return output

        hook = patch.object(bank, "forward_block", remember)
    else:
        original = model.blocks[-1].forward

        def remember(*args, **kwargs):
            output = original(*args, **kwargs)
            if record_outputs and phase == "extend_annotated":
                block_outputs.append(output)
            return output

        hook = patch.object(model.blocks[-1], "forward", remember)
    model.synchronize()
    with hook, profiler_capture():
        for _ in range(trace_warmups):
            prepare()
            warmup_scopes = ModelScopes(mode, phase.removesuffix("_annotated") + "_trace_warmup")
            with warmup_scopes("forward_misc"):
                model.forward(ids, scope=warmup_scopes)
        # The profiler stays active while restoring the same empty/cold/warm
        # cache. These operations finish before the measured forward begins.
        prepare()
        model.synchronize()
        record_outputs = True
        start = time.perf_counter()
        with scopes("forward_misc"):
            output = model.forward(ids, scope=scopes)
        wall = (time.perf_counter() - start) * 1000
    hidden = None
    full_graph = getattr(model, "_extend_graph", None)
    if full_graph is not None and phase == "extend_annotated":
        block_outputs = [(full_graph.last_hidden, full_graph.last_residual)]
    if block_outputs:
        # Independent output diagnostics occur after the captured forward.
        hidden = torch.cat(
            [
                rms_norm(
                    h.float() + residual.float(), model.final_norm, model.cfg.norm_eps
                ).bfloat16()
                for h, residual in block_outputs
            ]
        ).cpu()
    return (
        output.cpu(),
        hidden,
        scopes.calls(
            getattr(model, "_profile_graph_capture", None),
            getattr(model, "_profile_full_graph_capture", None),
        ),
        wall,
    )


def measurement_sources():
    paths = [Path(__file__), Path(__file__).parents[1] / "scripts/gap_profile.sh"]
    return {
        str(path.resolve().relative_to(common.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


@torch.inference_mode()
def run_profile(model, ids, args, result, receipt, *, trace_warmups):
    if not args.nsys:
        raise ValueError("minimal node profiling requires --nsys and cuda graph node tracing")
    manifest = measurement_sources()
    result["measurement_identity"] = {
        "kind": "deepseek-four-method-minimal-node-scopes-v2-in-capture-warmup",
        "source_sha256": manifest,
        "trace_warmups_per_capture": trace_warmups,
        "measured_captures_per_phase_method": 1,
        "cuda_graph_trace": "node; verified by complete native graph node lineage",
        "operator_wrappers_during_forward": False,
        "warmup_boundary": "profiler active throughout warmup, complete cache restoration, synchronization and measured forward; warmup/setup excluded by explicit nonoverlapping NVTX boundaries",
        "boundary": "same checked default model forward, native model scopes only, complete synchronization and commit included",
    }
    for relative in manifest:
        target = args.output / "measurement_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.ROOT / relative, target)
    common.write_json(args.output / "measurement_sources.json", manifest)
    result["profile_detail"] = "minimal_node_model_scopes"
    result["nsys_capture_order"] = ["graph_setup"] if args.compute_graphs else []
    result["correctness"].update(receipt["checks"]["comparisons"])
    calls = []
    for method in METHODS:
        control = torch.load(
            receipt["artifact_paths"][method + "_control.pt"], map_location="cpu", weights_only=True
        )
        prefix_control = torch.load(
            receipt["artifact_paths"][method + "_prefix_logits.pt"],
            map_location="cpu",
            weights_only=True,
        )
        prefix, _, records, prefix_wall = annotate(
            model,
            ids[: args.prefix],
            method,
            "prefill_annotated",
            prepare=lambda method=method: common.select_cache_method(model, method),
            trace_warmups=trace_warmups,
        )
        result["nsys_capture_order"].append(f"{method}/prefill_annotated")
        calls.extend(records)
        result["correctness"][method + "_profile_prefix_logits"] = common.comparison(
            prefix, prefix_control
        )
        prefix_metrics = [block.cache.metrics() for block in model.blocks]
        snapshot = model.snapshot_prefix()
        common.restore_extend_prefix(model, snapshot, args)
        common.profile_extend_graph(model, ids[args.prefix :], args, result, method)
        output, hidden, records, extend_wall = annotate(
            model,
            ids[args.prefix :],
            method,
            "extend_annotated",
            prepare=lambda snapshot=snapshot: common.restore_extend_prefix(model, snapshot, args),
            trace_warmups=trace_warmups,
        )
        result["nsys_capture_order"].append(f"{method}/extend_annotated")
        calls.extend(records)
        for key, actual in (("hidden", hidden), ("logits", output)):
            result["correctness"][method + "_profile_extend_" + key] = common.comparison(
                actual, control[key]
            )
        torch.save(
            {"hidden": hidden, "logits": output}, args.output / f"{method}_profile_output.pt"
        )
        result["measurements"][method] = {
            **common.cache_metrics(model, snapshot, prefix_metrics),
            "annotated_prefix_wall_ms": prefix_wall,
            "annotated_extend_wall_ms": extend_wall,
        }
        common.write_json(
            args.output / "operator_calls.json",
            {"schema_version": 3, "run_id": args.run_id, "calls": calls},
        )
        del snapshot
    if measurement_sources() != manifest:
        raise RuntimeError("minimal profile measurement source changed during execution")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--trace-warmups", type=int, default=1)
    options, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.trace_warmups < 0:
        parser.error("--trace-warmups must be nonnegative")
    _, checked_args = common.parse_run_args("profile", forwarded)
    if checked_args.extend_chunk_size != checked_args.extend:
        parser.error("minimal node gap profiling requires one complete extend chunk")

    def runner(model, ids, args, result, receipt):
        run_profile(model, ids, args, result, receipt, trace_warmups=options.trace_warmups)

    common.run("profile", forwarded, profile_runner=runner)


if __name__ == "__main__":
    main()
