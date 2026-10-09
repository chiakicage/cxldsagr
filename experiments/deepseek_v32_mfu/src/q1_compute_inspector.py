"""Run the complete formal pipeline without its initial compute-graph inspector."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

from experiments.deepseek_v32_mfu.src import gap_profile, profile_layers


def main():
    source = Path(__file__).resolve()
    relative = str(source.relative_to(profile_layers.ROOT))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original_prepare = profile_layers.prepare_graphs
    original_sources = gap_profile.measurement_sources
    original_profile = gap_profile.run_profile
    prepared = []

    def prepare(model, args, mode):
        if mode != "profile" or not args.nsys or not args.compute_graphs or prepared:
            raise ValueError("Control requires one profiled compute-graph preparation")
        with gap_profile.profiler_capture():
            result = original_prepare(model, args, "check")
        if result is not None:
            raise RuntimeError("Uninspected compute preparation produced an inspector")
        prepared.append(True)

    def sources():
        return {**original_sources(), relative: source_hash}

    def profile(model, ids, args, result, receipt, *, trace_warmups):
        if prepared != [True] or hasattr(model, "_profile_graph_capture"):
            raise RuntimeError("Unexpected compute-graph inspection state")
        original_profile(model, ids, args, result, receipt, trace_warmups=trace_warmups)
        result["diagnostic_control"] = {
            "kind": "initial-compute-graph-inspector-omitted-v1",
            "measurement_source": relative,
            "measurement_sha256": source_hash,
            "compute_preparation_calls": len(prepared),
            "all_four_method_prelude": "unchanged profile_layers.run",
            "measured_forward": "unchanged gap_profile.run_profile",
            "changed_step": "prepare_graphs uses check mode inside the same profiler range",
            "prefill_operator_attribution_available": False,
            "formal_gap_gate_available": False,
            "boundary": "Initial CaptureGraphOperators/InstrumentOperators omitted; later full-extend native inspection unchanged. Independent correctness receipt binds unchanged numerical execution. This diagnostic profile does not replace clean timing or the formal matrix.",
        }
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Diagnostic measurement source changed during execution")

    with (
        patch.object(profile_layers, "prepare_graphs", prepare),
        patch.object(gap_profile, "measurement_sources", sources),
        patch.object(gap_profile, "run_profile", profile),
    ):
        gap_profile.main()


if __name__ == "__main__":
    main()
