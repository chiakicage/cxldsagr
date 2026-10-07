"""Plot H64K/A128 candidate GPU kernel time from the retained HBM revisit profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.deepseek_v32_motivation.src.analyze_pipeline import analyze_capture

ROOT = Path(__file__).resolve().parents[3]
NAME = "compute_share_h64k_a128"
GROUPS = {
    "Indexer": ("indexer_qk", "exact_topk"),
    "Attention": ("mla_qk_pv",),
    "Projection": (
        "index_q_proj",
        "index_k_proj",
        "index_weights_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "q_absorb",
        "v_expand",
        "o_proj",
        "embedding",
        "lm_head",
    ),
    "MLP": ("mlp_gate", "mlp_up", "mlp_down"),
}
# Only explicitly identified auxiliary kernels are accepted. These signatures
# are checked against the retained projection/rotary implementation snapshots.
AUXILIARY_RULES = {
    "RoPE": (
        "BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel",
        "cos_kernel_cuda",
        "sin_kernel_cuda",
        "arange_cuda_out",
        "gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>",
        "CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)4>",
    ),
    "MLP": ("flashinfer::activation::act_and_mul_kernel",),
    "Norm": ("LocalPlainRMSNorm", "LocalFusedRMSNorm", "vectorized_layer_norm_kernel"),
    "Quantization / scaling": (
        "quantize_kernel",
        "AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>",
        "vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>",
    ),
    "Position / layout / checks": (
        "CUDAFunctor_add<long>",
        "CUDAFunctorOnSelf_add<long>",
        "direct_copy_kernel_cuda",
        "bfloat16_copy_kernel_cuda",
        "CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsigned int)2>",
        "CatArrayBatchedCopy_alignedK_contig",
        "MinMaxOps<long, long, int>",
    ),
}
IMPLEMENTATION_SOURCES = (
    "models/deepseek_v32/projections.py",
    "models/deepseek_v32/rotary.py",
    "models/deepseek_v32/execution/compute_graphs.py",
    "operators/flashinfer.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def identify_index_norm_casts(candidate):
    replays = defaultdict(list)
    for activity in candidate:
        scope = activity["scope"]
        if scope["stage"].startswith("compute_graph_projection_layer_"):
            replays[scope["label"]].append(activity)
    require(len(replays) == 10, "expected ten candidate projection replays")
    for sequence in replays.values():
        sequence.sort(key=lambda activity: activity["start"])
        indices = [
            i
            for i, activity in enumerate(sequence)
            if "vectorized_layer_norm_kernel" in activity["name"]
        ]
        require(len(indices) == 1, "expected one indexer LayerNorm per replay")
        index = indices[0]
        require(2 <= index < len(sequence) - 1, "incomplete indexer normalization chain")
        before, after = sequence[index - 1], sequence[index + 1]
        require(
            sequence[index - 2].get("graph_stage") == "index_k_proj",
            "indexer normalization must immediately follow its K projection",
        )
        require(
            "direct_copy_kernel_cuda" in before["name"] and "lambda(float)" in before["name"],
            "missing indexer FP32 conversion",
        )
        require("bfloat16_copy_kernel_cuda" in after["name"], "missing indexer BF16 conversion")
        before["index_norm_cast"] = after["index_norm_cast"] = True


def classify_kernel(activity, stage, stage_group):
    if stage in stage_group:
        return stage_group[stage], f"API: {stage}"
    matches = [
        (group, signature)
        for group, signatures in AUXILIARY_RULES.items()
        for signature in signatures
        if signature in activity["name"]
    ]
    require(len(matches) == 1, f"unknown or ambiguous auxiliary kernel: {activity['name']}")
    group, signature = matches[0]
    if group in ("RoPE", "Quantization / scaling"):
        require(
            stage.startswith("compute_graph_projection_layer_"),
            "rotary and indexer preprocessing must belong to a projection graph",
        )
    if group == "RoPE":
        if "BatchQKApplyRotary" in signature:
            if "<(bool)0," in activity["name"]:
                owner = "Indexer"
            else:
                require("<(bool)1," in activity["name"], "unknown rotary pairing")
                owner = "Attention"
        else:
            owner = "Projection"
    elif group == "Quantization / scaling":
        owner = "Indexer"
    elif group == "Norm":
        if "vectorized_layer_norm_kernel" in signature:
            require(
                stage.startswith("compute_graph_projection_layer_"), "unexpected index norm scope"
            )
            owner = "Indexer"
        elif stage.startswith("compute_graph_finish_layer_"):
            owner = "MLP"
        else:
            require(
                stage.startswith("compute_graph_projection_layer_")
                or stage == "final_norm_lm_head",
                "unexpected projection norm scope",
            )
            owner = "Projection"
    elif group == "MLP":
        require(stage.startswith("compute_graph_finish_layer_"), "unexpected activation scope")
        owner = "MLP"
    elif activity.get("index_norm_cast"):
        owner = "Indexer"
    else:
        require(group == "Position / layout / checks", "unknown auxiliary operation")
        require(
            stage.startswith("compute_graph_projection_layer_")
            or stage in ("forward", "final_norm_lm_head"),
            "unexpected preparation scope",
        )
        owner = "Projection"
    return owner, f"{group} -> {owner}: {signature}"


def draw(rows, output):
    with plt.rc_context({"font.size": 16, "svg.fonttype": "none", "svg.hashsalt": NAME}):
        figure, axis = plt.subplots(figsize=(5.5, 5.5))
        colors = ("#0072B2", "#D55E00", "#CCD2D7", "#6B7280")
        bottom = 0
        for index, row in enumerate(rows):
            height = row["share_pct"]
            axis.bar(
                0,
                height,
                bottom=bottom,
                width=0.8,
                color=colors[index],
                edgecolor="white",
                linewidth=1.5,
            )
            axis.text(
                0,
                bottom + height / 2,
                f"{row['group']}\n{height:.2f}%",
                ha="center",
                va="center",
                color="#28323C" if index == 2 else "white",
                fontsize=17,
                fontweight="bold",
            )
            bottom += height
        require(abs(bottom - 100) < 1e-10, "stacked bar must total 100 percent")
        combined = sum(row["share_pct"] for row in rows if row["group"] in ("Indexer", "Attention"))
        axis.plot(
            [0.47, 0.55, 0.55, 0.47],
            [0, 0, combined, combined],
            color="#28323C",
            linewidth=1.8,
            clip_on=False,
        )
        axis.text(
            0.88,
            combined / 2,
            f"Attention\n+ Indexer\n{combined:.2f}%",
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#28323C",
        )
        axis.set_xlim(-0.42, 1.18)
        axis.set_ylim(0, 100)
        axis.set_axis_off()
        figure.subplots_adjust(left=0.04, right=0.96, bottom=0.02, top=0.98)
        require(len(figure.axes) == 1, "expected one coordinate plot")
        for suffix in ("png", "svg", "pdf"):
            metadata = (
                {"Date": None}
                if suffix == "svg"
                else ({"CreationDate": None, "ModDate": None} if suffix == "pdf" else {})
            )
            figure.savefig(
                output / f"{NAME}.{suffix}", dpi=240, bbox_inches="tight", metadata=metadata
            )
        plt.close(figure)


def render(pipeline_path, output):
    pipeline_path = Path(pipeline_path).resolve(strict=True)
    output = Path(output).resolve()
    pipeline = json.loads(pipeline_path.read_text())
    captures = [c for c in pipeline["captures"] if (c["scheme"], c["phase"]) == ("hbm", "revisit")]
    require(len(captures) == 1, "expected one retained HBM revisit capture")
    capture = captures[0]
    inputs = {
        str(pipeline_path): digest(pipeline_path),
        capture["sqlite"]: capture["sqlite_sha256"],
        **capture["graph_input_sha256"],
    }
    for path, expected in inputs.items():
        require(digest(path) == expected, f"profile input changed: {path}")
    for path, expected in pipeline["analysis_sources"].items():
        require(digest(ROOT / path) == expected, f"accepted attribution source changed: {path}")
    metadata = json.loads((Path(capture["sqlite"]).parent / "metadata.json").read_text())
    implementation_paths = [
        Path(capture["sqlite"]).parent / "source" / rel for rel in IMPLEMENTATION_SOURCES
    ]
    for relative, path in zip(IMPLEMENTATION_SOURCES, implementation_paths, strict=True):
        require(
            digest(ROOT / relative) == digest(path),
            f"classification implementation differs from the retained profile: {relative}",
        )
        inputs[str(path)] = digest(path)
    config = metadata["config"]
    require(
        (config["history_tokens"], config["candidate_tokens"], config["layers"])
        == (65536, 128, 10),
        "expected H64K/A128 C10 profile",
    )
    selected = [c for c in metadata["captures"] if c["sqlite"] == Path(capture["sqlite"]).name]
    require(len(selected) == 1 and selected[0]["request_id"] == 16, "expected revisit request 16")
    counters = selected[0]["segment_counters"]["candidate"]
    require(
        counters["host_to_device_bytes"] == counters["device_to_host_bytes"] == 0,
        "HBM candidate must have no host KV fetch/writeback",
    )
    summary, _, _, activities = analyze_capture(capture["sqlite"])
    require(summary == capture, "raw capture attribution differs from the accepted pipeline")
    candidate = [a for a in activities if a["scope"] and a["scope"]["segment"] == "candidate"]
    require(
        len(candidate) == capture["segments"]["candidate"]["gpu_activity_count"],
        "candidate activity count mismatch",
    )
    stage_group = {stage: group for group, stages in GROUPS.items() for stage in stages}
    identify_index_norm_casts(candidate)
    totals = defaultdict(int)
    counts = defaultdict(int)
    stage_totals = defaultdict(lambda: [0, 0])
    records = []
    for activity in candidate:
        stage = activity.get("graph_stage", activity["scope"]["stage"])
        kind = activity["kind"]
        require(kind in ("kernel", "memcpy", "memset"), f"unexpected activity kind: {kind}")
        require(
            activity["category"] not in ("host_gather", "indexer_prefetch_fused"),
            "cannot separate fetch from fused kernels",
        )
        group, rule = (
            classify_kernel(activity, stage, stage_group)
            if kind == "kernel"
            else ("Excluded", kind)
        )
        duration = activity["end"] - activity["start"]
        require(duration > 0, "nonpositive GPU duration")
        totals[group] += duration
        counts[group] += 1
        stage_totals[group, stage, kind][0] += 1
        stage_totals[group, stage, kind][1] += duration
        records.append(
            {
                "group": group,
                "stage": stage,
                "kind": kind,
                "start_ns": activity["start"],
                "end_ns": activity["end"],
                "duration_ns": duration,
                "kernel_name": activity["name"],
                "classification_rule": rule,
            }
        )
    observed = {stage for group, stage, kind in stage_totals if kind == "kernel"}
    require(stage_group.keys() <= observed, "missing expected compute stages")
    total = sum(totals[group] for group in GROUPS)
    require(
        sum(counts[g] for g in GROUPS) == capture["segments"]["candidate"]["gpu_kernel_count"],
        "kernel count mismatch",
    )
    rows = [
        {
            "group": group,
            "kernel_count": counts[group],
            "duration_ns": totals[group],
            "duration_ms": totals[group] / 1e6,
            "share_pct": 100 * totals[group] / total,
        }
        for group in GROUPS
    ]
    stage_rows = [
        {
            "group": group,
            "stage": stage,
            "kind": kind,
            "activity_count": values[0],
            "duration_ns": values[1],
            "duration_ms": values[1] / 1e6,
        }
        for (group, stage, kind), values in sorted(stage_totals.items())
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "compute_share_data.csv", rows)
    write_csv(output / "compute_share_stages.csv", stage_rows)
    write_csv(output / "activities.csv", records)
    draw(rows, output)
    sources = [
        Path(__file__),
        *(ROOT / p for p in pipeline["analysis_sources"]),
        ROOT / "experiments/deepseek_v32_motivation/src/graph_instrumentation.py",
        *(ROOT / relative for relative in IMPLEMENTATION_SOURCES),
    ]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(relative)] = digest(source)
    narrow_ns = sum(
        row["duration_ns"]
        for row in records
        if row["kind"] == "kernel" and row["stage"] in ("indexer_qk", "exact_topk", "mla_qk_pv")
    )
    provenance = {
        "schema": "deepseek-c10-candidate-compute-share-v3",
        "run_id": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "profile_run_id": metadata["run_id"],
        "matching_bench_run_id": metadata["reference_run_id"],
        "matching_check_run_id": metadata["numerical_reference_run_id"],
        "scheme": "hbm",
        "phase": "revisit",
        "segment": "candidate",
        "request_id": 16,
        "history_tokens": 65536,
        "candidate_tokens": 128,
        "layers": 10,
        "input_sha256": inputs,
        "source_sha256": source_hashes,
        "source_base": str(output / "source"),
        "artifacts_base": str(output),
        "group_stages": GROUPS,
        "auxiliary_kernel_rules": AUXILIARY_RULES,
        "groups": rows,
        "total_kernel_ns": total,
        "combined_indexer_attention_pct": 100 * (totals["Indexer"] + totals["Attention"]) / total,
        "scores_topk_mla_only_pct": 100 * narrow_ns / total,
        "excluded_activity_count": counts["Excluded"],
        "excluded_duration_ns": totals["Excluded"],
        "attribution_matches_accepted_pipeline": True,
        "denominator": "Sum of all candidate GPU kernel durations; every kernel assigned exactly once.",
        "classification": "Four accounting groups; named API ownership, unique auxiliary signatures and verified index-normalization cast adjacency; unknown kernels rejected.",
        "group_boundaries": {
            "Indexer": "Scores/top-k, indexer rotary application, index LayerNorm with its FP32/BF16 casts, index record quantization and head-weight scaling.",
            "Attention": "Sparse MLA and MLA rotary application.",
            "Projection": "All previously separated input/output projection APIs, including Q absorption, V expansion and O projection; shared position/trig preparation, input/Q/K norm, KV layout, input checks, embedding, final norm and LM head. This accounting group includes request endpoints, not only linear kernels.",
            "MLP": "MLP APIs, SiLU, and the finish graph's post-attention normalization.",
        },
        "layout": "One vertical 100-percent stacked bar; four large in-bar labels plus a bracket and the combined Attention + Indexer share; no title, axes, legend or footnotes.",
        "boundary": (
            "CPU reanalysis of the retained 2026-10-06 instrumented HBM-only profile; no new GPU run. "
            "One revisit candidate over ten C10 blocks, not a repeated-run mean or the new matrix's profile. "
            "History reconstruction, all memcpy/memset and CPU gaps excluded. "
            "Projection API groups include their quantization/reduction kernels. "
            "Auxiliary kernels are folded into four documented accounting groups; the denominator is unchanged. "
            "Kernel duration shares are not FLOP shares, SM utilization or end-to-end latency shares."
        ),
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "artifacts_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "compute_share_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = render(args.pipeline, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "groups": result["groups"],
                "combined_pct": result["combined_indexer_attention_pct"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
