"""CPU audits for independently replayed NOSA attention operands and APIs."""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

from experiments.nosa_motivation.src.flops import model_dimensions, observe_selection
from experiments.nosa_motivation.src.matrix_baseline import invocation_geometry
from experiments.nosa_motivation.src.provenance import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_digest(value):
    import torch

    require(value.device.type == "cpu", "tensor hashing requires an explicit CPU copy")
    data = value.contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def descriptor(value):
    return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}


def evidence_path(directory, relative):
    relative = Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts, "unsafe attention evidence path"
    )
    return Path(directory) / relative


def load_payload(directory, entry):
    import torch

    path = evidence_path(directory, entry["file"])
    require(digest(path) == entry["sha256"], "attention operand file changed")
    return torch.load(path, map_location="cpu", weights_only=True)


def audit_layout(layout, shape, dtypes, name):
    require(
        layout["shape"] == list(shape)
        and layout["dtype"] in dtypes
        and len(layout["stride"]) == len(shape)
        and all(type(s) is int and s > 0 for s in layout["stride"]),
        f"captured {name} tensor contract differs",
    )
    extent = 1
    for size, stride in sorted(zip(shape, layout["stride"], strict=True), key=lambda x: x[1]):
        if size > 1:
            require(stride >= extent, f"captured {name} strides overlap")
            extent += (size - 1) * stride
    if name in ("q", "keys", "values"):
        require(layout["stride"][-1] == 1, f"captured {name} innermost stride differs")
        require(
            all(stride * 2 % 16 == 0 for stride in layout["stride"][:-1]),
            f"captured {name} row/head strides are not TMA aligned",
        )


def audit_tensor(value, shape, dtypes, name):
    require(
        value.device.type == "cpu"
        and list(value.shape) == list(shape)
        and str(value.dtype) in dtypes,
        f"saved {name} tensor contract differs",
    )


def selection_geometry(ids, valid, query_start, queries, kv_heads, query_heads, dimension, length):
    """Validate exact causal work and count unique logical KV without expansion."""
    import torch

    from layers.attention import BlockSelection

    require(ids.device.type == "cpu" and valid.device.type == "cpu", "selection audit runs on CPU")
    require(
        ids.dtype in (torch.int32, torch.int64) and valid.dtype == torch.bool,
        "attention selection dtype differs",
    )
    require(tuple(ids.shape) == (queries, kv_heads, 64), "attention selection geometry differs")
    observed = observe_selection(
        BlockSelection(ids, 64, valid), query_start=query_start, query_heads=query_heads
    )
    groups = query_heads // kv_heads
    slots = int(valid.sum(-1).max()) * 64
    blocks = (length + 63) // 64
    heads = torch.arange(kv_heads)[None, :, None].expand_as(ids)
    positions = torch.arange(query_start, query_start + queries)[:, None, None].expand_as(ids)
    maxima = torch.full((kv_heads * blocks,), -1, dtype=torch.int64)
    maxima.scatter_reduce_(
        0, (heads * blocks + ids)[valid].long(), positions[valid], reduce="amax", include_self=True
    )
    starts = torch.arange(blocks).repeat(kv_heads) * 64
    unique_tokens = int((maxima - starts + 1).clamp(0, 64).sum())
    causal_pairs = observed["causal_pairs_all_query_heads"]
    expanded_tokens = queries * kv_heads * slots
    return {
        "queries": queries,
        "kv_heads": kv_heads,
        "query_heads": query_heads,
        "head_dim": dimension,
        "selected_slots_per_query_head_group": slots,
        "causal_pairs_all_query_heads": causal_pairs,
        "useful_attention_flops": 4 * dimension * causal_pairs,
        "dense_executed_attention_flops": 4 * dimension * queries * query_heads * slots,
        "unique_selected_kv_tokens": unique_tokens,
        "materialized_kv_tokens": expanded_tokens,
        "materialized_bf16_kv_bytes": expanded_tokens * dimension * 2 * 2,
        "unique_bf16_kv_bytes": unique_tokens * dimension * 2 * 2,
        "padded_kv_duplication_ratio": expanded_tokens / unique_tokens,
        "useful_kv_duplication_ratio": (causal_pairs // groups) / unique_tokens,
    }


def audit_capture(directory, manifest, *, verify_files=True):
    """Check every call and its immutable-prefix relation to final layer KV."""
    cfg, contract = model_dimensions(manifest["model_config"]), manifest["config"]
    require(manifest["schema"] == "nosa-attention-capture-v1", "unknown attention capture schema")
    require(
        manifest["mode"] in ("full_request", "candidate_only"), "invalid attention capture mode"
    )
    require(
        manifest["identity"] and manifest["request_id"] >= 0, "missing attention capture identity"
    )
    require(manifest["request_output_exact"], "capture request has not matched its accepted output")
    require(
        len(manifest["request_output_sha256"]) == 64
        and manifest["request_output_sha256"] == manifest["accepted_output_sha256"],
        "captured request output identity differs",
    )
    geometry = invocation_geometry(
        contract["history_tokens"], contract["candidate_tokens"], contract["chunk_size"]
    )
    if manifest["mode"] == "candidate_only":
        if manifest["history_reference_sha256"] is not None:
            require(
                [
                    (row["layer"], row["query_start"], row["queries"])
                    for row in manifest["reused_history_calls"]
                ]
                == [
                    (layer, start, queries)
                    for start, queries in geometry[:-1]
                    for layer in range(cfg["num_hidden_layers"])
                ],
                "incomplete repeated-history operand validation",
            )
        else:
            require(not manifest["reused_history_calls"], "unexpected repeated-history evidence")
        geometry = geometry[-1:]
    else:
        require(
            manifest["history_reference_sha256"] is None and not manifest["reused_history_calls"],
            "full capture cannot reuse another history",
        )
    expected = [
        (layer, start, queries)
        for start, queries in geometry
        for layer in range(cfg["num_hidden_layers"])
    ]
    actual = [(row["layer"], row["query_start"], row["queries"]) for row in manifest["calls"]]
    require(actual == expected, "attention capture must cover every layer and invocation in order")
    require(
        set(manifest["layer_records"]) == {str(i) for i in range(cfg["num_hidden_layers"])},
        "missing final layer KV",
    )
    by_layer = {}
    for row in manifest["calls"]:
        by_layer.setdefault(row["layer"], []).append(row)
    useful = padded = 0
    for layer, calls in by_layer.items():
        final = (
            load_payload(directory, manifest["layer_records"][str(layer)]) if verify_files else None
        )
        final_length = contract["history_tokens"] + contract["candidate_tokens"]
        if verify_files:
            require(set(final) == {"keys", "values", "cis"}, "final KV payload fields differ")
            for name in ("keys", "values", "cis"):
                shape = [final_length, cfg["num_key_value_heads"]]
                if name != "cis":
                    shape.append(cfg["head_dim"])
                audit_tensor(final[name], shape, {"torch.bfloat16"}, name)
        for row in calls:
            require(
                row["length"] == row["query_start"] + row["queries"],
                "attention visible length differs",
            )
            shapes = {
                "q": [row["queries"], cfg["num_attention_heads"], cfg["head_dim"]],
                "keys": [row["length"], cfg["num_key_value_heads"], cfg["head_dim"]],
                "values": [row["length"], cfg["num_key_value_heads"], cfg["head_dim"]],
                "cis": [row["length"], cfg["num_key_value_heads"]],
                "ids": [row["queries"], cfg["num_key_value_heads"], 64],
                "valid": [row["queries"], cfg["num_key_value_heads"], 64],
            }
            dtypes = {
                name: {"torch.int32", "torch.int64"}
                if name == "ids"
                else {"torch.bool"}
                if name == "valid"
                else {"torch.bfloat16"}
                for name in shapes
            }
            require(set(row["layouts"]) == set(shapes), "captured layout fields differ")
            for name, shape in shapes.items():
                audit_layout(row["layouts"][name], shape, dtypes[name], name)
            if verify_files:
                payload = load_payload(directory, row)
                require(set(payload) == {"q", "ids", "valid"}, "call payload fields differ")
                for name, value in payload.items():
                    audit_tensor(value, shapes[name], dtypes[name], name)
                require(
                    {name: tensor_digest(value) for name, value in payload.items()}
                    == row["operand_hashes"],
                    "captured operand hash differs",
                )
                for name in ("keys", "values", "cis"):
                    view = final[name][: row["length"]]
                    require(
                        tensor_digest(view) == row["visible_hashes"][name],
                        "captured history is not an immutable slice of final KV/CIS",
                    )
                    require(
                        list(view.shape) == row["layouts"][name]["shape"],
                        "captured KV shape differs",
                    )
                require(
                    list(payload["q"].shape) == row["layouts"]["q"]["shape"],
                    "captured Q shape differs",
                )
                geometry_row = selection_geometry(
                    payload["ids"],
                    payload["valid"],
                    row["query_start"],
                    row["queries"],
                    cfg["num_key_value_heads"],
                    cfg["num_attention_heads"],
                    cfg["head_dim"],
                    row["length"],
                )
                require(
                    geometry_row == row["geometry"],
                    "captured attention work differs from selections",
                )
            require(len(row["output_sha256"]) == 64, "missing captured attention output hash")
            useful += row["geometry"]["useful_attention_flops"]
            padded += row["geometry"]["dense_executed_attention_flops"]
    return {
        "calls": len(expected),
        "useful_attention_flops": useful,
        "dense_executed_attention_flops": padded,
        "immutable_prefixes_verified": verify_files,
    }


def audit_history_reuse(first_dir, first, revisit_dir, revisit):
    """Reopen both captures before reusing first-request prefix API timings."""
    require(
        first["mode"] == "full_request"
        and revisit["mode"] == "candidate_only"
        and first["config"] == revisit["config"]
        and first["model_config"] == revisit["model_config"]
        and revisit["history_reference_sha256"] == digest(Path(first_dir) / "manifest.json"),
        "repeated-history reference identity differs",
    )
    history = first["config"]["history_tokens"]
    expected = [
        {
            name: row[name]
            for name in ("layer", "query_start", "queries", "operand_hashes", "output_sha256")
        }
        for row in first["calls"]
        if row["query_start"] < history
    ]
    require(revisit["reused_history_calls"] == expected, "repeated-history operand proof differs")
    for layer, entry in first["layer_records"].items():
        a = load_payload(first_dir, entry)
        b = load_payload(revisit_dir, revisit["layer_records"][layer])
        for field in ("keys", "values", "cis"):
            require(
                tensor_digest(a[field][:history]) == tensor_digest(b[field][:history]),
                "repeated-history final KV/CIS differs",
            )
    return {
        "history_calls_reused": len(expected),
        "final_prefix_layers_equal": len(first["layer_records"]),
    }


def summarize_attention_rows(manifest, rows):
    summaries = {}
    for name, selected in (
        ("captured_workload", rows),
        (
            "candidate_only",
            [row for row in rows if row["query_start"] == manifest["config"]["history_tokens"]],
        ),
    ):
        require(selected, "missing candidate attention timings")
        useful = sum(row["geometry"]["useful_attention_flops"] for row in selected)
        summaries[name] = {
            "useful_attention_flops": useful,
            "dense_executed_attention_flops": sum(
                row["geometry"]["dense_executed_attention_flops"] for row in selected
            ),
            "raw_qk_pv_cuda_ms": sum(
                row[op]["median_cuda_ms"] for row in selected for op in ("qk_bmm", "pv_bmm")
            ),
            "independent_fa3_cuda_ms": sum(row["fa3_eager"]["median_cuda_ms"] for row in selected),
            "independent_fa3_wall_ms": sum(row["fa3_eager"]["median_wall_ms"] for row in selected),
            "materialized_bf16_kv_bytes": sum(
                row["geometry"]["materialized_bf16_kv_bytes"] for row in selected
            ),
            "unique_bf16_kv_bytes_per_invocation_sum": sum(
                row["geometry"]["unique_bf16_kv_bytes"] for row in selected
            ),
        }
    return summaries


def audit_attention_timings(manifest, benchmark):
    """Recompute fixed-repeat statistics and complete useful-work coverage."""
    require(
        benchmark["schema"] == "nosa-attention-reference-v1", "unknown attention benchmark schema"
    )
    require(
        benchmark["capture_sha256"] == manifest["manifest_sha256"],
        "attention timing capture identity differs",
    )
    require(
        benchmark["warmup"] == 3 and benchmark["repeats"] == 7,
        "attention reference sampling policy changed",
    )
    rows = benchmark["rows"]
    require(
        [(r["layer"], r["query_start"], r["queries"]) for r in rows]
        == [(r["layer"], r["query_start"], r["queries"]) for r in manifest["calls"]],
        "attention timing coverage differs",
    )
    for row, capture in zip(rows, manifest["calls"], strict=True):
        require(
            row["geometry"] == capture["geometry"]
            and row["fa3_output_exact"]
            and row["fa3_output_sha256"] == capture["output_sha256"],
            "attention geometry or independent FA3 output differs",
        )
        validation = row["raw_product_validation"]
        batch = row["queries"] * row["geometry"]["kv_heads"]
        require(
            validation["passed"]
            and validation["batch_rows_checked"] == sorted({0, batch // 2, batch - 1})
            and validation["qk_atol"] == validation["qk_rtol"] == 2e-4
            and validation["pv_atol"] == validation["pv_rtol"] == 0.008
            and all(
                type(validation[f"{op}_max_tolerance_ratio"]) in (int, float)
                and math.isfinite(validation[f"{op}_max_tolerance_ratio"])
                and 0 <= validation[f"{op}_max_tolerance_ratio"] <= 1
                and math.isfinite(validation[f"{op}_max_abs"])
                and validation[f"{op}_max_abs"] >= 0
                for op in ("qk", "pv")
            ),
            "raw attention matrix validation failed",
        )
        dim = row["geometry"]["head_dim"]
        groups = row["geometry"]["query_heads"] // row["geometry"]["kv_heads"]
        slots = row["geometry"]["selected_slots_per_query_head_group"]
        for name, shape in {
            "q": [batch, groups, dim],
            "k": [batch, dim, slots],
            "v": [batch, slots, dim],
            "qk": [batch, groups, slots],
            "p": [batch, groups, slots],
            "pv": [batch, groups, dim],
        }.items():
            audit_layout(
                row["raw_layouts"][name],
                shape,
                {"torch.float32" if name == "qk" else "torch.bfloat16"},
                f"raw {name}",
            )
        for name in ("qk_bmm", "pv_bmm", "fa3_eager"):
            entry = row[name]
            for clock in ("cuda_ms", "wall_ms"):
                values = entry[f"samples_{clock}"]
                require(
                    len(values) == 7
                    and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
                    "invalid attention API samples",
                )
                require(
                    math.isclose(
                        entry[f"median_{clock}"], statistics.median(values), rel_tol=1e-12
                    ),
                    "attention API median differs",
                )
    summaries = summarize_attention_rows(manifest, rows)
    require(summaries == benchmark["summary"], "attention reference totals differ")
    return summaries
