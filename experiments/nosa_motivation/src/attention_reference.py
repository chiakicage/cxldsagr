"""Capture actual resident attention operands, then benchmark independent APIs.

Capture is intrusive and supplies no latency. Raw selected-QK/PV BMM uses
materialized per-query KV and is not an efficient FlashAttention ceiling.
The public FA3 API is timed eagerly; its unsupported graph capture is never used.
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

from experiments.nosa_motivation.src.attention_reference_audit import (
    audit_attention_timings,
    audit_capture,
    descriptor,
    load_payload,
    require,
    selection_geometry,
    summarize_attention_rows,
    tensor_digest,
)
from experiments.nosa_motivation.src.flops import model_dimensions
from experiments.nosa_motivation.src.provenance import digest, write_json


class AttentionCapture:
    def __init__(
        self, backend, directory, config, request_id, identity, mode, history_reference=None
    ):
        self.backend, self.directory = backend, Path(directory)
        require(backend.scheme == "hbm", "attention operand capture requires resident HBM")
        require(mode in ("full_request", "candidate_only"), "invalid capture mode")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "calls").mkdir()
        (self.directory / "layers").mkdir()
        self.history_reference = None
        if history_reference is not None:
            reference = json.loads((Path(history_reference) / "manifest.json").read_text())
            audit_capture(history_reference, reference)
            self.history_reference = {
                (row["layer"], row["query_start"]): row for row in reference["calls"]
            }
        self.manifest = {
            "schema": "nosa-attention-capture-v1",
            "model_config": model_dimensions(backend.config),
            "config": config,
            "request_id": request_id,
            "identity": identity,
            "mode": mode,
            "calls": [],
            "layer_records": {},
            "request_output_exact": False,
            "request_output_sha256": None,
            "accepted_output_sha256": None,
            "history_reference_sha256": None
            if history_reference is None
            else digest(Path(history_reference) / "manifest.json"),
            "reused_history_calls": [],
            "boundary": "Separate intrusive accepted-HBM replay. Q/selection are saved per invocation. Actual visible KV/CIS hashes from every invocation must match slices of each layer's final H+A snapshot. Attention output hashes and final request hidden are verified. Capture wall time is not a performance result.",
        }

    def accept_output(self, actual, expected):
        import torch

        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), atol=0, rtol=0)
        self.manifest["request_output_exact"] = True
        self.manifest["request_output_sha256"] = tensor_digest(actual.detach().cpu())
        self.manifest["accepted_output_sha256"] = tensor_digest(expected.detach().cpu())

    def observe(self, q, selection, cache, context, output):
        import torch

        require(
            cache._execution_resources is self.backend.resources,
            "capture observed a foreign backend",
        )
        history = self.manifest["config"]["history_tokens"]
        if self.manifest["mode"] == "candidate_only" and context.query_start != history:
            if self.history_reference is not None:
                reference = self.history_reference[(context.layer_idx, context.query_start)]
                ids = selection.block_ids.detach().cpu()
                valid = (
                    ids >= 0
                    if selection.valid_mask is None
                    else selection.valid_mask.detach().cpu()
                )
                hashes = {
                    "q": tensor_digest(q.detach().cpu()),
                    "ids": tensor_digest(ids),
                    "valid": tensor_digest(valid),
                }
                require(
                    hashes == reference["operand_hashes"]
                    and tensor_digest(output.detach().cpu()) == reference["output_sha256"],
                    "revisit history attention operands/output differ from first capture",
                )
                self.manifest["reused_history_calls"].append(
                    {
                        "layer": context.layer_idx,
                        "query_start": context.query_start,
                        "queries": len(q),
                        "operand_hashes": hashes,
                        "output_sha256": reference["output_sha256"],
                    }
                )
            return
        layer, start, queries = context.layer_idx, context.query_start, len(q)
        views = cache.layer_view(layer)
        records = {
            name: views[source].detach().cpu().contiguous()
            for name, source in (("keys", "keys"), ("values", "values"), ("cis", "cis_scores"))
        }
        require(
            len(records["keys"]) == start + queries, "capture requires the exact visible append"
        )
        ids = selection.block_ids.detach().cpu().contiguous()
        valid = (
            ids >= 0
            if selection.valid_mask is None
            else selection.valid_mask.detach().cpu().contiguous()
        )
        cfg = self.manifest["model_config"]
        geometry = selection_geometry(
            ids,
            valid,
            start,
            queries,
            cfg["num_key_value_heads"],
            cfg["num_attention_heads"],
            cfg["head_dim"],
            start + queries,
        )
        relative = f"calls/{len(self.manifest['calls']):05d}.pt"
        payload = {"q": q.detach().cpu().contiguous(), "ids": ids, "valid": valid}
        torch.save(payload, self.directory / relative)
        row = {
            "layer": layer,
            "query_start": start,
            "queries": queries,
            "length": start + queries,
            "file": relative,
            "sha256": digest(self.directory / relative),
            "layouts": {
                "q": descriptor(q),
                "ids": descriptor(selection.block_ids),
                "valid": descriptor(
                    valid if selection.valid_mask is None else selection.valid_mask
                ),
                **{
                    name: descriptor(views[source])
                    for name, source in (
                        ("keys", "keys"),
                        ("values", "values"),
                        ("cis", "cis_scores"),
                    )
                },
            },
            "visible_hashes": {name: tensor_digest(value) for name, value in records.items()},
            "operand_hashes": {name: tensor_digest(value) for name, value in payload.items()},
            "output_sha256": tensor_digest(output.detach().cpu()),
            "geometry": geometry,
        }
        self.manifest["calls"].append(row)
        if start == history:
            relative = f"layers/{layer:03d}.pt"
            torch.save(records, self.directory / relative)
            self.manifest["layer_records"][str(layer)] = {
                "file": relative,
                "sha256": digest(self.directory / relative),
            }

    def finish(self):
        self.manifest["audit"] = audit_capture(self.directory, self.manifest)
        write_json(self.directory / "manifest.json", self.manifest)


@contextmanager
def capture_attention_inputs(
    backend, directory, config, request_id, identity, *, mode="full_request", history_reference=None
):
    """Use only in a separate replay, and call capture.accept_output inside it."""
    from models.nosa.fixed_cache import NosaFixedAttention

    capture = AttentionCapture(
        backend, directory, config, request_id, identity, mode, history_reference
    )
    original = NosaFixedAttention.__call__

    def observed(attention, q, selection, cache, context):
        output = original(attention, q, selection, cache, context)
        capture.observe(q, selection, cache, context, output)
        return output

    with patch.object(NosaFixedAttention, "__call__", observed):
        yield capture
    capture.finish()


def restore_layout(value, layout, device):
    import torch

    require(
        list(value.shape) == layout["shape"] and str(value.dtype) == layout["dtype"],
        "saved tensor layout metadata differs",
    )
    restored = torch.empty_strided(
        tuple(layout["shape"]), tuple(layout["stride"]), dtype=value.dtype, device=device
    )
    restored.copy_(value)
    return restored


def materialize_selected(q, keys, values, ids, valid, cis, query_start):
    """Untimed gather preserving per-query causal membership and GQA sharing."""
    import torch

    queries, query_heads, dim = q.shape
    heads = keys.shape[1]
    groups = query_heads // heads
    slots = int(valid.sum(-1).max().item()) * 64
    order = (~valid).to(torch.int32).argsort(dim=-1, stable=True)
    compact_ids, compact_valid = ids.gather(-1, order), valid.gather(-1, order)
    tokens = (compact_ids[..., None] * 64 + torch.arange(64, device=q.device)).reshape(
        queries * heads, -1
    )[:, :slots]
    membership = (
        compact_valid[..., None].expand(-1, -1, -1, 64).reshape(queries * heads, -1)[:, :slots]
    )
    positions = torch.arange(query_start, query_start + queries, device=q.device).repeat_interleave(
        heads
    )[:, None]
    allowed = membership & (tokens >= 0) & (tokens < len(keys)) & (tokens <= positions)
    safe = tokens.clamp(0, len(keys) - 1)
    head = torch.arange(heads, device=q.device).repeat(queries)[:, None]
    selected_k = keys[safe, head].contiguous()
    selected_v = values[safe, head].contiguous()
    bias = cis[safe, head].float()
    return (
        q.reshape(queries * heads, groups, dim).contiguous(),
        selected_k.transpose(1, 2),
        selected_v,
        bias,
        allowed,
    )


def _samples(function, *, graph):
    import torch

    for _ in range(3):
        function()
    torch.cuda.synchronize()
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            function()
        stream.synchronize()
        operation = captured.replay
    else:
        operation = function
    values = {"cuda_ms": [], "wall_ms": []}
    for _ in range(7):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        operation()
        end.record()
        end.synchronize()
        values["cuda_ms"].append(start.elapsed_time(end))
        values["wall_ms"].append((time.perf_counter() - begin) * 1000)
    return {
        **{f"samples_{name}": items for name, items in values.items()},
        **{f"median_{name}": statistics.median(items) for name, items in values.items()},
    }


def _validate_products(query, key, value, scores, probabilities, pv):
    import torch

    rows = sorted({0, len(query) // 2, len(query) - 1})
    expected_qk = torch.bmm(query[rows].float(), key[rows].float())
    expected_pv = torch.bmm(probabilities[rows].float(), value[rows].float()).to(pv.dtype)
    torch.testing.assert_close(scores[rows], expected_qk, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(pv[rows], expected_pv, atol=0.008, rtol=0.008)
    return {
        "passed": True,
        "batch_rows_checked": rows,
        "qk_atol": 2e-4,
        "qk_rtol": 2e-4,
        "pv_atol": 0.008,
        "pv_rtol": 0.008,
        "qk_max_abs": float((scores[rows] - expected_qk).abs().max()),
        "pv_max_abs": float((pv[rows].float() - expected_pv.float()).abs().max()),
        "qk_max_tolerance_ratio": float(
            ((scores[rows] - expected_qk).abs() / (2e-4 + 2e-4 * expected_qk.abs())).max()
        ),
        "pv_max_tolerance_ratio": float(
            (
                (pv[rows].float() - expected_pv.float()).abs()
                / (0.008 + 0.008 * expected_pv.float().abs())
            ).max()
        ),
        "boundary": "Deterministic first/middle/last batch rows compared with IEEE FP32 products. All selection membership and useful-work counts are audited separately. Raw products do not assert complete attention/model equivalence.",
    }


def benchmark_attention_reference(capture_dir, output, *, device="cuda:0"):
    """Run after every serving backend closes; fixed three warmups/seven samples."""
    import torch

    from layers.attention import BlockSelection
    from operators.nosa.attention.device_only.api import nosa_block_sparse_attention
    from operators.nosa.attention.workspace import NosaAttentionWorkspace

    capture_dir, output = Path(capture_dir), Path(output)
    require(not output.exists(), "attention benchmark output already exists")
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    audit_capture(capture_dir, manifest)
    manifest["manifest_sha256"] = digest(capture_dir / "manifest.json")
    cfg = manifest["model_config"]
    device = torch.device(device)
    require(
        device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0),
        "independent attention reference requires Hopper",
    )
    require(
        cfg["head_dim"] == 128 and cfg["num_attention_heads"] == 16 * cfg["num_key_value_heads"],
        "independent attention reference requires D128/GQA16",
    )
    require(
        torch.get_float32_matmul_precision() == "highest"
        and not torch.backends.cuda.matmul.allow_tf32,
        "raw product validation requires IEEE FP32 settings",
    )
    workspace = NosaAttentionWorkspace(
        max(row["queries"] for row in manifest["calls"]),
        cfg["num_key_value_heads"],
        device=device,
        dtype=torch.bfloat16,
    )
    completed = {}
    with torch.inference_mode(), torch.cuda.device(device):
        for layer in range(cfg["num_hidden_layers"]):
            final = load_payload(capture_dir, manifest["layer_records"][str(layer)])
            calls = [row for row in manifest["calls"] if row["layer"] == layer]
            last = calls[-1]
            resident = {
                name: restore_layout(final[name], last["layouts"][name], device)
                for name in ("keys", "values", "cis")
            }
            for capture in calls:
                payload = load_payload(capture_dir, capture)
                q = restore_layout(payload["q"], capture["layouts"]["q"], device)
                ids, valid = (
                    restore_layout(payload[name], capture["layouts"][name], device)
                    for name in ("ids", "valid")
                )
                selection = BlockSelection(ids, 64, valid)
                keys, values, cis = (
                    resident[name][: capture["length"]] for name in ("keys", "values", "cis")
                )
                for name, value in (("keys", keys), ("values", values), ("cis", cis)):
                    require(
                        descriptor(value) == capture["layouts"][name],
                        "independent replay changed operand stride",
                    )

                fa3 = partial(
                    nosa_block_sparse_attention,
                    q,
                    keys,
                    values,
                    selection,
                    capture["query_start"],
                    cis,
                    workspace=workspace,
                )

                actual = fa3()
                require(
                    tensor_digest(actual.cpu()) == capture["output_sha256"],
                    "independent FA3 output differs from captured attention",
                )
                fa3_timing = _samples(fa3, graph=False)
                query, key, value, bias, allowed = materialize_selected(
                    q, keys, values, ids, valid, cis, capture["query_start"]
                )
                scores = torch.bmm(query, key, out_dtype=torch.float32)
                probabilities = (
                    (scores * (cfg["head_dim"] ** -0.5) + bias[:, None])
                    .masked_fill(~allowed[:, None], -torch.inf)
                    .softmax(-1)
                    .to(torch.bfloat16)
                )
                pv = torch.bmm(probabilities, value)
                validation = _validate_products(query, key, value, scores, probabilities, pv)
                qk_timing = _samples(
                    partial(torch.bmm, query, key, out_dtype=torch.float32, out=scores), graph=True
                )
                pv_timing = _samples(partial(torch.bmm, probabilities, value, out=pv), graph=True)
                completed[(layer, capture["query_start"])] = {
                    "layer": layer,
                    "query_start": capture["query_start"],
                    "queries": capture["queries"],
                    "geometry": capture["geometry"],
                    "fa3_output_exact": True,
                    "fa3_output_sha256": tensor_digest(actual.cpu()),
                    "raw_product_validation": validation,
                    "qk_bmm": qk_timing,
                    "pv_bmm": pv_timing,
                    "fa3_eager": fa3_timing,
                    "raw_layouts": {
                        name: descriptor(tensor)
                        for name, tensor in (
                            ("q", query),
                            ("k", key),
                            ("v", value),
                            ("qk", scores),
                            ("p", probabilities),
                            ("pv", pv),
                        )
                    },
                }
                del query, key, value, bias, allowed, scores, probabilities, pv, actual
            del resident, final
    result = {
        "schema": "nosa-attention-reference-v1",
        "capture_sha256": manifest["manifest_sha256"],
        "warmup": 3,
        "repeats": 7,
        "rows": [completed[(row["layer"], row["query_start"])] for row in manifest["calls"]],
        "boundary": "Independent calls on actual captured operands. Public resident FA3 is eager and includes validation, prepare/repair and submission gaps; no FA3 graph capture. Raw QK/PV are separate graph-submitted BF16-input BMMs (QK output FP32, PV output BF16). Gather, CIS/causal masking and FP32 softmax with BF16 probability rounding are outside raw timing. Materialized selected KV duplicates records per query while sharing them across GQA16; padding and duplication are explicit. Buffers are reused within each invocation's warmups/repeats; no artificial cache flush. Each layer's original resident KV is loaded once and a bounded per-invocation expanded arena is discarded before the next call. A sum of API medians is not an observed model request. Beating the materialized reference alone cannot satisfy an efficiency gate.",
    }
    result["summary"] = summarize_attention_rows(manifest, result["rows"])
    audit_attention_timings(manifest, result)
    write_json(output, result)
    return result
