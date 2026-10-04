"""Capture unique-page provenance and actual device work in a separate replay."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from experiments.nosa_motivation.src.provenance import digest


def expected_misses(ids, valid, prefix, tags, owner, head_dim, element_size):
    """Independent CPU page union, excluding complete pages owned by this session."""
    if ids.device.type != "cpu" or valid.device.type != "cpu" or tags.device.type != "cpu":
        raise ValueError("miss audit must use CPU copies outside the measured request")
    result = []
    for head in range(ids.shape[1]):
        blocks = ids[:, head][valid[:, head]]
        for block in sorted(set(blocks.tolist())):
            if block < 0 or block * 64 >= prefix:
                continue
            tokens = min(64, prefix - block * 64)
            if tokens == 64 and int(tags[block, head]) == owner:
                continue
            result.append(
                {"row": block * ids.shape[1] + head, "bytes": tokens * head_dim * element_size * 2}
            )
    return sorted(result, key=lambda row: row["row"])


@contextmanager
def capture_candidate_work(backend, config, records, *, evidence_dir, label):
    """Copy each layer's instrumentation before reuse; this replay provides no latency."""
    import torch

    from experiments.nosa_motivation.src.flops import observe_selection
    from experiments.nosa_offload_overlap.src.analyze import (
        stripe_interval_metrics,
        work_interval_metrics,
    )
    from operators.nosa.attention.offload._fused import build_info

    workspace = backend.resources.fetch_workspace
    if workspace is None:
        raise ValueError("internal work capture requires a sparse offload workspace")
    original = workspace.run
    stripes = build_info()["fetch_stripes"]

    def run(q, selection, host_k, host_v, suffix_k, suffix_v, cis, start, **kwargs):
        if start != config["history_tokens"] or len(q) != config["candidate_tokens"]:
            return original(q, selection, host_k, host_v, suffix_k, suffix_v, cis, start, **kwargs)
        tags, owner = kwargs.get("cache_tags"), kwargs.get("cache_owner")
        if tags is None or owner is None:
            raise ValueError("fixed-pool work audit requires pre-call page ownership tags")
        ids = selection.block_ids.detach().cpu()
        valid = (
            torch.ones_like(ids, dtype=torch.bool)
            if selection.valid_mask is None
            else selection.valid_mask.detach().cpu()
        )
        evidence = {
            "block_ids": ids,
            "valid_mask": valid,
            "tags": tags.detach().cpu(),
            "owner": owner,
            "head_dim": q.shape[-1],
            "element_size": q.element_size(),
            "query_heads": q.shape[1],
        }
        expected = expected_misses(
            ids, valid, start, evidence["tags"], owner, q.shape[-1], q.element_size()
        )
        geometry = {
            "prefix": start,
            "queries": len(q),
            "kv_heads": workspace.kv_heads,
            "expected_fetch_rows": expected,
            "expected_prefix_bytes": sum(row["bytes"] for row in expected),
        }
        workspace.profile_work_intervals = True
        try:
            evidence_path = evidence_dir / f"{label}_layer{len(records):02d}_selection.pt"
            torch.save(evidence, evidence_path)
            output = original(
                q, selection, host_k, host_v, suffix_k, suffix_v, cis, start, **kwargs
            )
            backend.synchronize()
            record = {
                "layer": len(records),
                "mode": "overlap" if backend.scheme == "overlap" else "serialized",
                "recorded_transfer_bytes": int(workspace.last_transfer_bytes.item()),
                "intervals": workspace.work_intervals(),
                "stripe_intervals": workspace.stripe_work_intervals(),
                "geometry": geometry,
                "fetch_stripes": stripes,
                "selection_file": evidence_path.name,
                "selection_sha256": digest(evidence_path),
                "selection_observation": observe_selection(
                    selection, query_start=start, query_heads=q.shape[1]
                ),
            }
            record["page_metrics"] = work_interval_metrics(geometry, record)
            if expected:
                record["stripe_metrics"] = stripe_interval_metrics(geometry, record, stripes)
            else:
                if record["stripe_intervals"] or any(
                    row["kind"] == 1 for row in record["intervals"]
                ):
                    raise ValueError("resident history emitted unexpected host-copy intervals")
                record["stripe_metrics"] = None
            if backend.scheme == "overlap" and expected:
                record["page_and_stripe_90pct"] = (
                    record["page_metrics"]["fetch_math_overlap_fraction"] >= 0.9
                    and record["stripe_metrics"]["fetch_stripe_math_overlap_fraction"] >= 0.9
                )
            else:
                record["page_and_stripe_90pct"] = None
            records.append(record)
            return output
        finally:
            workspace.profile_work_intervals = False

    with patch.object(workspace, "run", run):
        yield
