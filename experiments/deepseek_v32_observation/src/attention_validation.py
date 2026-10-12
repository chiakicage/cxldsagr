"""Independent bounded-memory correctness checks for resident observation."""

from __future__ import annotations

import torch

from operators.deepseek_v32.attention.reference.torch import reference_sparse_mla


def _all_finite(tensor, *, chunk_rows=16384):
    flat = tensor.reshape(-1, tensor.shape[-1])
    checks = []
    for start in range(0, len(flat), chunk_rows):
        block = flat[start : start + chunk_rows]
        if block.dtype == torch.float8_e4m3fn:
            block = block.float()
        checks.append(torch.isfinite(block).all())
    return bool(torch.stack(checks).all())


@torch.no_grad()
def validate_attention(workload):
    """Check full outputs/selection, with FP32 samples in every sequence.

    Full score masks, exact selection values/order/count/IDs, and complete
    MLA output finite status are checked for every row. At least one query
    in every independent sequence is compared against a bounded FP32 oracle;
    B1 extend additionally samples its middle and final query. Reference work
    and diagnostic comparisons are never part of formal timing.
    """
    workload.run()
    torch.cuda.synchronize(workload.device)
    full_finite = {
        name: _all_finite(getattr(workload, name))
        for name in (
            "index_q",
            "index_k",
            "index_scales",
            "index_weights",
            "main_q",
            "main_kv",
            "output",
        )
    }
    if not all(full_finite.values()):
        raise AssertionError(f"nonfinite resident operands/output: {full_finite}")
    rows = workload.batch * workload.queries
    count = min(2048, workload.tokens)
    ids = workload.indices.view(rows, count)
    values = workload.values
    full_rows = 0
    for start in range(0, rows, 32):
        stop = min(rows, start + 32)
        scores = workload.scores[start:stop]
        endpoints = workload.ends[start:stop]
        positions = torch.arange(scores.shape[1], device=workload.device)
        causal = positions[None] < endpoints[:, None]
        if not bool(torch.equal(torch.isfinite(scores), causal)):
            raise AssertionError("indexer finite score mask differs from exact causal visibility")
        if not bool(torch.isneginf(scores[~causal]).all()):
            raise AssertionError("causal/padded score tails must be -inf")
        selected = ids[start:stop]
        if not bool(((selected >= 0) & (selected < endpoints[:, None])).all()):
            raise AssertionError("selected local IDs leave their sequence's causal prefix")
        ordered_ids = selected.sort(dim=-1).values
        if bool((ordered_ids[:, 1:] == ordered_ids[:, :-1]).any()):
            raise AssertionError("exact top-k selected duplicate IDs")
        gathered = scores.gather(1, selected.long())
        torch.testing.assert_close(
            gathered.view(torch.int32), values[start:stop].view(torch.int32), atol=0, rtol=0
        )
        reference_values = torch.topk(scores, count, dim=-1).values
        torch.testing.assert_close(
            values[start:stop].view(torch.int32), reference_values.view(torch.int32), atol=0, rtol=0
        )
        if not bool((values[start:stop, :-1] >= values[start:stop, 1:]).all()):
            raise AssertionError("selected values must be descending")
        # Ordinary equal FP32 values use ascending IDs. Signed zero retains
        # FlashInfer's ordered floating-point key distinction.
        bits = values[start:stop].view(torch.int32)
        ties = bits[:, :-1] == bits[:, 1:]
        if bool((ties & (selected[:, :-1] > selected[:, 1:])).any()):
            raise AssertionError("SMALL tie policy must order equal score bits by local ID")
        full_rows += stop - start
    sampled_queries = sorted({0, workload.queries // 2, workload.queries - 1})
    samples = []
    maximum_indexer_error = 0.0
    maximum_mla_error = 0.0
    maximum_indexer_normalized_diff = 0.0
    legacy_elementwise_violations = 0
    # Keep this oracle FP32 and disable TF32 explicitly, restoring caller state.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            for sequence in range(workload.batch):
                for query in sampled_queries:
                    endpoint = workload.history + query + 1
                    row = sequence * workload.queries + query
                    q = workload.index_q[sequence, query].float()
                    weights = workload.index_weights[row].float()
                    sample_error = 0.0
                    error_squared = 0.0
                    operand_squared = 0.0
                    sample_legacy_violations = 0
                    for start in range(0, endpoint, 4096):
                        stop = min(endpoint, start + 4096)
                        keys = workload.index_k[sequence, start:stop].float()
                        raw = (q @ keys.T).relu()
                        expected = (raw * weights[:, None]).sum(0)
                        expected *= workload.index_scales[sequence, start:stop]
                        actual = workload.scores[row, start:stop]
                        error = actual.double() - expected.double()
                        error_squared += float(error.square().sum())
                        operand_squared += float(
                            actual.double().square().sum() + expected.double().square().sum()
                        )
                        sample_legacy_violations += int(
                            (error.abs() > 2e-3 + 1e-3 * expected.double().abs()).sum()
                        )
                        sample_error = max(sample_error, float(error.abs().max()))
                    normalized_diff = error_squared / max(operand_squared, 1e-30)
                    # DeepGEMM uses this criterion against its simulated,
                    # already quantized FP8 inputs. Weighted head reduction
                    # has a different floating-point tree from the oracle.
                    if normalized_diff >= 5e-6:
                        raise AssertionError(
                            f"Indexer normalized diff {normalized_diff} exceeds official 5e-6 criterion"
                        )
                    expected_output = reference_sparse_mla(
                        workload.main_q[sequence, query : query + 1],
                        workload.main_kv[sequence],
                        workload.indices[sequence, query : query + 1],
                        workload.cfg.attention_scale,
                    )
                    actual_output = workload.output[sequence, query : query + 1]
                    torch.testing.assert_close(actual_output, expected_output, atol=4e-3, rtol=2e-2)
                    mla_error = float((actual_output.float() - expected_output.float()).abs().max())
                    maximum_indexer_error = max(maximum_indexer_error, sample_error)
                    maximum_indexer_normalized_diff = max(
                        maximum_indexer_normalized_diff, normalized_diff
                    )
                    legacy_elementwise_violations += sample_legacy_violations
                    maximum_mla_error = max(maximum_mla_error, mla_error)
                    samples.append(
                        {
                            "sequence": sequence,
                            "query": query,
                            "causal_endpoint": endpoint,
                            "indexer_max_abs_error": sample_error,
                            "indexer_normalized_diff": normalized_diff,
                            "legacy_indexer_elementwise_violations": sample_legacy_violations,
                            "mla_max_abs_error": mla_error,
                        }
                    )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return {
        "passed": True,
        "case": workload.case,
        "full_finite": full_finite,
        "full_score_mask_rows": full_rows,
        "full_selection_rows": full_rows,
        "selected_entries_per_query": count,
        "full_output_shape": list(workload.output.shape),
        "reference_scope": "FP32 all visible indexer keys and full selected MLA slots for sampled queries in every sequence",
        "sampled_sequences": workload.batch,
        "sampled_query_count": len(samples),
        "indexer_criterion": {
            "normalized_diff_limit": 5e-6,
            "formula": "sum((actual-reference)^2) / sum(actual^2 + reference^2), independently per sampled query",
            "source": "3rdparty/DeepGEMM/tests/test_attention.py:mqa_logits_tolerances simulated FP8 criterion",
            "legacy_elementwise_diagnostic": {"atol": 2e-3, "rtol": 1e-3},
        },
        "mla_tolerances": {"atol": 4e-3, "rtol": 2e-2},
        "max_indexer_abs_error": maximum_indexer_error,
        "max_indexer_normalized_diff": maximum_indexer_normalized_diff,
        "legacy_indexer_elementwise_violations": legacy_elementwise_violations,
        "max_mla_abs_error": maximum_mla_error,
        "samples": samples,
        "leaves": validate_leaves(workload),
    }


@torch.no_grad()
def validate_leaves(workload):
    """Match fixed standalone operands to the already validated complete APIs.

    Full bitwise comparisons bind decomposition results to the current API
    outputs. The original attention validation independently checks selected
    query outputs against FP32; it is not replaced by these comparisons.
    """
    from experiments.deepseek_v32_observation.src.graph_validation import equal_output

    checks = {}
    if workload.paged:
        equal_output(
            workload.pack_only(),
            workload.prepared.packed.view(workload.batch * workload.capacity // 64, 8448),
        )
        equal_output(workload.block_table_only(), workload.prepared.block_table)
        checks["page64_pack"] = {
            "passed": True,
            "comparison": "all packed bytes vs complete prepare API",
        }
        checks["indexer_block_table"] = {
            "passed": True,
            "comparison": "full table vs complete prepare API",
        }
    raw_values, raw_ids = workload.topk_radix_only()
    # Unsorted native selection can change physical output order. Verify its
    # full selected values/IDs after the same exact ordering operation.
    if workload._cub_topk:
        from operators.deepseek_v32.indexer.q1_topk_cub import sort_mask_

        sort_mask_(raw_values, raw_ids)
    else:
        from operators.deepseek_v32.indexer.selection import _mask_nonfinite_indices_

        _mask_nonfinite_indices_(raw_values, raw_ids)
    expected_topk = [
        workload.values,
        workload.leaf_indices.view(workload.batch * workload.queries, -1),
    ]
    equal_output([raw_values, raw_ids], expected_topk)
    equal_output(workload.topk_clone_only(), [workload.leaf_raw_values, workload.leaf_raw_indices])
    equal_output(workload.topk_postprocess_only(), expected_topk)
    for name in ("exact_topk_radix", "topk_output_clone", "topk_postprocess"):
        checks[name] = {
            "passed": True,
            "comparison": "full selected value/ID bits vs complete exact top-k API",
        }
    equal_output(workload.operations["topk_values_clone"](), workload.leaf_raw_values)
    equal_output(workload.operations["topk_ids_clone"](), workload.leaf_raw_indices)
    for name in ("topk_values_clone", "topk_ids_clone"):
        checks[name] = {"passed": True, "comparison": "full cloned value/ID bits"}
    post_name = "topk_cub_sort_mask" if workload._cub_topk else "topk_nonfinite_mask"
    workload.before_call(post_name)()
    post_output = workload.operations[post_name]()
    equal_output(post_output, expected_topk if workload._cub_topk else expected_topk[1])
    checks[post_name] = {
        "passed": True,
        "comparison": "full output after actual postprocess on fresh selected inputs",
    }
    core, _, lse = workload.mla_core_only()
    split = workload.batch == workload.queries == 1
    if split:
        equal_output(workload.mla_repeat_only(), workload.leaf_repeated_q)
        equal_output(core, workload.leaf_partials)
        equal_output(lse, workload.leaf_lse)
        equal_output(
            workload.mla_combine_only().view_as(workload.leaf_mla_reference),
            workload.leaf_mla_reference,
        )
        checks["mla_query_repeat"] = {"passed": True, "comparison": "full repeated query bits"}
        checks["mla_lse_combine"] = {
            "passed": True,
            "comparison": "full output bits vs validated complete MLA API",
        }
    else:
        expected_ids = workload.leaf_native_indices.view_as(workload.leaf_indices)
        equal_output(workload.mla_indices_only(), expected_ids)
        equal_output(core.view_as(workload.leaf_mla_reference), workload.leaf_mla_reference)
        id_references = {
            "mla_id_nonnegative": workload.leaf_indices >= 0,
            "mla_id_upper_bound": workload.leaf_indices < workload.tokens,
            "mla_id_valid_and": (workload.leaf_indices >= 0)
            & (workload.leaf_indices < workload.tokens),
            "mla_sequence_arange": torch.arange(
                workload.batch, device=workload.device, dtype=torch.int64
            ),
            "mla_id_to_int64": workload.leaf_indices.to(torch.int64),
            "mla_sequence_offset_mul": torch.arange(
                workload.batch, device=workload.device, dtype=torch.int64
            )[:, None, None]
            * workload.tokens,
            "mla_id_offset_add": workload.leaf_native_indices.view_as(workload.leaf_indices).to(
                torch.int64
            ),
            "mla_id_where": workload.leaf_native_indices.view_as(workload.leaf_indices).to(
                torch.int64
            ),
            "mla_id_to_int32": expected_ids,
        }
        for name, expected in id_references.items():
            equal_output(workload.operations[name](), expected)
            checks[name] = {
                "passed": True,
                "comparison": "full intermediate ID operation bits vs accepted native IDs/explicit sequence offsets",
            }
        checks["mla_global_indices"] = {
            "passed": True,
            "comparison": "full IDs vs native prepared spans",
        }
    checks["sparse_mla_core"] = {
        "passed": True,
        "comparison": "full native output/LSE or merged output vs validated complete MLA API",
    }
    return {"passed": True, "operators": checks}
