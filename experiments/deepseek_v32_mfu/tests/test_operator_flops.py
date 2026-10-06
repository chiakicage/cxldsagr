"""Check causal, tiled, and mixed-precision accounting without CUDA."""

import pytest

from experiments.deepseek_v32_mfu.src.operator_flops import (
    build_operator_work,
    causal_pairs,
    indexer_work,
    linear_work,
    non_matmul_work,
    selected_pairs,
    sparse_mla_work,
)


@pytest.mark.parametrize("prefix,queries,slots", [(0, 131, 64), (47, 91, 64), (65536, 1024, 2048)])
def test_causal_count_matches_explicit_visibility(prefix, queries, slots):
    visible = [range(prefix + row + 1) for row in range(queries)]
    assert causal_pairs(queries, prefix) == sum(map(len, visible))
    assert selected_pairs(queries, prefix, slots) == sum(min(len(row), slots) for row in visible)


def test_indexer_odd_query_tail_and_key_tile_boundary():
    work = indexer_work(query_tokens=3, query_start=127)
    assert work.useful_flops == 2 * 64 * 128 * (128 + 129 + 130)
    # Groups [0,1] and [2,pad] each compute 2*256 key positions.
    assert work.executed_matmul_flops == 2 * 64 * 128 * (2 * 256 + 2 * 256)
    assert indexer_work(query_tokens=3, query_start=127, prefetch=True).name == "indexer_prefetch"


def test_indexer_official_resident_and_fused_prefetch_have_distinct_kv_padding():
    resident = indexer_work(query_tokens=1, query_start=0)
    fused = indexer_work(query_tokens=1, query_start=0, prefetch=True)
    assert resident.useful_flops == fused.useful_flops == 2 * 64 * 128
    assert resident.executed_matmul_flops == 2 * 64 * 128 * 2 * 256
    assert fused.executed_matmul_flops == 2 * 64 * 128 * 2 * 128


def test_indexer_chunked_prefix_preserves_useful_pairs():
    whole = indexer_work(query_tokens=1024, query_start=0)
    chunks = [indexer_work(query_tokens=256, query_start=start) for start in range(0, 1024, 256)]
    assert sum(chunk.useful_flops for chunk in chunks) == whole.useful_flops
    assert sum(chunk.executed_matmul_flops for chunk in chunks) == whole.executed_matmul_flops
    with pytest.raises(ValueError, match="causal endpoint"):
        indexer_work(query_tokens=10, query_start=20, kv_tokens=29)


def test_sparse_mla_full_extend_and_early_causal_padding():
    extend = sparse_mla_work(
        query_tokens=1024,
        heads=128,
        qk_dim=576,
        value_dim=512,
        selected_slots=2048,
        valid_selected_pairs=1024 * 2048,
    )
    expected = 2 * 1024 * 128 * 2048 * (576 + 512)
    assert extend.useful_flops == expected == extend.executed_matmul_flops
    assert extend.dimensions["qk_flops"] + extend.dimensions["pv_flops"] == expected
    early = sparse_mla_work(
        query_tokens=2,
        heads=64,
        qk_dim=576,
        value_dim=512,
        selected_slots=5,
        valid_selected_pairs=3,
    )
    assert early.useful_flops == 2 * 3 * 64 * (576 + 512)
    assert early.executed_matmul_flops == 2 * 2 * 64 * 128 * (576 + 512)


def test_split_offload_mla_conserves_work():
    def work(queries):
        return sparse_mla_work(
            query_tokens=queries,
            heads=128,
            qk_dim=576,
            value_dim=512,
            selected_slots=2048,
            valid_selected_pairs=queries * 2048,
        )

    whole = work(1024)
    leaves = [work(size) for size in (16, 16, 32, 64, 128, 256, 512)]
    assert sum(leaf.useful_flops for leaf in leaves) == whole.useful_flops
    assert sum(leaf.executed_matmul_flops for leaf in leaves) == whole.executed_matmul_flops


@pytest.mark.parametrize(
    "heads,slots,padded_slots",
    [(64, 65, 128), (128, 129, 256), (128, 2048, 2048)],
)
def test_mla_padding_follows_official_flashmla_selection_alignment(heads, slots, padded_slots):
    queries = 63
    work = sparse_mla_work(
        query_tokens=queries,
        heads=heads,
        qk_dim=576,
        value_dim=512,
        selected_slots=slots,
        valid_selected_pairs=queries * slots,
    )
    assert work.useful_flops == 2 * queries * slots * heads * (576 + 512)
    assert work.executed_matmul_flops == 2 * queries * heads * padded_slots * (576 + 512)
    assert work.dimensions["slots_padded"] == padded_slots
    assert work.dimensions["threads"] == 384


@pytest.mark.parametrize("heads,qk_dim,value_dim", [(65, 576, 512), (64, 160, 128)])
def test_mla_rejects_shapes_unsupported_by_official_sm90_kernel(heads, qk_dim, value_dim):
    with pytest.raises(ValueError, match="FlashMLA SM90"):
        sparse_mla_work(
            query_tokens=2,
            heads=heads,
            qk_dim=qk_dim,
            value_dim=value_dim,
            selected_slots=1,
            valid_selected_pairs=2,
        )


def test_official_fp8_projection_preserves_useful_work_without_inventing_runtime_padding():
    work = linear_work(
        "kv_a_proj",
        rows=33,
        in_features=129,
        out_features=576,
        precision="fp8",
        local_fp8_kernel=True,
    )
    assert work.useful_flops == 2 * 33 * 129 * 576
    assert work.executed_matmul_flops is None
    assert work.executed_formula is None
    assert "Official DeepGEMM" in work.notes
    cublas = linear_work("bf16", rows=33, in_features=129, out_features=576, precision="bf16")
    assert cublas.useful_flops == work.useful_flops
    assert cublas.executed_matmul_flops is None


def test_official_fp8_narrow_projection_does_not_assume_local_split_k():
    work = linear_work(
        "index_k_proj",
        rows=1024,
        in_features=1025,
        out_features=65,
        precision="fp8",
        local_fp8_kernel=True,
    )
    assert work.dimensions == {"M": 1024, "N": 65, "K": 1025}
    assert work.useful_flops == 2 * 1024 * 1025 * 65
    assert work.executed_matmul_flops is None


def test_mfu_uses_precision_peak_and_never_invents_nonmatmul_or_fp32_zero():
    fp8 = linear_work("gemm", rows=1000, in_features=1000, out_features=1000, precision="fp8")
    bf16 = linear_work("gemm", rows=1000, in_features=1000, out_features=1000, precision="bf16")
    peaks = {"bf16": 100, "fp8": 200}
    assert fp8.mfu(1, peaks)["mfu_pct"] == 1
    assert bf16.mfu(1, peaks)["mfu_pct"] == 2
    fp32 = linear_work("weights", rows=1000, in_features=1000, out_features=1000, precision="fp32")
    assert fp32.mfu(1, peaks)["mfu_pct"] is None
    assert "fp32" in fp32.mfu(1, peaks)["mfu_na_reason"]
    assert non_matmul_work("exact_topk").mfu(1, peaks)["mfu_pct"] is None
    with pytest.raises(ValueError, match="positive measured duration"):
        fp8.mfu(0, peaks)
    with pytest.raises(ValueError, match="finite and positive"):
        fp8.mfu(1, {"fp8": float("nan")})


CONFIG = {
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    "intermediate_size": 18432,
    "vocab_size": 129280,
}
DTYPES = dict.fromkeys(
    (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "index_q_proj",
        "index_k_proj",
        "o_proj",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    ),
    "fp8",
)


def test_real_dense_layer_dimensions_and_single_token_head():
    work = build_operator_work(
        CONFIG,
        query_tokens=1024,
        query_start=65536,
        linear_dtypes={**DTYPES, "index_k_proj": "bf16", "lm_head": "bf16"},
        lm_head_tokens=1,
    )
    assert len(work) == 15
    assert work["q_b_proj"].useful_flops == 2 * 1024 * 1536 * 128 * (128 + 64)
    assert (
        work["q_absorb"].useful_flops == work["v_expand"].useful_flops == 2 * 1024 * 128 * 128 * 512
    )
    assert work["lm_head"].useful_flops == 2 * 7168 * 129280
    assert (
        work["mlp_gate"].useful_flops
        == work["mlp_up"].useful_flops
        == work["mlp_down"].useful_flops
    )
    assert work["index_k_proj"].precision == "bf16"
    assert work["index_weights_proj"].precision == "fp32"
    assert work["sparse_mla"].dimensions["valid_pairs"] == 1024 * 2048
    assert work["indexer"].as_dict()["dimensions"]["KV"] == 66560


def test_missing_precision_is_not_inferred_from_backend_flag():
    with pytest.raises(ValueError, match="Observed linear precision"):
        build_operator_work(CONFIG, query_tokens=1024, query_start=0, linear_dtypes={})
