"""Causal effective FLOPs must be conserved across the execution boundary."""

from experiments.nosa_baseline_performance.src.dense.mfu import matrix_flops


def test_full_equals_prefix_plus_extend_flops():
    config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
    }
    full = matrix_flops(config, 0, 66560)
    prefix = matrix_flops(config, 0, 65536)
    extend = matrix_flops(config, 65536, 1024)
    assert all(full[name] == prefix[name] + extend[name] for name in full)
    assert set(extend) == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "attention_core"}
    # Sum the effective work of the constituent projections, without inventing
    # individual timing denominators for the combined GEMMs.
    assert extend["qkv_proj"] == 1099511627776 + 2 * 68719476736
    assert extend["gate_up_proj"] == 2 * 4398046511104
    assert full["attention_core"] == 1161376605143040
    assert sum(extend.values()) == 50990120173568
    assert sum(full.values()) == 2170865718394880
