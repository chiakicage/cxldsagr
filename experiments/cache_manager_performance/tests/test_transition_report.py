"""The layer subtotal must be reduced within samples before taking a median."""

from experiments.cache_manager_performance.src.transition_report import summarize


def test_layer_sum_median_is_not_sum_of_layer_medians():
    windows = [
        {
            "source_kind": "standalone_replay",
            "run_id": "run",
            "scheme": "echo",
            "phase": "extend_cold",
            "layer": layer,
            "sample": sample,
            "topk_to_mla_us": duration,
        }
        for sample, durations in enumerate(((1, 100), (100, 1), (50, 50)))
        for layer, duration in enumerate(durations)
    ]
    rows = summarize(windows)
    total = next(row for row in rows if row["layer"] == "all")
    assert total["samples"] == 3
    assert total["median_us"] == 101
    assert total["q1_us"] == 100.5
    assert total["q3_us"] == 101
    assert sum(row["median_us"] for row in rows if row["layer"] != "all") == 100


def test_one_profile_has_degenerate_quantiles_not_invented_repeats():
    row = summarize(
        [
            {
                "source_kind": "complete_model",
                "run_id": "v10",
                "scheme": "echo",
                "phase": "extend_cold",
                "layer": 0,
                "sample": 0,
                "topk_to_mla_us": 42,
            }
        ]
    )[0]
    assert row["samples"] == 1
    assert row["q1_us"] == row["median_us"] == row["q3_us"] == 42
