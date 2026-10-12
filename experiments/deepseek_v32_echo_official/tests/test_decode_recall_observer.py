from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_official.src.decode_recall_observer import (
    I32_MAX,
    RECORD_BYTES,
    RecallObserver,
    classify_recall,
)


def test_copy_source_classification_includes_host_zero_and_validates_staging_identity():
    result = classify_recall(
        [[0, 4, 7, 8, -1, 9]],
        [[100, 101, 102, I32_MAX, 100, 2]],
        [[0, 5, 7]],
        100,
        5,
    )
    assert result["h2d_columns"] == [1, 3, 5]
    assert result["d2d_columns"] == [0, 2]
    assert result["invalid_columns"] == [4]
    assert result["dram_h2d_bytes"] == 3 * RECORD_BYTES
    assert result["prefetch_d2d_bytes"] == 2 * RECORD_BYTES
    assert result["total_recall_records"] == 5


def test_zero_miss_counts_zero_without_reading_invalid_slot_tags():
    result = classify_recall([[-1, -1]], [[I32_MAX - 1, -2]], [[-1]], 100, 0)
    assert result["total_recall_copy_bytes"] == 0


@pytest.mark.parametrize(
    "hosts,slots,staged,start,counter,message",
    [
        ([[1]], [[100]], [[1]], 100, 0, "device counter"),
        ([[1]], [[101]], [[1]], 100, 1, "exceeds"),
        ([[1], [2]], [[100], [101]], [[1], [2]], 100, 2, "query=1"),
        ([[1]], [[100, 101]], [[1]], 100, 1, "geometry"),
    ],
)
def test_invalid_observation_fails(hosts, slots, staged, start, counter, message):
    with pytest.raises(ValueError, match=message):
        classify_recall(hosts, slots, staged, start, counter)


def test_formal_replay_binds_actual_selected_graph_once():
    observer = object.__new__(RecallObserver)
    graph = object()
    captured, snapshots = {"capture_index": 2}, ["snapshot"]
    observer.capture_snapshots = [(graph, captured, snapshots)]
    observer.recording_enabled = False
    observer.observed_replays = 0
    runner = SimpleNamespace(graphs={1: graph}, bs=1)
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
    calls = []

    def original(*args):
        calls.append(args)
        return "original-result"

    wrapped = observer.replay_wrapper(original)
    assert wrapped(runner, batch) == "original-result"
    assert observer.observed_replays == 0
    observer.recording_enabled = True
    assert wrapped(runner, batch) == "original-result"
    assert observer.observed_replays == 1
    assert observer.selected_snapshots is snapshots
    with pytest.raises(ValueError, match="exactly one"):
        wrapped(runner, batch)
    assert len(calls) == 3
