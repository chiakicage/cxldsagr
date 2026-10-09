"""Official decode EMA preserves prefill state and follows live graph inputs."""

import pytest
import torch

from operators.deepseek_v32.indexer.decode_hint import update_decode_hint


def test_cpu_decode_hint_updates_only_its_owned_scalar():
    offset = torch.arange(16, dtype=torch.float32)
    expected = offset.clone()
    expected[1] = 0.5 * expected[1] + 0.5 * -3.0
    update_decode_hint(torch.tensor([[7.0, 2.0, -3.0]]), offset)
    assert torch.equal(offset, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_decode_hint_is_bitwise_official_ema_with_changed_graph_inputs():
    values = torch.tensor([[8.0, 4.0, -3.25]], device="cuda")
    offset = torch.arange(16, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        update_decode_hint(values, offset)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update_decode_hint(values, offset)
        for score, previous in ((-3.25, 7.125), (0.0, -0.0), (2.75, -5.5), (1e20, 1e-20)):
            values[0, -1] = score
            offset[1] = previous
            expected = offset.clone()
            expected[1] = expected[1] * 0.5 + values[0, -1] * 0.5
            graph.replay()
            torch.testing.assert_close(
                offset.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
            )
