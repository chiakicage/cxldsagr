import pytest
import torch

from models.deepseek_v32.config import Config
from models.deepseek_v32.rotary import (
    apply_rope,
    apply_rope_pair,
    prepare_rotary_cache,
    rotary_frequencies,
)


@pytest.mark.parametrize("interleaved", [False, True])
@torch.inference_mode()
def test_rotary_pair_cpu_preserves_unrotated_suffix(interleaved):
    q = torch.randn(5, 3, 8).bfloat16()
    k = torch.randn(5, 8).bfloat16()
    angles = torch.arange(5).float()[:, None] * torch.tensor([[1.0, 0.01]])
    actual = apply_rope_pair(q, k, angles, interleaved=interleaved)
    for source, result in zip((q, k), actual):
        expected = apply_rope(source[..., :4], angles, interleaved=interleaved)
        torch.testing.assert_close(result[..., :4], expected, rtol=0, atol=0)
        torch.testing.assert_close(result[..., 4:], source[..., 4:], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count,start", [(1, 0), (5, 4094), (1024, 65536)])
@pytest.mark.parametrize("interleaved", [False, True])
@torch.inference_mode()
def test_flashinfer_rotary_pair_matches_model_yarn(count, start, interleaved):
    torch.manual_seed(917)
    # Main MLA rotates a sliced projection with noncontiguous head strides;
    # indexer rotates a prefix and must preserve its nonrotary features.
    width, heads = (64, 128) if interleaved else (128, 64)
    packed_q = torch.randn(count, heads, 192, device="cuda").bfloat16()
    packed_k = torch.randn(count, 576, device="cuda").bfloat16()
    q = packed_q[..., -width:]
    k = packed_k[..., -width:]
    before = (q.clone(), k.clone())
    positions = torch.arange(start, start + count, device="cuda").float()
    angles = positions[:, None] * rotary_frequencies(Config(), device="cuda")[None, :]
    actual = apply_rope_pair(
        q, k, angles, interleaved=interleaved, cache=prepare_rotary_cache(angles)
    )
    for source, saved, result in zip((q, k), before, actual):
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
        assert result.shape == source.shape and result.dtype == source.dtype
        expected = apply_rope(source[..., :64], angles, interleaved=interleaved)
        # FP32 fused multiply-add can round BF16 boundary cases differently.
        torch.testing.assert_close(result[..., :64], expected, rtol=0.008, atol=0.0001)
        error = (result[..., :64].float() - expected.float()).norm()
        assert error / expected.float().norm() < 0.0005
        torch.testing.assert_close(result[..., 64:], source[..., 64:], rtol=0, atol=0)


def _same_rotary_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [1, 121, 128, 1024])
@pytest.mark.parametrize("offset", [0, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_flashinfer_rotary_destination_matches_ordinary_adapter(count, offset, dtype):
    from operators.flashinfer import rotary_pair, rotary_pair_into

    torch.manual_seed(942)
    q = torch.randn(count, 128, 192 + offset, device="cuda", dtype=dtype)[..., 128 + offset :]
    k = torch.randn(count, 576 + offset, device="cuda", dtype=dtype)[..., 512 + offset :]
    q[0, 0, :2] = torch.tensor([0.0, -0.0], device="cuda", dtype=dtype)
    k[0, :2] = torch.tensor([-0.0, 0.0], device="cuda", dtype=dtype)
    saved_q, saved_k = q.clone(), k.clone()
    positions = torch.arange(65536, 65536 + count, device="cuda").float()
    angles = positions[:, None] * rotary_frequencies(Config(), device="cuda")[None, :]
    ids, trig = prepare_rotary_cache(angles)
    expected_q, expected_k = rotary_pair(q, k, ids, trig, interleaved=True)
    final = torch.full((count, 128, 576), 0.625, device="cuda", dtype=dtype)
    destination = final[..., 512:]
    actual_q, actual_k = rotary_pair_into(q, k, ids, trig, destination, interleaved=True)
    assert actual_q is destination
    assert actual_k.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
    _same_rotary_bytes(actual_q, expected_q)
    _same_rotary_bytes(actual_k, expected_k)
    _same_rotary_bytes(q, saved_q)
    _same_rotary_bytes(k, saved_k)
    assert torch.all(final[..., :512] == 0.625)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [128, 1024])
@torch.inference_mode()
def test_flashinfer_rotary_destination_graph_stream_and_owned_key(count):
    from operators.flashinfer import rotary_pair, rotary_pair_into

    torch.manual_seed(943)
    q = torch.randn(count, 128, 192, device="cuda", dtype=torch.bfloat16)[..., 128:]
    k = torch.randn(count, 576, device="cuda", dtype=torch.bfloat16)[..., 512:]
    source_q, source_k = q.clone(), k.clone()
    angles = torch.arange(65536, 65536 + count, device="cuda").float()[:, None]
    angles = angles * rotary_frequencies(Config(), device="cuda")[None, :]
    positions, trig = prepare_rotary_cache(angles)
    final = torch.full((count, 128, 576), 0.625, device="cuda", dtype=q.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            rotary_pair_into(q, k, positions, trig, final[..., 512:], interleaved=True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = rotary_pair_into(q, k, positions, trig, final[..., 512:], interleaved=True)
    retained = []
    for step in range(3):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            q.copy_(source_q * (step + 1))
            k.copy_(source_k * (step + 1))
            positions.copy_(torch.arange(count, device="cuda").roll(step))
            graph.replay()
            owned_key = actual[1].clone()
            done = torch.cuda.Event()
            done.record()
        torch.cuda.current_stream().wait_event(done)
        expected = rotary_pair(q, k, positions, trig, interleaved=True)
        for got, want in zip(actual, expected, strict=True):
            _same_rotary_bytes(got, want)
        assert torch.all(final[..., :512] == 0.625)
        for previous, snapshot in retained:
            _same_rotary_bytes(previous, snapshot)
        retained.append((owned_key, owned_key.clone()))
    torch.cuda.synchronize()
