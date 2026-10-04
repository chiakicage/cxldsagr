from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.deepseek_v32.echo_model import CheckpointAttention


class _CudaLayoutOnCPU(torch.Tensor):
    """Select the CUDA layout branch while all tensor dispatch remains on CPU."""

    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("queries", [1, 5, 128, 1024])
@pytest.mark.parametrize("strided_input", [False, True])
def test_output_writes_expansion_into_projection_layout(monkeypatch, queries, strided_input):
    """Check layout/ownership with CPU math; this is not CUDA numerical acceptance."""
    generator = torch.Generator().manual_seed(71)
    heads, latent, values, hidden = 3, 7, 5, 11
    source = torch.randn(
        queries, heads, latent * (2 if strided_input else 1), generator=generator
    ).bfloat16()
    attention = source[..., ::2] if strided_input else source
    before = attention.clone()
    expansion_weight = torch.randn(heads, latent, values, generator=generator).bfloat16()
    projection_weight = torch.randn(hidden, heads * values, generator=generator).bfloat16()
    expected_heads = torch.bmm(attention.transpose(0, 1), expansion_weight).transpose(0, 1)
    expected_input = expected_heads.reshape(queries, heads * values)
    expected_output = F.linear(expected_input, projection_weight)
    observed = {}
    original_bmm = torch.bmm

    def bmm(left, right, *, out):
        assert "destination" not in observed
        observed["destination"] = out
        assert out.shape == (heads, queries, values)
        assert out.stride() == (values, heads * values, 1)
        return original_bmm(left, right, out=out)

    class Projection:
        weight = projection_weight

        def __call__(self, flattened):
            assert flattened.shape == (queries, heads * values)
            assert flattened.is_contiguous()
            assert (
                flattened.untyped_storage().data_ptr()
                == observed["destination"].untyped_storage().data_ptr()
            )
            torch.testing.assert_close(flattened, expected_input, rtol=0, atol=0)
            return F.linear(flattened, self.weight)

    model = object.__new__(CheckpointAttention)
    model.cfg = SimpleNamespace(n_heads=heads, kv_lora_rank=latent, v_head_dim=values)
    model.wv_b = expansion_weight
    model.wo = Projection()
    monkeypatch.setattr(torch, "bmm", bmm)
    actual = CheckpointAttention.output(model, attention.as_subclass(_CudaLayoutOnCPU))
    torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(attention, before, rtol=0, atol=0)
    assert observed["destination"].untyped_storage().nbytes() == queries * heads * values * 2
