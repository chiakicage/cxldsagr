import pytest
import torch

from models.deepseek_v32.inputs import prepare_token_ids


@pytest.mark.parametrize("ids", [[], [[1, 2]], [-1], [23], [1, 2, 3, 4]])
def test_invalid_host_sequence_fails_before_device_transfer(ids):
    # No GPU is needed: invalid input must be rejected before touching CUDA.
    with pytest.raises(ValueError):
        prepare_token_ids(ids, device="cuda", vocab_size=23, max_tokens=3)


def test_valid_cpu_tensor_retains_storage_and_integer_conversion():
    ids = torch.tensor([0, 7, 22])
    result = prepare_token_ids(ids, device="cpu", vocab_size=23, max_tokens=3)
    assert result is ids
    result = prepare_token_ids(ids.int(), device="cpu", vocab_size=23, max_tokens=3)
    assert result.dtype == torch.long
    torch.testing.assert_close(result, ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("values,valid", [([0, 22], True), ([-1, 22], False), ([0, 23], False)])
def test_cuda_input_keeps_vocabulary_validation(values, valid):
    ids = torch.tensor(values, device="cuda")
    if valid:
        assert prepare_token_ids(ids, device=ids.device, vocab_size=23, max_tokens=2) is ids
    else:
        with pytest.raises(ValueError, match="vocabulary"):
            prepare_token_ids(ids, device=ids.device, vocab_size=23, max_tokens=2)
