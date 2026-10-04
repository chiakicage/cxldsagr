"""Owned token encoding preserves identity, strict types and admission order."""

import gc
import struct
import sys
import weakref
from array import array

import pytest
import torch

from serving.persistent import (
    PersistentGRRunner,
    _packed_prefix_digest,
    _prepare_token_input,
    token_digest,
)
from serving.tests.test_persistent import Backend, request


def test_packed_input_keeps_exact_int64_values_digest_and_private_lifetime():
    ids = [0, 1, 65536, 2**63 - 1]
    expected = list(ids)
    signature, tensor = _prepare_token_input(ids, 3)
    assert signature == (3, token_digest(expected[:3]))
    ids[:] = [8]
    gc.collect()
    assert tensor.dtype == torch.long and tensor.tolist() == expected
    tensor[0] = 9
    assert ids == [8] and signature == (3, token_digest(expected[:3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_input_device_copy_owns_values_after_cpu_buffer_release():
    ids = [0, 1, 65536, 2**63 - 1]
    expected = torch.tensor(ids, dtype=torch.long)
    signature, host = _prepare_token_input(ids, 3)
    host_reference = weakref.ref(host)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Match the runner's blocking transfer before admission, including a
        # non-default execution stream with outstanding earlier work.
        torch.cuda._sleep(2_000_000)
        device = host.to(device="cuda")
    host.zero_()
    ids[:] = [8]
    del host
    gc.collect()
    assert host_reference() is None
    stream.synchronize()
    torch.testing.assert_close(device.cpu(), expected, rtol=0, atol=0)
    assert signature == (3, token_digest(expected[:3].tolist()))


def test_big_endian_prefix_hash_uses_copy_and_preserves_native_tensor_buffer(monkeypatch):
    values = [0, 17, 2**63 - 1, 99]
    packed = array("q", values)
    if sys.byteorder == "little":
        packed.byteswap()  # Simulate the native byte representation on a big-endian host.
    before = packed.tobytes()
    monkeypatch.setattr(sys, "byteorder", "big")
    assert _packed_prefix_digest(packed, 3) == token_digest(values[:3])
    assert packed.tobytes() == before


def test_int64_overflow_retains_prefix_struct_error_and_candidate_tensor_error():
    with pytest.raises(struct.error) as prefix_error:
        _prepare_token_input([2**63, 1], 1)
    with pytest.raises(ValueError, match="Overflow when unpacking long long") as candidate_error:
        _prepare_token_input([1, 2**63], 1)
    assert prefix_error.value.__context__ is candidate_error.value.__context__ is None


class IntSubclass(int):
    pass


@pytest.mark.parametrize("bad", [True, 1.0, -1, IntSubclass(1)])
def test_faster_validation_still_requires_exact_nonnegative_python_ints(bad):
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        with pytest.raises(ValueError, match="nonnegative integers"):
            runner.execute(request(prefix=(bad, 2)))
        assert len(runner.pool) == 0 and backend.built == 0


def test_invalid_token_type_cannot_invoke_metaclass_hash_or_equality():
    class ForeignType(type):
        def __hash__(cls):
            raise AssertionError("validation must not hash user-defined types")

        def __eq__(cls, other):
            raise AssertionError("validation must compare types by identity")

    class ForeignToken(metaclass=ForeignType):
        pass

    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        with pytest.raises(ValueError, match="nonnegative integers"):
            runner.execute(request(prefix=(ForeignToken(), 2)))
        assert len(runner.pool) == 0


def test_device_transfer_failure_keeps_existing_session_before_admission(monkeypatch):
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        runner.execute(request())
        original = runner.pool._entries[0]

        def forbidden_acquire(*args, **kwargs):
            raise AssertionError("input transfer must complete before admission")

        def failed_transfer(*args, **kwargs):
            raise RuntimeError("input transfer failed")

        monkeypatch.setattr(runner.pool, "acquire", forbidden_acquire)
        monkeypatch.setattr(torch.Tensor, "to", failed_transfer)
        with pytest.raises(RuntimeError, match="input transfer failed"):
            runner.execute(request(candidate=(5, 6)))
        assert runner.pool._entries[0] is original
        assert runner.visits[0] == 1 and backend.released == 0


def test_candidate_overflow_keeps_existing_session_before_admission():
    backend = Backend()
    with PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner:
        runner.execute(request())
        original = runner.pool._entries[0]
        with pytest.raises(ValueError, match="Overflow when unpacking long long"):
            runner.execute(request(candidate=(2**63,)))
        assert runner.pool._entries[0] is original
        assert runner.visits[0] == 1 and backend.released == 0
