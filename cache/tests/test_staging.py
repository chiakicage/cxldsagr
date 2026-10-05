"""Fixed storage, serial ownership and asynchronous double-buffer dependencies."""

import pytest
import torch

from cache.staging import DoubleBufferStaging


def make_staging(*, capacity=129, device="cpu", shapes=None):
    return DoubleBufferStaging(
        capacity,
        {"keys": (2, 3), "values": (2, 3)} if shapes is None else shapes,
        dtype=torch.float32,
        device=device,
    )


def source_records(staging, length, value):
    return {
        name: torch.full(
            (length, *shape),
            value + offset,
            dtype=staging.dtype,
            pin_memory=staging.device.type == "cuda",
        )
        for offset, (name, shape) in enumerate(staging.record_shapes.items())
    }


def test_estimate_is_pure_and_matches_independent_storage_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("estimate must not allocate or initialize CUDA")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(torch.cuda, "current_device", forbidden)
        estimated = DoubleBufferStaging.estimate_bytes(
            129, {"keys": (2, 3), "values": (2, 3)}, dtype=torch.float32
        )
    staging = make_staging()
    storage = {tensor.untyped_storage().data_ptr(): tensor for tensor in staging.storage_tensors()}
    actual = sum(tensor.untyped_storage().nbytes() for tensor in storage.values())
    assert actual == estimated == 2 * 129 * 2 * 2 * 3 * 4
    assert staging.shared_bytes() == {"hbm": actual, "dram": 0}
    pointers = tuple(storage)
    with staging.lease(object()) as lease:
        lease.prefetch(0, source_records(staging, 65, 1))
        lease.wait_ready(0)
        lease.reset()
    assert tuple(t.untyped_storage().data_ptr() for t in staging.storage_tensors()) == pointers
    staging.close()
    staging.close()
    assert staging.storage_tensors() == ()
    assert staging.shared_bytes() == {"hbm": 0, "dram": 0}
    with pytest.raises(RuntimeError, match="closed"):
        staging.lease(object())


@pytest.mark.parametrize(
    "capacity,shapes",
    [(0, {"a": (2,)}), (True, {"a": (2,)}), (1, {}), (1, {"": (2,)}), (1, {"a": (0,)})],
)
def test_invalid_layout_does_not_allocate(monkeypatch, capacity, shapes):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid layout must fail before allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(ValueError):
        DoubleBufferStaging(capacity, shapes, dtype=torch.float32, device="cpu")


def test_generic_record_shape_and_zero_history():
    staging = make_staging(capacity=7, shapes={"records": (5,), "scale": ()})
    with staging.lease(object(), generation=2) as lease:
        lease.prefetch(0, source_records(staging, 0, 0))
        views = lease.wait_ready(0)
        views["records"][:1].fill_(17)
        views["scale"][:1].fill_(3)
        assert lease.view(0, end=1)["records"].shape == (1, 5)
        assert lease.view(0, end=0)["scale"].shape == (0,)
        assert torch.all(lease.view(0, end=1)["records"] == 17)


def test_serial_lease_identity_rejects_foreign_stale_and_live_close():
    staging = make_staging()
    a, b = object(), object()
    first = staging.lease(a, generation=3)
    first.prefetch(0, source_records(staging, 65, 10))
    identity = first.slot_identity(0)
    assert identity.session is a
    assert (identity.generation, identity.layer, identity.cycle) == (3, 0, 0)
    with pytest.raises(RuntimeError, match="active execution lease"):
        staging.lease(b)
    with pytest.raises(RuntimeError, match="live execution lease"):
        staging.close()
    first.close()
    with staging.lease(b, generation=4) as second:
        assert second.slot_identity(0) is None
        with pytest.raises(RuntimeError, match="closed or stale"):
            first.wait_ready(0)
        with pytest.raises(RuntimeError, match="does not own"):
            second.wait_ready(0)
        second.prefetch(0, source_records(staging, 17, 20))
        assert second.slot_identity(0).session is b
        torch.testing.assert_close(second.wait_ready(0)["keys"][:17], torch.full((17, 2, 3), 20.0))
    with staging.lease(a, generation=5) as revisit:
        revisit.prefetch(0, source_records(staging, 129, 30))
        assert revisit.slot_identity(0).generation == 5
        torch.testing.assert_close(revisit.wait_ready(0)["keys"], torch.full((129, 2, 3), 30.0))


def test_ready_and_layer_ownership_are_required_before_consumption_and_reuse():
    staging = make_staging()
    with staging.lease(object()) as lease:
        lease.prefetch(0, source_records(staging, 65, 1))
        with pytest.raises(RuntimeError, match="wait_ready"):
            lease.view(0)
        with pytest.raises(RuntimeError, match="consumer waits ready"):
            lease.prefetch(2, source_records(staging, 65, 3))
        lease.wait_ready(0)
        lease.prefetch(1, source_records(staging, 65, 2))
        lease.wait_ready(1)
        lease.prefetch(2, source_records(staging, 65, 3))
        with pytest.raises(RuntimeError, match="does not own"):
            lease.view(0)
        assert torch.all(lease.wait_ready(2)["keys"][:65] == 3)
        with pytest.raises(ValueError, match="view end"):
            lease.view(2, end=130)


def test_multiple_chunks_reset_ownership_within_one_prefill_lease():
    staging = make_staging()
    with staging.lease(object(), generation=7) as lease:
        for cycle, length in enumerate((0, 17, 65, 129)):
            if cycle:
                lease.reset()
                with pytest.raises(RuntimeError, match="does not own"):
                    lease.view(0)
            for layer in range(4):
                lease.prefetch(layer, source_records(staging, length, cycle * 10 + layer))
                views = lease.wait_ready(layer)
                assert torch.all(views["keys"][:length] == cycle * 10 + layer)
                assert lease.slot_identity(layer % 2).cycle == cycle


def test_invalid_prefetch_does_not_change_valid_slot():
    staging = make_staging()
    with staging.lease(object()) as lease:
        good = source_records(staging, 17, 4)
        lease.prefetch(0, good)
        before = lease.wait_ready(0)["keys"][:17].clone()
        identity = lease.slot_identity(0)
        invalid = [
            {"keys": good["keys"]},
            {**good, "keys": torch.empty(130, 2, 3)},
            {**good, "keys": torch.empty(17, 2, 4)},
            {**good, "keys": torch.empty(17, 2, 3, dtype=torch.float64)},
        ]
        for records in invalid:
            with pytest.raises(ValueError):
                lease.prefetch(2, records)
            assert lease.slot_identity(0) is identity
            torch.testing.assert_close(lease.view(0, end=17)["keys"], before)


def test_partial_submission_error_requires_drain_before_reuse(monkeypatch):
    staging = make_staging()
    original = torch.Tensor.copy_
    calls = 0

    def fail_second(destination, source, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy error")
        return original(destination, source, *args, **kwargs)

    with staging.lease(object()) as lease:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", fail_second)
            with pytest.raises(RuntimeError, match="injected copy"):
                lease.prefetch(0, source_records(staging, 17, 4))
        assert lease.submitted_copy_bytes == 17 * 2 * 3 * 4
        with pytest.raises(RuntimeError, match="submission failed"):
            lease.wait_ready(0)
        lease.reset()
        lease.prefetch(0, source_records(staging, 17, 8))
        assert torch.all(lease.wait_ready(0)["keys"][:17] == 8)
        assert lease.submitted_copy_bytes == 3 * 17 * 2 * 3 * 4
    assert not staging.poisoned


def test_exception_returns_lease_but_unconfirmed_join_poisons_and_retains_storage():
    staging = make_staging()
    with pytest.raises(ValueError, match="middle layer"), staging.lease(object()) as lease:
        lease.prefetch(0, source_records(staging, 17, 1))
        lease.wait_ready(0)
        lease.prefetch(1, source_records(staging, 17, 2))
        raise ValueError("middle layer failed")
    assert staging.active_lease is None
    assert not staging.poisoned

    class FakeStream:
        def __init__(self, fail):
            self.fail, self.calls = fail, 0

        def synchronize(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError("injected CUDA failure")

    lease = staging.lease(object())
    lease.prefetch(0, source_records(staging, 17, 3))
    owner = lease.slot_identity(0)
    caller, copy = FakeStream(True), FakeStream(False)
    lease.caller_stream, staging._copy_stream = caller, copy
    with pytest.raises(RuntimeError, match="injected CUDA failure"):
        lease.close()
    assert caller.calls == copy.calls == 1
    assert lease.closed and staging.poisoned
    assert staging.active_lease is lease
    assert staging._slot_identities[0] is owner
    assert staging.storage_tensors() and lease._sources
    with pytest.raises(RuntimeError, match="poisoned"):
        staging.lease(object())
    with pytest.raises(RuntimeError, match="injected CUDA failure"):
        staging.close()
    assert staging.storage_tensors()
    caller.fail = False
    staging.close()
    assert staging.storage_tensors() == ()


@pytest.mark.parametrize("body_fails", [False, True])
def test_all_stream_failures_and_body_failure_survive_with_borrowed_storage(body_fails):
    staging = make_staging()
    body_error = ValueError("model failed")
    caller_error = RuntimeError("caller stream failed")
    copy_error = KeyboardInterrupt("copy stream interrupted")
    joined = []

    class FailingStream:
        def __init__(self, error):
            self.error = error

        def synchronize(self):
            joined.append(self.error)
            raise self.error

    with pytest.raises(BaseExceptionGroup) as caught, staging.lease(object()) as lease:
        lease.prefetch(0, source_records(staging, 17, 3))
        lease.caller_stream = FailingStream(caller_error)
        staging._copy_stream = FailingStream(copy_error)
        if body_fails:
            raise body_error
    group = caught.value
    if body_fails:
        assert group.exceptions[0] is body_error
        group = group.exceptions[1]
    assert group.exceptions == (caller_error, copy_error)
    assert joined == [caller_error, copy_error]
    assert staging.poisoned and lease.closed
    assert staging.active_lease is lease
    assert staging.storage_tensors() and lease._sources
    with pytest.raises(RuntimeError, match="poisoned"):
        staging.lease(object())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_delayed_copy_slot_consumer_and_nondefault_session_handoff():
    staging = make_staging(device="cuda")
    sessions = object(), object()
    originals, outputs = [], []
    for request, length in enumerate((65, 17, 129)):
        caller = torch.cuda.Stream()
        with (
            torch.cuda.stream(caller),
            staging.lease(sessions[request % 2], generation=request) as lease,
        ):
            with torch.cuda.stream(staging._copy_stream):
                torch.cuda._sleep(50_000_000)
            lease.prefetch(0, source_records(staging, length, request * 10))
            for layer in range(4):
                current = lease.wait_ready(layer)
                # The next prefetch uses the other slot. Later reuse of this
                # slot must wait for this deliberately delayed consumer.
                if layer < 3:
                    lease.prefetch(
                        layer + 1, source_records(staging, length, request * 10 + layer + 1)
                    )
                torch.cuda._sleep(5_000_000)
                outputs.append(current["keys"][:length].clone())
                originals.append(torch.full((length, 2, 3), float(request * 10 + layer)))
    for output, expected in zip(outputs, originals, strict=True):
        torch.testing.assert_close(output.cpu(), expected)
    staging.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_exception_joins_unwaited_speculative_copy_and_consumer():
    staging = make_staging(device="cuda")
    copy_done, consumer_done = torch.cuda.Event(), torch.cuda.Event()
    caller = torch.cuda.Stream()
    with (
        pytest.raises(ValueError, match="middle layer"),
        torch.cuda.stream(caller),
        staging.lease(object()) as lease,
    ):
        lease.prefetch(0, source_records(staging, 65, 1))
        ready = lease.wait_ready(0)
        torch.cuda._sleep(5_000_000)
        output = ready["keys"][:65].clone()
        consumer_done.record(caller)
        with torch.cuda.stream(staging._copy_stream):
            torch.cuda._sleep(50_000_000)
        lease.prefetch(1, source_records(staging, 65, 2))
        copy_done.record(staging._copy_stream)
        assert not copy_done.query(), "the test must leave a copy outstanding"
        raise ValueError("middle layer failed before waiting for layer one")
    assert copy_done.query() and consumer_done.query()
    assert staging.active_lease is None and not staging.poisoned
    with staging.lease(object()) as lease:
        lease.prefetch(0, source_records(staging, 17, 7))
        lease.wait_ready(0)
    torch.testing.assert_close(output.cpu(), torch.ones(65, 2, 3))
    staging.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_reset_joins_speculative_copy_before_new_chunk():
    staging = make_staging(device="cuda")
    done = torch.cuda.Event()
    with torch.cuda.stream(torch.cuda.Stream()), staging.lease(object()) as lease:
        lease.prefetch(0, source_records(staging, 65, 1))
        output = lease.wait_ready(0)["keys"][:65].clone()
        with torch.cuda.stream(staging._copy_stream):
            torch.cuda._sleep(50_000_000)
        lease.prefetch(1, source_records(staging, 65, 2))
        done.record(staging._copy_stream)
        assert not done.query(), "the test must leave a copy outstanding"
        lease.reset()
        assert done.query()
        assert lease.slot_identity(0) is lease.slot_identity(1) is None
        lease.prefetch(0, source_records(staging, 17, 3))
        current = lease.wait_ready(0)["keys"][:17].clone()
    torch.testing.assert_close(output.cpu(), torch.ones(65, 2, 3))
    torch.testing.assert_close(current.cpu(), torch.full((17, 2, 3), 3.0))
    staging.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_lease_rejects_wrong_stream_and_unpinned_async_source():
    staging = make_staging(device="cuda")
    caller = torch.cuda.Stream()
    with torch.cuda.stream(caller):
        lease = staging.lease(object())
        with pytest.raises(ValueError, match="pinned"):
            lease.prefetch(0, {"keys": torch.ones(1, 2, 3), "values": torch.ones(1, 2, 3)})
        lease.prefetch(0, source_records(staging, 17, 3))
        lease.wait_ready(0)
    with pytest.raises(RuntimeError, match="captured caller stream"):
        lease.view(0)
    # Return may happen from a finally block outside the caller's stream context.
    lease.close()
    staging.close()
