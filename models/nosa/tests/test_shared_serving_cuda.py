"""Small native SM90 models exercise shared serving storage and failure boundaries.

These checks use four randomly initialized layers, not the NOSA checkpoint.
Full 32-layer checkpoint acceptance is maintained in ``test_serving.py``.
"""

from contextlib import contextmanager

import pytest
import torch

from cache.prefix_pool import CacheFootprint
from models.nosa.serving import NosaServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_serving_resources import storages
from models.nosa.tests.test_sparse_model import initialized_sparse_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SCHEMES = NosaServingBackend.schemes
CAPACITY, CANDIDATE, CHUNK = 224, 81, 47
GEOMETRY = ((210, 129), (160, 79))


@pytest.fixture(scope="module")
def native_model():
    from operators.nosa._native import native_enabled

    assert torch.cuda.get_device_capability() == (9, 0), "shared native test requires SM90"
    assert native_enabled(), "shared native test requires CXLDSAGR_SM90_BACKEND=native"
    config = tiny_config(
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=127,
        max_position_embeddings=CAPACITY + 32,
    )
    # The helper initializes FP32 CPU samples before copying to BF16 parameters;
    # do not use CPU BF16 random initialization, which changes the input weights.
    model = initialized_sparse_model(
        config, device="cuda:0", dtype=torch.bfloat16, backend="triton"
    )
    torch.cuda.synchronize()
    yield model
    torch.cuda.synchronize()


@contextmanager
def allocated(model, scheme):
    backend = NosaServingBackend(model, scheme, chunk_size=CHUNK)
    plan = backend.plan_resources(
        CacheFootprint(1 << 30, 1 << 30),
        {"max_session_capacity": CAPACITY, "max_candidate_tokens": CANDIDATE},
    )
    backend.allocate_shared(plan)
    try:
        yield backend, plan
    finally:
        for session in tuple(backend.resources._sessions):
            backend.release_session(session)
        backend.close()


def tokens(model, length, salt=0):
    return (torch.arange(length, device="cuda:0") * 19 + 11 + salt * 23) % model.config.vocab_size


def shared_storage(backend):
    # The live lease keeps borrowed source/session references until its copies
    # complete. Follow the owned buffers, not those temporary borrower handles.
    return storages(
        backend.resources,
        stop=(
            backend,
            backend.model,
            backend.resources.staging_lease,
            *backend.resources._sessions,
        ),
    )


def session_storage(backend, session):
    return storages(session, stop=(backend, backend.model, backend.resources))


def assert_session_accounting(backend, session, reservation):
    storage = session_storage(backend, session)
    logical = {
        "hbm": sum(size for (device, _), size in storage.items() if device.startswith("cuda:")),
        "dram": sum(size for (device, _), size in storage.items() if device == "cpu"),
    }
    assert backend.session_storage_bytes(session) == logical
    # This CUDA fixture owns only pinned host K/V and a pinned scalar flag.
    # Independently account their physical allocator bins rather than asking
    # the production estimator to confirm itself.
    actual = {
        **logical,
        "dram": sum(
            1 << (size - 1).bit_length()
            for (device, _), size in storage.items()
            if device == "cpu" and size
        ),
    }
    assert backend.session_bytes(session) == actual
    assert set(storage).isdisjoint(shared_storage(backend))
    for tier in ("hbm", "dram"):
        assert actual[tier] <= reservation[tier], (backend.scheme, tier, actual, reservation)


def committed_state(session):
    """Copy committed history and derived views, excluding the unpublished tail."""
    records = {
        name: tensor[:, : session.length].cpu().clone() for name, tensor in session.buffers.items()
    }
    records["cis_scores"] = session.cis_scores[:, : session.length].cpu().clone()
    states, derived = [], []
    for layer in range(session.spec.num_layers):
        state = session.indexer_cache.layer_state(layer)
        states.append((state.validated_tokens, dict(state.lengths)))
        derived.append(
            {
                name: value.cpu().clone()
                for name, value in session.indexer_cache.layer_view(layer).items()
            }
        )
    return session.length, session.indexer_cache.length, records, states, derived


def assert_committed_equal(left, right):
    assert left[:2] == right[:2]
    assert left[3] == right[3]
    torch.testing.assert_close(left[2], right[2], atol=0, rtol=0)
    torch.testing.assert_close(left[4], right[4], atol=0, rtol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
@torch.inference_mode()
def test_cuda_shared_schemes_alternate_users_queries_and_retain_outputs(native_model, scheme):
    with (
        allocated(native_model, scheme) as (backend, plan),
        allocated(native_model, "hbm") as (control, _),
    ):
        shared = shared_storage(backend)
        assert sum(shared.values()) == plan.metadata["logical_shared_hbm_bytes"]
        assert sum(shared.values()) <= plan.shared.hbm
        assert backend.shared_bytes() == {"hbm": sum(shared.values()), "dram": 0}
        backend.allocate_shared(plan)
        assert shared_storage(backend) == shared
        sessions = [backend.create_session(capacity) for capacity, _ in GEOMETRY]
        controls = [control.create_session(capacity) for capacity, _ in GEOMETRY]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        reservations = [
            backend.estimate_session_bytes(capacity, prefix) for capacity, prefix in GEOMETRY
        ]
        samples = []

        def observe(*_):
            session = backend.resources._active_session
            if session is not None:
                index = sessions.index(session)
                assert_session_accounting(backend, session, reservations[index])
                assert shared_storage(backend) == shared
                samples.append(index)

        hooks = [layer.register_forward_hook(observe) for layer in native_model.model.layers]
        retained = []
        try:
            for user, (_, prefix) in enumerate(GEOMETRY):
                with torch.cuda.stream(streams[user]):
                    history = tokens(native_model, prefix, user)
                    control.prefill(controls[user], history)
                    backend.prefill(sessions[user], history)
            assert set(session_storage(backend, sessions[0])).isdisjoint(
                session_storage(backend, sessions[1])
            )
            prefixes = [committed_state(session) for session in sessions]
            for visit, (user, query) in enumerate(((0, 81), (1, 17), (0, 33), (1, 65), (0, 1))):
                with torch.cuda.stream(streams[user]):
                    candidate = tokens(native_model, query, 7 + visit)
                    expected = control.extend(controls[user], candidate)
                    actual = backend.extend(sessions[user], candidate)
                    assert actual.shape == (query, native_model.config.hidden_size)
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    retained.append((actual, expected.cpu().clone()))
                    if scheme == "dense_prefetch":
                        assert sessions[user].last_prefetch_bytes == (
                            2 * GEOMETRY[user][1] * 4 * 2 * 128 * torch.bfloat16.itemsize
                        )
                    backend.truncate(sessions[user], GEOMETRY[user][1])
                    control.truncate(controls[user], GEOMETRY[user][1])
                assert_committed_equal(committed_state(sessions[user]), prefixes[user])
                assert shared_storage(backend) == shared
            for output, expected in retained:
                torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
            assert set(samples) == {0, 1}
            other_storage = session_storage(backend, sessions[1])
            backend.release_session(sessions[0])
            assert session_storage(backend, sessions[0]) == {}
            assert backend.session_bytes(sessions[0]) == {"hbm": 0, "dram": 0}
            assert session_storage(backend, sessions[1]) == other_storage
            assert shared_storage(backend) == shared
        finally:
            for hook in hooks:
                hook.remove()


@pytest.mark.parametrize("scheme", SCHEMES)
@torch.inference_mode()
def test_cuda_shared_middle_layer_failure_preserves_prefix_and_recovers(
    native_model, monkeypatch, scheme
):
    with allocated(native_model, scheme) as (backend, _):
        a, b = backend.create_session(224), backend.create_session(160)
        caller = torch.cuda.Stream()
        with torch.cuda.stream(caller):
            backend.prefill(a, tokens(native_model, 129))
            backend.prefill(b, tokens(native_model, 79, 1))
            candidate = tokens(native_model, 33, 4)
            expected = backend.extend(a, candidate).cpu().clone()
            backend.truncate(a, 129)
        prefix = committed_state(a)
        shared = shared_storage(backend)
        original_attention = native_model.main_attention
        copy_done = torch.cuda.Event() if scheme == "dense_prefetch" else None

        def fail(*args, **kwargs):
            raise RuntimeError("injected shared middle-layer failure")

        with monkeypatch.context() as patch:
            patch.setattr(native_model.model.layers[2], "forward", fail)
            if scheme == "dense_prefetch":
                original_prefetch = a._prefetch

                def delayed_prefetch(layer_idx):
                    stream = backend.resources.staging._copy_stream
                    if layer_idx == 2:
                        with torch.cuda.stream(stream):
                            torch.cuda._sleep(50_000_000)
                    original_prefetch(layer_idx)
                    if layer_idx == 2:
                        copy_done.record(stream)

                patch.setattr(a, "_prefetch", delayed_prefetch)
            with torch.cuda.stream(caller), pytest.raises(RuntimeError, match="injected shared"):
                backend.extend(a, candidate)
        assert native_model.main_attention is original_attention
        assert backend.resources._active_session is None
        assert not backend.resources.poisoned
        assert a._pending_end is a.indexer_cache._pending_end is None
        assert_committed_equal(committed_state(a), prefix)
        if copy_done is not None:
            assert copy_done.query()
            assert backend.resources.staging.active_lease is None
        with torch.cuda.stream(torch.cuda.Stream()):
            backend.extend(b, tokens(native_model, 17, 7))
            backend.truncate(b, 79)
            recovered = backend.extend(a, tokens(native_model, 33, 4))
        torch.testing.assert_close(recovered.cpu(), expected, atol=0, rtol=0)
        assert shared_storage(backend) == shared


@pytest.mark.parametrize("scheme", SCHEMES)
@torch.inference_mode()
def test_cuda_shared_invalid_query_and_disabled_native_fail_before_cache_write(
    native_model, monkeypatch, scheme
):
    with allocated(native_model, scheme) as (backend, plan):
        session = backend.create_session(CAPACITY)
        backend.prefill(session, tokens(native_model, 129))
        prefix = committed_state(session)
        shared = shared_storage(backend)
        invalid_queries = [tokens(native_model, CANDIDATE + 1), tokens(native_model, 0)]
        invalid_queries.append(tokens(native_model, 1).reshape(1, 1))
        valid = tokens(native_model, 17, 2)

        def forbidden(*args, **kwargs):
            pytest.fail("invalid execution must fail before begin_step or cache allocation")

        with monkeypatch.context() as patch:
            patch.setattr(session, "begin_step", forbidden)
            for candidate in invalid_queries:
                with pytest.raises(ValueError):
                    backend.extend(session, candidate)
            patch.setenv("CXLDSAGR_SM90_BACKEND", "triton")
            with pytest.raises(NotImplementedError, match="native SM90"):
                backend.extend(session, valid)
            with pytest.raises(NotImplementedError, match="native SM90"):
                backend.plan_resources(CacheFootprint(1 << 30, 1 << 30), plan.metadata)
        assert backend.resources._active_session is None
        assert session._pending_end is session.indexer_cache._pending_end is None
        assert backend.resources.plan == plan and shared_storage(backend) == shared
        assert_committed_equal(committed_state(session), prefix)
        result = backend.extend(session, valid)
        assert result.shape == (17, native_model.config.hidden_size)


def _unique_historical_payload(selection, prefix, record_bytes):
    """Independent per-KV-head union of actually consumed logical selections."""
    ids = selection.block_ids.cpu()
    valid = (
        selection.valid_mask.cpu()
        if selection.valid_mask is not None
        else torch.ones_like(ids, dtype=torch.bool)
    )
    count = 0
    for head in range(ids.shape[1]):
        blocks = set(ids[:, head][valid[:, head]].tolist())
        count += sum(min(64, prefix - block * 64) for block in blocks if 0 <= block * 64 < prefix)
    return count * record_bytes


@pytest.mark.parametrize("scheme", SCHEMES)
@torch.inference_mode()
def test_cuda_shared_transfer_metrics_match_payload_and_consumed_union(
    native_model, monkeypatch, scheme
):
    from models.nosa.attention import NosaSparseAttention
    from models.nosa.serving import _DensePrefetchAttention

    with allocated(native_model, scheme) as (backend, _):
        session = backend.create_session(210)
        fields = (
            "main_kv_host_to_device_bytes",
            "main_kv_device_to_host_bytes",
            "indexer_host_to_device_bytes",
        )
        expected = [{name: 0 for name in fields} for _ in range(2)]
        adapter = _DensePrefetchAttention if scheme == "dense_prefetch" else NosaSparseAttention
        original = adapter.__call__

        def observe(self, q, selection, access, context):
            output = original(self, q, selection, access, context)
            if access is not session or scheme == "hbm":
                return output
            values = expected[session._transfer_metrics._phase]
            prefix, queries = context.query_start, len(q)
            width_bytes = 2 * 128 * torch.bfloat16.itemsize
            values["main_kv_device_to_host_bytes"] += 2 * queries * width_bytes
            values["main_kv_host_to_device_bytes"] += (
                2 * prefix * width_bytes
                if scheme == "dense_prefetch"
                else _unique_historical_payload(
                    selection, prefix, 2 * 128 * torch.bfloat16.itemsize
                )
            )
            old = max(0, prefix // 16 - 1)
            new = max(0, (prefix + queries) // 16 - 1)
            if old < new:
                values["indexer_host_to_device_bytes"] += (prefix - old * 16) * width_bytes
            return output

        def check(status):
            observed = backend.session_metrics(session)
            assert observed["transfer_metrics_status"] == status
            for index, label in enumerate(("prefix", "candidate")):
                for field in fields:
                    assert observed[f"{label}_{field}"] == expected[index][field]
                assert observed[f"{label}_host_to_device_bytes"] == (
                    expected[index][fields[0]] + expected[index][fields[2]]
                )
                assert observed[f"{label}_device_to_host_bytes"] == expected[index][fields[1]]
            for field in fields:
                assert observed[field] == expected[0][field] + expected[1][field]
            assert observed["host_to_device_bytes"] == sum(
                values[fields[0]] + values[fields[2]] for values in expected
            )
            return observed

        monkeypatch.setattr(adapter, "__call__", observe)
        with torch.cuda.stream(torch.cuda.Stream()):
            backend.prefill(session, tokens(native_model, 129))
            check("prefill_ready")
            backend.extend(session, tokens(native_model, 81, 1))
            completed = check("complete")
            backend.truncate(session, 129)
            assert backend.session_metrics(session) == completed
            expected[:] = [{name: 0 for name in fields} for _ in range(2)]
            # The diagnostic entry uses the same default candidate phase.
            candidate = tokens(native_model, 17, 2)
            with backend.execution_lease(session):
                backend._forward(session, candidate)
            check("complete")
        if scheme in ("serial_sparse", "overlap"):
            assert session._transfer_metrics.storage_bytes == 16
            assert session._transfer_metrics._sparse_fetch_bytes.shape == (2,)
        else:
            assert session._transfer_metrics.storage_bytes == 0


@pytest.mark.parametrize("scheme", ("serial_sparse", "dense_prefetch", "overlap"))
@torch.inference_mode()
def test_cuda_shared_partial_d2h_failure_counts_only_submitted_payload(
    native_model, monkeypatch, scheme
):
    with allocated(native_model, scheme) as (backend, _):
        session = backend.create_session(210)
        backend.prefill(session, tokens(native_model, 129))
        destination = session.buffers["values"].untyped_storage().data_ptr()
        original = torch.Tensor.copy_

        def fail_value_copy(target, source, *args, **kwargs):
            if target.device.type == "cpu" and target.untyped_storage().data_ptr() == destination:
                raise RuntimeError("injected second host record copy")
            return original(target, source, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", fail_value_copy)
            with pytest.raises(RuntimeError, match="second host record"):
                backend.extend(session, tokens(native_model, 17, 1))
        metrics = backend.session_metrics(session)
        assert metrics["transfer_metrics_status"] == "failed"
        assert (
            metrics["candidate_main_kv_device_to_host_bytes"]
            == 17 * 2 * 128 * torch.bfloat16.itemsize
        )
        assert session.length == session.indexer_cache.length == 129
        backend.extend(session, tokens(native_model, 17, 1))
        recovered = backend.session_metrics(session)
        assert recovered["transfer_metrics_status"] == "complete"
        assert (
            recovered["prefix_host_to_device_bytes"]
            == recovered["prefix_device_to_host_bytes"]
            == 0
        )
        assert recovered["candidate_main_kv_device_to_host_bytes"] == (
            2 * 4 * 17 * 2 * 128 * torch.bfloat16.itemsize
        )


@pytest.mark.parametrize("scheme", ("serial_sparse", "overlap"))
@torch.inference_mode()
def test_cuda_shared_counter_allocation_stream_cannot_overwrite_request_totals(
    native_model, monkeypatch, scheme
):
    with allocated(native_model, scheme) as (backend, _):
        creator, caller = torch.cuda.Stream(), torch.cuda.Stream()
        original = torch.empty
        initialized = torch.cuda.Event()
        injected = []

        def delayed_counter(*args, **kwargs):
            result = original(*args, **kwargs)
            if result.shape == (2,) and result.dtype == torch.int64 and result.is_cuda:
                # A delayed creation-stream initialization must precede reset
                # and add on the different request stream, even when first use
                # begins immediately after create_session returns.
                torch.cuda._sleep(50_000_000)
                result.fill_(123456789)
                initialized.record(creator)
                injected.append(result)
            return result

        with torch.cuda.stream(creator), monkeypatch.context() as patch:
            patch.setattr(torch, "empty", delayed_counter)
            session = backend.create_session(210)
        assert len(injected) == 1 and not initialized.query()
        with torch.cuda.stream(caller):
            backend.prefill(session, tokens(native_model, 129))
            backend.extend(session, tokens(native_model, 33, 1))
        assert initialized.query()
        metrics = backend.session_metrics(session)
        bytes_per_token = 2 * 2 * 128 * torch.bfloat16.itemsize
        assert metrics["prefix_main_kv_host_to_device_bytes"] == 4 * (47 + 94) * bytes_per_token
        assert metrics["candidate_main_kv_host_to_device_bytes"] == 4 * 129 * bytes_per_token


@torch.inference_mode()
def test_cuda_shared_dense_partial_h2d_failure_counts_only_submitted_record(
    native_model, monkeypatch
):
    with allocated(native_model, "dense_prefetch") as (backend, _):
        session = backend.create_session(210)
        backend.prefill(session, tokens(native_model, 129))
        source_storage = session.buffers["values"].untyped_storage().data_ptr()
        original = torch.Tensor.copy_

        def fail_value_fetch(target, source, *args, **kwargs):
            if (
                target.is_cuda
                and source.device.type == "cpu"
                and source.untyped_storage().data_ptr() == source_storage
            ):
                raise RuntimeError("injected second prefetch record")
            return original(target, source, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", fail_value_fetch)
            with pytest.raises(RuntimeError, match="second prefetch record"):
                backend.extend(session, tokens(native_model, 17, 1))
        metrics = backend.session_metrics(session)
        assert metrics["transfer_metrics_status"] == "failed"
        assert (
            metrics["candidate_main_kv_host_to_device_bytes"]
            == 129 * 2 * 128 * torch.bfloat16.itemsize
        )
        assert metrics["candidate_main_kv_device_to_host_bytes"] == 0
        assert session.length == session.indexer_cache.length == 129
        assert backend.resources.staging.active_lease is None
        backend.extend(session, tokens(native_model, 17, 1))
        assert backend.session_metrics(session)["transfer_metrics_status"] == "complete"
