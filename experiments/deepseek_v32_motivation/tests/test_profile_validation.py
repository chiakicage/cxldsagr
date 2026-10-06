"""Captured requests must observe the validator selected before instrumentation."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from experiments.deepseek_v32_mfu.src import operator_instrumentation
from experiments.deepseek_v32_motivation.src.profile import InstrumentServing, Scopes
from serving import token_validation
from serving.persistent import PersistentGRRunner
from serving.tests.test_persistent import Backend, SharedBackend, request


@pytest.mark.parametrize("body_failure", [False, True])
@pytest.mark.parametrize("stop_failure", [False, True])
def test_capture_completion_preserves_body_and_stop_errors(body_failure, stop_failure):
    from experiments.deepseek_v32_motivation.src.profile import capture_range

    body, stop = KeyboardInterrupt("profile interrupted"), OSError("capture stop failed")
    calls = []

    def close():
        calls.append("stop")
        if stop_failure:
            raise stop

    runtime = SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("start"), cudaProfilerStop=close
    )

    def execute():
        with capture_range(runtime):
            calls.append("body")
            if body_failure:
                raise body

    if body_failure and stop_failure:
        with pytest.raises(BaseExceptionGroup) as failed:
            execute()
        assert failed.value.exceptions == (body, stop)
    elif body_failure or stop_failure:
        expected = body if body_failure else stop
        with pytest.raises(type(expected)) as failed:
            execute()
        assert failed.value is expected
    else:
        execute()
    assert calls == ["start", "body", "stop"]


@pytest.mark.parametrize(
    ("failure_stage", "cleanup_failure", "reporting_failure", "backend_failure"),
    [
        (stage, cleanup, False, False)
        for stage in ("finalize", "capture_stop", "ledger_write")
        for cleanup in (False, True)
    ]
    + [("ledger_write", False, True, backend) for backend in (False, True)],
)
def test_graph_setup_failure_releases_runner_before_backend_close(
    tmp_path, monkeypatch, failure_stage, cleanup_failure, reporting_failure, backend_failure
):
    from evaluation import provenance
    from experiments.deepseek_v32_motivation.src import graph_instrumentation, profile
    from experiments.nosa_motivation.src import validation
    from models.deepseek_v32.execution import adapter
    from serving import persistent

    setup_error = KeyboardInterrupt(failure_stage)
    cleanup_error = OSError("runner cleanup failed")
    reporting_error = OSError("failure metadata write failed")
    backend_error = OSError("backend close failed")
    calls = []
    reference, output = tmp_path / "reference", tmp_path / "temporary"
    reference.mkdir()
    output.mkdir()
    config = {
        "enable_compute_graphs": True,
        "layers": 1,
        "chunk_size": 4,
        "sparse_pool_tokens": 8,
        "host_arena_tokens": 16,
        "workspace_query_tokens": 4,
    }
    checkpoint = {
        "path": str(reference),
        "files": {},
        "identity_boundary": "checkpoint path and shard stat inventory, not weights hashes",
    }
    workload = SimpleNamespace(manifest={"workload_sha256": "input"}, write=lambda *_: None)
    metadata = {"run_id": "reference", "config": config, "checkpoint": checkpoint}
    monkeypatch.setattr(validation, "reference_directory", lambda *_args, **_kwargs: reference)
    monkeypatch.setattr(profile, "load_reference", lambda *_: (metadata, workload, {}))
    monkeypatch.setattr(
        profile, "reference_provenance", lambda *_: {"reference_source_sha256": "reference"}
    )
    monkeypatch.setattr(profile.tempfile, "mkdtemp", lambda **_: str(output))
    monkeypatch.setattr(profile, "snapshot_sources", lambda *_: "profile")
    monkeypatch.setattr(profile, "precision_settings", dict)
    monkeypatch.setattr(provenance, "_git", lambda *_: "fixture")
    monkeypatch.setattr(provenance, "backend_provenance", dict)
    monkeypatch.setattr(profile.measure, "configure_precision", lambda *_: None)
    monkeypatch.setattr(profile.measure, "resource_limits", lambda *_: {})
    monkeypatch.setattr(profile.measure, "warmup", lambda *_: {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: SimpleNamespace(
            name="fixture", uuid="fixture", total_memory=1, multi_processor_count=1
        ),
    )

    def stop_capture():
        calls.append("capture_stop")
        if failure_stage == "capture_stop":
            raise setup_error

    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: SimpleNamespace(
            cudaProfilerStart=lambda: calls.append("capture_start"),
            cudaProfilerStop=stop_capture,
        ),
    )

    class BackendFixture:
        owner = None

        def __init__(self, *_args, **_kwargs):
            pass

        def synchronize(self):
            pass

        def configure_scheme(self, _scheme):
            pass

        def close(self):
            calls.append("backend_close")
            assert self.owner is None
            if backend_failure:
                raise backend_error

    class RunnerFixture:
        def __init__(self, backend, **_kwargs):
            self.backend = backend
            backend.owner = self
            calls.append("runner_created")

        def close(self):
            calls.append("runner_close")
            self.backend.owner = None
            if cleanup_failure:
                raise cleanup_error

    class GraphFixture:
        def __init__(self, _backend):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def finalize(self):
            calls.append("finalize")
            if failure_stage == "finalize":
                raise setup_error
            return {}

    original_write = profile.measure.write_json

    def write_json(path, value):
        if path.name == "graph_capture_ledger.json":
            calls.append("ledger_write")
            if failure_stage == "ledger_write":
                raise setup_error
        if path.name == "metadata.json" and value["status"] == "failed" and reporting_failure:
            raise reporting_error
        original_write(path, value)

    monkeypatch.setattr(adapter, "DeepSeekServingBackend", BackendFixture)
    monkeypatch.setattr(persistent, "PersistentGRRunner", RunnerFixture)
    monkeypatch.setattr(graph_instrumentation, "CaptureGraphOperators", GraphFixture)
    monkeypatch.setattr(profile.measure, "write_json", write_json)
    group_expected = cleanup_failure or reporting_failure or backend_failure
    with pytest.raises(BaseExceptionGroup if group_expected else KeyboardInterrupt) as failed:
        profile.main(
            [
                "--run-id",
                "setup_failure",
                "--reference-run",
                str(reference),
                "--output-dir",
                str(tmp_path / "published"),
                "--scheme",
                "echo",
                "--nsys",
            ]
        )
    propagated = failed.value
    if backend_failure:
        propagated, actual_backend_error = propagated.exceptions
        assert actual_backend_error is backend_error
    if reporting_failure:
        propagated, actual_reporting_error = propagated.exceptions
        assert actual_reporting_error is reporting_error
    if cleanup_failure:
        propagated, actual_cleanup_error = propagated.exceptions
        assert actual_cleanup_error is cleanup_error
    assert propagated is setup_error
    assert calls.count("runner_created") == calls.count("runner_close") == 1
    assert calls[-2:] == ["runner_close", "backend_close"]
    expected_status = "running" if reporting_failure else "failed"
    assert json.loads((output / "metadata.json").read_text())["status"] == expected_status
    assert not (tmp_path / "published").exists()


def test_profile_retains_formal_run_and_separate_numerical_reference(tmp_path):
    from experiments.deepseek_v32_motivation.src.profile import reference_provenance, sha

    formal, numerical = tmp_path / "bench", tmp_path / "check"
    for path in (formal, numerical):
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "run_id": path.name,
                    "source_sha256": "frozen_source",
                    "config": {"history_tokens": 65536},
                }
            )
        )
    result = reference_provenance(formal, numerical)
    assert result["reference_run_id"] == "bench"
    assert result["reference_run"] == str(formal)
    assert result["reference_metadata_sha256"] == sha(formal / "metadata.json")
    assert result["numerical_reference_run_id"] == "check"
    assert result["numerical_reference_run"] == str(numerical)
    assert result["numerical_reference_metadata_sha256"] == sha(numerical / "metadata.json")
    metadata = json.loads((numerical / "metadata.json").read_text())
    metadata["config"]["history_tokens"] = 4096
    (numerical / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="configurations differ"):
        reference_provenance(formal, numerical)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_profile_wraps_selected_validator_once_and_restores_it(monkeypatch, native, invalid):
    monkeypatch.setattr(token_validation, "prepare", lambda: token_validation.reference)
    monkeypatch.setattr(token_validation, "runtime_info", lambda: {"backend": "cpython_native"})
    # Validation runs real runner code. Matrix instrumentation is outside this
    # CPU test; the full InstrumentServing context still installs/restores hooks.
    monkeypatch.setattr(operator_instrumentation, "InstrumentOperators", lambda *_: nullcontext())
    backend = Backend()
    backend._forward = lambda *args, **kwargs: None
    backend.extend_candidate = backend.extend
    backend.session_metrics = lambda *_: {}
    backend.attentions, backend.blocks = (), ()
    backend.head_weight = torch.empty(0)
    scopes = Scopes("echo", "cold", 1, nvtx=False)
    with PersistentGRRunner(
        backend,
        hbm_budget_bytes=64,
        dram_budget_bytes=64,
        native_token_validation=native,
    ) as runner:
        original = runner._validate
        original_local = "_validate" in vars(runner)
        assert not original_local
        assert runner.token_validation_identity["requested_native"] is native
        instrument = InstrumentServing(backend, scopes, runner)
        value = request()
        if invalid:
            value["input_ids"] = [True, 2, 3, 4]
        expected = pytest.raises(ValueError) if invalid else nullcontext()
        with expected, instrument:
            runner._validate(value)
        assert len(scopes.calls) == 1
        assert scopes.calls[0]["stage"] == "request_validation"
        assert scopes.calls[0]["segment"] == "admission"
        assert bool(scopes.calls[0].get("raised")) is invalid
        assert not scopes.active
        assert instrument.targets.count("PersistentGRRunner._validate") == 1
        assert runner._validate.__func__ is original.__func__
        assert runner._validate.__self__ is runner
        assert ("_validate" in vars(runner)) is original_local
        assert runner._token_ids_valid is token_validation.reference
        runner._validate(request())
        assert len(scopes.calls) == 1


def test_late_instrumentation_observes_existing_diagnostics_and_preserves_owner(monkeypatch):
    from executor.adapters import BackendAdapter

    monkeypatch.setattr(operator_instrumentation, "InstrumentOperators", lambda *_: nullcontext())

    class ProfileBackend(SharedBackend):
        def __init__(self):
            super().__init__()
            self._forward = lambda *args, **kwargs: None
            self.attentions, self.blocks = (), ()
            self.head_weight = torch.empty(0)
            self.metric_reads = 0
            self.authorized_calls = []
            self.session_metrics = self.read_metrics

        def runtime_driver(self, policy):
            return BackendAdapter(
                self,
                shared=True,
                candidate_mode="gpu_transient",
                session_length=lambda session: len(session["tokens"]),
                chunk_size=self.max_seq_len,
                diagnostics=lambda session: self.session_metrics(session),
                owner_aware=True,
            )

        def authorize(self, stage, owner):
            assert owner is self.owner and owner is not None
            self.authorized_calls.append(stage)

        def allocate_shared(self, plan, *, owner=None):
            self.authorize("allocate", owner)
            return super().allocate_shared(plan)

        def create_session(self, capacity, *, owner=None):
            self.authorize("create", owner)
            return super().create_session(capacity)

        def prefill(self, session, ids, *, owner=None):
            self.authorize("prefill", owner)
            return super().prefill(session, ids)

        def extend_candidate(self, session, ids, *, owner=None):
            self.authorize("candidate", owner)
            history = len(session["tokens"])
            result = super().extend(session, ids)
            super().truncate(session, history)
            return result

        def truncate(self, session, history, *, owner=None):
            self.authorize("truncate", owner)
            return super().truncate(session, history)

        def release_session(self, session, *, owner=None):
            self.authorize("release", owner)
            return super().release_session(session)

        def read_metrics(self, session):
            self.metric_reads += 1
            return {"retained_tokens": len(session["tokens"])}

    backend = ProfileBackend()
    scopes = Scopes("echo", "cold", 1, nvtx=False)
    with PersistentGRRunner(backend, hbm_budget_bytes=80, dram_budget_bytes=64) as runner:
        assert backend.metric_reads == 0
        with InstrumentServing(backend, scopes, runner) as instrument:
            result = runner.execute(request())
        assert result.hidden.tolist() == [[6], [10]]
        assert (
            backend.metric_reads == 2
        )  # One diagnostic history read and the existing runner read.
        assert instrument.segment_counters == {
            "history": {"retained_tokens": 2},
            "candidate": {"retained_tokens": 2},
        }
        for stage in ("history_prefill", "candidate_extend", "candidate_counters"):
            assert sum(call["stage"] == stage for call in scopes.calls) == 1
        assert not scopes.active
    assert backend.authorized_calls == [
        "allocate",
        "create",
        "prefill",
        "candidate",
        "truncate",
        "release",
    ]


@pytest.mark.parametrize("max_ctas,counted", [(None, False), (80, True), (128, True)])
def test_gather_instrumentation_preserves_counted_dispatch_without_scalar_read(
    monkeypatch, max_ctas, counted
):
    from torch.utils._python_dispatch import TorchDispatchMode

    from operators.common import kv_transfer

    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if function is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("profile must not read the device record count")
            return function(*args, **(kwargs or {}))

    forwarded = []
    expected_result = object()

    def gather(host, device, host_ids, device_ids, *, max_ctas=None, valid_count=None):
        forwarded.append((host, device, host_ids, device_ids, max_ctas, valid_count))
        return expected_result

    monkeypatch.setattr(kv_transfer, "gather_host_records", gather)
    monkeypatch.setattr(operator_instrumentation, "InstrumentOperators", lambda *_: nullcontext())
    backend = Backend()
    backend._forward = lambda *args, **kwargs: None
    backend.extend_candidate = backend.extend
    backend.session_metrics = lambda *_: {}
    backend.attentions, backend.blocks = (), ()
    backend.head_weight = torch.empty(0)
    host, device = torch.empty((64, 576), dtype=torch.bfloat16), torch.empty((64, 576))
    host_ids, device_ids = torch.empty(64, dtype=torch.int64), torch.empty(64, dtype=torch.int64)
    count = torch.tensor(3, dtype=torch.int32) if counted else None
    scopes = Scopes("dense_prefetch" if max_ctas == 80 else "echo", "revisit", 1, nvtx=False)
    with (
        PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner,
        InstrumentServing(backend, scopes, runner),
        RejectScalarRead(),
    ):
        result = kv_transfer.gather_host_records(
            host, device, host_ids, device_ids, max_ctas=max_ctas, valid_count=count
        )
    assert result is expected_result
    assert len(forwarded) == 1
    for actual, expected in zip(
        forwarded[0], (host, device, host_ids, device_ids, max_ctas, count)
    ):
        assert actual is expected
    assert kv_transfer.gather_host_records is gather
    call = next(row for row in scopes.calls if row["stage"] == "host_gather")
    assert call["id_buffer_capacity"] == 64 and call["record_bytes"] == 1152
    assert call["max_ctas"] == max_ctas
    assert call["records"] == (None if counted else 64)
    assert call["requested_bytes"] == (None if counted else 64 * 1152)
    assert call["count_source"] == ("device_valid_count" if counted else "id_buffer_length")


def test_dma_instrumentation_preserves_exact_contiguous_addresses_without_scalar_read(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from operators.common import kv_transfer

    class RejectScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            if function is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("DMA annotations must not read a device scalar")
            return function(*args, **(kwargs or {}))

    forwarded = []
    expected_result = object()

    def copy(host, device, *, host_start, device_start, count):
        forwarded.append((host, device, host_start, device_start, count))
        return expected_result

    monkeypatch.setattr(kv_transfer, "copy_host_records_async", copy)
    monkeypatch.setattr(operator_instrumentation, "InstrumentOperators", lambda *_: nullcontext())
    backend = Backend()
    backend._forward = lambda *args, **kwargs: None
    backend.extend_candidate = backend.extend
    backend.session_metrics = lambda *_: {}
    backend.attentions, backend.blocks = (), ()
    backend.head_weight = torch.empty(0)
    host, device = (torch.empty((64, 576), dtype=torch.bfloat16) for _ in range(2))
    scopes = Scopes("dense_prefetch", "revisit", 1, nvtx=False)
    with (
        PersistentGRRunner(backend, hbm_budget_bytes=64, dram_budget_bytes=64) as runner,
        InstrumentServing(backend, scopes, runner),
        RejectScalarRead(),
    ):
        result = kv_transfer.copy_host_records_async(
            host, device, host_start=4, device_start=1, count=8
        )
    assert result is expected_result
    assert len(forwarded) == 1
    assert forwarded[0][0] is host and forwarded[0][1] is device
    assert forwarded[0][2:] == (4, 1, 8)
    assert kv_transfer.copy_host_records_async is copy
    call = next(row for row in scopes.calls if row["stage"] == "host_dma")
    assert call["records"] == 8 and call["record_bytes"] == 1152
    assert call["requested_bytes"] == 8 * 1152
    assert call["host_address_start"] == host.data_ptr() + 4 * 1152
    assert call["host_address_end"] == host.data_ptr() + 12 * 1152
    assert call["device_address_start"] == device.data_ptr() + 1152
    assert call["device_address_end"] == device.data_ptr() + 9 * 1152
    assert call["host_contiguous"] and call["device_contiguous"]
    assert call["transfer_implementation"] == "cudaMemcpyAsync"
