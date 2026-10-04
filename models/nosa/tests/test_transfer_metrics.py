"""CPU reference request phases never pretend that host/device transfers ran."""

import pytest

from models.nosa.tests.test_serving_resources import SCHEMES, allocated_backend, tokens


@pytest.mark.parametrize("scheme", SCHEMES)
def test_cpu_transfer_metrics_preserve_request_phases_without_gpu_payload(monkeypatch, scheme):
    with allocated_backend(scheme) as (backend, _):
        session = backend.create_session(160)
        assert backend.session_metrics(session)["transfer_metrics_status"] == "idle"
        assert session._transfer_metrics.storage_bytes == 0
        backend.prefill(session, tokens(79))
        assert backend.session_metrics(session)["transfer_metrics_status"] == "prefill_ready"
        backend.extend(session, tokens(33, 1))
        complete = backend.session_metrics(session)
        assert complete["transfer_metrics_status"] == "complete"
        assert all(value == 0 for name, value in complete.items() if name.endswith("_bytes"))
        backend.truncate(session, 79)
        assert backend.session_metrics(session) == complete
        with pytest.raises(ValueError):
            backend.extend(session, tokens(82))
        assert backend.session_metrics(session) == complete

        def fail(*args, **kwargs):
            raise RuntimeError("injected transfer failure")

        with monkeypatch.context() as patch:
            patch.setattr(backend.model.model.layers[2], "forward", fail)
            with pytest.raises(RuntimeError, match="injected transfer"):
                backend.extend(session, tokens(17, 2))
        assert backend.session_metrics(session)["transfer_metrics_status"] == "failed"
        backend.extend(session, tokens(17, 2))
        assert backend.session_metrics(session)["transfer_metrics_status"] == "complete"
        backend.release_session(session)
        assert session._transfer_metrics is None
