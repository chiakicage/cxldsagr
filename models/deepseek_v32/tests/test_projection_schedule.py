from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v32.projections import CheckpointAttention


class Tensor:
    device = "cuda:0"

    def __init__(self, count=1):
        self.count = count
        self.recorded = []

    def __len__(self):
        return self.count

    def record_stream(self, stream):
        self.recorded.append(stream)


@pytest.mark.parametrize("failure", [None, "body", "side", "body_and_join"])
def test_projection_branch_join_and_failure_ownership(monkeypatch, failure):
    primary, cleanup = RuntimeError("execution"), RuntimeError("join")
    waits = []

    def join(side):
        waits.append(side)
        if failure == "body_and_join":
            raise cleanup

    current = SimpleNamespace(wait_stream=join)
    side = SimpleNamespace(wait_stream=lambda stream: waits.append(stream))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    owner = object.__new__(CheckpointAttention)
    owner._q1_projection_stream = side
    kv, latent, k_pe, index_k, x = (Tensor() for _ in range(5))

    def project_kv(value):
        assert value is x
        if failure == "side":
            raise primary
        return kv, latent, k_pe

    owner._project_kv = project_kv
    owner._project_index_k = lambda value: index_k
    caught = None
    try:
        with owner._projection_branches(x) as result:
            assert result == (kv, latent, k_pe, index_k)
            if failure in ("body", "body_and_join"):
                raise primary
    except BaseException as error:  # noqa: BLE001 -- compare the actual exception objects
        caught = error
    assert waits == [current, side]
    assert x.recorded == [side]
    assert owner._q1_projection_stream is side
    if failure == "body_and_join":
        assert isinstance(caught, BaseExceptionGroup)
        assert caught.exceptions == (primary, cleanup)
    elif failure:
        assert caught is primary
    else:
        assert caught is None
        assert all(value.recorded == [current] for value in (kv, latent, index_k))


@pytest.mark.parametrize("count,capture", [(2, True), (1, False)])
def test_projection_schedule_leaves_other_calls_on_caller(monkeypatch, count, capture):
    owner = object.__new__(CheckpointAttention)
    owner._q1_projection_stream = object()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    with owner._projection_branches(Tensor(count)) as result:
        assert result is None
