"""Submission, tag invalidation and completion errors retain all failure evidence."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import tvm_ffi

from models.attention_contracts import BlockSelection
from operators.nosa.attention.offload import _fused, api


def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [item for child in error.exceptions for item in leaves(child)]
    return [error]


@pytest.mark.parametrize("invalidate_fails", [False, True])
@pytest.mark.parametrize("record_fails", [False, True])
def test_failed_submission_preserves_cleanup_errors_and_prevents_unsafe_reuse(
    monkeypatch, invalidate_fails, record_fails
):
    workspace = api.NosaFetchWorkspace(
        65, 1, 128, device="cpu", dtype=torch.bfloat16, bounded=True, max_queries=1
    )
    q = torch.empty(1, 16, 128, dtype=torch.bfloat16)
    host = torch.empty(64, 1, 128, dtype=torch.bfloat16)
    suffix = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    selection = BlockSelection(torch.tensor([[[0]]], dtype=torch.int32), 64)
    tags = torch.ones(2, 1, dtype=torch.int64)
    body = ValueError("submission failure")
    invalidate = RuntimeError("tag reset failure")
    record = RuntimeError("completion record failure")
    calls = []

    def prepare(*_args):
        calls.append("submit")
        raise body

    def finish(_stream):
        calls.append("record")
        if record_fails:
            raise record

    original_zero = torch.Tensor.zero_

    def zero(tensor):
        if tensor is tags:
            calls.append("invalidate")
            if invalidate_fails:
                raise invalidate
        return original_zero(tensor)

    workspace._last_done = SimpleNamespace(record=finish, synchronize=lambda: calls.append("sync"))
    monkeypatch.setattr(workspace, "_validate", lambda *_: None)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    monkeypatch.setattr(torch.Tensor, "zero_", zero)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *_: SimpleNamespace(wait_event=lambda *_: None)
    )
    monkeypatch.setattr(tvm_ffi, "use_torch_stream", nullcontext)
    monkeypatch.setattr(api, "_module", lambda: SimpleNamespace(prepare_cached=prepare))
    monkeypatch.setattr(_fused, "_module", lambda: object())
    arguments = (q, selection, host, host, suffix, suffix, None, 64)
    with pytest.raises(BaseException) as caught:
        workspace.run(*arguments, cache_tags=tags, cache_owner=1)
    expected = [
        body,
        *([invalidate] if invalidate_fails else []),
        *([record] if record_fails else []),
    ]
    assert leaves(caught.value) == expected
    assert calls == ["submit", "invalidate", "record"]
    assert workspace.failed is (invalidate_fails or record_fails)
    if workspace.failed:
        assert any(alias is host for alias in workspace._failed_aliases)
        assert any(alias is q for alias in workspace._failed_aliases)
        with pytest.raises(RuntimeError, match="poisoned"):
            workspace.run(*arguments, cache_tags=tags, cache_owner=1)
        with pytest.raises(RuntimeError, match="poisoned"):
            workspace.reserve(1)
        with pytest.raises(RuntimeError, match="unknown|poisoned"):
            workspace.synchronize()
    else:
        assert not tags.any()
        workspace.synchronize()
        assert calls[-1] == "sync"
