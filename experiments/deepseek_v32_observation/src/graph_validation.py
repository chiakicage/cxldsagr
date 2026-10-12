"""Numerical acceptance of the static CUDA Graph used for API observation."""

from dataclasses import fields, is_dataclass

import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if is_dataclass(value):
        return {field.name: snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    return value


def equal_output(actual, expected):
    if isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError("Graph output shape or dtype differs from eager output")
        # Compare bits, including FP8 and -inf score tails. The static fixture
        # uses unchanged inputs, so graph execution should be deterministic.
        if not torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        ):
            raise AssertionError("Static graph output bits differ from eager output")
        return
    if is_dataclass(actual):
        actual = {field.name: getattr(actual, field.name) for field in fields(actual)}
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("Graph output fields differ")
        for key in expected:
            equal_output(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("Graph output arity differs")
        for item, reference in zip(actual, expected, strict=True):
            equal_output(item, reference)
    elif actual != expected:
        raise AssertionError("Graph output scalar metadata differs")


def validate_graph(call, *, transform=None, before_call=None):
    if transform is None:
        transform = lambda output: output
    torch.cuda.synchronize()
    if before_call is not None:
        before_call()
    expected = snapshot(transform(call()))
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if before_call is not None:
            before_call()
        call()
    stream.synchronize()
    if before_call is not None:
        before_call()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    stream.synchronize()
    for _ in range(3):
        if before_call is not None:
            before_call()
            torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        equal_output(transform(output), expected)
    del graph, output, expected
    return {
        "passed": True,
        "replays": 3,
        "comparison": "full static graph output bits vs numerically accepted eager fixture",
    }
