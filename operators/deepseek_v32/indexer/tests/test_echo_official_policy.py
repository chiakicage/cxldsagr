"""Policy differentials against AST-extracted pinned ECHO implementation.

The hint check isolates the finite-input update expression. Causal/nonfinite
padding is the separately disclosed local guard, not a claim that the official
unclean logits buffer has defined values outside its causal range.
"""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from cache.sparse_token_pool import MISSING, SharedSparseTokenPool
from operators.deepseek_v32.indexer import cache_ops
from operators.deepseek_v32.indexer.echo import UPSTREAM_REVISION
from operators.deepseek_v32.indexer.prefetch_hint import update_prefetch_hint
from operators.deepseek_v32.indexer.tests.test_echo_cache_ops import _official_function

ROOT = Path(__file__).resolve().parents[4]
OFFICIAL = ROOT / "3rdparty/ECHO"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable; scripts/run_tests.sh gpu requires Hopper before collection",
)


def require_reference_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.fail("explicit official-policy differential requires Hopper SM90")
    revision = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == UPSTREAM_REVISION
    paths = [
        "sglang/python/sglang/srt/mem_cache/memory_pool_host.py",
        "sglang/python/sglang/srt/mem_cache/recall_ops.py",
        "sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py",
        "sglang/sgl-kernel/csrc/elementwise/topk.cu",
    ]
    assert not subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "status", "--porcelain", "--", *paths], text=True
    ).strip(), "official helper sources must be unchanged from the pinned commit"


@pytest.fixture(scope="module")
def official_graph_free(tmp_path_factory):
    """Build the original bounded selector and load actual graph-path helpers."""
    require_reference_gpu()
    import triton
    import triton.language as tl
    from torch.utils.cpp_extension import load

    directory = tmp_path_factory.mktemp("echo_official_argmin")
    binding = directory / "binding.cpp"
    binding.write_text(
        "#include <torch/extension.h>\n"
        "void fast_argmin_bounded_interface(at::Tensor, at::Tensor, at::Tensor);\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        '  m.def("argmin", &fast_argmin_bounded_interface);\n'
        "}\n"
    )
    native = load(
        name="echo_pinned_official_argmin_policy_test",
        sources=[
            str(binding),
            str(OFFICIAL / "sglang/sgl-kernel/csrc/elementwise/topk.cu"),
        ],
        build_directory=str(directory),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--extended-lambda", "-gencode=arch=compute_90,code=sm_90"],
    )
    namespace = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "I32_MAX": MISSING,
        "Optional": Optional,
        "_fast_argmin_bounded": native.argmin,
    }
    recall = OFFICIAL / "sglang/python/sglang/srt/mem_cache/recall_ops.py"
    for name in (
        "_protect_kernel",
        "protect",
        "_free_update_kernel",
        "free_update",
        "get_free_loc",
    ):
        _official_function(recall, name, namespace)
    return _official_function(
        OFFICIAL / "sglang/python/sglang/srt/mem_cache/memory_pool_host.py",
        "free_device_pool_cuda_graph",
        namespace,
    )


@pytest.mark.parametrize("official_mode", ["torch", "cuda_graph"])
@pytest.mark.parametrize(
    "resident,protected,needed,tied",
    [
        (9, [], 0, False),
        (9, [0, 3], 0, False),
        (9, [0], 1, False),
        (9, [0, 3, 7], 3, False),
        (9, list(range(9)), 0, False),
        (9, [0, 3, 7], 3, True),
        (6, [0], 4, False),
    ],
)
def test_cuda_selected_protection_and_victims_match_official_free_helper(
    resident, protected, needed, tied, official_mode, official_graph_free
):
    require_reference_gpu()
    free_device_pool = _official_function(
        OFFICIAL / "sglang/python/sglang/srt/mem_cache/memory_pool_host.py",
        "free_device_pool",
        {"torch": torch, "Optional": Optional, "I32_MAX": MISSING},
    )
    pool = SharedSparseTokenPool(64, 4, 1, 9, device="cuda", metadata_ops=cache_ops)
    session = pool.allocate_session(64)
    cache = session.layer(0)
    layer = pool.layers[0]
    host_ids = 16 + torch.arange(resident, device="cuda", dtype=torch.int64)
    physical = torch.arange(1, resident + 1, device="cuda", dtype=torch.int64)
    layer.host_to_device[host_ids] = physical.int()
    layer.device_to_host[physical] = host_ids
    layer.free[physical] = False
    priorities = torch.arange(resident, device="cuda", dtype=torch.int64)
    if tied:
        priorities //= 3
    layer.priority[physical] = priorities
    layer.clock = 20
    layer.clock_tensor.fill_(20)
    protected_ids = 16 + torch.tensor(protected, device="cuda", dtype=torch.int64)
    protected_slots = layer.host_to_device[protected_ids].long()
    original_map = layer.host_to_device.clone()
    original_priority = layer.priority.clone()
    official_freed = []

    def note_freed(ids, size=None):
        official_freed.append(ids.clone() if size is None else ids[: int(size)].clone())

    reference = SimpleNamespace(
        device_pool_priority=layer.priority.int().unsqueeze(0).clone(),
        fifo_counter=torch.tensor([20], dtype=torch.int32, device="cuda"),
        device_token_to_host=layer.device_to_host.unsqueeze(0).clone(),
        host_token_to_device=layer.host_to_device.unsqueeze(0).clone(),
        device_pool_allocator=[SimpleNamespace(free=note_freed)],
        device_pool_loc_small_priority=torch.empty(10, device="cuda", dtype=torch.int32),
        free_index_device_buf=torch.empty(9, device="cuda", dtype=torch.int32),
    )
    deficit = max(0, needed - (9 - resident))
    # The reference performs its own protection stamp, excludes existing free
    # slots, chooses victims with torch.topk, and invalidates both global maps.
    if official_mode == "torch":
        free_device_pool(reference, deficit, 0, protected_slots)
    else:
        # The pinned graph helper's Triton protection kernel reads its complete
        # 128-element tile; official top-k tables are padded to this extent.
        padded_selection = torch.full((1, 128), MISSING, device="cuda", dtype=torch.int32)
        padded_selection[0, : len(protected_slots)] = protected_slots.int()
        official_graph_free(
            reference,
            torch.tensor([deficit], device="cuda", dtype=torch.int32),
            0,
            padded_selection,
        )
    with cache.operation():
        pool.protect(0, protected_ids)
        allocated = cache._available_slots(protected_ids, needed)
    assert len(allocated) == needed
    lost = torch.where((original_map != MISSING) & (layer.host_to_device == MISSING))[0]
    reference_lost = torch.where(
        (original_map != MISSING) & (reference.host_token_to_device[0] == MISSING)
    )[0]
    assert len(lost) == len(reference_lost) == deficit
    assert len(official_freed) == 1
    assert not torch.isin(lost, protected_ids).any()
    assert pool.layers[0].clock == int(reference.fifo_counter[0])
    if not tied:
        torch.testing.assert_close(layer.host_to_device, reference.host_token_to_device[0])
        torch.testing.assert_close(layer.device_to_host, reference.device_token_to_host[0])
        torch.testing.assert_close(
            layer.priority, reference.device_pool_priority[0].long(), rtol=0, atol=0
        )
    else:
        # torch.topk and stable argsort may choose different slots at a tied
        # boundary. The victim priority multiset, count, and protected ownership
        # must still agree with the actual official implementation.
        local_priorities = original_priority[original_map[lost].long()].sort().values
        official_priorities = original_priority[original_map[reference_lost].long()].sort().values
        torch.testing.assert_close(local_priorities, official_priorities)
    live = torch.where(layer.device_to_host != MISSING)[0]
    torch.testing.assert_close(layer.host_to_device[layer.device_to_host[live]].long(), live)
    assert layer.priority[0] == MISSING and not layer.free[0]
    session.release()
    pool.close()


def _official_hint_statements(path):
    tree = ast.parse(path.read_text())
    # Execute the exact upstream offset initialization and request loop from
    # _get_topk_ragged, without importing the SGLang scheduler or model.
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_topk_ragged"
    )
    for parent in ast.walk(method):
        body = getattr(parent, "body", [])
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if (
                isinstance(node, ast.For)
                and "extend_logits_offsets" in ast.unparse(node)
                and "torch.mean" in ast.unparse(node)
            ):
                return body[index - 1 : index + 1]
    raise AssertionError(f"could not locate the real hint update in {path}")


@pytest.mark.parametrize("rows", [1, 2, 4, 7])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_cuda_finite_hint_update_matches_pinned_official_last_four_row_mean(rows, grad_enabled):
    require_reference_gpu()
    reference_path = OFFICIAL / "sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py"
    scores = torch.arange(rows * 13, device="cuda", dtype=torch.float32).reshape(rows, 13)
    scores = (scores.remainder(19) - 7) * 0.25
    official_self = SimpleNamespace(
        extend_logits_window=4,
        extend_logits_offsets=torch.full((1, 16), 999.0, device="cuda"),
    )
    local_self = SimpleNamespace(offset=torch.full((16,), 999.0, device="cuda"))
    namespace = {
        "torch": torch,
        "self": official_self,
        "forward_batch": SimpleNamespace(batch_size=1),
        "extend_seq_lens_cpu": [rows],
        "logits": scores,
    }
    module = ast.Module(body=_official_hint_statements(reference_path), type_ignores=[])
    exec(compile(module, str(reference_path), "exec"), namespace)  # noqa: S102 — pinned/read-only source differential
    # Call the production helper, covering its inference-native and explicit
    # grad-enabled reference dispatch without reproducing either formula.
    with torch.set_grad_enabled(grad_enabled):
        update_prefetch_hint(scores, local_self.offset)
    torch.testing.assert_close(local_self.offset[0], official_self.extend_logits_offsets[0, 0])
    assert local_self.offset[1:].eq(999).all()  # Only offset zero is consumed locally.
