"""Native 33/64 selection, dtype ties, pool layouts, and joint dispatch."""

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture(autouse=True)
def native_backend(monkeypatch):
    monkeypatch.setenv("CXLDSAGR_SM90_BACKEND", "native")


def _pool(cis, blocks):
    padded = torch.nn.functional.pad(cis.T.float(), (1, 4 * blocks - len(cis)), value=-torch.inf)
    return padded.unfold(-1, 5, 4).amax(-1).T.to(cis.dtype)


def _oracle(work, cis, positions, pooled_cis=None):
    """Two independent stable full sorts over exactly the supplied storage."""
    work, cis, positions = work.cpu(), cis.cpu(), positions.cpu()
    rows, heads = len(positions), cis.shape[1]
    blocks = work.shape[1]
    last = positions[:, None, None] // 64
    block = torch.arange(blocks)
    qa = work.reshape(rows, heads, blocks)
    first = qa.argsort(dim=-1, descending=True, stable=True)[..., :33]
    raw = _pool(cis, blocks).to(work.dtype)
    if pooled_cis is not None:
        raw[: len(pooled_cis)] = pooled_cis.cpu().to(work.dtype)
    shared = raw.T[None].expand_as(qa).clone()
    mandatory = (block == 0) | (last <= block + 16)
    shared.masked_fill_(mandatory, torch.inf).masked_fill_(block > last, -torch.inf)
    shared.scatter_(-1, first, torch.where(first <= last, torch.inf, -torch.inf).to(work.dtype))
    selected = shared.argsort(dim=-1, descending=True, stable=True)[..., :64].sort(-1).values
    return selected.masked_fill(selected > last, -1)


def _prefix_ranking(pool, first):
    """Independent stable BF16 ordering, encoded in the native candidate format."""
    bits = pool[:first].view(torch.int16).to(torch.int32) & 65535
    bits = torch.where((bits & 32767) == 0, 0, bits)
    keys = torch.where(bits & 32768 != 0, bits ^ 65535, bits ^ 32768).T
    ids = keys.argsort(dim=-1, descending=True, stable=True)[:, :64].sort(-1).values
    return ((keys.gather(1, ids) << 12) | (4095 - ids)).to(torch.int32)


def _inputs(blocks, dtype, *, heads=2, rows=5, contiguous=False):
    count = blocks * 4 - 2
    generator = torch.Generator().manual_seed(1901 + blocks)
    choices = torch.tensor([-torch.inf, -2.0, -0.0, 0.0, 1.0, 2.0, torch.inf])
    work = choices[torch.randint(len(choices), (rows, heads, blocks), generator=generator)].to(
        dtype
    )
    cis = choices[torch.randint(len(choices), (count, heads), generator=generator)].to(dtype)
    if contiguous:
        positions = torch.arange(blocks * 64 - rows, blocks * 64)
    else:
        positions = torch.tensor([0, 32 * 64, 63 * 64, (blocks - 2) * 64, blocks * 64 - 1])
    last = positions[:, None, None] // 64
    block = torch.arange(blocks)
    mandatory = (block == 0) | ((block <= last) & (last <= block + 16))
    work.masked_fill_(mandatory, torch.inf).masked_fill_(block > last, -torch.inf)
    return work.flatten(0, 1), cis, positions


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "blocks", [65, 128, 129, 256, 257, 512, 513, 1024, 1040, 1056, 1057, 2048, 2049, 4096]
)
@pytest.mark.parametrize("cached", [False, True])
@torch.inference_mode()
def test_native_selection_exact_sort_ties_masks_and_launch_boundaries(dtype, blocks, cached):
    from operators.nosa.indexer._selection_cuda import select_pooled_blocks

    work, cis, positions = _inputs(blocks, dtype)
    pool = _pool(cis, blocks)[: blocks - 2] if cached else None
    expected = _oracle(work, cis, positions, pool)
    ids, valid = select_pooled_blocks(
        work.cuda(),
        cis.cuda(),
        positions.to(device="cuda", dtype=torch.int32),
        0,
        len(positions),
        cis.shape[1],
        pooled_cis=pool.cuda() if cached else None,
        return_valid_mask=True,
    )
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(valid.cpu(), expected >= 0, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cis_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["strided", "broadcast", "partial_pool", "empty_pool"])
@torch.inference_mode()
def test_native_selection_cis_pool_precision_strides_and_scalar_positions(dtype, cis_dtype, layout):
    from operators.nosa.indexer._selection_cuda import select_pooled_blocks

    blocks = 1040
    work, cis, positions = _inputs(blocks, dtype, contiguous=True)
    # Dense close values expose incorrectly rounded mixed-precision ordering.
    cis = (torch.arange(len(cis), dtype=torch.float32)[:, None].expand(-1, 2) * 0.00001).to(
        cis_dtype
    )
    pool = _pool(cis, blocks)[: blocks - 2]
    if layout == "strided":
        cis_storage = torch.empty(len(cis) * 2, 4, dtype=cis_dtype, device="cuda")
        cis_cuda = cis_storage[::2, ::2].copy_(cis)
        pool_storage = torch.empty(len(pool) * 2, 4, dtype=cis_dtype, device="cuda")
        pool_cuda = pool_storage[::2, ::2].copy_(pool)
    elif layout == "broadcast":
        cis_cuda = cis[:, :1].cuda().expand(-1, 2)
        pool_cuda = pool[:1, :1].cuda().expand(len(pool), 2)
    else:
        cis_cuda = cis.cuda()
        pool_cuda = pool[: 31 if layout == "partial_pool" else 0].cuda()
    expected = _oracle(work, cis_cuda, positions, pool_cuda)
    ids = select_pooled_blocks(
        work.cuda(),
        cis_cuda,
        None,
        int(positions[0]),
        len(positions),
        2,
        pooled_cis=pool_cuda,
        return_valid_mask=False,
    )
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)


@requires_cuda
@torch.inference_mode()
def test_native_selection_nondefault_stream_graph_and_empty_rows():
    from operators.nosa.indexer._selection_cuda import select_pooled_blocks

    work, cis, positions = _inputs(1040, torch.bfloat16)
    pool = _pool(cis, 1040)[:1038]
    expected = _oracle(work, cis, positions, pool)
    work, cis, positions, pool = (tensor.cuda() for tensor in (work, cis, positions, pool))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):

        def call():
            return select_pooled_blocks(
                work, cis, positions, 0, 5, 2, pooled_cis=pool, return_valid_mask=True
            )

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ids, valid = call()
        ids.fill_(-99)
        valid.zero_()
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(valid.cpu(), expected >= 0, rtol=0, atol=0)
    empty_ids, empty_mask = select_pooled_blocks(
        work[:0], cis, None, 0, 0, 2, pooled_cis=pool, return_valid_mask=True
    )
    assert empty_ids.shape == empty_mask.shape == (0, 2, 64)


@requires_cuda
@pytest.mark.parametrize(
    ("blocks", "start", "rows"),
    [
        (66, 4096, 128),
        (131, 8191, 128),
        (1040, 65535, 1024),
        (1040, 65472, 1088),
        (1040, 65408, 1152),
    ],
)
@pytest.mark.parametrize("distribution", ["mixed_infinities", "all_equal"])
@torch.inference_mode()
def test_native_shared_cis_prefix_ranking_exact_coverage_and_ties(
    blocks, start, rows, distribution
):
    from operators.nosa.indexer._selection_cuda import select_pooled_blocks

    generator = torch.Generator().manual_seed(3311 + blocks)
    choices = torch.tensor([-torch.inf, -2.0, -0.0, 0.0, 1.0, 2.0, torch.inf])
    qa = choices[torch.randint(len(choices), (rows, 2, blocks), generator=generator)].bfloat16()
    cis = choices[torch.randint(len(choices), (blocks * 4 - 2, 2), generator=generator)].bfloat16()
    if distribution == "all_equal":
        qa.zero_()
        cis.fill_(-torch.inf)
    positions = torch.arange(start, start + rows)
    last = positions[:, None, None] // 64
    block = torch.arange(blocks)
    qa.masked_fill_((block == 0) | (last <= block + 16), torch.inf)
    qa.masked_fill_(block > last, -torch.inf)
    work = qa.flatten(0, 1)
    pool = _pool(cis, blocks)[: blocks - 2]
    expected = _oracle(work, cis, positions, pool)
    ids, valid = select_pooled_blocks(
        work.cuda(),
        cis.cuda(),
        None,
        start,
        rows,
        2,
        pooled_cis=pool.cuda(),
        return_valid_mask=True,
    )
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(valid.cpu(), expected >= 0, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize(
    "layout", ["q_broadcast", "k_broadcast", "q_misaligned", "q_stride", "k_stride"]
)
@torch.inference_mode()
def test_joint_indexer_preserves_score_layout_fallback(layout, monkeypatch):
    from operators.nosa.indexer import _indexer_cuda as joint
    from operators.nosa.indexer._scores_cuda import supports
    from operators.nosa.indexer.api import select_contiguous_blocks

    rows, heads, length = 128, 2, 32768
    count, blocks = length // 16 - 1, length // 64
    query = torch.zeros(rows, heads, 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros(count, heads, 128, device="cuda", dtype=query.dtype)
    if layout == "q_broadcast":
        query = query[:1, :, :1].expand_as(query)
    elif layout == "k_broadcast":
        keys = keys[:1, :1].expand_as(keys)
    elif layout == "q_misaligned":
        query = torch.zeros(query.numel() + 1, device="cuda", dtype=query.dtype)[1:].view_as(query)
    elif layout == "q_stride":
        query = torch.zeros(rows, heads, 16, 129, device="cuda", dtype=query.dtype)[..., :128]
    else:
        keys = torch.zeros(count, heads, 129, device="cuda", dtype=query.dtype)[..., :128]
    cis = (
        torch.arange(count, device="cuda", dtype=torch.float32)[:, None]
        .expand(-1, heads)
        .to(query.dtype)
    )
    pool = _pool(cis, blocks)[: blocks - 2]
    workspace = torch.empty(rows * heads * blocks, device="cuda", dtype=query.dtype)
    assert not supports(query, keys, workspace)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported TMA layout entered joint native submission")

    monkeypatch.setattr(joint, "select", forbidden)
    ids, valid = select_contiguous_blocks(
        query,
        keys,
        cis,
        length - rows,
        length,
        pooled_cis=pool,
        workspace=workspace,
        return_valid_mask=True,
    )
    last = torch.arange(length - rows, length)[:, None, None] // 64
    block = torch.arange(blocks)
    qa = torch.zeros(rows, heads, blocks, dtype=query.dtype)
    qa.masked_fill_((block == 0) | (last <= block + 16), torch.inf)
    qa.masked_fill_(block > last, -torch.inf)
    expected = _oracle(qa.flatten(0, 1), cis, torch.arange(length - rows, length), pool)
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(valid.cpu(), expected >= 0, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize(("rows", "heads"), [(128, 2), (1024, 2), (1025, 2), (1088, 2), (1024, 1)])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("length", [32768, 66560])
@torch.inference_mode()
def test_joint_indexer_matches_separate_native_calls_and_graph(
    monkeypatch, rows, heads, prepared, length
):
    from operators.nosa.indexer import _indexer_cuda as joint
    from operators.nosa.indexer._selection_cuda import select_pooled_blocks
    from operators.nosa.indexer.api import pooled_scores, select_contiguous_blocks

    count, blocks = length // 16 - 1, length // 64
    generator = torch.Generator(device="cuda").manual_seed(2129)
    # Positive outer strides retain the packed projection / cache layout.
    query = torch.randn(rows, 4608, device="cuda", dtype=torch.bfloat16, generator=generator)
    query = query[:, : heads * 16 * 128].view(rows, heads, 16, 128)
    keys = torch.randn(count, heads, 144, device="cuda", dtype=query.dtype, generator=generator)
    keys = keys[..., :128]
    cis = torch.randn(count, heads, device="cuda", dtype=query.dtype, generator=generator)
    pool = _pool(cis, blocks)[: blocks - 2]
    workspace = torch.empty(rows * heads * blocks, device="cuda", dtype=query.dtype)
    matrix = workspace.view(rows * heads, blocks)
    pooled_scores(query, keys, None, length - rows, length, matrix)
    expected = select_pooled_blocks(
        matrix, cis, None, length - rows, rows, heads, pooled_cis=pool, return_valid_mask=True
    )
    ranking = _prefix_ranking(pool, (length - rows) // 64) if prepared else None
    saved_ranking = ranking.clone() if prepared else None
    submitted = []
    original = joint.select

    def record(*args, **kwargs):
        submitted.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(joint, "select", record)

    def call():
        return select_contiguous_blocks(
            query,
            keys,
            cis,
            length - rows,
            length,
            pooled_cis=pool,
            workspace=workspace,
            return_valid_mask=True,
            prepared_ranking=ranking,
        )

    actual = call()
    assert submitted
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = call()
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    for result, reference in zip(captured, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    if prepared:
        torch.testing.assert_close(ranking, saved_ranking, rtol=0, atol=0)


@requires_cuda
@torch.inference_mode()
def test_prepared_ranking_rejects_overlap_and_unsupported_prefix_before_score_writes():
    from operators.nosa.indexer.api import select_contiguous_blocks

    rows, heads, length = 128, 2, 32768
    query = torch.zeros((rows, heads, 16, 128), device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((length // 16 - 1, heads, 128), device="cuda", dtype=query.dtype)
    cis = torch.zeros(keys.shape[:2], device="cuda", dtype=query.dtype)
    pool = torch.zeros((length // 64 - 2, heads), device="cuda", dtype=query.dtype)
    workspace = torch.full((rows * heads * (length // 64),), 123, device="cuda", dtype=query.dtype)
    saved = workspace.clone()
    ranking = workspace[: heads * 128].view(torch.int32).reshape(heads, 64)
    for start, message in ((length - rows, "overlap"), (0, "prefix path")):
        with pytest.raises(ValueError, match=message):
            select_contiguous_blocks(
                query,
                keys,
                cis,
                start,
                length,
                pooled_cis=pool,
                workspace=workspace,
                prepared_ranking=ranking,
                return_valid_mask=True,
            )
        torch.testing.assert_close(workspace, saved, rtol=0, atol=0)
