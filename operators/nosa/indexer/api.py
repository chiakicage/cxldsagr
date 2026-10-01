"""SM90 NOSA QK/pooling and stable two-stage block selection.

BF16/FP16 QK uses Tensor Cores and FP32 normalization. Two passes avoid a
Q-head-sized probability allocation. The second pass rounds each GQA sum to
the model dtype before pooling, preserving the standalone score semantics.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_MAX_SEQUENCE_LENGTH = 262144
_MAX_COMPRESSED_WINDOWS = _MAX_SEQUENCE_LENGTH // 16 - 1


@triton.jit
def _query_position(POS, row, QUERY_START, CONTIGUOUS: tl.constexpr):
    if CONTIGUOUS:
        return QUERY_START + row
    else:
        return tl.load(POS + row)


@triton.jit
def _masked_pool(value, block, qblock):
    causal = block <= qblock
    mandatory = (block == 0) | (causal & (qblock <= block + 16))
    value = tl.where(causal, tl.where(mandatory, float("inf"), value), -float("inf"))
    # FlashInfer's radix ordering distinguishes signed zeros. Stable reference
    # sorting regards them as equal and resolves ties by the smaller block ID.
    return tl.where(value == 0.0, 0.0, value)


@triton.jit(do_not_specialize=["COUNT", "ROWS", "QUERY_START", "BLOCKS"])
def _scores(
    Q,
    K,
    POS,
    OUT,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QG: tl.constexpr,
    KC: tl.constexpr,
    KH: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    COUNT,
    BG: tl.constexpr,
    BN: tl.constexpr,
    ROWS,
    BQ: tl.constexpr,
    QUERY_START,
    BLOCKS,
    CONTIGUOUS: tl.constexpr,
    POOL_OUT: tl.constexpr,
    STATIC_COUNT: tl.constexpr,
):
    count = STATIC_COUNT if STATIC_COUNT > 0 else COUNT
    row, head = tl.program_id(0) * BQ, tl.program_id(1)
    g, d, n = tl.arange(0, BG), tl.arange(0, DIM), tl.arange(0, BN)
    PG: tl.constexpr = BG // BQ
    qr = row + g // PG
    group = g % PG
    q = tl.load(
        Q + qr[:, None].to(tl.int64) * QR + head * QH + group[:, None] * QG + d[None, :],
        (group[:, None] < GROUPS) & (qr[:, None] < ROWS),
        other=0,
    )
    if CONTIGUOUS:
        position = QUERY_START + qr
    else:
        position = tl.load(POS + qr, qr < ROWS, other=-1)
    maximum = tl.full((BG,), -float("inf"), tl.float32)
    denominator = tl.zeros((BG,), tl.float32)
    # Softmax is per Q head; reducing logits before normalization changes GQA.
    # Both passes round the FP32 scale before subtracting the max. Fusing
    # scale/subtract can bias a uniform softmax when logits are large.
    for start in range(tl.cdiv(count, BN)):
        c = start * BN + n
        valid = (c[None, :] < count) & (c[None, :] * 16 + 31 <= position[:, None])
        k = tl.load(K + c[None, :] * KC + head * KH + d[:, None], c[None, :] < count, other=0)
        score = libdevice.mul_rn(tl.dot(q, k), DIM**-0.5)
        score = tl.where(valid, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(next_max == -float("inf"), 0.0, next_max)
        denominator = denominator * tl.exp(maximum - safe) + tl.sum(
            tl.exp(score - safe[:, None]), 1
        )
        maximum = next_max
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    inv = 1.0 / tl.where(denominator > 0, denominator, 1.0)
    out_row = row + tl.arange(0, BQ)
    if POOL_OUT:
        pool_lane = tl.arange(0, BN // 4)
        if CONTIGUOUS:
            out_position = QUERY_START + out_row
        else:
            out_position = tl.load(POS + out_row, out_row < ROWS, other=-1)
        qblock = out_position // 64
        halo = tl.full((BQ,), -float("inf"), tl.float32)
    for start in range(tl.cdiv(count, BN)):
        c = start * BN + n
        valid = (c[None, :] < count) & (c[None, :] * 16 + 31 <= position[:, None])
        k = tl.load(K + c[None, :] * KC + head * KH + d[:, None], c[None, :] < count, other=0)
        score = libdevice.mul_rn(tl.dot(q, k), DIM**-0.5)
        probability = tl.exp(score - safe[:, None]) * inv[:, None]
        probability = tl.where(valid & (group[:, None] < GROUPS), probability, 0.0)
        reduced = tl.sum(tl.reshape(probability, (BQ, PG, BN)), 1)
        if POOL_OUT:
            # Preserve the old score HBM store's rounding even with an FP32
            # selection workspace (e.g. BF16 QK and FP32 CIS).
            rounded = reduced.to(Q.dtype.element_ty).to(tl.float32)
            rounded = tl.where(c[None, :] < count, rounded, -float("inf"))
            pooled = tl.max(tl.reshape(rounded, (BQ, BN // 4, 4)), 2)
            left_index = tl.maximum(pool_lane * 4 - 1, 0)
            left_index = tl.broadcast_to(left_index[None, :], (BQ, BN // 4))
            left = tl.gather(rounded, left_index, 1)
            left = tl.where(pool_lane[None, :] == 0, halo[:, None], left)
            pooled = tl.maximum(pooled, left)
            block = start * (BN // 4) + pool_lane
            pooled = _masked_pool(pooled, block[None, :], qblock[:, None])
            offset = (out_row[:, None].to(tl.int64) * HEADS + head) * BLOCKS + block[None, :]
            tl.store(OUT + offset, pooled, (out_row[:, None] < ROWS) & (block[None, :] < BLOCKS))
            # Carry the last tile slot, not its last valid window. An absent
            # final slot must remain -inf for an empty trailing logical block.
            halo = tl.sum(tl.where(n[None, :] == BN - 1, rounded, 0.0), 1)
        else:
            offset = (out_row[:, None].to(tl.int64) * HEADS + head) * count + c[None, :]
            tl.store(OUT + offset, reduced, (out_row[:, None] < ROWS) & (c[None, :] < count))
    if POOL_OUT:
        # ceil(L/64) can exceed 32*ceil(C/128): this block has only a left
        # halo, or no complete compression window at all (e.g. L=65537).
        block = tl.cdiv(count, BN) * (BN // 4)
        if block < BLOCKS:
            pooled = _masked_pool(halo, block, qblock)
            offset = (out_row.to(tl.int64) * HEADS + head) * BLOCKS + block
            tl.store(OUT + offset, pooled, out_row < ROWS)


@triton.jit(do_not_specialize=["COUNT", "BLOCKS", "QUERY_START"])
def _pool_qa(
    SCORE,
    POS,
    OUT,
    COUNT,
    HEADS: tl.constexpr,
    BLOCKS,
    QUERY_START,
    CONTIGUOUS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1) * TILE + tl.arange(0, TILE)
    qblock = _query_position(POS, row // HEADS, QUERY_START, CONTIGUOUS) // 64
    pooled = tl.full((TILE,), -float("inf"), tl.float32)
    for window in tl.static_range(5):
        c = block * 4 - 1 + window
        value = tl.load(
            SCORE + row.to(tl.int64) * COUNT + c,
            (block < BLOCKS) & (c >= 0) & (c < COUNT),
            other=-float("inf"),
        ).to(tl.float32)
        pooled = tl.maximum(pooled, value)
    tl.store(
        OUT + row.to(tl.int64) * BLOCKS + block, _masked_pool(pooled, block, qblock), block < BLOCKS
    )


@triton.jit(do_not_specialize=["COUNT", "BLOCKS", "QUERY_START", "CACHED_BLOCKS"])
def _prepare_cis(
    WORK,
    CIS,
    CACHED,
    QA_VALUES,
    QA_IDS,
    POS,
    COUNT,
    HEADS: tl.constexpr,
    BLOCKS,
    QUERY_START,
    CONTIGUOUS: tl.constexpr,
    CS: tl.constexpr,
    CH: tl.constexpr,
    CACHED_BLOCKS,
    PS: tl.constexpr,
    PH: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % HEADS
    block = tl.program_id(1) * TILE + tl.arange(0, TILE)
    qblock = _query_position(POS, row // HEADS, QUERY_START, CONTIGUOUS) // 64
    n = tl.arange(0, 64)
    values = tl.load(QA_VALUES + row * 33 + n, n < 33, other=float("inf")).to(tl.float32)
    ids = tl.load(QA_IDS + row * 33 + n, n < 33, other=-1)
    threshold = tl.min(values, 0)
    threshold_id = tl.max(tl.where((n < 33) & (values == threshold), ids, -1), 0)
    offset = row.to(tl.int64) * BLOCKS + block
    old_qa = tl.load(WORK + offset, block < BLOCKS, other=-float("inf")).to(tl.float32)
    chosen = (old_qa > threshold) | ((old_qa == threshold) & (block <= threshold_id))
    pooled = tl.full((TILE,), -float("inf"), tl.float32)
    if HAS_CACHE:
        pooled = tl.load(
            CACHED + block * PS + head * PH,
            (block < BLOCKS) & (block < CACHED_BLOCKS),
            other=-float("inf"),
        ).to(tl.float32)
    # Cache only stable, query-independent pools. At most two trailing blocks
    # remain when consuming the cache's stable prefix.
    for window in tl.static_range(5):
        c = block * 4 - 1 + window
        valid = (block < BLOCKS) & (c >= 0) & (c < COUNT)
        if HAS_CACHE:
            valid = valid & (block >= CACHED_BLOCKS)
        value = tl.load(CIS + c * CS + head * CH, valid, other=-float("inf")).to(tl.float32)
        pooled = tl.maximum(pooled, value)
    pooled = _masked_pool(pooled, block, qblock)
    pooled = tl.where(chosen & (block <= qblock), float("inf"), pooled)
    tl.store(WORK + offset, pooled, block < BLOCKS)


@triton.jit(do_not_specialize=["QUERY_START"])
def _finish_selection(
    SELECTED,
    POS,
    OUT,
    VALID,
    HEADS: tl.constexpr,
    QUERY_START,
    CONTIGUOUS: tl.constexpr,
    WRITE_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.arange(0, 64)
    ids = tl.load(SELECTED + row * 64 + n).to(tl.int32)
    ids = tl.sort(ids, descending=False)
    qblock = _query_position(POS, row // HEADS, QUERY_START, CONTIGUOUS) // 64
    valid = ids <= qblock
    tl.store(OUT + row * 64 + n, tl.where(valid, ids, -1))
    if WRITE_MASK:
        tl.store(VALID + row * 64 + n, valid)


@triton.jit(do_not_specialize=["QUERY_START", "BLOCKS"])
def _select_all_blocks(
    POS,
    OUT,
    VALID,
    HEADS: tl.constexpr,
    BLOCKS,
    WIDTH: tl.constexpr,
    QUERY_START,
    CONTIGUOUS: tl.constexpr,
    WRITE_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.arange(0, 64)
    qblock = _query_position(POS, row // HEADS, QUERY_START, CONTIGUOUS) // 64
    valid = (block < BLOCKS) & (block <= qblock)
    offset = row * WIDTH + block
    tl.store(OUT + offset, tl.where(valid, block, -1), block < WIDTH)
    if WRITE_MASK:
        tl.store(VALID + offset, valid, block < WIDTH)


def _launch_scores(query, keys, positions, output, query_start, blocks, *, pool_output):
    rows, heads, groups, dim = query.shape
    if not rows or not len(keys):
        return
    from operators.nosa._native import native_enabled
    from operators.nosa.indexer._scores_cuda import scores_out, supports

    # The two-kernel native schedule amortizes its extra launch on long keys;
    # short prefill chunks retain the single-kernel Triton schedule.
    native_min_count = 2047 if rows >= 1024 else 511
    if native_enabled() and len(keys) >= native_min_count and supports(query, keys, output):
        scores_out(query, keys, positions, output, query_start, blocks, pool_output=pool_output)
        return
    query_block = 4 if groups == 16 else 1
    if groups == 16 and dim == 128 and rows >= 1024:
        query_block = 8
    # Large batches benefit from a constant compression mask: runtime masks
    # raise SM90 register pressure even with a constant loop tile count. Small
    # batches keep runtime lengths, avoiding a new decode kernel every window.
    _scores.run(
        query,
        keys,
        positions,
        output,
        *query.stride()[:3],
        *keys.stride()[:2],
        heads,
        groups,
        dim,
        len(keys),
        query_block * max(16, triton.next_power_of_2(groups)),
        128,
        rows,
        query_block,
        query_start,
        blocks,
        positions is None,
        pool_output,
        len(keys) if rows >= 1024 else 0,
        grid=(triton.cdiv(rows, query_block), heads),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )


def _validate_query(query, keys):
    if query.ndim != 4 or keys.ndim != 3 or query.shape[1] != keys.shape[1] or query.shape[1] <= 0:
        raise ValueError("Expected grouped Q [Q,H,G,D] and compressed K [C,H,D]")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("Q/K head dimensions must agree")
    if len(query) > _MAX_SEQUENCE_LENGTH or len(keys) > _MAX_COMPRESSED_WINDOWS:
        raise ValueError(
            "SM90 NOSA indexer supports at most 262144 query/KV tokens (16383 windows)"
        )
    if not query.is_cuda or keys.device != query.device:
        raise ValueError("SM90 NOSA indexer requires CUDA Q/K on one device")
    if query.requires_grad or keys.requires_grad:
        raise ValueError("SM90 NOSA indexer supports inference tensors only")
    if query.dtype not in (torch.bfloat16, torch.float16) or keys.dtype != query.dtype:
        raise ValueError("SM90 NOSA indexer requires matching BF16/FP16 Q/K")
    if query.shape[-1] not in (64, 128) or not 1 <= query.shape[2] <= 32:
        raise ValueError("SM90 NOSA indexer supports head_dim 64/128 and 1..32 Q heads per KV head")
    if query.stride(-1) != 1 or keys.stride(-1) != 1:
        raise ValueError("Fused Indexer requires contiguous last dimensions")
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        raise RuntimeError("SM90 NOSA indexer requires a Hopper GPU")


def _validate_positions(positions, rows, device, total_length=None, *, check_values=True):
    if positions.shape != (rows,) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be an integer tensor with one entry per query")
    if positions.device != device or positions.stride(-1) != 1 or positions.requires_grad:
        raise ValueError("positions must be contiguous CUDA inference tensors on the input device")
    if (
        check_values
        and total_length is not None
        and ((positions < 0) | (positions >= total_length)).any()
    ):
        raise ValueError("Query positions must be within total_length")


def compressed_scores(query, keys, positions):
    """Standalone compressed scores, retaining the original model-dtype output."""
    _validate_query(query, keys)
    _validate_positions(positions, len(query), query.device, check_values=False)
    output = torch.empty((*query.shape[:2], len(keys)), dtype=query.dtype, device=query.device)
    with torch.cuda.device(query.device):
        _launch_scores(query, keys, positions, output, 0, 0, pool_output=False)
    return output


def _validate_cis(cis, rows, heads, count, device, total_length, *, pooled_cis=None):
    if type(total_length) is not int or total_length <= 0:
        raise ValueError("total_length must be a positive integer")
    if total_length > _MAX_SEQUENCE_LENGTH or rows > _MAX_SEQUENCE_LENGTH:
        raise ValueError("SM90 NOSA indexer supports at most 262144 query/KV tokens")
    if count != max(0, (total_length - 32) // 16 + 1) or cis.shape != (count, heads):
        raise ValueError("Compressed score/CIS dimensions must match total_length and KV heads")
    if cis.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NOSA selection requires floating-point scores and CIS")
    if cis.device != device or cis.requires_grad:
        raise ValueError("SM90 NOSA selection requires CUDA inference tensors on one device")
    if pooled_cis is not None:
        blocks = (total_length + 63) // 64
        if pooled_cis.ndim != 2 or pooled_cis.shape[1] != heads or len(pooled_cis) > blocks:
            raise ValueError("pooled_cis must be a [stable blocks,KV heads] prefix")
        if (
            pooled_cis.device != device
            or pooled_cis.requires_grad
            or pooled_cis.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                "pooled_cis requires floating-point CUDA inference storage on the input device"
            )


def _validate_scores(
    scores, cis, positions, total_length, *, check_storage=True, check_positions=True
):
    if scores.ndim != 3 or scores.shape[1] <= 0:
        raise ValueError("Expected [Q,KV head,compressed token] scores")
    rows, heads, count = scores.shape
    if scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NOSA selection requires floating-point scores and CIS")
    if not scores.is_cuda or scores.requires_grad:
        raise ValueError("SM90 NOSA selection requires CUDA inference tensors on one device")
    if torch.cuda.get_device_capability(scores.device)[0] != 9:
        raise RuntimeError("SM90 NOSA selection requires a Hopper GPU")
    if check_storage and not scores.is_contiguous():
        raise ValueError("NOSA scores must be contiguous")
    _validate_cis(cis, rows, heads, count, scores.device, total_length)
    _validate_positions(positions, rows, scores.device, total_length, check_values=check_positions)


def _score_workspace(workspace, rows, heads, blocks, dtype, device):
    elements = rows * heads * blocks
    if workspace is None:
        workspace = torch.empty(elements, device=device, dtype=dtype)
    elif (
        workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.numel() < elements
        or workspace.device != device
        or workspace.dtype != dtype
        or workspace.requires_grad
        or workspace.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "workspace must be 16-byte aligned contiguous flat CUDA inference storage with promoted dtype and sufficient capacity"
        )
    # A prefix of flat capacity is fully contiguous for every B. Slicing a
    # [Rmax,Bmax] allocation would retain Bmax row strides and fail top_k.
    return workspace[:elements].view(rows * heads, blocks)


def pooled_scores(query, keys, positions, query_start, total_length, workspace):
    """Launch QK and five-window QA pooling into a validated [Q*H,B] workspace."""
    _launch_scores(
        query, keys, positions, workspace, query_start, (total_length + 63) // 64, pool_output=True
    )
    return workspace


def topk_qa(workspace):
    """Choose stable smaller-ID top33; returned order need not be sorted."""
    import flashinfer

    return flashinfer.top_k(
        workspace, 33, sorted=False, deterministic=False, tie_break=flashinfer.TopKTieBreak.SMALL
    )


def prepare_cis(workspace, cis, qa_values, qa_ids, positions, query_start, heads, pooled_cis=None):
    """Overwrite QA scores with masked CIS and promotion, reading each QA cell once."""
    rows, blocks = workspace.shape
    _prepare_cis.run(
        workspace,
        cis,
        pooled_cis,
        qa_values,
        qa_ids,
        positions,
        len(cis),
        heads,
        blocks,
        query_start,
        positions is None,
        *cis.stride(),
        len(pooled_cis) if pooled_cis is not None else 0,
        *(pooled_cis.stride() if pooled_cis is not None else (0, 0)),
        pooled_cis is not None,
        256,
        grid=(rows, triton.cdiv(blocks, 256)),
        warmup=False,
        num_warps=4,
    )
    return workspace


def topk_cis(workspace):
    """Complete stable top64 using the same contiguous score workspace."""
    import flashinfer

    return flashinfer.top_k(
        workspace, 64, sorted=False, deterministic=False, tie_break=flashinfer.TopKTieBreak.SMALL
    )[1]


def finish_selection(selected, positions, query_start, rows, heads, *, return_valid_mask=False):
    ids = torch.empty((rows, heads, 64), device=selected.device, dtype=torch.int64)
    valid = torch.empty_like(ids, dtype=torch.bool) if return_valid_mask else None
    if rows:
        _finish_selection.run(
            selected,
            positions,
            ids,
            valid,
            heads,
            query_start,
            positions is None,
            return_valid_mask,
            grid=(rows * heads,),
            warmup=False,
            num_warps=4,
        )
    return (ids, valid) if return_valid_mask else ids


def _short_selection(positions, heads, blocks, *, rows, device, query_start=0, return_valid_mask):
    width = 64 if return_valid_mask else blocks
    ids = torch.empty((rows, heads, width), device=device, dtype=torch.int64)
    valid = torch.empty_like(ids, dtype=torch.bool) if return_valid_mask else None
    if rows:
        _select_all_blocks.run(
            positions,
            ids,
            valid,
            heads,
            blocks,
            width,
            query_start,
            positions is None,
            return_valid_mask,
            grid=(rows * heads,),
            warmup=False,
            num_warps=4,
        )
    return (ids, valid) if return_valid_mask else ids


def _select_workspace(
    workspace,
    cis,
    positions,
    query_start,
    rows,
    heads,
    *,
    pooled_cis,
    return_valid_mask,
    prepared_ranking=None,
):
    from operators.nosa._native import native_enabled

    if native_enabled() and workspace.dtype in (torch.bfloat16, torch.float16):
        from operators.nosa.indexer._selection_cuda import select_pooled_blocks

        return select_pooled_blocks(
            workspace,
            cis,
            positions,
            query_start,
            rows,
            heads,
            pooled_cis=pooled_cis,
            return_valid_mask=return_valid_mask,
            prepared_ranking=prepared_ranking,
        )
    if not rows:
        selected = torch.empty((0, 64), device=workspace.device, dtype=torch.int64)
    else:
        qa_values, qa_ids = topk_qa(workspace)
        prepare_cis(workspace, cis, qa_values, qa_ids, positions, query_start, heads, pooled_cis)
        selected = topk_cis(workspace)
    return finish_selection(
        selected, positions, query_start, rows, heads, return_valid_mask=return_valid_mask
    )


def _select_validated_scores(scores, cis, positions, total_length, *, return_valid_mask=False):
    rows, heads, count = scores.shape
    blocks = (total_length + 63) // 64
    if blocks <= 64:
        return _short_selection(
            positions,
            heads,
            blocks,
            rows=rows,
            device=scores.device,
            return_valid_mask=return_valid_mask,
        )
    if cis.shape != (count, heads) or not scores.is_contiguous():
        raise ValueError("Expected contiguous [Q,H,C] scores and shared [C,H] CIS")
    workspace = _score_workspace(
        None, rows, heads, blocks, torch.promote_types(scores.dtype, cis.dtype), scores.device
    )
    if rows:
        _pool_qa.run(
            scores,
            positions,
            workspace,
            count,
            heads,
            blocks,
            0,
            False,
            256,
            grid=(rows * heads, triton.cdiv(blocks, 256)),
            warmup=False,
            num_warps=4,
        )
    return _select_workspace(
        workspace,
        cis,
        positions,
        0,
        rows,
        heads,
        pooled_cis=None,
        return_valid_mask=return_valid_mask,
    )


def select_from_scores(scores, cis, positions, total_length):
    """Preserve reference pooling, causal masks and smaller-ID ties for given scores."""
    _validate_scores(scores, cis, positions, total_length)
    with torch.cuda.device(scores.device):
        return _select_validated_scores(scores, cis, positions, total_length)


def _select_blocks(
    query,
    keys,
    cis,
    positions,
    total_length,
    *,
    check_positions,
    return_valid_mask=False,
    query_start=0,
    pooled_cis=None,
    workspace=None,
    prepared_ranking=None,
):
    rows, heads = query.shape[:2]
    _validate_cis(cis, rows, heads, len(keys), query.device, total_length, pooled_cis=pooled_cis)
    if positions is not None:
        _validate_positions(
            positions, rows, query.device, total_length, check_values=check_positions
        )
    blocks = (total_length + 63) // 64
    # Validate supplied storage even on the short path, without allocating
    # scratch space that this path does not consume.
    if workspace is not None or blocks > 64:
        workspace = _score_workspace(
            workspace,
            rows,
            heads,
            blocks,
            torch.promote_types(query.dtype, cis.dtype),
            query.device,
        )
    if prepared_ranking is not None:
        from operators.nosa._native import native_enabled
        from operators.nosa.indexer._selection_cuda import _validate_prepared_ranking

        if not native_enabled() or blocks <= 64:
            raise ValueError("Prepared CIS ranking requires the native contiguous prefix path")
        _validate_prepared_ranking(
            prepared_ranking, workspace, positions, query_start, rows, heads, pooled_cis
        )
    with torch.cuda.device(query.device):
        if blocks <= 64:
            return _short_selection(
                positions,
                heads,
                blocks,
                rows=rows,
                device=query.device,
                query_start=query_start,
                return_valid_mask=return_valid_mask,
            )
        from operators.nosa._native import native_enabled
        from operators.nosa.indexer._scores_cuda import supports

        # Joint submission must preserve the ordinary score dispatch's layout
        # fallback. Broadcast, unaligned and non-TMA strides remain supported
        # by the Triton score path followed by native selection.
        if (
            native_enabled()
            and rows >= 128
            and query.dtype == torch.bfloat16
            and workspace.dtype == torch.bfloat16
            and positions is None
            and pooled_cis is not None
            and pooled_cis.dtype == torch.bfloat16
            and len(pooled_cis) >= blocks - 2
            and 2047 <= len(keys)
            and blocks <= 1056
            and return_valid_mask
            and supports(query, keys, workspace)
        ):
            from operators.nosa.indexer._indexer_cuda import select

            return select(
                query,
                keys,
                cis,
                pooled_cis,
                workspace,
                query_start,
                prepared_ranking=prepared_ranking,
            )
        pooled_scores(query, keys, positions, query_start, total_length, workspace)
        return _select_workspace(
            workspace,
            cis,
            positions,
            query_start,
            rows,
            heads,
            pooled_cis=pooled_cis,
            return_valid_mask=return_valid_mask,
            prepared_ranking=prepared_ranking,
        )


def select_blocks(query, keys, cis, positions, total_length):
    """Explicit positions are range-checked once before launching QK/pooling."""
    _validate_query(query, keys)
    return _select_blocks(query, keys, cis, positions, total_length, check_positions=True)


def select_contiguous_blocks(
    query,
    keys,
    cis,
    query_start,
    total_length,
    *,
    return_valid_mask=False,
    pooled_cis=None,
    workspace=None,
    prepared_ranking=None,
):
    """Select a contiguous range using scalar positions and reusable flat scratch.

    ``pooled_cis`` optionally provides [S,H] raw, query-independent five-window
    pools from the already-rounded compressed CIS. Blocks [S,B) are pooled
    directly, so callers can cache only the stable prefix. ``workspace`` must
    have at least Q*H*ceil(total_length/64) elements, 16-byte alignment, and
    dtype promoted from Q and CIS. FlashInfer allocates its values/ID outputs
    internally.

    ``prepared_ranking`` may supply packed int32 [H,64] candidates from native
    ranked preparation for this exact query start and stable pool. Its storage
    must stay separate from ``workspace`` until this call finishes consuming it.
    """
    _validate_query(query, keys)
    if type(total_length) is not int or total_length <= 0:
        raise ValueError("total_length must be a positive integer")
    if type(query_start) is not int or query_start < 0:
        raise ValueError("query_start must be a nonnegative integer")
    if query_start + len(query) > total_length:
        raise ValueError("Query positions must be within total_length")
    return _select_blocks(
        query,
        keys,
        cis,
        None,
        total_length,
        check_positions=False,
        return_valid_mask=return_valid_mask,
        query_start=query_start,
        pooled_cis=pooled_cis,
        workspace=workspace,
        prepared_ranking=prepared_ranking,
    )
