"""SM90 NOSA selector, ported from cxl-recsys's tiled Triton indexer.

BF16/FP16 QK uses Tensor Cores and FP32 normalization. Two passes avoid a
Q-head-sized logits/probability allocation. Absolute positions support a
temporary suffix after a cached prefix. Numerical equality to the FP32
reference is not guaranteed; ranking ties themselves remain deterministic.
Selection preserves the source's inclusive 17-local-block, 33/64 policy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
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
    COUNT: tl.constexpr,
    BG: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    BQ: tl.constexpr,
):
    row, head = tl.program_id(0) * BQ, tl.program_id(1)
    g, d, n = tl.arange(0, BG), tl.arange(0, DIM), tl.arange(0, BN)
    PG: tl.constexpr = BG // BQ
    qr = row + g // PG
    group = g % PG
    q = tl.load(
        Q + qr[:, None] * QR + head * QH + group[:, None] * QG + d[None, :],
        (group[:, None] < GROUPS) & (qr[:, None] < ROWS),
        other=0,
    )
    position = tl.load(POS + qr, qr < ROWS, other=-1)
    maximum = tl.full((BG,), -float("inf"), tl.float32)
    denominator = tl.zeros((BG,), tl.float32)
    # Each Q head has its own normalizer; summing logits before softmax would
    # change NOSA's GQA probability-sum semantics.
    for start in range(tl.cdiv(COUNT, BN)):
        c = start * BN + n
        valid = (c[None, :] < COUNT) & (c[None, :] * 16 + 31 <= position[:, None])
        k = tl.load(K + c[None, :] * KC + head * KH + d[:, None], c[None, :] < COUNT, other=0)
        score = tl.dot(q, k) * (DIM**-0.5)
        score = tl.where(valid, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(next_max == -float("inf"), 0.0, next_max)
        denominator = denominator * tl.exp(maximum - safe) + tl.sum(
            tl.exp(score - safe[:, None]), 1
        )
        maximum = next_max
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    inv = 1.0 / tl.where(denominator > 0, denominator, 1.0)
    for start in range(tl.cdiv(COUNT, BN)):
        c = start * BN + n
        valid = (c[None, :] < COUNT) & (c[None, :] * 16 + 31 <= position[:, None])
        k = tl.load(K + c[None, :] * KC + head * KH + d[:, None], c[None, :] < COUNT, other=0)
        score = tl.dot(q, k) * (DIM**-0.5)
        probability = tl.exp(score - safe[:, None]) * inv[:, None]
        probability = tl.where(valid & (group[:, None] < GROUPS), probability, 0.0)
        reduced = tl.sum(tl.reshape(probability, (BQ, PG, BN)), 1)
        out_row = row + tl.arange(0, BQ)
        tl.store(
            OUT + (out_row[:, None] * HEADS + head) * COUNT + c[None, :],
            reduced,
            (out_row[:, None] < ROWS) & (c[None, :] < COUNT),
        )


@triton.jit
def _rank_key(value, index):
    # Monotone signed float-bit key, then lower ID wins. Canonicalize +/-0,
    # which compare equal under the reference stable argsort.
    value = tl.where(value == 0.0, 0.0, value).to(tl.float32)
    bits = value.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, (~bits) ^ (-2147483648), bits).to(tl.int64)
    return (ordered << 32) | (4294967295 - index.to(tl.int64))


@triton.jit
def _pool(
    SCORE,
    CIS,
    POS,
    QKEY,
    CKEY,
    COUNT: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCKS: tl.constexpr,
    CS: tl.constexpr,
    CH: tl.constexpr,
    B: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    block = tl.program_id(2) * B + tl.arange(0, B)
    qblock = tl.load(POS + row) // 64
    qs = tl.full((B,), -float("inf"), tl.float32)
    cs = tl.full((B,), -float("inf"), tl.float32)
    for window in tl.static_range(5):  # pyright: ignore[reportGeneralTypeIssues]
        c = block * 4 - 1 + window
        valid = (block < BLOCKS) & (c >= 0) & (c < COUNT)
        qs = tl.maximum(
            qs,
            tl.load(SCORE + (row * HEADS + head) * COUNT + c, valid, other=-float("inf")).to(
                tl.float32
            ),
        )
        cs = tl.maximum(
            cs,
            tl.load(CIS + c * CS + head * CH, valid, other=-float("inf")).to(tl.float32),
        )
    mandatory = (block == 0) | ((block <= qblock) & (qblock <= block + 16))
    qs = tl.where(mandatory, float("inf"), qs)
    cs = tl.where(mandatory, float("inf"), cs)
    qs = tl.where(block <= qblock, qs, -float("inf"))
    cs = tl.where(block <= qblock, cs, -float("inf"))
    offset = (row * HEADS + head) * BLOCKS + block
    tl.store(QKEY + offset, _rank_key(qs, block), block < BLOCKS)
    tl.store(CKEY + offset, _rank_key(cs, block), block < BLOCKS)


@triton.jit
def _promote(CKEY, SELECTED, POS, HEADS: tl.constexpr, BLOCKS: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, 64)
    ids = tl.load(SELECTED + row * 33 + n, n < 33, other=0)
    qblock = tl.load(POS + row // HEADS) // 64
    value = tl.where(ids <= qblock, float("inf"), -float("inf"))
    tl.store(CKEY + row * BLOCKS + ids, _rank_key(value, ids), n < 33)


@triton.jit
def _select(
    SCORE,
    CIS,
    POS,
    OUT,
    COUNT: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCKS: tl.constexpr,
    CS: tl.constexpr,
    CH: tl.constexpr,
    B: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    block = tl.arange(0, B)
    qblock = tl.load(POS + row) // 64
    qs = tl.full((B,), -float("inf"), tl.float32)
    cs = tl.full((B,), -float("inf"), tl.float32)
    for window in tl.static_range(5):  # pyright: ignore[reportGeneralTypeIssues]
        c = block * 4 - 1 + window
        valid = (block < BLOCKS) & (c >= 0) & (c < COUNT)
        qs = tl.maximum(
            qs,
            tl.load(SCORE + (row * HEADS + head) * COUNT + c, valid, other=-float("inf")).to(
                tl.float32
            ),
        )
        cs = tl.maximum(
            cs,
            tl.load(CIS + c * CS + head * CH, valid, other=-float("inf")).to(tl.float32),
        )
    causal = block <= qblock
    mandatory = (block == 0) | (causal & (qblock <= block + 16))
    qs = tl.where(causal, tl.where(mandatory, float("inf"), qs), -float("inf"))
    cs = tl.where(causal, tl.where(mandatory, float("inf"), cs), -float("inf"))
    qkey = tl.where(block < BLOCKS, _rank_key(qs, block), -9223372036854775807 - 1)
    best = tl.topk(qkey, 64)  # pyright: ignore[reportArgumentType]
    n = tl.arange(0, 64)
    threshold = tl.sum(tl.where(n == 32, best, 0), 0)
    cs = tl.where((qkey >= threshold) & causal, float("inf"), cs)
    ckey = tl.where(block < BLOCKS, _rank_key(cs, block), -9223372036854775807 - 1)
    final = tl.topk(ckey, 64)  # pyright: ignore[reportArgumentType]
    ids = tl.sort(4294967295 - (final & 4294967295), descending=False)  # pyright: ignore[reportArgumentType]
    tl.store(OUT + (row * HEADS + head) * 64 + n, tl.where(ids <= qblock, ids, -1))


def compressed_scores(query, keys, positions):
    if (
        query.ndim != 4
        or keys.ndim != 3
        or query.shape[1] != keys.shape[1]
        or query.shape[-1] != keys.shape[-1]
        or query.shape[-1] not in (64, 128)
        or not 1 <= query.shape[2] <= 32
        or query.dtype not in (torch.bfloat16, torch.float16)
        or keys.dtype != query.dtype
        or positions.shape != (len(query),)
        or positions.dtype not in (torch.int64, torch.int32)
    ):
        raise ValueError("Unsupported fused Indexer geometry/dtype")
    if any(
        not t.is_cuda or t.device != query.device or t.stride(-1) != 1 or t.requires_grad
        for t in (query, keys, positions)
    ):
        raise ValueError(
            "Fused Indexer requires CUDA inference tensors with contiguous last dimensions"
        )
    rows, heads, groups, dim = query.shape
    output = torch.empty((rows, heads, len(keys)), dtype=query.dtype, device=query.device)
    if rows and len(keys):
        query_block = 4 if groups == 16 else 1
        with torch.cuda.device(query.device):
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
                grid=(triton.cdiv(rows, query_block), heads),
                warmup=False,
                num_warps=4,
                num_stages=2,
            )
    return output


def _validate_scores(scores, cis, positions, total_length, *, check_storage=True):
    if type(total_length) is not int or total_length <= 0:
        raise ValueError("total_length must be a positive integer")
    if scores.ndim != 3 or scores.shape[1] <= 0:
        raise ValueError("Expected [Q,KV head,compressed token] scores")
    rows, heads, count = scores.shape
    expected_count = max(0, (total_length - 32) // 16 + 1)
    if count != expected_count or cis.shape != (count, heads):
        raise ValueError("Compressed score/CIS dimensions must match total_length and KV heads")
    if positions.shape != (rows,) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be an integer tensor with one entry per query")
    if scores.dtype not in (torch.float16, torch.bfloat16, torch.float32) or cis.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("NOSA selection requires floating-point scores and CIS")
    if any(
        not tensor.is_cuda or tensor.device != scores.device or tensor.requires_grad
        for tensor in (scores, cis, positions)
    ):
        raise ValueError("SM90 NOSA selection requires CUDA inference tensors on one device")
    if torch.cuda.get_device_capability(scores.device)[0] != 9:
        raise RuntimeError("SM90 NOSA selection requires a Hopper GPU")
    if (check_storage and not scores.is_contiguous()) or positions.stride(-1) != 1:
        raise ValueError("NOSA scores and positions must be contiguous")
    if ((positions < 0) | (positions >= total_length)).any():
        raise ValueError("Query positions must be within total_length")


def select_from_scores(scores, cis, positions, total_length):
    """Exactly preserve reference pooling/causality/ties for identical scores.

    Unique integer composite keys allow unsorted partial Top-K without changing
    tie behavior. Only the final <=64 block IDs are sorted.
    """
    _validate_scores(scores, cis, positions, total_length)
    with torch.cuda.device(scores.device):
        return _select_validated_scores(scores, cis, positions, total_length)


def _select_validated_scores(scores, cis, positions, total_length):
    rows, heads, count = scores.shape
    blocks = (total_length + 63) // 64
    if blocks <= 64:
        ids = torch.arange(blocks, device=scores.device).expand(rows, heads, -1)
        return ids.masked_fill(ids > positions[:, None, None] // 64, -1)
    if cis.shape != (count, heads) or not scores.is_contiguous():
        raise ValueError("Expected contiguous [Q,H,C] scores and shared [C,H] CIS")
    if blocks <= 4096:
        selected = torch.empty((rows, heads, 64), dtype=torch.int64, device=scores.device)
        _select.run(
            scores,
            cis,
            positions,
            selected,
            count,
            heads,
            blocks,
            *cis.stride(),
            triton.next_power_of_2(blocks),
            grid=(rows, heads),
            warmup=False,
            num_warps=4,
        )
        return selected
    qkeys = torch.empty((rows, heads, blocks), dtype=torch.int64, device=scores.device)
    ckeys = torch.empty_like(qkeys)
    _pool.run(
        scores,
        cis,
        positions,
        qkeys,
        ckeys,
        count,
        heads,
        blocks,
        *cis.stride(),
        128,
        grid=(rows, heads, triton.cdiv(blocks, 128)),
        warmup=False,
    )
    chosen = torch.topk(qkeys, 33, dim=-1, sorted=False).indices
    _promote.run(ckeys, chosen, positions, heads, blocks, grid=(rows * heads,), warmup=False)
    selected = torch.topk(ckeys, 64, dim=-1, sorted=False).indices.sort(-1).values
    return selected.masked_fill(selected > positions[:, None, None] // 64, -1)


def select_blocks(query, keys, cis, positions, total_length):
    if query.ndim != 4 or keys.ndim != 3 or query.shape[1] != keys.shape[1]:
        raise ValueError("Expected grouped Q [Q,H,G,D] and compressed K [C,H,D]")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("Q/K head dimensions must agree")
    if not query.is_cuda or keys.device != query.device:
        raise ValueError("SM90 NOSA indexer requires CUDA Q/K on one device")
    if query.requires_grad or keys.requires_grad:
        raise ValueError("SM90 NOSA indexer supports inference tensors only")
    if query.dtype not in (torch.bfloat16, torch.float16) or keys.dtype != query.dtype:
        raise ValueError("SM90 NOSA indexer requires matching BF16/FP16 Q/K")
    if query.shape[-1] not in (64, 128) or not 1 <= query.shape[2] <= 32:
        raise ValueError("SM90 NOSA indexer supports head_dim 64/128 and 1..32 Q heads per KV head")
    # Validate all metadata before launching either the scoring kernel or the
    # short-context path. In particular, explicit CUDA must never silently
    # execute a CPU fallback just because the context is short.
    probe = torch.empty((), dtype=query.dtype, device=query.device).expand(
        *query.shape[:2], len(keys)
    )
    _validate_scores(probe, cis, positions, total_length, check_storage=False)
    if (total_length + 63) // 64 <= 64:
        blocks = (total_length + 63) // 64
        ids = torch.arange(blocks, device=query.device).expand(*query.shape[:2], -1)
        return ids.masked_fill(ids > positions[:, None, None] // 64, -1)
    scores = compressed_scores(query, keys, positions)
    return select_from_scores(scores, cis, positions, total_length)
