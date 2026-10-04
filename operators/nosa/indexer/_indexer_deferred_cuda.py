"""Guarded short-prefix submission preserving the ordinary score schedule.

Only an owning model transaction may publish these asynchronous outputs: its
device flag must be checked before cache commit or returning model output.
The synchronous preparation/selection APIs keep their original behavior.
"""

import torch
import triton
import triton.language as tl

from operators.nosa._native import load_module, native_enabled
from operators.nosa.indexer._prepare_cuda import _validate, prepare_out
from operators.nosa.indexer._prepare_ranked_cuda import _geometry_supported, prepare_ranked_out
from operators.nosa.indexer._scores_cuda import supports as scores_supported
from operators.nosa.indexer.api import _scores, _select_all_blocks


@triton.jit(do_not_specialize=["COUNT", "ROWS", "QUERY_START", "BLOCKS"])
def _guarded_scores(
    FINITE,
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
    # This uniform branch dominates every score input read and output write.
    # Keep the existing arithmetic, constexpr arguments, and launch unchanged.
    if tl.load(FINITE):
        _scores(
            Q,
            K,
            POS,
            OUT,
            QR,
            QH,
            QG,
            KC,
            KH,
            HEADS,
            GROUPS,
            DIM,
            COUNT,
            BG,
            BN,
            ROWS,
            BQ,
            QUERY_START,
            BLOCKS,
            CONTIGUOUS,
            POOL_OUT,
            STATIC_COUNT,
        )


@triton.jit(do_not_specialize=["QUERY_START", "BLOCKS"])
def _guarded_all_blocks(FINITE, OUT, VALID, HEADS: tl.constexpr, BLOCKS, QUERY_START):
    if tl.load(FINITE):
        _select_all_blocks(None, OUT, VALID, HEADS, BLOCKS, 64, QUERY_START, True, True)
    else:
        offset = tl.program_id(0) * 64 + tl.arange(0, 64)
        tl.store(OUT + offset, -1)
        tl.store(VALID + offset, False)


def _launch_guarded_scores(query, keys, output, normalizers, finite, query_start, blocks):
    """Match _launch_scores exactly, adding only a uniform finite guard."""
    rows, heads, groups, dim = query.shape
    if not rows or not len(keys):
        return
    native_min_count = 2047 if rows >= 1024 else 511
    if native_enabled() and len(keys) >= native_min_count and scores_supported(query, keys, output):
        import tvm_ffi

        with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
            load_module("nosa_scores").scores_guarded_out(
                query, keys, query, output, normalizers, finite, query_start, blocks, True, True
            )
        return
    query_block = 4 if groups == 16 else 1
    if groups == 16 and dim == 128 and rows >= 1024:
        query_block = 8
    _guarded_scores.run(
        finite,
        query,
        keys,
        None,
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
        True,
        True,
        len(keys) if rows >= 1024 else 0,
        grid=(triton.cdiv(rows, query_block), heads),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )


def _span(tensor):
    """Include gaps in positive-stride inputs and full supplied capacities."""
    begin = tensor.data_ptr()
    elements = (
        0
        if not tensor.numel()
        else 1
        + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    )
    return begin, begin + elements * tensor.element_size()


def _validate_disjoint(inputs, outputs):
    inputs = [(name, *_span(tensor)) for name, tensor in inputs]
    outputs = [(name, *_span(tensor)) for name, tensor in outputs if tensor is not None]
    for index, (name, begin, end) in enumerate(outputs):
        if begin == end:
            continue
        for other, left, right in (*inputs, *outputs[index + 1 :]):
            if left != right and begin < right and left < end:
                raise ValueError(f"Writable indexer buffers must not overlap: {name}, {other}")


def _validate_tensor(tensor, shape, dtype, device, name, alignment=1):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or tensor.data_ptr() % alignment
    ):
        raise ValueError(f"Invalid deferred indexer {name} shape, dtype, device, or storage")


def select_prepared_out(
    query,
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    workspace,
    normalizers,
    block_ids,
    valid_mask,
    ranking,
    *,
    query_start,
    validated_start,
    compressed_start,
    pooled_start,
    scratch,
):
    """Prepare an owned short append and produce guarded 64-wide selection.

    All capacities and alias checks run before preparation launches. Invalid
    inputs leave derived records, ranking, scores, and normalizers untouched;
    only validation scratch/flag and safe IDs=-1/mask=false may be written.
    Full output capacities, rather than visible-prefix slices, are required.
    No host finite read occurs here; the owner must check scratch.finite.
    """
    import tvm_ffi

    count, stable = _validate(
        query,
        keys,
        cis,
        compressed_keys,
        compressed_cis,
        pooled_cis,
        scratch,
        validated_start,
        compressed_start,
        pooled_start,
    )
    rows, heads = len(query), keys.shape[1]
    blocks = (len(keys) + 63) // 64
    if (
        not native_enabled()
        or type(query_start) is not int
        or query_start < 0
        or query_start != validated_start
        or query_start + rows != len(keys)
        or query.shape != (rows, heads, 16, 128)
        or query.dtype != torch.bfloat16
        or query.stride(-1) != 1
        or query.data_ptr() % 16
        or any(stride <= 0 or stride % 8 for stride in query.stride()[:3])
        or count >= 2047
    ):
        raise ValueError("Deferred short indexer requires an owned BF16/D128/GQA16 append")
    _validate_tensor(block_ids, (rows, heads, 64), torch.int64, query.device, "IDs")
    _validate_tensor(valid_mask, (rows, heads, 64), torch.bool, query.device, "mask")
    if scratch.partial.requires_grad or scratch.finite.requires_grad:
        raise ValueError("Deferred validation scratch must be inference storage")
    if blocks <= 64:
        if workspace is not None or normalizers is not None or ranking is not None:
            raise ValueError("Short all-block selection does not accept score or ranking buffers")
    else:
        if not _geometry_supported(
            query,
            keys,
            query_start=query_start,
            validated_start=validated_start,
            pooled_start=pooled_start,
        ):
            raise ValueError("Deferred scored selection requires ready prefix ranking")
        _validate_tensor(workspace, (rows * heads, blocks), query.dtype, query.device, "scores", 16)
        if normalizers is not None or count >= (2047 if rows >= 1024 else 511):
            _validate_tensor(
                normalizers, (1, rows, heads, 16, 2), torch.float32, query.device, "normalizers"
            )
        _validate_tensor(ranking, (heads, 64), torch.int32, query.device, "ranking", 4)
        if not scores_supported(query, compressed_keys[:count], workspace):
            raise ValueError("Unsupported deferred native score strides")
    _validate_disjoint(
        (("Q", query), ("K", keys), ("CIS", cis)),
        (
            ("compressed K capacity", compressed_keys),
            ("compressed CIS capacity", compressed_cis),
            ("pooled CIS capacity", pooled_cis),
            ("scores", workspace),
            ("normalizers", normalizers),
            ("IDs", block_ids),
            ("mask", valid_mask),
            ("ranking", ranking),
            ("validation partials", scratch.partial),
            ("finite flag", scratch.finite),
        ),
    )
    # Resolve native modules before the first producer launch.
    if blocks > 64:
        selection = load_module("nosa_selection")
        if count >= (2047 if rows >= 1024 else 511):
            load_module("nosa_scores")
    options = {"ranking": ranking, "query_start": query_start} if blocks > 64 else {}
    prepare = prepare_ranked_out if blocks > 64 else prepare_out
    prepare(
        query,
        keys,
        cis,
        compressed_keys,
        compressed_cis,
        pooled_cis,
        **options,
        validated_start=validated_start,
        compressed_start=compressed_start,
        pooled_start=pooled_start,
        scratch=scratch,
    )
    with torch.cuda.device(query.device):
        if blocks <= 64:
            _guarded_all_blocks.run(
                scratch.finite,
                block_ids,
                valid_mask,
                heads,
                blocks,
                query_start,
                grid=(rows * heads,),
                warmup=False,
                num_warps=4,
            )
        else:
            _launch_guarded_scores(
                query,
                compressed_keys[:count],
                workspace,
                normalizers,
                scratch.finite,
                query_start,
                blocks,
            )
            with tvm_ffi.use_torch_stream():
                selection.select_ranked_guarded(
                    workspace,
                    compressed_cis[:count],
                    pooled_cis[:stable],
                    workspace,
                    block_ids,
                    valid_mask,
                    ranking,
                    scratch.finite,
                    query_start,
                )
    return block_ids, valid_mask
