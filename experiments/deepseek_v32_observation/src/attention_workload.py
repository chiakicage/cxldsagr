"""Resident indexer and sparse MLA observation with independent batch histories.

Inputs are deterministic synthetic operator operands at checkpoint dimensions.
No model activation quality or full-model resident capacity is inferred. The
single-layer resident KV allocation scales with B*(H+Q); sequences never share
history tensors. Correctness is separate from the formal timing closures.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path

import torch

from models.deepseek_v32.config import Config
from operators.deepseek_v32.attention._config import DECODE_SPLITS
from operators.deepseek_v32.attention.device_only.batch import global_indices, sparse_mla_batch
from operators.deepseek_v32.attention.device_only.mla import _flash_mla, _prepare_inputs
from operators.deepseek_v32.attention.device_only.mla import build_info as mla_build_info
from operators.deepseek_v32.indexer import resident_batch
from operators.deepseek_v32.indexer.selection import exact_topk

ROOT = Path(__file__).resolve().parents[3]
POLICY = "resident-independent-sequences-page64-q1-bf16-mla-v1"


def _case_integer(case, name):
    value = case[name]
    if type(value) is not int or value < 1:
        raise ValueError(f"case[{name!r}] must be a positive integer")
    return value


def _file_identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _normal_chunks(tensor, generator, chunk_tokens=16384):
    flat = tensor.view(-1, tensor.shape[-1])
    for start in range(0, len(flat), chunk_tokens):
        block = flat[start : start + chunk_tokens]
        if block.dtype == torch.float8_e4m3fn:
            temporary = torch.empty(block.shape, dtype=torch.bfloat16, device=block.device)
            temporary.normal_(generator=generator)
            block.copy_(temporary.to(block.dtype))
        else:
            block.normal_(generator=generator)


def _sample_signature(tensor, samples_per_sequence=32):
    """Hash bounded, uniformly spaced operand bytes per independent sequence.

    This explicitly is a sampled digest, not a claim to have hashed the full
    multi-GiB main cache. Deterministic RNG, generation policy and shape are
    separately bound in input_identity.
    """
    flat = tensor.reshape(tensor.shape[0], -1)
    count = min(flat.shape[1], samples_per_sequence)
    positions = (
        torch.arange(count, device=tensor.device, dtype=torch.int64)
        * (flat.shape[1] - 1)
        // max(count - 1, 1)
    )
    # FP8 advanced indexing is not available in every supported Torch build.
    byte_positions = (
        positions[:, None] * tensor.element_size()
        + torch.arange(tensor.element_size(), device=tensor.device)[None, :]
    ).view(-1)
    sample = flat.view(torch.uint8)[:, byte_positions]
    raw = sample.contiguous().cpu().numpy().tobytes()
    return {"sampled_sha256": hashlib.sha256(raw).hexdigest(), "sample_bytes": len(raw)}


def _work(
    name,
    *,
    shapes,
    useful_flops=None,
    precision=None,
    formula="N/A (non-matmul)",
    operator_kind=None,
    operator_family="indexer",
    parent_api=None,
    boundary=None,
):
    return {
        "name": name,
        "precision": precision,
        "useful_flops": useful_flops,
        "formula": formula,
        "input_shapes": shapes,
        "scope": "module_aggregate" if operator_kind == "api_aggregate" else "operator",
        "operator_kind": operator_kind or ("gemm" if useful_flops is not None else "non_gemm"),
        "operator_family": operator_family,
        "parent_api": parent_api or name,
        "boundary": boundary or "complete callable API",
    }


class AttentionWorkload:
    def __init__(self, case, *, device="cuda:2", seed=947):
        self.case = {name: _case_integer(case, name) for name in ("B", "Q", "H")}
        self.batch, self.queries, self.history = (self.case[name] for name in ("B", "Q", "H"))
        if self.batch != 1 and self.queries != 1:
            raise ValueError("observation currently supports Q1 batch decode or B1 extend")
        self.device = torch.device(device)
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(self.device) != (9, 0):
            raise RuntimeError("resident observation requires available Hopper SM90")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.tokens = self.history + self.queries
        self.capacity = (self.tokens + 63) // 64 * 64
        self.score_width = (self.tokens + 255) // 256 * 256
        self.paged = self.queries == 1
        self.closed = False
        self.cfg = Config()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.cuda.device(self.device), torch.no_grad():
            self.index_q = torch.empty(
                (self.batch, self.queries, 64, 128),
                device=self.device,
                dtype=torch.float8_e4m3fn,
            )
            self.index_k = torch.empty(
                (self.batch, self.capacity, 128), device=self.device, dtype=torch.float8_e4m3fn
            )
            self.index_scales = torch.empty(
                (self.batch, self.capacity), device=self.device, dtype=torch.float32
            )
            self.index_weights = torch.empty(
                (self.batch * self.queries, 64), device=self.device, dtype=torch.float32
            )
            self.main_q = torch.empty(
                (self.batch, self.queries, 128, 576), device=self.device, dtype=torch.bfloat16
            )
            self.main_kv = torch.empty(
                (self.batch, self.tokens, 576), device=self.device, dtype=torch.bfloat16
            )
            for tensor in (self.index_q, self.index_k, self.main_q, self.main_kv):
                _normal_chunks(tensor, generator)
            self.index_scales.uniform_(0.5, 1.5, generator=generator)
            self.index_weights.normal_(std=1 / 64, generator=generator)
            self.index_k[:, self.tokens :].zero_()
            self.index_scales[:, self.tokens :].zero_()
            ends = self.history + torch.arange(
                1, self.queries + 1, device=self.device, dtype=torch.int32
            )
            self.context_lens = ends.unsqueeze(0).expand(self.batch, -1).contiguous()
            self.ends = self.context_lens.view(-1)
            self.starts = torch.zeros_like(self.ends)
        self.prepared = None
        self.metadata = None
        self.scores = None
        self.values = None
        self.indices = None
        self.output = None
        self.operations = {}
        self.operator_metadata = {}
        if self.paged:
            self.operations["indexer_prepare"] = self.prepare
            self.operations["indexer_metadata"] = self.make_metadata
            self.operator_metadata["indexer_prepare"] = _work(
                "indexer_prepare",
                operator_kind="api_aggregate",
                operator_family="cache_preparation",
                boundary="page64 packing and block-table creation; overlaps independent children",
                shapes={"K": list(self.index_k.shape), "scales": list(self.index_scales.shape)},
            )
            self.operator_metadata["indexer_metadata"] = _work(
                "indexer_metadata",
                operator_family="cache_preparation",
                shapes={"context_lens": list(self.context_lens.shape)},
            )
        self.operations.update(
            indexer=self.score,
            indexer_causal_mask=self.mask,
            exact_topk=self.select,
            sparse_mla=self.attend,
        )
        pairs = self.batch * (self.queries * self.history + self.queries * (self.queries + 1) // 2)
        selected_pairs = self.batch * self.queries * min(2048, self.tokens)
        self.operator_metadata.update(
            indexer=_work(
                "indexer",
                boundary="official fused FP8 QK scoring, ReLU and weighted head reduction",
                precision="fp8",
                useful_flops=2 * 64 * 128 * pairs,
                formula="2 * 64 * 128 * B * (Q * H + Q * (Q + 1) / 2)",
                shapes={
                    "Q": list(self.index_q.shape),
                    "K": list(self.index_k.shape),
                    "weights": list(self.index_weights.shape),
                },
            ),
            indexer_causal_mask=_work(
                "indexer_causal_mask",
                shapes={"scores": [self.batch * self.queries, self.score_width]},
            ),
            exact_topk=_work(
                "exact_topk",
                operator_kind="api_aggregate",
                operator_family="topk",
                boundary="exact selection, ordering and nonfinite ID mask; overlaps independent children",
                shapes={"scores": [self.batch * self.queries, self.score_width], "K": 2048},
            ),
            sparse_mla=_work(
                "sparse_mla",
                operator_kind="api_aggregate",
                operator_family="sparse_attention",
                boundary="complete batch MLA adapter; overlaps independent preparation and fused core",
                precision="bf16",
                useful_flops=2 * selected_pairs * 128 * (576 + 512),
                formula="2 * B * Q * 2048 * 128 * (576 + 512)",
                shapes={
                    "Q": list(self.main_q.shape),
                    "KV": list(self.main_kv.shape),
                    "local_indices": [self.batch, self.queries, min(2048, self.tokens)],
                },
            ),
        )
        # Materialize every dependency once before standalone check/timing.
        self.run()
        self._build_leaf_fixtures()

    def _open(self):
        if self.closed:
            raise RuntimeError("resident attention workload is closed")

    def prepare(self):
        self._open()
        self.prepared = resident_batch.prepare_keys(
            self.index_k, self.index_scales, logical_tokens=self.tokens
        )
        return self.prepared.packed

    def make_metadata(self):
        self._open()
        self.metadata = resident_batch.schedule(self.context_lens)
        return self.metadata

    def score(self):
        self._open()
        if self.paged:
            self.scores = resident_batch.paged_logits(
                self.index_q,
                self.prepared,
                self.index_weights,
                self.context_lens,
                self.metadata,
                score_width=self.score_width,
            )
        else:
            self.scores = resident_batch.single_sequence_logits(
                self.index_q[0],
                self.index_k[0, : self.tokens],
                self.index_weights,
                self.index_scales[0, : self.tokens],
                self.starts,
                self.ends,
                score_width=self.score_width,
            )
        return self.scores

    def mask(self):
        self._open()
        return resident_batch.mask_causal_(self.scores, self.ends, history_length=self.history)

    def select(self):
        self._open()
        self.values, selected = exact_topk(self.scores, min(2048, self.tokens))
        self.indices = selected.view(self.batch, self.queries, -1)
        return self.indices

    def attend(self):
        self._open()
        self.output = sparse_mla_batch(
            self.main_q, self.main_kv, self.indices, self.cfg.attention_scale
        )
        return self.output

    def run(self):
        names = ("indexer_prepare", "indexer_metadata") if self.paged else ()
        for name in (*names, "indexer", "indexer_causal_mask", "exact_topk", "sparse_mla"):
            self.operations[name]()
        return self.output

    def _add_leaf(
        self,
        name,
        call,
        *,
        shapes,
        parent_api,
        operator_family,
        useful_flops=None,
        precision=None,
        formula="N/A (non-matmul)",
        boundary,
        operator_kind=None,
    ):
        self.operations[name] = call
        self.operator_metadata[name] = _work(
            name,
            shapes=shapes,
            useful_flops=useful_flops,
            precision=precision,
            formula=formula,
            operator_family=operator_family,
            parent_api=parent_api,
            boundary=boundary,
            operator_kind=operator_kind,
        )

    def _build_leaf_fixtures(self):
        # These are fixed, already accepted operator operands. They do not
        # participate in run(), model execution, or cache transactions.
        from operators.deepseek_v32.indexer.selection import _official_topk_int32

        self.leaf_scores = self.scores
        self.leaf_indices = self.indices
        self.leaf_mla_reference = self.output
        self.leaf_raw_values, self.leaf_raw_indices = _official_topk_int32(
            self.leaf_scores,
            min(2048, self.tokens),
            sorted_output=not self._cub_topk,
            deterministic=not self._cub_topk,
        )
        if self.paged:
            self._add_leaf(
                "page64_pack",
                self.pack_only,
                shapes={"K": list(self.index_k.shape), "scales": list(self.index_scales.shape)},
                parent_api="indexer_prepare",
                operator_family="cache_preparation",
                boundary="existing page64 kernel calls, final packed allocation; no block table",
            )
            self._add_leaf(
                "indexer_block_table",
                self.block_table_only,
                shapes={"B": self.batch, "pages_per_sequence": self.capacity // 64},
                parent_api="indexer_prepare",
                operator_family="cache_preparation",
                boundary="int32 arange for the independent-sequence page table",
            )
        self._add_leaf(
            "exact_topk_radix",
            self.topk_radix_only,
            shapes={"scores": list(self.leaf_scores.shape), "K": min(2048, self.tokens)},
            parent_api="exact_topk",
            operator_family="topk",
            boundary=(
                "official SMALL unsorted selection API; CUB ordering excluded"
                if self._cub_topk
                else "official SMALL sorted selection API; upstream ordering kernels included"
            ),
        )
        self._add_leaf(
            "topk_output_clone",
            self.topk_clone_only,
            operator_kind="api_aggregate",
            shapes={
                "values": list(self.leaf_raw_values.shape),
                "indices": list(self.leaf_raw_indices.shape),
            },
            parent_api="exact_topk",
            operator_family="topk",
            boundary="two explicit input clones used only by the postprocess fixture",
        )
        self._add_leaf(
            "topk_postprocess",
            self.topk_postprocess_only,
            operator_kind="api_aggregate",
            shapes={
                "values": list(self.leaf_raw_values.shape),
                "indices": list(self.leaf_raw_indices.shape),
            },
            parent_api="exact_topk",
            operator_family="topk",
            boundary=(
                "fresh input clones plus actual CUB composite sort and nonfinite mask; "
                "clones restore unsorted inputs each invocation; overlaps topk_output_clone"
                if self._cub_topk
                else "fresh input clones plus actual nonfinite ID mask; overlaps topk_output_clone"
            ),
        )
        for name, call, tensor in (
            ("topk_values_clone", lambda: self.leaf_raw_values.clone(), self.leaf_raw_values),
            ("topk_ids_clone", lambda: self.leaf_raw_indices.clone(), self.leaf_raw_indices),
        ):
            self._add_leaf(
                name,
                call,
                shapes={"input": list(tensor.shape)},
                parent_api="topk_output_clone",
                operator_family="topk",
                boundary="one actual Torch clone; fixture reset work, not production top-k",
            )
        self.leaf_post_values = self.leaf_raw_values.clone()
        self.leaf_post_indices = self.leaf_raw_indices.clone()
        if self._cub_topk:
            self._add_leaf(
                "topk_cub_sort_mask",
                self.topk_cub_sort_mask_only,
                shapes={
                    "values": list(self.leaf_post_values.shape),
                    "indices": list(self.leaf_post_indices.shape),
                },
                parent_api="exact_topk",
                operator_family="topk",
                boundary="one existing fused CUB sort/nonfinite-mask kernel; unsorted input buffers "
                "reset by copies before each call/replay outside measured time and DRAM counters",
            )
        else:
            self._add_leaf(
                "topk_nonfinite_mask",
                self.topk_nonfinite_mask_only,
                shapes={
                    "values": list(self.leaf_raw_values.shape),
                    "indices": list(self.leaf_raw_indices.shape),
                },
                parent_api="exact_topk",
                operator_family="topk",
                boundary="one existing nonfinite ID mask kernel; input buffers reset by copies before "
                "each call/replay outside measured time and DRAM counters",
            )
        if self.batch == self.queries == 1:
            self.leaf_flat_q, self.leaf_flat_kv, self.leaf_native_indices = _prepare_inputs(
                self.main_q.view(1, 128, 576),
                self.main_kv.view(self.tokens, 576),
                self.leaf_indices[0],
                splits=DECODE_SPLITS,
            )
            self.leaf_repeated_q = self.leaf_flat_q.repeat(DECODE_SPLITS, 1, 1)
            self._add_leaf(
                "mla_query_repeat",
                self.mla_repeat_only,
                shapes={"Q": list(self.leaf_flat_q.shape), "splits": DECODE_SPLITS},
                parent_api="sparse_mla",
                operator_family="sparse_attention",
                boundary="actual Torch query repeat for the split16 sparse decode API",
            )
            self.leaf_partials, _, self.leaf_lse = self.mla_core_only()
            self._add_leaf(
                "mla_lse_combine",
                self.mla_combine_only,
                shapes={
                    "partials": list(self.leaf_partials.shape),
                    "LSE": list(self.leaf_lse.shape),
                },
                parent_api="sparse_mla",
                operator_family="sparse_attention",
                boundary="existing natural-log LSE combine kernel and output allocation",
            )
        else:
            translated = global_indices(self.leaf_indices, self.tokens)
            self.leaf_flat_q, self.leaf_flat_kv, self.leaf_native_indices = _prepare_inputs(
                self.main_q.view(self.batch * self.queries, 128, 576),
                self.main_kv.view(self.batch * self.tokens, 576),
                translated.view(self.batch * self.queries, -1),
            )
            self._add_leaf(
                "mla_global_indices",
                self.mla_indices_only,
                operator_kind="api_aggregate",
                shapes={
                    "local_indices": list(self.leaf_indices.shape),
                    "tokens_per_sequence": self.tokens,
                },
                parent_api="sparse_mla",
                operator_family="sparse_attention",
                boundary="actual validity mask, int64 local-to-global translation and int32 conversion",
            )
            self._build_id_leaves()
        self.operator_metadata["sparse_mla"]["inactive_adaptations"] = (
            "Q/KV contiguous and aligned, IDs int32, selection width 2048 already aligned: "
            "contiguous/view/unsqueeze do not launch kernels; no pad, narrow or alignment clone executes"
        )
        original = self.operator_metadata["sparse_mla"]
        self._add_leaf(
            "sparse_mla_core",
            self.mla_core_only,
            shapes={
                "Q": list(
                    (
                        self.leaf_repeated_q
                        if self.batch == self.queries == 1
                        else self.leaf_flat_q
                    ).shape
                ),
                "KV": list(self.leaf_flat_kv.shape),
                "indices": list(self.leaf_native_indices.shape),
            },
            useful_flops=original["useful_flops"],
            precision="bf16",
            formula=original["formula"],
            parent_api="sparse_mla",
            operator_family="sparse_attention",
            boundary="official fused sparse QK/PV and softmax API with prepared operands; "
            "includes native outputs/LSE, excludes adapter ID conversion, query repeat and combine",
        )

    def _build_id_leaves(self):
        ids = self.leaf_indices
        self.leaf_id_nonnegative = ids >= 0
        self.leaf_id_upper_bound = ids < self.tokens
        self.leaf_id_valid = self.leaf_id_nonnegative & self.leaf_id_upper_bound
        self.leaf_sequence_arange = torch.arange(self.batch, device=self.device, dtype=torch.int64)
        self.leaf_id_int64 = ids.to(torch.int64)
        self.leaf_sequence_offsets = self.leaf_sequence_arange[:, None, None] * self.tokens
        self.leaf_id_translated = self.leaf_id_int64 + self.leaf_sequence_offsets
        self.leaf_id_where = torch.where(self.leaf_id_valid, self.leaf_id_translated, -1)
        leaves = (
            ("mla_id_nonnegative", lambda: ids >= 0, [ids], "Torch local_ids >= 0"),
            (
                "mla_id_upper_bound",
                lambda: ids < self.tokens,
                [ids],
                "Torch local_ids < tokens_per_sequence",
            ),
            (
                "mla_id_valid_and",
                lambda: self.leaf_id_nonnegative & self.leaf_id_upper_bound,
                [self.leaf_id_nonnegative, self.leaf_id_upper_bound],
                "Torch boolean mask conjunction",
            ),
            (
                "mla_sequence_arange",
                lambda: torch.arange(self.batch, device=self.device, dtype=torch.int64),
                [],
                "Torch int64 sequence arange",
            ),
            (
                "mla_id_to_int64",
                lambda: ids.to(torch.int64),
                [ids],
                "Torch int32 to int64 ID conversion",
            ),
            (
                "mla_sequence_offset_mul",
                lambda: self.leaf_sequence_arange[:, None, None] * self.tokens,
                [self.leaf_sequence_arange],
                "Torch sequence offset multiplication; views excluded",
            ),
            (
                "mla_id_offset_add",
                lambda: self.leaf_id_int64 + self.leaf_sequence_offsets,
                [self.leaf_id_int64, self.leaf_sequence_offsets],
                "Torch local ID and sequence offset addition",
            ),
            (
                "mla_id_where",
                lambda: torch.where(self.leaf_id_valid, self.leaf_id_translated, -1),
                [self.leaf_id_valid, self.leaf_id_translated],
                "Torch invalid-ID masking with where",
            ),
            (
                "mla_id_to_int32",
                lambda: self.leaf_id_where.to(torch.int32),
                [self.leaf_id_where],
                "Torch int64 to int32 final ID conversion",
            ),
        )
        for name, call, inputs, boundary in leaves:
            self._add_leaf(
                name,
                call,
                shapes={
                    "inputs": [list(t.shape) for t in inputs],
                    "B": self.batch,
                    "N": self.tokens,
                },
                parent_api="mla_global_indices",
                operator_family="sparse_attention",
                boundary=boundary,
            )

    def before_call(self, name):
        if name not in {"topk_cub_sort_mask", "topk_nonfinite_mask"}:
            return None

        def reset_postprocess_inputs():
            self.leaf_post_values.copy_(self.leaf_raw_values)
            self.leaf_post_indices.copy_(self.leaf_raw_indices)

        return reset_postprocess_inputs

    def topk_cub_sort_mask_only(self):
        self._open()
        from operators.deepseek_v32.indexer.q1_topk_cub import sort_mask_

        sort_mask_(self.leaf_post_values, self.leaf_post_indices)
        return self.leaf_post_values, self.leaf_post_indices

    def topk_nonfinite_mask_only(self):
        self._open()
        from operators.deepseek_v32.indexer.selection import _mask_nonfinite_indices_

        _mask_nonfinite_indices_(self.leaf_post_values, self.leaf_post_indices)
        return self.leaf_post_indices

    @property
    def _cub_topk(self):
        return self.batch * self.queries == 1 and self.score_width >= 32768

    def graph_transform(self, name):
        if name == "exact_topk_radix" and self._cub_topk:

            def ordered_selection(output):
                from operators.deepseek_v32.indexer.q1_topk_cub import sort_mask_

                values, indices = (tensor.clone() for tensor in output)
                sort_mask_(values, indices)
                return values, indices

            return ordered_selection
        return None

    def pack_only(self):
        self._open()
        return resident_batch._pack_independent_keys(
            self.index_k, self.index_scales, self.capacity // 64
        )

    def block_table_only(self):
        self._open()
        pages = self.capacity // 64
        return torch.arange(self.batch * pages, device=self.device, dtype=torch.int32).view(
            self.batch, pages
        )

    def topk_radix_only(self):
        self._open()
        from operators.deepseek_v32.indexer.selection import _official_topk_int32

        return _official_topk_int32(
            self.leaf_scores,
            min(2048, self.tokens),
            sorted_output=not self._cub_topk,
            deterministic=not self._cub_topk,
        )

    def topk_clone_only(self):
        self._open()
        return self.leaf_raw_values.clone(), self.leaf_raw_indices.clone()

    def topk_postprocess_only(self):
        self._open()
        values, indices = self.topk_clone_only()
        if self._cub_topk:
            from operators.deepseek_v32.indexer.q1_topk_cub import sort_mask_

            sort_mask_(values, indices)
        else:
            from operators.deepseek_v32.indexer.selection import _mask_nonfinite_indices_

            _mask_nonfinite_indices_(values, indices)
        return values, indices

    def mla_repeat_only(self):
        self._open()
        return self.leaf_flat_q.repeat(DECODE_SPLITS, 1, 1)

    def mla_indices_only(self):
        self._open()
        return global_indices(self.leaf_indices, self.tokens)

    def mla_core_only(self):
        self._open()
        split = self.batch == self.queries == 1
        q = self.leaf_repeated_q if split else self.leaf_flat_q
        ids = (
            self.leaf_native_indices.view(DECODE_SPLITS, 1, -1)
            if split
            else self.leaf_native_indices.unsqueeze(1)
        )
        return _flash_mla().flash_mla_sparse_fwd(
            q,
            self.leaf_flat_kv.unsqueeze(1),
            ids,
            float(self.cfg.attention_scale),
            512,
            topk_length=None,
        )

    def mla_combine_only(self):
        self._open()
        from operators.deepseek_v32.attention.device_only.decode import _combine_kernel, _observed

        output = torch.empty((1, 128, 512), dtype=self.main_q.dtype, device=self.device)
        compiled = _combine_kernel()[(128,)](
            self.leaf_partials,
            self.leaf_lse,
            output,
            128,
            DECODE_SPLITS,
            512,
            num_warps=4,
        )
        _observed[compiled.hash] = compiled
        return output

    def validate_leaves(self):
        from experiments.deepseek_v32_observation.src.attention_validation import validate_leaves

        return validate_leaves(self)

    def input_identity(self):
        self._open()
        tensors = {
            name: getattr(self, name)
            for name in ("index_q", "index_k", "index_scales", "index_weights", "main_q", "main_kv")
        }
        return {
            "case": self.case,
            "seed": self.seed,
            "policy": POLICY,
            "generation": "Torch CUDA Philox, BF16 normal chunks of 16384 tokens; independent storage",
            "torch_version": torch.__version__,
            "scope": "deterministic generation + sampled bytes",
            "tensors": {
                name: {"shape": list(t.shape), "dtype": str(t.dtype), **_sample_signature(t)}
                for name, t in tensors.items()
            },
        }

    def validate(self):
        from experiments.deepseek_v32_observation.src.attention_validation import validate_attention

        return validate_attention(self)

    def source_identity(self):
        self._open()
        files = [
            Path(__file__),
            Path(__file__).with_name("attention_validation.py"),
            ROOT / "models/deepseek_v32/config.py",
            ROOT / "operators/deepseek_v32/indexer/resident_batch.py",
            ROOT / "operators/deepseek_v32/indexer/page64.py",
            ROOT / "operators/deepseek_v32/indexer/echo.py",
            ROOT / "operators/deepseek_v32/indexer/selection.py",
            ROOT / "operators/deepseek_v32/indexer/q1_topk_cub.py",
            ROOT / "operators/deepseek_v32/indexer/csrc/q1_topk_sort.cu",
            ROOT / "operators/deepseek_v32/attention/device_only/batch.py",
            ROOT / "operators/deepseek_v32/attention/reference/torch.py",
        ]
        result = {"policy": POLICY, "source": [_file_identity(path) for path in files]}
        result["flash_mla"] = mla_build_info()
        for package_name, native_name in (("deep-gemm", "deep_gemm._C"),):
            package = importlib.import_module(native_name)
            result[package_name] = {
                "version": importlib.metadata.version(package_name),
                "native": _file_identity(package.__file__),
            }
        from operators.deepseek_v32.attention.device_only.decode import runtime_info

        result["decode_runtime"] = runtime_info()
        return result

    def close(self):
        if self.closed:
            return
        # Failed synchronization propagates before releasing tensors. Storage
        # must remain owned while asynchronous completion is uncertain.
        torch.cuda.synchronize(self.device)
        self.closed = True
        for name in tuple(vars(self)):
            if name.startswith("leaf_"):
                setattr(self, name, None)
        for name in (
            "prepared",
            "metadata",
            "scores",
            "values",
            "indices",
            "output",
            "index_q",
            "index_k",
            "index_scales",
            "index_weights",
            "main_q",
            "main_kv",
            "context_lens",
            "ends",
            "starts",
        ):
            setattr(self, name, None)


def build_attention_workload(case, *, device="cuda:2", seed=947):
    return AttentionWorkload(case, device=device, seed=seed)
