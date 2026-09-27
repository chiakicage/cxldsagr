"""Reversible NOSA stage scopes for profiling a serial resident request.

These scopes instrument the existing model, indexer and attention calls; they
do not reproduce inference. CUDA events measure inclusive stream intervals.
Indexer subscopes are nested and must not be added to ``indexer_total``.
Shape/selection audits run separately, with timing disabled.
"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps

import torch


class _CudaClock:
    def __init__(self, device):
        self.device = device

    def record(self):
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        return event

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    @staticmethod
    def elapsed_ms(start, end):
        return start.elapsed_time(end)


class _NullNVTX:
    def range_push(self, label):
        pass

    def range_pop(self):
        pass


class _WrappedCallable:
    """Keep policy/backend attributes visible through a temporary call wrapper."""

    def __init__(self, original, call):
        self.original = original
        self.call = call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


@dataclass(frozen=True)
class _LayerCall:
    layer_idx: int
    call_id: int
    query_start: int
    query_length: int


class SparseScopes:
    """Collect per-layer NVTX scopes and optional deferred CUDA event timings.

    ``with SparseScopes(model, phase="extend", timing=True) as scopes: ...``
    wraps only work performed inside the context. ``collect()`` synchronizes
    once for all new events, either inside or after the context, and returns
    JSON-compatible records. Repeated collection without new work is cached.
    Context exit never synchronizes and always restores the original calls.

    ``audit=True`` requires ``timing=False``. Audit records add ``details`` with
    Q/K/V shapes, selection shape/budget and valid-block statistics. Reductions
    and host transfers used by auditing are excluded from timing runs.

    Every record carries ``stage``, ``phase``, ``layer_idx``, ``call_id``,
    ``query_start``, ``query_length``, ``scope_id``, ``parent_scope_id``,
    ``inclusive=True``, ``status`` and ``cuda_elapsed_ms`` (null when untimed).
    Query metadata describes the enclosing attention-layer invocation, even
    for a smaller query tile processed inside the indexer. ``call_id`` is a
    unique layer invocation within this context, not a repetition/run ID.

    The optional ``clock`` supplies record(), synchronize(), elapsed_ms(a,b),
    and ``nvtx`` supplies range_push(label)/range_pop(). They allow CPU tests
    to validate ordering and restoration without claiming GPU measurements.
    Module patches are process-global: overlapping contexts are rejected, and
    the experiment must keep the project's serial model execution contract.
    """

    _active_instance = None

    def __init__(self, model, *, phase="extend", timing=True, audit=False, clock=None, nvtx=None):
        if phase not in ("extend", "full_prefill"):
            raise ValueError("phase must be extend or full_prefill")
        if audit and timing:
            raise ValueError("Selection audit must run separately with timing=False")
        if getattr(model, "attention_mode", None) != "sparse":
            raise ValueError("SparseScopes requires an existing NOSA sparse model")
        if not callable(model.indexer) or not callable(model.main_attention):
            raise TypeError("SparseScopes requires callable model indexer and main_attention")
        self.model = model
        self.phase = phase
        self.timing = timing
        self.audit = audit
        self.device = next(model.parameters()).device
        if timing and clock is None and self.device.type != "cuda":
            raise ValueError("CUDA stage timing requires a CUDA model")
        self.clock = clock if clock is not None else (_CudaClock(self.device) if timing else None)
        self.nvtx = (
            nvtx
            if nvtx is not None
            else (torch.cuda.nvtx if self.device.type == "cuda" else _NullNVTX())
        )
        self._patches = None
        self._entered = False
        self._layer_calls = []
        self._scope_stack = []
        self._compression_calls = []
        self._next_call_id = 0
        self._records = []
        self._pending_events = []
        self._collected_count = 0

    def __enter__(self):
        if self._entered:
            raise RuntimeError("A SparseScopes instance can only be entered once")
        if SparseScopes._active_instance is not None:
            raise RuntimeError("Overlapping SparseScopes contexts are not supported")
        self._entered = True
        SparseScopes._active_instance = self
        self._patches = ExitStack()
        try:
            self._install()
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._restore()
        return False

    def _restore(self):
        try:
            if self._patches is not None:
                self._patches.close()
        finally:
            self._layer_calls.clear()
            self._scope_stack.clear()
            self._compression_calls.clear()
            if SparseScopes._active_instance is self:
                SparseScopes._active_instance = None

    def _patch(self, owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self._patches.callback(setattr, owner, name, original)

    @staticmethod
    def _argument(args, kwargs, name, position):
        return kwargs[name] if name in kwargs else args[position]

    def _attention_enter(self, layer_idx):
        def hook(module, args, kwargs):
            x = self._argument(args, kwargs, "x", 0)
            cache = self._argument(args, kwargs, "cache", 4)
            frame = _LayerCall(
                layer_idx=layer_idx,
                call_id=self._next_call_id,
                query_start=0 if cache is None else cache.length,
                query_length=x.shape[0],
            )
            self._next_call_id += 1
            self._layer_calls.append(frame)

        return hook

    def _attention_exit(self, module, args, kwargs, output):
        if self._layer_calls:
            self._layer_calls.pop()

    def _frame(self, context=None):
        if not self._layer_calls:
            raise RuntimeError("Instrumented calls must run inside the model attention layer")
        frame = self._layer_calls[-1]
        if context is not None and (
            frame.layer_idx != context.layer_idx
            or frame.query_start != context.query_start
            or frame.query_length != context.query_length
        ):
            raise RuntimeError("Attention context disagrees with the instrumented layer call")
        return frame

    @contextmanager
    def _scope(self, stage, frame):
        scope_id = len(self._records)
        record = {
            "stage": stage,
            "phase": self.phase,
            "layer_idx": frame.layer_idx,
            "call_id": frame.call_id,
            "query_start": frame.query_start,
            "query_length": frame.query_length,
            "scope_id": scope_id,
            "parent_scope_id": self._scope_stack[-1] if self._scope_stack else None,
            "inclusive": True,
            "status": "ok",
            "cuda_elapsed_ms": None,
        }
        self._records.append(record)
        label = (
            f"NOSA/{self.phase}/layer_{frame.layer_idx}/{stage}"
            f"/q{frame.query_start}+{frame.query_length}"
        )
        self.nvtx.range_push(label)
        self._scope_stack.append(scope_id)
        start = None
        try:
            if self.timing:
                start = self.clock.record()
            yield record
        except BaseException as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            raise
        finally:
            try:
                if self.timing and start is not None:
                    end = self.clock.record()
                    self._pending_events.append((scope_id, start, end))
            finally:
                self._scope_stack.pop()
                self.nvtx.range_pop()

    def _install(self):
        import models.nosa.indexer as indexer_module
        import models.nosa.scoring as scoring_module
        import operators.sm90.nosa_indexer as operator_module
        import operators.sm90.nosa_validation as validation_module

        original_indexer = self.model.indexer
        original_attention = self.model.main_attention

        def indexer(q, cache_access, context):
            frame = self._frame(context)
            details = self._audit_qkv(q, cache_access, context) if self.audit else None
            with self._scope("indexer_total", frame) as record:
                self._compression_calls.append(0)
                try:
                    selection = original_indexer(q, cache_access, context)
                finally:
                    self._compression_calls.pop()
                if self.audit:
                    record["details"] = details | self._audit_selection(selection)
                return selection

        def attention(q, selection, cache_access, context):
            frame = self._frame(context)
            details = None
            if self.audit:
                details = self._audit_qkv(q, cache_access, context) | self._audit_selection(
                    selection
                )
            with self._scope("block_sparse_attention", frame) as record:
                if details is not None:
                    record["details"] = details
                return original_attention(q, selection, cache_access, context)

        self._patch(self.model, "indexer", _WrappedCallable(original_indexer, indexer))
        self._patch(self.model, "main_attention", _WrappedCallable(original_attention, attention))
        self._wrap_function(scoring_module, "cis_scores", "cis_projection")
        self._wrap_function(indexer_module, "compress_sequence", "compression")
        self._wrap_function(indexer_module, "prepare_indexer_inputs", "indexer_cache_update")
        self._wrap_function(validation_module, "all_finite", "indexer_validate")
        self._wrap_function(operator_module, "compressed_scores", "compressed_scores")
        self._wrap_function(operator_module, "_select_validated_scores", "select_from_scores")
        for stage in ("pooled_scores", "topk_qa", "prepare_cis", "topk_cis", "finish_selection"):
            self._wrap_function(operator_module, stage, stage)
        for layer_idx, layer in enumerate(self.model.model.layers):
            before = layer.self_attn.register_forward_pre_hook(
                self._attention_enter(layer_idx), with_kwargs=True
            )
            self._patches.callback(before.remove)
            after = layer.self_attn.register_forward_hook(
                self._attention_exit, with_kwargs=True, always_call=True
            )
            self._patches.callback(after.remove)

    def _wrap_function(self, module, name, stage):
        original = getattr(module, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            # Calls unrelated to this model remain untouched by global patches.
            if not self._layer_calls or (stage != "cis_projection" and not self._compression_calls):
                return original(*args, **kwargs)
            label = stage
            if stage == "compression":
                index = self._compression_calls[-1]
                self._compression_calls[-1] += 1
                label = ("compression_k", "compression_cis")[index] if index < 2 else "compression"
            with self._scope(label, self._frame()) as record:
                if self.audit and args and isinstance(args[0], torch.Tensor):
                    record["details"] = {"input_shape": list(args[0].shape)}
                result = original(*args, **kwargs)
                if self.audit and isinstance(result, torch.Tensor):
                    record["details"] = record.get("details", {}) | {
                        "output_shape": list(result.shape)
                    }
                return result

        self._patch(module, name, wrapped)

    @staticmethod
    def _audit_qkv(q, cache_access, context):
        records = cache_access.layer_view(context.layer_idx)
        return {
            "q_shape": list(q.shape),
            "k_shape": list(records["keys"].shape),
            "v_shape": list(records["values"].shape),
            "dtype": str(q.dtype),
            "device": str(q.device),
        }

    @staticmethod
    def _audit_selection(selection):
        ids = selection.block_ids
        valid = ids >= 0
        if selection.valid_mask is not None:
            valid &= selection.valid_mask
        counts = valid.sum(dim=-1).cpu()
        return {
            "selection_shape": list(ids.shape),
            "block_size": selection.block_size,
            "block_budget": ids.shape[-1],
            "valid_blocks_min": int(counts.min()) if counts.numel() else 0,
            "valid_blocks_max": int(counts.max()) if counts.numel() else 0,
            "valid_blocks_mean": float(counts.float().mean()) if counts.numel() else 0.0,
            "valid_entries": int(counts.sum()),
        }

    def collect(self):
        """Synchronize new event pairs once, preserving inclusive nesting metadata."""
        if self._scope_stack:
            raise RuntimeError("Cannot collect while an instrumented scope is executing")
        if len(self._pending_events) > self._collected_count:
            self.clock.synchronize()
            for scope_id, start, end in self._pending_events[self._collected_count :]:
                self._records[scope_id]["cuda_elapsed_ms"] = float(
                    self.clock.elapsed_ms(start, end)
                )
            self._collected_count = len(self._pending_events)
        return deepcopy(self._records)
