# C5: direct value-expansion output layout

Status: the original C5 candidate passed CPU and component GPU checks; see
[the checkpoint](deepseek_output_layout_checkpoint.md). Later production
factoring for C6 requires separate validation. C4 is frozen separately at
`/tmp/deepseek-motivation-c4-frozen-48yhnrcs`; its results must not be presented
as C5 results. The root agent schedules every GPU run.

## Evidence and scope

`CheckpointAttention.output` currently computes `torch.bmm` into a contiguous
head-major `[heads,Q,v_dim]` tensor, transposes to `[Q,heads,v_dim]`, then
reshapes to `[Q,heads*v_dim]` for `o_proj`. Except for degenerate dimensions,
the reshape copies the full value-expansion result. At Q=1024, heads=128 and
v_dim=128, this is a 32 MiB BF16 tensor per layer call.

The accepted C3 full profile has 640 history value-expansion calls per cold
request. Their API-active time is about 34 ms. The surrounding graph finish
also contains noncontiguous BF16 packing. Its full 133–135 ms auxiliary total
includes several operations and is not the expected gain from this candidate.
The C2 query-layout work already validated cuBLAS strided output for its own
shape; that evidence motivates C5 but does not validate C5's dimensions,
algorithm choice, numerical outputs or performance.

Scope is one CUDA-only layout change in `models/deepseek_v32/echo_model.py`.
Keep CPU behavior and all validation unchanged. Allocate contiguous
`[Q,heads,v_dim]`, give its `[heads,Q,v_dim]` transpose as `torch.bmm(out=...)`,
then pass its zero-copy `[Q,heads*v_dim]` view to the existing `o_proj`.
The vendor BMM, independent checkpoint weights, BF16 result and projection
API remain the same; the strided destination may select a different vendor
algorithm and therefore requires explicit exact-output validation.

Before-edit SHA-256 of `echo_model.py`:
`cb4bfcd40f7675ccb76097a2c7eaa36f13cfe27eeb497ac709866849f7a8a0a7`.
The exact backup is `/tmp/deepseek_output_layout/echo_model_before.py`.

## Invariants

- Input attention is unchanged, and output rows retain token/head/value order.
- The BMM writes disjoint output elements despite interleaved head batches:
  destination offset is `token * heads * v_dim + head * v_dim + value`.
- The flattened projection input aliases the BMM destination and is contiguous.
- No cache/session state, graph input ownership, attention numerical formula,
  precision setting, norm/residual order or third-party source changes.
- Graph capture still uses its private allocator for the intermediate result;
  actual private/reserved capacity is audited by the existing graph checks.
- Instrumentation already forwards BMM `out=` and identifies the operator by
  the unchanged second-operand weight pointer. Useful FLOPs remain unchanged.

## Verification and measurement gate

1. CPU/static verification checks destination strides, storage identity, input
   preservation and exact baseline values, including Q=1 and noncontiguous
   attention. Existing model/block/profiler tests cover unchanged CPU behavior
   and BMM instrumentation. These checks cannot establish CUDA correctness.
2. After root assigns a GPU, compare baseline and candidate with actual source
   checkpoint layers 0–2, FP8 projection backends and Q=128/1024 plus small/tail
   shapes. Check expanded heads and complete `output` results byte-for-byte.
   Record vendor kernel identities and actual `out=` destination use; confirm
   no hidden output copy replaces the removed reshape.
3. Compare complete value expansion plus `o_proj` and captured finish-graph
   timings, including allocation/setup rules, at Q=128 and Q=1024. Keep the
   candidate only if exact outputs pass and measured cost improves.
4. Freeze C5 separately, rerun real-checkpoint eager/graph and complete serving
   acceptance, then use a new formal/profile run before updating affected
   performance claims. Retain currently valid artifacts until publication.

## Separate C6 idea

Root is considering moving only value expansion before the finish graph,
writing directly into a smaller static `[Q,heads,v_dim]` graph input. That
could remove the larger static attention-input copy because FlashMLA has no
`out=` API. It changes the eager/graph boundary, resource plan and static-input
contract; an extra eager launch may hurt Q=128. C5 does not make this change or
factor new public methods solely for it. C6 requires its own measurements and
root-owned graph-resource edits.

## C5 CPU handoff

The CUDA-only direct destination is implemented without changing CPU dispatch
or factoring new graph APIs. `test_output_layout.py` selects the CUDA layout
branch with a tensor subclass that retains CPU dispatch. It checks exact
expanded/projection values, input preservation, destination strides and
zero-copy contiguous flattening for Q=1/5/128/1024 with contiguous and strided
inputs. This is layout/ownership evidence, not CUDA numerical validation.

Existing model/block and operator instrumentation tests plus these new cases
completed with **44 passed** in 2.30 seconds under `CUDA_VISIBLE_DEVICES=''`.
Ruff check and format check passed. Subsequent component GPU results are in
the checkpoint; they identify the original C5 source explicitly.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  models/deepseek_v32/tests/test_output_layout.py \
  models/deepseek_v32/tests/test_echo_model.py \
  models/deepseek_v32/tests/test_echo_block.py \
  experiments/deepseek_v32_echo_prefill/tests/test_operator_instrumentation.py
```

| Source | SHA-256 |
|---|---|
| `models/deepseek_v32/echo_model.py` | `a50b73b283b79acf6628d708564af1d9fd66f4b0269920f42ef5a05f78c99e27` |
| `models/deepseek_v32/tests/test_output_layout.py` | `f8243524d62dc3723672586b8c612255b31547b6e15cf0fa44c62f845b675471` |
