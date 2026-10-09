# Stateless Q1 preparation: independent production review

Review date: 2026-10-09. Reviewer: `/root/production_review`.

The final integration patch and the bounded fresh-cohort evidence reviewed below
are approved for publication alongside root's completed numerical and report
audit. No concrete publication blocker remains. This review covers source,
runtime participation, native artifacts, raw graph continuity, retained samples,
and reported memory boundaries under the
[integration plan](fused_prepare_integration_plan.md).

## Reviewed identity

Reviewed [patch](fused_prepare_integration.patch): 43,876 bytes, SHA256
`6d35542bb9aea59a50ad29b6a9ce5637096d6702996fe9467b038281fe3bf624`.
The reviewer independently checked that all nine applied files exactly match
the reviewed staging copies and the `source_sha256` entries in
`/tmp/cxldsagr-checks/q1-fused-prepare-production/integration_stage_20261009_01/applied_sources.json`.
The original files remain in that directory's `old/` tree for this acceptance.

| File | Applied SHA256 |
| --- | --- |
| `operators/deepseek_v32/indexer/csrc/official_prefetch.cu` | `020daf60e2acdeb3195b34485f036fe1074f2b5501754ba95bc72c94e4e3e1e3` |
| `operators/deepseek_v32/indexer/official_prefetch.py` | `c4efd30949e1dbef774e98383c8dc7ac33371bfa0afa1d2c4bacdfaef7871929` |
| `operators/deepseek_v32/indexer/echo.py` | `d451c53d4dae4b13ef7f3b8486bf8e0a06bf5f416162504da7d0851f88ea4203` |
| `experiments/deepseek_v32_mfu/src/run_contract.py` | `8949468a482e5c1128032e5e01170768c82e2a3835f419cf62b7cd5999c608ee` |

## Independent equality checks

The reviewer compared the applied CUDA file with the pre-integration copy.
Every byte before the original namespace close is unchanged, including
`prepare_kernel`, `validate_promotion_kernel`, `copy_publish_kernel`,
`clear_kernel`, their host launchers, constants, and stream/error helpers.
The original three exports remain present. The addition does not modify or
copy an official ECHO header.

The new fused template and its host launcher exactly match the accepted private
`experiments/deepseek_v32_echo_official/src/q1_fused_prepare.cu` body after only
renaming `prepare_kernel` to `prepare_keys_kernel` and the host `prepare` to
`prepare_keys`. The namespace and export names are the production adaptation.
This source comparison does not normalize or equate native ELF identities.

AST comparisons, excluding source-position attributes, independently confirmed
that these definitions are unchanged:

- `echo.py`: `_check_tensor`, `_validate_prefetch`, `_PreparedPrefetch`,
  `_BoundedPreparedPrefetch`, `_uses_official_q1`, and `_paged_q1_logits`.
- `official_prefetch.py`: `_Staging`, `_call`, and `_promote`.

The `echo.logits` diff changes only the existing official-Q1 branch from separate
packing/table/schedule preparation to `logits_from_keys`. Bounds creation,
validation, the physical output-tail view, resident dispatch, and generic
offload dispatch retain their existing code. The original packed-input `logits`
and `_prepare` interfaces remain usable as controls.

## Native and transaction reasoning

The fused kernel copies FP8 key and FP32 scale bits without arithmetic, emits
zero page padding, rebuilds the identity block table and current logical host
table, and resets the original 64 staging IDs and counter on every launch.
Only initialized history reads the host page table; the suffix receives no host
record. The context assertion precedes key/table reads, and host-page assertions
retain their original bounds. Signed-int32 index and tag bounds remain checked.

Python retains shape, dtype, device, contiguous/no-gradient, SM90, lease-capacity,
and ownership checks. Valid key storage requires four-byte alignment; the native
launcher also checks the actual pointer and selects 16-byte vector copies only
when that pointer permits them. Page and row strides preserve that alignment.
The FFI call uses `tvm_ffi.use_torch_stream()`, and the kernel obtains the current
stream through `TVMFFIEnvGetStream` for the same device.

All new packing allocations finish before cleanup registration. The original
staging owner and callback are registered before any launch can publish tags.
Bounded tokens are consumed once with the explicit 64-slot consumer limit;
full tokens retain their original consumption path. Unprepared leases retain
the full journal/statistics reset. Promotion, host-ID-zero cleanup, finalize,
owner retention after uncertain completion, and exception propagation remain
the existing caller contract. No retry, fallback, persistent packed history, or
new storage class is introduced.

## Runtime participation finding and resolution

The first staged runtime gate only waived resident page64 evidence for cold A1
with `P-H>=64`. That was too narrow: official Q1 can also use full preparation,
and a warm or chunked request can reach the same current-key entry. The reviewer
and implementation agent resolved this before application.

The final gate requires a declared preparation version, a normally returned
`logits_from_keys` entry marker, matching native source identity, and exactly one
matching mapped adapter path/name/SHA256. Merely loading the packed-input control
or declaring build capability supplies no marker. The marker is stable across
measurement phases and does not contain invocation counts.

Cold A1 with at least 64 legal prefetch slots requires the marker even when
`P-H<64`; warming an unused page64 specialization cannot satisfy that gate.
An observed valid marker can cover a large-Q1 extend boundary, including
single-query chunks and a final one-query tail. Resident/legacy paths and a
resident Q1 prefill tail still require the page64 specialization. This proves
process participation, not every call or complete graph equivalence; formal
profile lineage must establish actual production nodes.

The new kernel stays in namespace `official_prefetch`, whose existing timeline
rule classifies it as GPU control before broad operator ownership. Both alignment
specializations have classification cases. Existing `indexer_fused` hooks include
the work once; the preparation adds neither matrix FLOPs nor H2D payload.

## Acceptance boundary

The implementation agent reported 23 focused runtime-gate tests passing with
21 deselected, then 261 production CPU tests passing with 24 expected GPU skips
and 151 GPU1/SM90 tests passing with no skips. Root reported 5,090 global CPU
tests and 58 subtests passing, with 1,596 optional skips. Scoped Ruff and
`git diff --check` passed. These test outcomes are the executing agents' terminal
evidence; the reviewer inspected the test cases and independently ran the
equality/hash checks above. The reviewer did not run GPU work.

The reviewer also inspected root's scope audit and independently rehashed all
58 recorded inputs, verified the 12 MFU and 12 C10 configuration rows exclude
Q1 indexer batches, re-evaluated all five `execution_reservation` samples, and
recomputed the Q1 allocation dimensions. All checks passed.
`/tmp/cxldsagr-checks/q1-fused-prepare-production/scope_audit.json` has SHA256
`c843b58df96f0c0bcb49f6d418eb5ea39d000994a5dd916ca10433d2727d2052`.
The A128 single point and post-top-k entries preserve their original boundaries.
This does not refresh C10 request memory, capacity-fill acceptance, or graph
reserved/device-used measurements.

## Fresh production cohort: publication gate passed

The final independent auditor completed successfully before the review turn was
interrupted. Its script and output are:

- `/tmp/cxldsagr-checks/q1-fused-prepare-production/independent_review_20261009_01/audit.py`
- `/tmp/cxldsagr-checks/q1-fused-prepare-production/independent_review_20261009_01/native_graph.json`

The final output contains `native`, `graph`, and `cohort` sections, with input
hashes. The cohort is `deepseek_h64k_a1_fused_prepare_20261009_01`; the comparison
uses the original `deepseek_h64k_a1_isolated_20261008_01` artifacts. The reviewer
performed no GPU work and did not modify producer sources.

Independent `cuobjdump` extraction reproduced the complete 1,218,840-byte SM90a
CUBIN, SHA256
`0131578b8d4ab4e55e9cf1cd728387cec7eff2d36032835344d7847173410e43`.
It is byte-identical to the preserved formal baseline. All 223 non-null ELF
sections, including 44 `.text.*` sections, were independently parsed and checked.
The generic build closure changes only the `echo.py` source hash; other build
fields are equal. The full generic ELF remains a distinct artifact, SHA256
`afafc5d3bee190963b64821cce3b18ef852ff00be425cf77015916480c6b08f1`.
The saved post-acceptance mappings are explicitly separate from the focused
pytest process, which did not archive its process mappings.

For both the minimal and operator NSYS captures, an independent SQLite reader
selected the unique formal graph launch through its NVTX range and CUDA
correlation, then resolved every GPU node through recorded clone lineage and
checked PID/device ownership. The complete graph changes from 263 to 257 nodes.
Preparation changes from nine nodes to three, with one fused preparation per
layer. The remaining 254 nodes have equal full owner/signature multisets.
The retained query-bounds `arange` uses grid 1; the removed block-table `arange`
uses grid 17. They are not conflated by kernel name.

Each production layer owns 83 nodes, so L0-L2 owns 249; eight shared nodes bring
the complete graph to 257. Minimal-profile preparation is 6.176, 6.272, and
6.208 microseconds for L0-L2, totaling 18.656 microseconds. Operator-profile
preparation is 6.944, 5.984, and 6.112 microseconds, totaling 19.040 microseconds.
These are intrusive profile kernel sums, not clean end-to-end speedup estimates.

The fresh-phase and measurement checks also passed:

- All four ECHO phases carry the same observed `logits_from_keys` marker, declared
  preparation identity, native source identity, and mapped immutable adapter.
  Archived/current preparation sources and the applied operator hashes match.
  ECHO does not claim an unused page64 specialization; the three resident-indexer
  methods retain that specialization and have no fused-entry marker.
- All 12 offload children bind the independently checked generic ELF. The
  cohort contains 16 distinct process IDs, each reporting only its selected
  method's preparation and warmup. This is not a claim of continuous hardware
  isolation.
- All four methods retain three prefill and five extend samples; all medians
  recompute exactly. The corresponding cache sample counts are complete.
  All 15 ECHO layer samples satisfy cap 64, zero evictions, complete selection
  accounting, and the actual H2D/D2H byte formulas.
- For every reported `measurements.<method>.extend_graph_runtime`, the reviewer
  checked one graph, 62,914,560 bytes of private reserved memory, 512 bytes of
  static allocation, and separately reported
  `allocated <= reserved <= device_used <= device_total` values.

Root's full report audit supplies the numerical, receipt, saved-output,
process/traffic, and publication checks beyond this bounded independent review.
The remaining boundaries are explicit: this reviewer did not reconstruct raw
allocator segments, prove complete graph-edge equality, infer a causal wall-time
gain from the preparation sum, subtract formal-cohort medians to estimate
candidate speedup, or establish cross-implementation numerical equivalence.
None is required for the scoped publication claims. The publication gate passes;
superseded outputs can be replaced under the experiment publication rules.
