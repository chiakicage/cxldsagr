# Exact Q1 mean: bounded production integration

Status: private component/model gates accepted; bounded production integration
is in progress. Root owns GPU execution and result publication. The private
model timing and raw profile are accepted in [the model result](q1_hint_model_result.md).
The private component has accepted numerical/timing evidence in
[the component result](q1_hint_candidate_result.md). The private full-model
check is `q1_hint_model_check_20261008_01`; its independent observer reread
passed all 16 executions / 48 layer proofs, including 42 official Q1 layers
with prepared/effective/copied cap 64 and six actual A2 consumer layers.
The relocated production implementation requires fresh acceptance and formal
remeasurement; private receipts cannot stand in for those gates.

## Implementation boundary

1. Copy the accepted arithmetic into
   `operators/deepseek_v32/indexer/csrc/q1_hint_exact.cu` and its small adapter
   into `operators/deepseek_v32/indexer/q1_hint_exact.py`. Point `SOURCE` at
   `csrc/`; retain the exact reduction tree, finite mask, integer count,
   `__fadd_rn`, `__fdiv_rn`, `--ftz=false` and `--fmad=false`. No second kernel
   candidate, arithmetic reassociation or EMA fusion belongs in this change.
2. Dispatch from `prefetch_hint.update_prefetch_hint` after its existing ABI
   validation and CPU/grad/unsupported reference gate. The specialized input
   is SM90 `(9,0)`, gradients disabled, FP32 `scores[1,65537]`, unit inner
   stride and a 16-byte-aligned input pointer, with the existing same-device
   contiguous FP32 `offset[16]`. A larger row stride on a one-row view is valid.
   All other supported shapes/layouts retain the existing normal dispatch.
3. Import the new adapter only after that gate; import TVM only inside actual
   build/call paths. The adapter must not import `prefetch_hint` and call back
   into it, which would make the private fallback recursive after relocation.
   A direct unsupported native entry should reject its input. Preserve the
   current Torch device and stream; native build/execution errors propagate
   with no retry or backend fallback.
4. Keep `models/deepseek_v32/attention.py` and its `prefetch_hint` scope intact.
   Keep official ECHO kernels, predictive threshold, exact top-k, 64-slot
   preparation, cache transitions, FlashMLA and `decode_hint.py` unchanged.
   No shared cache/model/serving or dependency-version change is required.

Relocation changes source identity, debug paths and native cache identity. The
production DSO requires fresh acceptance; the private component receipt binds
the original DSO only. Keep the frozen private candidate/harness and its
evidence unchanged. That harness calls production `prefetch_hint` as its
baseline, so after promotion it cannot silently serve as the old baseline.
Production correctness uses the installed GPU Torch expression as the oracle
and an explicitly bound integrated-model check.

## Storage and capacity

The eligible generic mean requests 262,148 B for masked FP32 scores, 260 B for
65 int32 partial counts and 4 B for the FP32 sum. The specialization removes
262,412 B of requested transient tensor storage per eligible call. It adds no
global/persistent tensor allocation; its 512 floats plus 512 integers use
4,096 B of on-chip static shared memory. This is neither a per-session saving
nor a simultaneous peak multiplied by layer count.

Keep both conservative formulas unchanged:

- `models/deepseek_v32/execution/cache_resources.py:execution_reservation`;
- `evaluation/cache_memory_audit.py:WorkspaceAllowance.echo`.

Their generic allowance, `16*Q*columns + 64*Q*selected + 32*columns`, also
covers noneligible paths and takes the maximum with Q1 paged indexer/staging
and attention storage. At Q1/N65537 the current indexer allowance is
12,289,584 B and total execution allowance is 12,291,888 B. Planned capacity
delta is therefore zero. Measure post-integration graph-private reserved
segments, static storage, allocated bytes and device usage; allocator rounding
and reuse may leave reserved storage unchanged. Do not infer new P/NH/session
capacity from the removed temporary tensors.

The five `experiments/cache_management` plans use H65536/A128/Q1024 with
N65664 and `E(1024)=1,214,517,248 B`; their requests do not execute this
specialization. Record a new bounded source/scope and unchanged-plan audit if
needed for publication, preserving the distinction from a physical capacity
fill. `experiments/cache_manager_performance`, retained MFU A128/matrix runs
and `experiments/deepseek_v32_motivation` also need an explicit configuration
scope check; retain their valid Q>1 results if no N65537/Q1 tail is reachable.
No NOSA planner or executable path changes in this bounded integration.

## Provenance and classification

`measure.source_manifest()` already archives the new non-test operator source
files recursively. Also list both new files in `backend_provenance.source_files`
for its standalone manifest. Add a stable build identity following
`_q1_topk_identity`: rehash declared source closure, compare any loaded build's
source identity, and return the same build identity before/after loading.
Add observed hint runtime information via `sys.modules` and `runtime_info()`
without compiling an unused provider. Require the actual hint DSO for eligible
ECHO execution and equality across check, benchmark and profile identities.

The shared native collector already records mapped `cxldsagr_*.so`; the new
DSO is an `other_local_native`. Keep the shared collector and `_native_cache`
unchanged. The candidate's declared CUDA/compiler/TVM closure does not include
325 system C/C++ headers seen in a later dependency inspection; preserve that
reproducibility limitation instead of inventing build-time hashes or expanding
the shared cache in this task.

`audit_shape_matrix.runtime_records()` only visits `{path,sha256}` records;
native records use `artifact_path`, `artifact_sha256` and `source_sha256`.
Add a narrow hint-specific verification (or normalized file records) for the
source closure, cache key/build record, exact DSO and mapped-library match.
The existing mapped-DSO audit remains necessary but is not a substitute for
checking the native build closure.

`timeline.py` already classifies all `prefetch_hint` stage kernels as GPU
control. Add the new symbol to the existing classifier regression; no FLOPs
definition or classifier rule change is needed. Verify that mean launches fall
from three to one and the complete hint chain from four to two per ECHO layer,
including unchanged EMA. Confirm actual graph-node ownership rather than
hard-coding a six-node model reduction. The compact timeline already has an
ECHO hint lane.

## Acceptance and publication sequence

1. Run CPU import-independence/dispatch and failure-propagation checks. Extend
   existing operator tests for N65537 aligned and padded views, all 16 offsets,
   finite/nonfinite/subnormal/overflow/signed-zero bits, guard boundaries,
   changed captured inputs and ordered nondefault streams. Keep the separate
   EMA and official-policy tests. Reuse the component's meaningful cases;
   do not clone its arithmetic as a test oracle.
2. Obtain new source/DSO-bound GPU operator and complete-model acceptance,
   including real Q1-to-A2 consumption, official prepared/effective cap 64,
   exact outputs/selections/cache state and clean graph verification. Unit
   tests and private component timings do not replace formal experiment gates.
3. Run the existing schema-3 `shape_matrix.sh` with H65536/A1 under a new
   cohort ID on GPU0 / CPU0-7, the FREE environment and token111090 request.
   Retain current four-method prebinding warmup, warmups1, prefill samples3,
   extend samples5 and full-extend graph boundary. All four methods are warmed
   before runtime identity collection, so every formal child records the new
   DSO although only eligible ECHO executes it.
4. Run a separate operator profile through `scripts/profile_layers.sh`, bound
   to the new check and benchmark. Run existing `audit_shape_matrix`,
   `report_shape_matrix` and `report_full_graph_mfu`, then independently verify
   raw windows, output acceptance, source/native identities, node/FLOPs/traffic
   conservation and actual graph/private memory. Profile activity durations
   remain separate from clean latency samples.
5. After acceptance, publish replacement `deepseek_v32_mfu/report/h64k_a1`
   and `report/h64k_a1_mfu`, regenerate the official `report/combined_decode`
   comparison and any diagnosis asset that embeds those local measurements,
   then update both experiment READMEs. Preserve existing published results
   until the replacement is audited. Official Engine results and their
   numerical-acceptance-false boundary remain unchanged; continue to state
   different input, GPU, residency and framework boundaries. Unaffected
   prefill/A128 comparisons and valid source-bound component controls remain.

Do not implement method-isolation CLI/schema/cohort changes in this refresh.
[Method isolation](../deepseek_q1_control/method_isolation_plan.md) is a later,
separate measurement design. In the current formal boundary, HBM wall-time
variation cannot be attributed to the ECHO mean kernel. The accepted
single-method controls measure HBM after different preludes, not offload speed.
