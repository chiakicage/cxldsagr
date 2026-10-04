# T3 production integration plan

Status: deferred at the user's C10 closure on 2026-10-04. The component work
below completed, but T3 was not promoted. All five owned production/test/
provenance paths were archived and restored to frozen C10. See
[the integration checkpoint](t3_integration_checkpoint.md).

Root authorized this scope after the temporary T3 artifact audit passed.
The component screen passed correctness and improved all six complete-API
wall/event medians. Production integration remains subject to public API
correctness and timing, followed by root's complete-model and formal/profile
gates. No cache/serving code or existing report is in this change.

Files owned by graph_profile:

- `operators/deepseek_v32/indexer/selection.py`: keep the complete C5 fallback,
  add T3 selection/finalization and scoped build/runtime identity.
- `operators/deepseek_v32/indexer/_selection_kernel.py`: copy the validated
  short/long kernel bodies into a lazily imported production module.
- `operators/deepseek_v32/indexer/tests/test_selection.py`: preserve existing
  public tests and add threshold/ownership, fallback, import and identity checks.
- `experiments/deepseek_v32_echo_prefill/src/backend_provenance.py` and its
  existing test file: collect `exact_topk_selection` build identity and
  `topk_selection_triton` live artifacts without loading or compiling kernels.

The public wrapper preserves validation and official SMALL selected-set
semantics. Outside uint32 packing, it calls the complete retained C5 wrapper
before any new selector, flag or identity capture. Supported calls retain
the official stable value sort and use the validated two-stage repair.
Capture source/compiler identity before lazy kernel import, then observe the
actual compiled objects returned by both launches. Reuse the existing scoped
Identity engine without editing linear operators or scanning other model
source trees. Importing the selection wrapper must not import Triton or
initialize CUDA.

- [x] Implement the scoped files above and review the diff against the frozen
  temporary candidate. Record any integration-only changes explicitly.
- [x] Run focused CPU metadata/import/provenance checks and Ruff after root
  releases its CPU timing window. Do not run hash sweeps, compilation or timing
  during another agent's measurement window.
- [x] Under a new root GPU grant, run actual public API tests plus the bounded
  production correctness driver, including changed-input graph replay, retained
  outputs, strided inputs, all K values, threshold/warp/padding/signed-zero
  cases and complete C5 fallback. Inspect both live production kernel handles.
- [x] Compare the complete production API against the frozen C10 C5 module from
  `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0/`, with ten warmups
  and forty alternating repeats on the same six cases. Include identity capture
  fast-path checks and both observe calls in the API; report wall/event medians
  and paired deltas, with source/runtime identity and isolation observations.
- [x] Hand completed component results and the deferred-source restoration audit
  to root. Root owns complete-model, formal/profile acceptance
  and any report publication/replacement. Component acceptance alone does not
  establish a new model-level result.
