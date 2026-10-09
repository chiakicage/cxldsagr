# QKV cleanup: current dependencies

The rejected QKV candidates, timing runs, prototype entry points and unapplied
integration mirror were removed by `cleanup_cub_20261008_01.json`. The two
superseded proposal documents were subsequently removed by
`cleanup_qkv_plans_20261008_01.json`. Their rejection reason and correctness
scope remain in `checkpoint.md`: clean complete-model timing regressed, and
no QKV patch was applied.

The following remain necessary during the active measurement investigation:

- `experiments/deepseek_v32_mfu/src/q1_qkv.py` supplies `runtime`, `save`,
  `tensor_digest` and `weight_identity` to the accepted `q1_selection_model.py`
  experiment and is explicitly fingerprinted by its receipts. Do not rename
  or remove this helper without fresh acceptance of the affected experiment.
- `q1_qkv_model_profile_20261008_02` and
  `q1_qkv_clean_model_profile_20261008_01` under the MFU experiment's
  `output/data/` and `output/profile/` retain the original discrepancy inputs.
  Only their baseline arms are diagnostic inputs; the rejected fused arms
  are not valid optimization results. Remove these run pairs after the
  first-replay/collection control establishes a replacement explanation.
- `deepseek_h64k_a1_official_20261008_03_h65536_a1_profile` is also retained
  temporarily for the original discrepancy. Current production reporting
  already uses `deepseek_h64k_a1_cub_20261008_01_h65536_a1_profile`.
- The signed `request.json` under
  `deepseek_h64k_a1_official_20261008_03_h65536_a1_bench` remains an input to
  accepted selection and controlled-measurement receipts. The signed request
  under `deepseek_h64k_a1_q1_20261008_02_h65536_a1_bench` also remains in use.
  Neither may be removed merely because its parent run was superseded.
- `profile_discrepancy_review.md` and the two temporary
  `/tmp/cxldsagr-checks/q1_qkv_profile_boundary_*.json` audits describe the
  original observation. Current controls have excluded construction profiler
  state, the graph inspector and internal external-event nodes as its cause;
  profiler state during the first graph replays remains under investigation.

Accepted `q1_selection_model_*` records belong to CUB top-k, despite their
shared helper and older NVTX label. They are outside QKV cleanup. Temporary
engineering correctness checks are not experimental performance deliverables.
