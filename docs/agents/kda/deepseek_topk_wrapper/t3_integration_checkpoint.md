# T3 integration deferred at C10 closure

On 2026-10-04 the user closed optimization scope at C10. T3 is deferred and
is absent from current production. No complete-model T3 gate or formal/profile
publication was performed. Published experiment reports were not edited by
this component work.

The restored C10 implementation subsequently completed its matching profile
and independent analysis; see [the accepted C10 profile checkpoint](../../system/deepseek_motivation_c10_profile.md).
That acceptance applies to C10 and does not promote this deferred candidate.

Before the scope change, the actual production wrapper passed seven CPU
selection tests, fifteen provenance tests, Ruff, and twenty-eight public GPU
tests. The independent production driver passed 41 main shape/pattern cases,
27 K representatives, 26 synthetic fixture batches, six saved real batches,
fallback and changed-input graph/stream/output checks. It recorded 34 actual
production Triton specializations. These completed checks concern the archived
candidate; they are not new C10 validation.

The already-running production timing finished with exit 0. It included the
wrapper's identity capture/observe overhead, used frozen C10 C5 as baseline,
and retained forty alternating pairs after ten warmups. The six wall medians
in milliseconds were:

| Q / N | Pattern | C10 C5 | T3 candidate | Median paired T3 minus C5 |
|---|---|---:|---:|---:|
| 1,024 / 1,024 | causal | 0.107074 | 0.087668 | -0.018996 |
| 1,024 / 32,768 | causal | 0.336102 | 0.300280 | -0.035270 |
| 1,024 / 65,536 | causal | 0.442042 | 0.403643 | -0.038814 |
| 128 / 65,664 | causal | 0.098212 | 0.095731 | -0.002508 |
| 128 / 65,664 | ties | 0.120933 | 0.114612 | -0.006310 |
| 128 / 65,664 | invalid | 0.153530 | 0.149059 | -0.004007 |

These are deferred component measurements. The complete post-run runtime
artifact audit was not executed after the scope change, and no model-level
performance conclusion or promotion follows. At 01:24:46 UTC, the post-run
all-device process snapshot was empty; GPU1 reported 0% utilization. The
exclusive performance window was explicitly released before archival work.

Evidence is under `/tmp/deepseek_topk_order_t3_integration_20261004/`:

- `production_correctness.json`, SHA256
  `51fa1555f82031e500783d9f4072fef5e93e3576a225399c8d990c3ae8b9d917`.
- `production_timing.json`, SHA256
  `ccd031320f1625c91760ed56d985733c229cc126f6ca799424a228b1e642e893`.
- `deferred_timing_summary.json` retains wall/event medians and paired deltas;
  the raw JSON retains every timing sample. Separate stdout/stderr logs and
  before/after observations are preserved.
- `deferred_source/` preserves all five changed files at their original
  relative paths. Its `manifest.json` has SHA256
  `31e1045fa258cd5a2c3b1b0141c04de68bc9d91fd5fad1c47af3596cf38eff3d`.
- `restore_audit.json` has SHA256
  `5265520380138cd04bc2d3771ef86e89e89e607e1ea8261c36f331ca82f9766a`.

Before restoration, every owned live file matched its production-launch
snapshot and its newly written archive. Four existing files were then copied
from `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0/`; the new
`_selection_kernel.py` was removed only after archival verification. The final
audit confirms exact C10 hashes for selection, its tests, shared provenance
and its tests, and confirms that the extra kernel is absent. No other agent's
files were reverted. No new tests or timing ran after the closure request.
