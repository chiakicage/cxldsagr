# Exact hint: private checkpoint L0–L2 gate

The private candidate passed correctness, independent clean timing and raw NSYS
review. Root accepted this gate for bounded production integration. These are
complete calls on the first three checkpoint layers, not all 61 DeepSeek layers.

The runs are `q1_hint_model_{check,bench,profile}_20261008_01`; correctness is
under `/tmp/cxldsagr-checks/q1-hint-model/`, and timing/profile are under the
official experiment's `output/`. The frozen driver SHA256 is
`72e8d2d067a6b26f681ea9f7caaee2b922b6a7d696f1821a29381823d9e8fd07`.
Both arms use official ECHO, H65536/A1, independent model/cache instances,
actual preparation and prefetch cap 64, and cold prefix/hint restore outside
timing. Only the mean implementation differs; the decode EMA remains separate.

Correctness covers tokens 111090/111091/111092, eager and graph execution,
all 16 offset bits, exact outputs/selection and per-execution cache transition
proofs. Diagnostic graphs are destroyed before clean graphs. Separate models
verify actual committed Q1 then A2 without restoring/truncating their history
between calls. Independent CPU reviews checked 16 saved executions and 48
layer proofs; runtime comparisons include KV bytes not all retained in the
compact proof, so the independent reread does not recreate all raw KV checks.

The clean benchmark contains all 200 samples from 100 balanced AB/BA pairs.
Baseline/candidate medians are 2.6486815/2.6053275 ms, paired median change
−45.828 µs, with 87/100 candidate wins. AB/BA paired medians are
−50.089/−36.471 µs. All outliers remain. Both graph private pools reserve
62,914,560 B. The 600 per-layer traffic records are retained individually.

The separate NSYS replay has 269 baseline versus 263 candidate GPU nodes.
Native capture ownership and clone lineage identify three mean kernels
replaced by one in each layer. The other 260 nodes, including three EMA
kernels, 12 copies and one memset, have identical signatures and ownership;
independent review also checked their preserved capture order. Complete hint
activity sums are 12.096/12.192/12.032 → 4.928/5.024/4.896 µs by layer.
These intrusive durations are not clean API latency or an attribution of the
entire wall-time improvement. Actual profile L2 recall is 2007 versus 2002
records; its 5,760 B difference remains explicit. Official prediction is
schedule dependent. The prior FREE formal ECHO template also contains 269
full-graph nodes; 261 is its L0–L2 subset after excluding eight shared nodes.
Matching counts alone do not establish cross-run signature or edge equality.

The frozen CPU analyzer is `src/analyze_q1_hint_model.py`, SHA256
`e348160c4e912697a3900ce4d5d4bfdfd933c59a4f9bd4e609a29589ba4627ae`.
Its result is `output/data/q1_hint_model_analysis_20261008_01/result.json`,
SHA256 `ee0246a98ca8c5bf9f5bf5757cffdb34864cd0f1bf7a729098db2630d54d834b`.
Independent reviews are under `/tmp/cxldsagr-checks/`:

- `q1_hint_model_profile_raw_independent.json`, SHA256
  `b388784500d1e85e59c991e60c2f919b153233eac8a56150779ed7e14a78b50a`:
  all 532 raw nodes/clone edges, ten clipped integer unions and all gaps,
  signature/owner conservation, and all 606 timing/profile traffic records.
- `q1_hint_model_profile_identity_independent.json`, SHA256
  `051a3cf03583d03880559dd30c5cacda9dbf0f57371af26c894b0350a182b1e8`:
  1,380 current/archive sources, seven DSOs, 30 assemblies, two templates,
  three SQLite exports and three NSYS reports; exact check/bench/profile identity.
- `q1_hint_model_root_raw_review.json`: root's direct SQL reread of the hint
  activities and per-arm kernel counts.

Production relocation creates a distinct source/native identity. Its component
and model correctness must be reaccepted, then the current four-method formal
cohort remeasured and audited before its figures are replaced. The private
receipts remain immutable. See [integration plan](q1_hint_integration_plan.md).
