# V10 complete and per-layer extend diagnostics

All four complete extend windows and every L0 window fail the strict 10% gap
gate. Only dense-prefetch L1 passes among twelve layer windows. All three
offload methods pass the full-prefill absolute-gap limit of 1.2 times HBM.
These diagnostics do not establish final optimization-goal acceptance or a
new formal operator-MFU result.

The table combines independent benchmark medians with conservative non-IO
gap upper bounds from a separate intrusive profile. All three prefill and
five extend samples per method are retained in `timing_samples.csv`. There is
one measured profile per method/phase; no confidence interval or paired speedup
is inferred from these windows or unpaired medians.

| Method | Bench prefill ms | Bench extend ms | Full gap % | L0 % | L1 % | L2 % |
|---|---:|---:|---:|---:|---:|---:|
| hbm | 645.067243 | 3.461295 | 20.291484 | 36.249708 | 10.278381 | 10.269745 |
| echo | 647.961466 | 5.433349 | 40.416182 | 62.795000 | 26.527816 | 26.422064 |
| serial_sparse | 646.863175 | 3.906473 | 34.556835 | 49.881971 | 22.556216 | 24.109421 |
| dense_prefetch | 644.243363 | 5.827201 | 19.424940 | 36.474134 | 1.569232 | 11.355556 |

Full-prefill absolute-gap/HBM ratios, in ECHO / serial sparse / dense prefetch
order: 1.0335831993653863 / 1.0153742398862482 / 1.0223021043960159. `summary.json` retains complete, per-layer,
shared-tail and last-prefill-chunk metrics, run IDs, all timing samples and gate
states at full precision.

## Labeled figures

- [Complete extend SVG](complete_extend.svg) · [PNG](complete_extend.png)
- [L0 extend SVG](layer_0/timeline_extend.svg) · [PNG](layer_0/timeline_extend.png)
- [L1 extend SVG](layer_1/timeline_extend.svg) · [PNG](layer_1/timeline_extend.png)
- [L2 extend SVG](layer_2/timeline_extend.svg) · [PNG](layer_2/timeline_extend.png)

Every figure identifies the single intrusive NSYS profile and its separation
from independent benchmark wall times. Complete extend includes startup,
synchronization and commit. L0 begins at the complete forward boundary. Each
figure uses a common time scale across the four methods. Dense L2 states
“no following-layer prefetch”. Numbered computation labels in the complete
view, and direct labels in the layer views, identify approximate purpose;
original activity bars and their internal gaps remain visible. The final shared
computation is “Final norm / LM head”.

Gap is the full window outside the union of Compute, actual host IO and
indivisible Compute + IO. GPU metadata/control, D2D copies and idle remain gap
unless covered by that union. Pure-IO-only time is removed from the denominator;
fused compute/IO yields conservative ratio bounds. Proven-empty gathers are
control; a proven-zero-IO fused indexer is compute. The six append preparation
and slot-selection kernels per sparse extend remain control. No profiler cost
is subtracted. SVGs contain no embedded raster. These wide engineering previews
have not been formatted for paper-column scaling.

## Last-prefill-chunk detail

- [L0 prefill SVG](layer_0/timeline_prefill.svg) · [PNG](layer_0/timeline_prefill.png)
- [L1 prefill SVG](layer_1/timeline_prefill.svg) · [PNG](layer_1/timeline_prefill.png)
- [L2 prefill SVG](layer_2/timeline_prefill.svg) · [PNG](layer_2/timeline_prefill.png)

These panels show only chunk 64/64, with 1,024 queries at positions
64,512–65,535. They exclude preceding chunks, full-prefill startup and the
shared final tail. L0 begins at the previous chunk's final-layer compute end;
L1/L2 begin at the preceding layer's compute end. All twelve endpoints and gap
values match the accepted last-chunk audit. Counter-backed zero-IO classification
is unchanged. The full-prefill gate uses all 64 chunks plus startup and tail;
the last-chunk plots cannot substitute for that gate. Every compute activity
has an approximate-purpose label, including the new causal mask within
“Indexer logits”. All seven PNGs receive visual review; figure/data/source
hashes and the review record are bound in `provenance.json`.

## Workload and verification

V10 uses real checkpoint layers 0–2, H=65,536, A=128, history chunk=1,024,
extend chunk=128, pool=65,664 per layer and cold offload residency. It executes
ordinary persistent append. The graph policy is
`deepseek-compute-islands-v4-bound-inputs`. The production resident-indexer
causal-tail mask uses one Triton call; cache recall and free-append paths retain
their accepted implementations. Arbitrary legal slot choices at equal priority
are permitted; indexer exact top-k and its tie semantics are unchanged.

| Purpose | Run ID |
|---|---|
| Correctness | `deepseek_gap_v10_a128_check_20261006_01` |
| Independent timing | `deepseek_gap_v10_a128_bench_20261006_01` |
| Intrusive profile | `deepseek_gap_v10_a128_minimal_20261006_01` |

The check has 13 passing receipt flags. Twelve saved tensors were independently
reread, checked for shape and finiteness, and compared by raw bytes in nine
cross-method hidden/extend-logit/prefill-logit comparisons. Four default-forward
versus control flags have no separately saved default output and do not count
as independent saved-tensor rereads. Eight saved profile/control hidden/logit
tensors were additionally checked for dtype, shape, finiteness and raw-byte
equality.

The audit checks 3,855 execution-source records, 58,395 measured GPU activities
across eight captures, and 788 layer/tail partition windows. It independently
recomputes interval unions and raw-row/correlation coverage while reusing the
accepted semantic classifier and graph-lineage reader. Each prefill contains
192 actual resident mask launches. HBM, serial and dense extend contain three
each; fused ECHO extend contains zero. Those masks are Compute under
`indexer_qk`, without an IO annotation. The replaced arange, comparison and
masked-fill helpers are absent within `indexer_qk`.

All three actually loaded local native libraries and bridge dependencies match
their recorded identities. A post-capture FP32 RMSNorm compilation addition
appears in none of the nine native captures. Check/bench/profile observers
have 12/17/36 clean discrete samples, with maximum gaps of 2.211002 / 2.201770 /
3.390163 seconds. Run IDs are explicitly bound in `MFU_RUN_ID` argv. Samples do
not establish continuous GPU isolation, exclusive CPU use or a uniform
observer cost.

## Resident-mask runtime identity boundary

V10 records exact production source, actual native libraries, matched model
receipt, saved outputs and actual profiled mask activities. The generic runtime
collector records `linear_quantization_triton`; it does not separately save
the production resident mask's live JIT/CUBIN artifacts. The successful public
production-operator acceptance also lacks a separate live mask CUBIN capture.
Prior private full-model correctness and paired-timing evidence has actual
returned live mask artifacts. Its production-source equivalence and public
acceptance support this diagnostic, but do not turn that earlier private
artifact record into a fresh V10 production CUBIN attestation. No concrete
identity mismatch was found; no rerun was made solely to enrich this metadata.

`provenance.json` binds the three accepted audits, raw gap data, runtime-boundary
evidence and all seven figures with their exact source/data manifests. It keeps
the original evidence paths. This staged report copies fourteen image files
and compact tables/metadata; raw captures, source archives, tensors and activity
rows stay in their evidence directories. The v3 formal operator-MFU report and
its ECHO scale-stage limitation retain their original identity. Cache manager
optimization remains the first priority before further model optimization.
