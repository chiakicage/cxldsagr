# C3 integration and current measurement

The full user objective remains active: offload first-visit latency must converge
with contemporaneous HBM-only, and every scheme's end-to-end MFU must approach
its measured matrix API MFU. Component speedups and bit-exact graphs do not prove
that end state.

## Candidate

C3 combines the preexisting trusted-resident metadata implementation, validated
CTA-local exact-union reduction (C3b), and backend-owned pure compute graphs.
The graphs cover input norm/projection and output projection/norm/MLP. Cache,
indexer, top-k, exact recall, MLA and transactions remain eager. Graph outputs
used by D2H get owned copies. The graph banks have independent weight addresses
for all ten block copies and prepare H1024/A128 before request timing.

Source/focused evidence:

- [Cache metadata checkpoint](../kda/deepseek_cache_metadata/checkpoint.md):
  236 module tests, 16 relocation checks; target resident ensure 0.787 to 0.100 ms,
  actual kernel activity 366.36 to 57.48 us and launches 63 to 2.
- [Graph validation](deepseek_motivation_compute_graph_validation.md):
  full H=P65536/NH131072/C1024/A128 plus A121 fallback, four schemes eager/graph,
  exact complete prefill/candidate hidden/logits, interleaved users, retained
  outputs/indexer/host history, no candidate D2H, cross-stream and fault checks.
- Root integration check: 41 model/measurement tests passed after propagating a
  failed graph completion event to backend poisoning. The active execution and
  admission owner, graph and KV storage stay retained when completion is unknown.

## Frozen formal run

Run ID: `motivation_c3_20261004_u16_r2_01`.
H65536/A128/C1024/P65536/NH16777216, sixteen users, two rounds, all four schemes,
three warmup requests per scheme, ten independent checkpoint dense-block copies.
TF32 disabled. Physical GPU0, PyTorch reports H200/SM90.

Execution source is an isolated copy at
`/tmp/deepseek-motivation-c3-frozen-x1ruyzmz`, made from the current worktree.
Its `frozen_source_identity.json` records 452 copied files checked against the
original at snapshot completion. Third-party pinned trees, Git metadata and
installed environment are shared read-only; executed project modules are copied.
This protects the measurement from concurrent edits to profiler analysis tools.
The formal runner also collects and verifies its own executed-source manifest,
installed backend identities, outputs, warmup paths and exact graph replay counts.

Command from that isolated root:

```bash
CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0 bash experiments/deepseek_v32_motivation/scripts/run.sh --run-id motivation_c3_20261004_u16_r2_01 --compute-graphs
```

Exec session 86671 completed successfully. Accepted output and logs are now in
the original experiment's `output/data/motivation_c3_20261004_u16_r2_01` and
`output/log/motivation_c3_20261004_u16_r2_01`. The source manifest SHA256 is
`d7eda579163ab9d51a211b2a5d7434f22d597d726ce2c665fa826144cec27d02`.
The independent saved-output audit rechecked all 128 requests, all 96 offload
comparisons and all 12 warmup requests. All 32 new HBM outputs also match the
original `_02` outputs byte-for-byte with identical request token hashes.

| Scheme | First mean ms | Revisit mean ms | First E2E MFU % |
|---|---:|---:|---:|
| hbm | 2526.549 | 2526.522 | 38.2896 |
| echo | 2631.164 | 43.412 | 36.7673 |
| serial_sparse | 2570.870 | 37.905 | 37.6295 |
| dense_prefetch | 2575.018 | 84.533 | 37.5689 |

All offload first-visit means are within 1.8%–4.1% of contemporaneous HBM.
Do not equate this result with full goal completion. Dense revisit request16
took 674.951 ms (663.633 ms candidate execution); all later dense revisits are
44.6–45.7 ms. The outlier stays in the mean and requires investigation. Serial
first-visit request0 also reached 2762.581 ms, versus typical 2554–2562 ms.
Other GPU validation/diagnostic jobs ran concurrently on separate GPUs during
parts of this trace; a later diagnostic observed a long cudaMalloc stall,
but that does not establish the cause of these outliers. One external py-spy
stack snapshot was taken while ECHO was entering measured execution. Final
performance promotion needs a fresh trace with other local CUDA jobs idle.

Static FLOPs/E2E MFU analysis used the isolated executed sources and is stored
in `output/data/motivation_c3_flops_20261004_01`. Graph-private reserved storage
is 7,260,340,224 B; static graph inputs are about 1.945 GB. Process allocator
peaks are 15.724/17.562/17.563/17.563 GiB allocated and
26.408/27.059/27.059/27.068 GiB reserved, in table scheme order. These are
observed capacities, not an invented fixed overhead deducted from P/NH.
Keep prior accepted publication until complete replacement evidence is ready.

## Next evidence

Profile actual graph replay nodes with capture-time node
sets plus Nsight's recorded clone lineage; do not borrow eager timings or assign
API duration by graph wall interval. Compare first/revisit matrix work exactly.
The C3 graph profile is running from another isolated source copy,
`/tmp/deepseek-motivation-c3-profile-frozen-4d3qdyh4`. Run ID is
`motivation_c3_profile_20261004_01`, exec session36497. Staging:
`/tmp/deepseek-motivation-profile-motivation_c3_profile_20261004_01.CezGLP`.
Re-poll its handle/process before restarting. Four schemes require12 Nsight
traces: setup traces1/4/7/10 plus request traces2/3/5/6/8/9/11/12.

The independent same-GPU HBM eager/graph diagnostic confirms little history
gain. Actual graph execution reduced internal gaps but added about54.6 ms of
D2D copies at graph boundaries. About138.7 ms of CompareFunctor/masked_fill
work belongs to the resident indexer API's causal-tail initialization. Current
code scans all Q×N entries, while only the suffix beginning at query_start can
be invalid. C4 is investigating suffix-only masking with unchanged official
DeepGEMM, exact top-k, logical visibility, and physical row-stride semantics.
The large top-k/cast/nonmatrix costs remain in end-to-end MFU.
