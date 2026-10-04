# ECHO controlled GR diagnostic entry

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> Experiment location: [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md).
> Historical commands, source paths and hashes below remain unchanged; use the
> [migration record](echo_cache_experiment_migration.md) to locate moved artifacts.
> This directory migration is not a new measurement.

Date: 2026-10-03. Status: sampled v2 **C1024/W1024 and C2048/W2048 GPU
diagnostics accepted**, each including a separate CPU artifact audit. The stopped
v1 attempt and failed v2 observer attempt produced no published result.
This document records engineering scope and acceptance boundaries. It is not a
performance result, chunk-size selection, or a replacement for a serving report.

## Entry and fixed contract

`experiments/deepseek_v32_echo_cache/src/echo_diagnostics.py` reuses `build_workload`,
`PersistentGRRunner`, and `DeepSeekServingBackend`. It does not implement model
or cache computations. The fixed case is `/preset-models`, 10 independently
allocated checkpoint blocks using the source-0/1/2 input-replay contract,
65,536 history + 128 changing candidate tokens, 16 synthetic user IDs in order
twice, 4 GiB HBM / 64 GiB DRAM cache caps, NH=1,050,624 and per-layer P=32,768.
The four schemes are `hbm`, `echo`, `serial_sparse`, and `dense_prefetch`.
Prefill C and reserved workspace Q are explicit CLI arguments; extend remains
one actual 128-token batch. Each scheme starts with independent empty cache
ownership; weights are reused. All 32 requests execute and save complete outputs.
Only fixed request IDs `[0,15,16,31]` receive intrusive layer/batch observation.
These cover the first/last initial visit and first/last revisit; actual cache
hit versus rebuild remains an observed property, not an assumed sample label.

Accepted C1024 command, executed from freeze 04 on GPU 1:

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs CUDA_VISIBLE_DEVICES=1 \
bash experiments/gr_serving/scripts/echo_diagnostics.sh 20261003_echo_gr_diagnostics_c1024_04 \
  --chunk-size 1024 --workspace-query-tokens 1024
```

The calling experiment script owns run IDs, log redirection and publication into
`output/{data,log,profile}`. `echo_diagnostics.sh` explicitly prepends `.venv/bin`
to PATH for JIT tools such as `ninja`, defaults OMP/MKL/OpenBLAS CPU threads to
8 unless provided, separates stdout/stderr, reruns the saved-artifact audit in
a separate CPU process, and publishes only after success. Failed capture/audit
keeps its exit status and external staging; no experiment output is published.
The entry exclusively creates the supplied data
directory and removes that directory on failure. It never deletes old reports
or other runs. Its CPU auditor can be rerun with `--audit-only` and the same CLI
arguments. `--help` was checked with CUDA hidden.

## Observation and saved evidence

- Schema `echo-gr-diagnostics-v2` records `parameters.profile_request_ids` and a
  strict per-request `observed` boolean. Unobserved requests must have empty
  `batches`, but their full outputs are still reopened and compared. Counts,
  scopes and exact-selection parity describe the sampled requests only; they
  cannot be extrapolated to traffic or selection parity for the whole trace.
- In each sampled request, temporary wrappers capture every attention `forward` call, with request,
  phase, layer and logical query range. Prefix and extend cannot overwrite each
  other's evidence. A cache hit has no fabricated prefix execution rows.
- HBM, ECHO and serial-sparse use `collect_cache_diagnostics=True`. The observer
  saves exact selected unions, residency before fused prefetch/append, residency
  after append, and each successful consumer group's IDs and actual missing mask.
  CPU auditing reconstructs union/hit/miss, complete group coverage, recall and
  cross-group reread counts, append D2H and sparse H2D byte equations.
- Dense staging has no sparse global map and bypasses sparse recall. The observer
  uses the actual `dense_fetched_records` deltas to require one complete history
  transfer per physical layer and outer batch. Initial selected history is marked
  staged, with a separate stage label; this is not retained-HBM hit evidence.
- Every ordered per-query top-k ID in observed batches is checked live for causal visibility,
  expected cardinality and duplicates. SHA256 covers the complete int32 matrix,
  including its shape and -1 padding. The auditor pairs all batch hashes against
  HBM, including every prefix batch that was actually built. Raw top-k matrices
  are not retained; offline union/count reconstruction and live per-query
  invariant validation are explicitly distinct evidence boundaries.
- All schemes save all 32 requests' actual full candidate hidden `[128,7168]` and last-token
  logits `[1,129280]`. The auditor reopens every tensor, verifies file hashes,
  dtype, shape and finiteness, and recomputes HBM comparisons under the existing
  `rtol=0.01, atol=0.02` contract. It does not rely on online correctness flags.
- Prepare/finalize/recall have separate CUDA scope records and CPU submission
  durations for every batch. Scope totals are recomputed from calls; all failed
  over-capacity probes and successful consumer groups must appear. These are
  intrusive diagnostic intervals, not formal request latency or overlap evidence.
  No warm-operator performance claim is made. Contaminated runner latency fields
  and its incomplete last-batch diagnostic summary are removed from saved rows.
- Complete source bytes and installed official backend identity are captured,
  including `include_official=True` build inputs and the directly reused
  `experiments/deepseek_v32_echo_prefill/src/measure.py` scope implementation;
  current-source and installed-backend checks must pass at the end. The saved
  deterministic workload is authenticated against exact token IDs, fixed user
  histories and changing candidate suffixes. Per-case NH/P/workspace and all
  request cache ledger caps are checked. Diagnostic copies/evidence storage do
  not constitute transient cache or process-peak memory acceptance.
- Wrappers are installed only for a sampled request, including already existing
  sessions reached by a revisit, and restored when that request finishes.
  `PersistentGRRunner` captures allocation/release callbacks while constructing
  its pool, before observation starts. The observer therefore wraps the actual
  `pool._allocate` and `pool._release` callbacks as well as backend methods,
  preserving and restoring each original call target.
  Unobserved requests use the original methods and capture hooks. References
  are also dropped at each session release,
  before the model drops its cache. Otherwise the observer itself could retain
  evicted HBM records. The CPU lifetime test checks both runner and record tensors
  with weak references. Error exit also restores methods and capture hooks.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  experiments/gr_serving/tests/test_echo_diagnostics.py \
  experiments/gr_serving/tests/test_echo_diagnostics_script.py -q -p no:cacheprovider
```

Latest combined entry/shell result: **16 passed in 2.19 s**. The tests exercise the real CPU
`EchoAttentionRunner` and cache path for batch capture and release lifetime,
malformed selections, non-last hidden/logit errors, missing prefix batches,
forged count/timing summaries, sparse cross-group reread evidence, dense
full-history accounting and source-guard rejection when the reused scope code
changes. New checks cover attachment to an existing session, complete restoration
before unobserved requests, strict sampling flags/phase coverage, numerical
failure in an unobserved request, JIT PATH setup, separated logs, and capture/audit
failure without publication. A regression constructs the real
`PersistentGRRunner` before observation and checks cold construction, cached
revisit, unobserved transition, sampled eviction and immediate weak-reference
release of the evicted runner/record storage. Ruff format/check, shell syntax
and CLI help passed.

## Accepted C1024/W1024 diagnostic

Run `20261003_echo_gr_diagnostics_c1024_04` completed with shell exit 0. Both the
collector's audit and the separate `CUDA_VISIBLE_DEVICES='' --audit-only`
process accepted. All **128 scheme/request outputs** were independently reopened;
each complete candidate hidden `[128,7168]` and last-token logits `[1,129280]`
matches HBM bitwise, with zero max absolute error and relative L2. Both stderr
logs are empty. All **16 observed scheme/request pairs** cover the fixed sample
IDs `[0,15,16,31]`; the other 112 requests have no intrusive batches.

| Scheme | Observed layer-batches | Observed prefix state |
|---|---:|---|
| HBM | 2,600 | All four samples rebuild; 650 batches each |
| ECHO | 1,320 | IDs 0/15 rebuild (650 each); IDs 16/31 hit (10 each) |
| serial_sparse | 1,320 | IDs 0/15 rebuild (650 each); IDs 16/31 hit (10 each) |
| dense_prefetch | 2,600 | All four samples rebuild; 650 batches each |

The audit accepted **7,840 observed layer-batches**, including complete actual
prefix/extend coverage, ordered selection hashes against HBM, saved
union/residency/group evidence and scheme-specific traffic equations. These
counts describe the sampled mechanism evidence, not complete-trace traffic.

The run used NVIDIA H200 SM90 on freeze
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-04`, detached HEAD
`1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`, with 749 overlay files and SHA256
`d5c7de6e8dc7b577ac69322cbb3d219510ec116bf71b18edb04dd8b347d77b0a`.
Its diagnostic source manifest covers **1,390 files**, SHA256
`60fc2e1cbd26155a6d347d3a817910c10ed11d8e0b3af21b3dde06ed93025662`.
The workload SHA256 is
`4885ab864bed148f9f4a54230c3dced1c9a3749ac29ca92e13faf1edb019b9d1`.
Collector source/backend guards and the post-run frozen-source verification
passed. Gate/sweep source identities remain the accepted freeze-02 identities;
this diagnostic does not itself select a serving C or accept the standalone sweep.

Accepted data, logs and empty profile directory are under the corresponding
`experiments/gr_serving/output/{data,log,profile}/20261003_echo_gr_diagnostics_c1024_04`
in freeze 04 and main. Original data contains 9,364 files totaling 1,041,460,197
bytes. Import into main preserved all **9,369 original data/log/profile files**;
source-before, destination-after and source-after inventories match SHA256
`f9a4da91db6ed159f9a3b5225b0bfb35c08908a52ba5dfa64bf93a9d891ceda7`.
The import adds only `artifact_import_receipt.json`, excluded explicitly from the
original-file inventory; metadata and audit retain their original source
identity. Final report generation will audit the imported raw evidence again.

Concurrent activity is explicitly recorded in `output/log/<run_id>/execution_context.json`:
external NOSA PID 933314 also used GPU 1 during collection; other GPU tasks were
observed and left running. This run has **no formal request-latency, overlap or
process-peak memory claim**. Diagnostic CUDA scopes remain intrusive evidence.

## Accepted C2048/W2048 diagnostic

Run `20261003_echo_gr_diagnostics_c2048_04` used the same freeze 04 source and
workload identities on H200 GPU 4. The execution owner confirmed session 82252
exited 0 after the collector audit and separate CPU `--audit-only` process
passed. All **128 scheme/request outputs** were reopened; complete candidate
hidden and last-token logits are bitwise equal to HBM within C2048/W2048, with
zero max absolute error and relative L2. Both stderr logs are empty. This is a
within-configuration comparison, not a cross-C numerical conclusion.

All four schemes' sampled request IDs `[0,15,16,31]` rebuilt prefix, each with
330 layer-batches (32 prefix batches plus one extend batch, each across 10
layers). Thus each scheme contributes 1,320 observed layer-batches and the
audit accepted **16 observed scheme/request pairs, 5,280 layer-batches**. Unlike
C1024/W1024 ECHO and serial_sparse, the sampled C2048/W2048 revisits are cache
misses. These remain revisits and are not relabeled first visits. The complete
32-request visit/cache-state trace is retained for matching against formal runs.

The accepted command was:

```bash
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs CUDA_VISIBLE_DEVICES=4 \
bash experiments/gr_serving/scripts/echo_diagnostics.sh 20261003_echo_gr_diagnostics_c2048_04 \
  --chunk-size 2048 --workspace-query-tokens 2048
```

Data, logs and an empty profile directory are under the corresponding
`experiments/gr_serving/output/{data,log,profile}/20261003_echo_gr_diagnostics_c2048_04`
in freeze 04 and main. Original data contains 6,804 files totaling 955,006,255
bytes. Main import preserved all **6,808 original data/log/profile files**;
source-before, source-after and destination-after inventories match SHA256
`47b29206d9a05b9376fa894f96e60f1aee2094b7a75aaff284200f18116833a5`.
Only the separate import receipt was added. The same 1,390-file diagnostic source
SHA256 and workload SHA256 stated above were independently checked. After C2048
completed, freeze 04 verification again passed for 749 copied source files,
26 imports and the unchanged gate/sweep source/backend identities. Other GPU
diagnostics and memory checks ran
concurrently; this run makes no formal latency, overlap or peak-memory claim.

The parameterized temporary import helper is
`/tmp/cxldsagr_import_accepted_run.py`. It checks the accepted-run markers and
requires the execution owner's wrapper-exit record before `--copy`, refuses
existing destinations and symlinks, hashes every original data/log/profile file
before copying, and compares destination-after and source-after inventories.
It adds a separate `artifact_import_receipt.json` and never rewrites original
metadata or audits. It supports diagnostic, standalone sweep and formal GR
artifacts; it does not replace their raw-artifact auditors or generate new
measurement results. Large artifacts remain on SSD.

## Unaccepted attempts

The stopped C1024 v1 attempt used run ID `20261003_echo_gr_diagnostics_c1024_02`
on freeze 02, GPU 1; SIGTERM was requested before acceptance, process exit 143.
Its unaccepted staging remains outside experiments at
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/20261003_echo_gr_diagnostics_c1024_02.drs5cl`.
No request latency from that intrusive run is a performance result. The running
memory-gate freeze was not edited; v2 tools require a separately identified overlay.

Run `20261003_echo_gr_diagnostics_c1024_03` on freeze 03 exited 1 at the first
request with `ValueError: diagnostics omit prefix or extend layers`. Patching
backend methods alone missed callbacks already captured by the persistent pool.
The failed data directory was removed; logs remain only in external staging
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/echo-gr-diagnostics-20261003_echo_gr_diagnostics_c1024_03.tBSEpd`.
The matching C2048 attempt failed for the same reason. Freeze 04 adds the pool
callback observer fix and its real-runner lifecycle regression; DeepSeek runtime
and gate/sweep source identities were unchanged. No failed attempt is an accepted
numerical or performance result.

## Review findings sent to the parent

The preexisting serving measurement saved only HBM hidden references, omitted
last-logit comparisons, and its reference auditor could not independently
recompute actual-vs-HBM numerical checks. Its per-request counters begin anew at
extend and per-layer diagnostic dictionaries retain only the final outer batch.
The legacy token-hit metric is observed after fused prefetch and append.
Parent-owned changes address formal measurement/audit artifacts; this separate
entry supplies intrusive phase-complete evidence. Neither path should substitute
the fixed-NH/P three-layer chunk sweep for the controlled 10-block serving trace.
