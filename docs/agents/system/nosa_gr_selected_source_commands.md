# Selected-source NOSA budget-serving handoff

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Prepared on 2026-10-04. Root released the GPU0 sequence after fresh fixed full32
acceptance. The four selected shared tests passed without skips; the budget
checkpoint gate also passed with all16 candidate outputs exact. H4K formal
execution reached all 804 rows but failed the old report accounting gate. The
reviewed correction is now frozen, and the complete `_02` sequence has resumed. This is the ordinary
`NosaServingBackend` budget experiment; fixed P/NH results do not replace it.

Current completion and evidence are tracked in
[the execution checkpoint](nosa_gr_selected_source_execution.md).

## Source and retained workload identity

The CPU-only freeze is outside experiments at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_selected_source_20261004_02/`.
Its `gr_formal_scope/` contains the exact 280 files selected by the existing GR
measurement snapshot, including the cross-model trees and runtime tests.
The formal GR digest is
`8dc00e23469a67ebee67eb603859c9de9987e9151520a607d1bb20a498e547f9`.
The same checkout's narrower 114-file motivation digest is
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
Different manifest scopes explain the different digests. `source_freeze.json`
records the reviewed report/audit correction. Exactly `report.py`, `audit.py`
and `test_nosa_shared_accounting.py` changed relative to the failed-run freeze;
model/runtime sources did not change. Recheck the entire GR
inventory before launch and after each formal/profile process; additions count
as changes as well as modifications. Never relabel an old source snapshot.

The external `verify_freeze.py` rechecks both file sets, saved bytes and current
source hashes without starting CUDA or rewriting the freeze. From the repository
root, run it before the selected GPU sequence:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  /mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_selected_source_20261004_02/verify_freeze.py
```

The completed prior CPU trace proof was copied unchanged to
`prior_workload_reopen_explained.json`, SHA256
`887552d6f15e76c8111c209eee9b3ed3d8ce1524d0315d7486c6bafd008344e5`.
It covers all 21 populations and 444 exact complete request records, hence
1,776 four-scheme executions. Its old workload identities were checked against
the retained originals again. New manifest hashes differ because their config
explicitly includes `sampling=weighted` and `heat_field=interaction_count`;
those two additions explain every new hash. Preserve both identities.

H4K/H16K each retain 201 requests per scheme: the one-user trace ends at nine
visits under the original cap, and each other population has 32 draws. H64K
retains six draws in each of seven populations, 42 per scheme. H64K uses the
existing automatic 65,664-token context override and permits empty revisit
groups. All runs retain A128, C1024, seed42, Beauty interaction-count heat,
max-revisits8, two warmups, exact hidden comparison, HBM4 GiB and DRAM16 GiB.

## Placement and boundaries

The proposed physical GPU0 is `GPU-1bdee8b4-22ac-536c-208b-bfb4ed38b878`,
PCI `0000:18:00.0`, NUMA0. CPUs0–7 are distinct physical cores; their SMT
siblings are96–103. GPU5 is `GPU-a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`,
PCI `0000:ab:00.0`, NUMA1, with the fixed-serving job on CPUs48–55
(siblings144–151). Both allocations are local and disjoint, with different
PCI host roots. Driver/pinned-registration services, storage/page-cache traffic
and chassis power remain shared. Placement does not prove zero timing
interference; root decides whether concurrent formal jobs are appropriate.
The independent raw topology receipt is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_selected_source_preflight_review_20261004/topology_preflight.json`,
SHA256 `24498bc457cdbb6f33352f8f6aaf95bc47aa04151065ab93ba5b39a5acc942dc`.

The GR metadata records GPU UUID, CPU affinity and Torch thread count. Its
static `host_memory_placement` text is not an observed NUMA placement record.
Save the actual command/environment and process placement separately. The
current GR harness does not save `_allocator_snapshot.runtime_info()` or
the selected adapter's mapped-library identity. The requested register8
configuration and another process's receipt cannot establish that a GR process
used `private_cpp` without fallback. Do not make that claim without a receipt
from that target process. The cache ledger bounds active allocations and reports
allocator allocated/reserved separately; it does not close inactive cache,
direct-library or driver overhead into a physical HBM capacity proof.

## Existing correctness coverage

The fresh fixed full32 gate exercises the changed reserved FA3 helper with
all32 checkpoint layers, but uses `NosaFixedServingBackend` and compute graphs.
It does not validate ordinary budget admission/LRU or persistent-candidate
truncate. The selected existing four-case native check below exercises the
changed HBM/dense shared paths on two nondefault streams, varying query sizes,
retained outputs, truncate and middle-layer rollback. It uses a four-layer
random model and is not a checkpoint or performance result.

The optional existing checkpoint command below covers the budget backend's
independently built histories, all32 layers, all four schemes, two users and
two visits at H65536/A128. It checks all hidden elements and storage bounds;
it does not execute the formal heat trace or establish physical peak capacity.
Root selects and schedules these checks; no skipped case counts as acceptance.

```bash
gr_repo=/mnt/ssd-wlcb/chenkaiqi/cxldsagr
cd "$gr_repo"
gr_python="$gr_repo/.venv/bin/python"
gr_venv="$gr_repo/.venv/bin"
gr_base_path="$(python3 -c 'import os; from pathlib import Path; v=str(Path.cwd()/".venv/bin"); print(":".join(x for x in os.environ["PATH"].split(":") if x != v))')"
gr_tmp=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp
gr_output="$gr_repo/experiments/gr_serving/output"
gr_run_env=(env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
  -u FLASHINFER_DISABLE_JIT
  -u PYTORCH_NO_CUDA_MEMORY_CACHING -u PYTORCH_NO_HIP_MEMORY_CACHING
  -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES
  PATH="$gr_base_path" CUDA_VISIBLE_DEVICES=0 CXLDSAGR_SM90_BACKEND=native
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  TMPDIR="$gr_tmp"
  PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
  numactl --physcpubind=0-7 --membind=0)

"${gr_run_env[@]}" env PATH="$gr_venv:$gr_base_path" "$gr_python" -B -m pytest -q -s -p no:cacheprovider \
  'models/nosa/tests/test_shared_serving_cuda.py::test_cuda_shared_schemes_alternate_users_queries_and_retain_outputs[hbm]' \
  'models/nosa/tests/test_shared_serving_cuda.py::test_cuda_shared_schemes_alternate_users_queries_and_retain_outputs[dense_prefetch]' \
  'models/nosa/tests/test_shared_serving_cuda.py::test_cuda_shared_middle_layer_failure_preserves_prefix_and_recovers[hbm]' \
  'models/nosa/tests/test_shared_serving_cuda.py::test_cuda_shared_middle_layer_failure_preserves_prefix_and_recovers[dense_prefetch]'

"${gr_run_env[@]}" env PATH="$gr_venv:$gr_base_path" \
  NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=128 \
  "$gr_python" -B -m pytest -q -s -p no:cacheprovider \
  models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits
```

The existing CPU admission tests are
`test_exact_shared_plus_session_budget_controls_user_admission` and
`test_one_byte_below_minimum_user_capacity_cannot_allocate_or_evict` in
`models/nosa/tests/test_serving_resources.py`, plus
`tests/integration/test_gr_persistent.py`. Their code path did not change with
the stream-wrapper patch; do not repeat them solely for the new broad digest.

## Formal traces and matching profiles

The six IDs below were unused during preparation. Each shell wrapper checks
all three destination categories and refuses collisions. The wrappers add
`.venv/bin` themselves and keep separate stdout/stderr. They stage failures
outside experiments. Run the formal process to acceptance before its matching
profile. Do not overlap either with another process on GPU0. Check the actual
GPU UUID and idle state again immediately before root authorizes execution.

```bash
"${gr_run_env[@]}" env GR_AUDIT_BEFORE_PUBLISH=1 bash experiments/gr_serving/scripts/run.sh \
  gr_nosa_flags_h4k_20261004_02 --models nosa --device cuda:0 \
  --nosa-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --users 1 8 32 64 128 256 512 --history-tokens 4096 --candidate-tokens 128 \
  --requests 32 --max-revisits 8 --chunk-size 1024 --warmup 2 --seed 42 \
  --sampling weighted --heat-dataset beauty --heat-field interaction_count \
  --hbm-budget-gib 4 --dram-budget-gib 16 --atol 0 --rtol 0
"${gr_run_env[@]}" bash experiments/gr_serving/scripts/profile.sh \
  gr_nosa_flags_h4k_profile_20261004_02 --device cuda:0 \
  --latency-data "$gr_output/data/gr_nosa_flags_h4k_20261004_02" --num-users 8

"${gr_run_env[@]}" env GR_AUDIT_BEFORE_PUBLISH=1 bash experiments/gr_serving/scripts/run.sh \
  gr_nosa_flags_h16k_20261004_02 --models nosa --device cuda:0 \
  --nosa-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --users 1 8 32 64 128 256 512 --history-tokens 16384 --candidate-tokens 128 \
  --requests 32 --max-revisits 8 --chunk-size 1024 --warmup 2 --seed 42 \
  --sampling weighted --heat-dataset beauty --heat-field interaction_count \
  --hbm-budget-gib 4 --dram-budget-gib 16 --atol 0 --rtol 0
"${gr_run_env[@]}" bash experiments/gr_serving/scripts/profile.sh \
  gr_nosa_flags_h16k_profile_20261004_02 --device cuda:0 \
  --latency-data "$gr_output/data/gr_nosa_flags_h16k_20261004_02" --num-users 8

"${gr_run_env[@]}" env GR_AUDIT_BEFORE_PUBLISH=1 bash experiments/gr_serving/scripts/run.sh \
  gr_nosa_flags_h64k_20261004_02 --models nosa --device cuda:0 \
  --nosa-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --users 1 8 32 64 128 256 512 --history-tokens 65536 --candidate-tokens 128 \
  --requests 6 --max-revisits 8 --allow-empty-revisits \
  --chunk-size 1024 --warmup 2 --seed 42 \
  --sampling weighted --heat-dataset beauty --heat-field interaction_count \
  --hbm-budget-gib 4 --dram-budget-gib 16 --atol 0 --rtol 0
"${gr_run_env[@]}" bash experiments/gr_serving/scripts/profile.sh \
  gr_nosa_flags_h64k_profile_20261004_02 --device cuda:0 \
  --latency-data "$gr_output/data/gr_nosa_flags_h64k_20261004_02" --num-users 1
```

Each profile selects the first three real revisits in its chosen saved
population, observes layer31, and constructs independent sparse prefixes for
serial and overlap. It retains the full-batch union-once serial comparison.
`profile.sh` runs its saved-output verification before copying accepted data.
Per-sample page and stripe ratios determine the 90% claim; a valid result that
fails this performance claim remains reportable. No `--nsys` is added to these
original native-interval diagnostics.

Retain all three old trace/profile/report families until the new matching
families pass correctness and measurement audits and are published. Then update
the README, selected report assets and affected old outputs in one replacement.
The source, placement, implementation changes and physical-budget limitations
must remain explicit; the six runs have not yet supplied new results.

## Execution update

Root accepted the fixed full32 evidence at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_fa3_stream_full32_20261004_01/evidence.json`,
SHA256 `865979653553a92e5c5f74dfa51b062a418edb537448b66c1f513dc0a78aa852`.
It contains four eager references and sixteen graph requests with exact hidden
comparisons, real misses and cleanup. The four selected ordinary shared-path
tests then passed on GPU0 (four passed, no skips). The budget checkpoint gate
passed the existing test at H65536/A128: all16 candidate outputs were exact
across four schemes, two users and two visits, with no skips. External launch/environment/actual-process
placement receipts are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_gpu0_execution_20261004_01/`.
Each stage saves its launcher source and rechecks the frozen model/harness
source before and after. These receipts explicitly record concurrent separate
GPU5/NUMA1 work and do not assert an unrecorded target allocator provider.

The first H4K attempt, `gr_nosa_flags_h4k_20261004_01`, ended with exit1
after all 804 request rows and numerical records were written. Its 201 HBM
rows correctly observed zero actual host-flag bytes at short history, while
the validator incorrectly required the conservative one-byte flag reservation
to be allocated. All 603 offload rows satisfied the existing accounting
formula. The failed data remain outside experiments at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/gr-serving-gr_nosa_flags_h4k_20261004_01.TpL70Q/`;
no profile or later history length was launched. These are failure diagnostics,
not accepted experiment results. The narrow report/audit fix passed 167 focused CPU tests and root review.
The new freeze is `8dc00e23469a67ebee67eb603859c9de9987e9151520a607d1bb20a498e547f9`;
H4K `_02` has started. New stage receipts are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_gpu0_execution_20261004_02/`.
The unchanged model/runtime GPU gates remain valid for their recorded source.
