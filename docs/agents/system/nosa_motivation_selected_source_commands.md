# Selected-source formal NOSA command handoff

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Execution update: root completed the promoted-source 11-case GPU check and the
full32 fixed gate (4 eager references plus 16 exact graph requests), then accepted
`nosa_motivation_sm90_20261004_03` at runtime digest
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
Independent review checked all 128 saved outputs, timing groups, cache semantics,
source/dependency identity and mapped libraries. Receipt:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_motivation_formal_review_20261004_03/formal_measurement_review.json`,
SHA256 `74364c1ad795d8d99e9719825faa79b88c3c34965d2769ff3614928db081212f`.
The matching `nosa_motivation_profile_sm90_20261004_03` and independent
`nosa_attention_reference_sm90_20261004_02` also completed and passed their
independent audits. The accepted fixed package is published, and exact superseded
output cleanup is complete;
the [fixed report](../../../experiments/nosa_motivation/README.md) and
[fixed publication record](nosa_motivation_publication.md) own its final installed
state and cleanup receipt.

The full-request FA3-composition MFU ratios are 97.99%/99.29% for HBM requests
0/16, while the matched candidate-only ratios are 75.93%/72.50%. Candidate MFU,
tail behavior and compute/IO dominance remain unresolved. Async failed both the
formal overall-latency comparison and all 96 applicable samples' two 90% overlap
gates; valid failed gates remain reportable. This does not complete the overall
efficiency goal.

The independent GR H4K/H16K/H64K `_02` families at source `8dc00e...` have
completed, been published and had their exact superseded outputs cleaned.
See the [GR execution receipt](nosa_gr_selected_source_execution.md).
Its reporting-only host-flag fix does not change this 114-file runtime identity.

The commands below were prepared on 2026-10-04; the execution update above
supersedes their initial pending state. Root selected the FA3 reserved-workspace stream wrapper after its paired
comparison. No Q128 native candidate is selected. The source digest for this
sequence is `a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`,
captured after that patch; its fresh full32 gate passed before formal scheduling.
`02b5d0...` and the earlier formal `6e3dfd...` digests cannot label the new run.
The fresh full32 wrapper is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_fa3_stream_wrapper_20261004_01/promotion/full32_gate.py`,
SHA256 `b60b48acebadf901a9f290a5f19497ad662499e7365b8233b7c1fb2cb96d89a1`.
The accepted gate has four independent eager-reference requests
and sixteen four-scheme graph requests; its output is the fresh sibling
`nosa_fa3_stream_full32_20261004_01/`. The old `/tmp/nosa_integrated_allocator_full32_20261004.py`
hard-codes an existing output path and must not be rerun or edited in place.

Root's selected placement is physical GPU5, CPUs48–55, NUMA1 memory, eight CPU and
pinned-registration threads. All matched measurement/profile/API processes must
keep the same new source, allocator, precision and GPU UUID. The intended GPU5
UUID is `a5cd5bab-33a4-a7e2-4a3c-78c2b08a8872`; CUDA names it H200. The commands
use the existing entry points and preserve fixed P/NH semantics: H65536/A128,
chunk1024, P65536, NH16777216, sixteen users/two rounds, seed42, all32 checkpoint
layers and compute graphs. Measurement performs the existing three warmups and
32 measured requests per method. Profile uses three independent samples of the
first/revisit requests for every method. The independent API program retains its
fixed three warmups/seven repeats and both FA3 and materialized-QK/PV compositions.

## Environment and unused run IDs

Run from the repository root. The two shell wrappers add the repository venv
prefix themselves; their incoming PATH excludes it. Direct Python commands add
exactly one prefix explicitly. The SSD TMPDIR is required by attention capture.
The IDs were unused when this handoff was prepared and now identify completed
runs. Preserve the commands as execution records; use fresh IDs for any later
authorized rerun. Each entry point rejects an existing destination.

```bash
nosa_repo=/mnt/ssd-wlcb/chenkaiqi/cxldsagr
cd "$nosa_repo"
nosa_python="$nosa_repo/.venv/bin/python"
nosa_venv="$nosa_repo/.venv/bin"
nosa_base_path="$(python3 -c 'import os; from pathlib import Path; v=str(Path.cwd()/".venv/bin"); print(":".join(x for x in os.environ["PATH"].split(":") if x != v))')"
nosa_measure_id=nosa_motivation_sm90_20261004_03
nosa_profile_id=nosa_motivation_profile_sm90_20261004_03
nosa_api_id=nosa_attention_reference_sm90_20261004_02
nosa_output="$nosa_repo/experiments/nosa_motivation/output"
nosa_tmp=/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp
nosa_run_env=(env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_HIP_ALLOC_CONF
  -u FLASHINFER_DISABLE_JIT
  -u PYTORCH_NO_CUDA_MEMORY_CACHING -u PYTORCH_NO_HIP_MEMORY_CACHING
  -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES
  PATH="$nosa_base_path" CUDA_VISIBLE_DEVICES=5 CXLDSAGR_SM90_BACKEND=native
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  TMPDIR="$nosa_tmp"
  PYTORCH_ALLOC_CONF=backend:native,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
  numactl --physcpubind=48-55 --membind=1)
```

Run the correctness wrapper first, after root applies the selected patch. Its
preserved driver deliberately sets PyTorch intra-op threads to one; that is the
historical correctness fixture's setting, not the eight-thread formal timing
configuration. It keeps pinned registration at eight threads. Unsetting
`FLASHINFER_DISABLE_JIT` permits the new source-derived build keys.

```bash
"${nosa_run_env[@]}" env PATH="$nosa_venv:$nosa_base_path" "$nosa_python" -B \
  /mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_fa3_stream_wrapper_20261004_01/promotion/full32_gate.py
```

## Measurement, matching profile and independent APIs

Run sequentially after the fresh full32 gate and source freeze. Inspect each
program's success/acceptance before the next command. Failed diagnostics stay
outside the experiment deliverables; neither a failed overlap gate in a valid
profile nor an unfavorable API comparison may be hidden or relabeled as a pass.

```bash
"${nosa_run_env[@]}" bash experiments/nosa_motivation/scripts/run.sh \
  --run-id "$nosa_measure_id" --device cuda:0 \
  --model-path /mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  --num-users 16 --rounds 2 --history-tokens 65536 --candidate-tokens 128 \
  --chunk-size 1024 --sparse-pool-tokens 65536 --host-arena-tokens 16777216 \
  --seed 42 --compute-graphs --peak-bf16-tflops 989.5

"${nosa_run_env[@]}" bash experiments/nosa_motivation/scripts/profile.sh \
  --run-id "$nosa_profile_id" \
  --reference-run "$nosa_output/data/$nosa_measure_id" \
  --device cuda:0 --repeats 3

mkdir -p "$nosa_tmp/${nosa_api_id}_logs"
"${nosa_run_env[@]}" env PATH="$nosa_venv:$nosa_base_path" "$nosa_python" -B \
  -m experiments.nosa_motivation.src.profile_attention_reference \
  --run-id "$nosa_api_id" --device cuda:0 \
  --reference-run "$nosa_output/data/$nosa_measure_id" \
  --matrix-profile-run "$nosa_output/data/$nosa_profile_id" \
  --matrix-profile-dir "$nosa_output/profile/$nosa_profile_id" \
  --output-root "$nosa_output" \
  >"$nosa_tmp/${nosa_api_id}_logs/stdout.log" \
  2>"$nosa_tmp/${nosa_api_id}_logs/stderr.log"
```

`profile.sh` without `--method` covers all four methods and also produces the
matching GEMM/BMM reference. `--matrix-profile-run` takes the **data directory**,
not its run-ID string. The independent attention reference must bind the new
measurement and new profile; the old `_02`/old-reference pair cannot be reused.
The profile includes complete API prepare/repair and launch gaps; its timing
cannot replace the uninstrumented request trace.

## CPU reopen and report staging

These commands do not launch GPU work. Use fresh temporary report destinations,
then publish selected report assets and replace old results only after the
complete source/measurement/profile/API evidence is accepted. The report and
plot commands reopen every saved candidate tensor and memory-accounting record.

```bash
CUDA_VISIBLE_DEVICES= "$nosa_python" -B -m experiments.nosa_motivation.src.profile_audit \
  "$nosa_output/data/$nosa_profile_id" \
  --profile-dir "$nosa_output/profile/$nosa_profile_id" \
  --reference-run "$nosa_output/data/$nosa_measure_id"

CUDA_VISIBLE_DEVICES= "$nosa_python" -B -m experiments.nosa_motivation.src.profile_attention_reference \
  --audit-data "$nosa_output/data/$nosa_api_id" \
  --reference-run "$nosa_output/data/$nosa_measure_id" \
  --matrix-profile-run "$nosa_output/data/$nosa_profile_id" \
  --matrix-profile-dir "$nosa_output/profile/$nosa_profile_id"

CUDA_VISIBLE_DEVICES= "$nosa_python" -B -m experiments.nosa_motivation.src.report \
  "$nosa_output/data/$nosa_measure_id" \
  --output-dir "$nosa_tmp/${nosa_measure_id}_report"
CUDA_VISIBLE_DEVICES= "$nosa_python" -B -m experiments.nosa_motivation.src.plot \
  "$nosa_output/data/$nosa_measure_id" \
  --output-dir "$nosa_tmp/${nosa_measure_id}_figures"
```

Do not average away failed layer/sample overlap values: every applicable sample
must pass both page-envelope and nonempty-stripe-copy ratios. Request latency,
candidate extend latency, complete-API execution spans and sums of independent
API medians retain separate denominators. A valid profile may report that a
performance gate failed.

An external CPU-only staging helper combines these existing audits/renderers
with source checks, selected diagnostic tables, artifact hashes and an old-output
inventory. It refuses an existing destination or any destination inside the
repository. It stages files for review; it does not publish, delete old outputs
or update the README. Use it after the complete new trio exists:

```bash
CUDA_VISIBLE_DEVICES= "$nosa_python" -B \
  /mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_selected_source_publication_tools_20261004/stage_publication.py \
  --measurement "$nosa_output/data/$nosa_measure_id" \
  --profile-data "$nosa_output/data/$nosa_profile_id" \
  --profile-dir "$nosa_output/profile/$nosa_profile_id" \
  --attention-data "$nosa_output/data/$nosa_api_id" \
  --stage-dir "$nosa_tmp/${nosa_measure_id}_publication_stage"
```

This helper's preparation record covered CLI/AST and interface checks only.
The complete accepted trio is published. Use the fixed publication record above
for the actual staging and
install receipts; do not infer execution of this particular helper from its
presence here.

## Replacement and snapshot inventory

The inventory below records the pre-replacement state, including the then-empty
report directory and old IDs. It is not a current retention list. Required frozen
Q128 inputs were copied outside experiments as described below; their dependency
is closed. Fixed and GR installation and exact cleanup are complete; their
publication records preserve the receipts.

| Family | Existing retained identity | Required action after new acceptance |
| --- | --- | --- |
| Formal fixed P/NH measurement | `experiments/nosa_motivation/output/data/nosa_motivation_sm90_20261004_02/` and matching `output/log/` | Replace with the new four-method run, then remove the superseded measurement/log directories in the same publication update. Retain all old measurements, numerical tensors, memory, workload and `source/`/`source_manifest.json` until then. |
| Matching old profile and raw matrices | `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_motivation_current_profile/{data,profile,log}/nosa_motivation_current_diagnostic_20261004_02/` | Produce a new profile/data/trace family from the new measurement, with source/native/CPU/GPU identity matching. Retain old external diagnostics and any frozen Q128 evidence that depends on them; do not relabel them as new-source evidence. |
| Independent attention/API reference | `experiments/nosa_motivation/output/data/nosa_attention_reference_sm90_20261004_01/` | Rebuild after the new matrix profile. It contains source snapshots, raw matrix reference, 2,112 call captures, reuse proofs and request comparisons. Do not attach these old captures/timings to a new-source formal measurement. |
| Human report | `experiments/nosa_motivation/README.md`; no files currently exist under its `report/` directory | Replace the original-source result text only after acceptance. Publish the selected new tables/figures/data under `report/<new_run_id>/`, not ignored output directories. The old measurement's internal `output/data/.../report/` is retained with that measurement until replacement. |
| Fixed/budget shared reserved attention | Current `nosa_motivation` and NOSA `gr_serving` rerun entry points | The modified helper is exercised by shared HBM/dense paths. H4K/H16K/H64K were independently rerun at `8dc00e...` and published with their original workload limits; see the [GR execution receipt](nosa_gr_selected_source_execution.md). Fixed P/NH and budget-mode results remain separate. |

New runs must create new source snapshots; never overwrite a retained run's
`source/`, `source_manifest.json`, metadata, build identities or binaries. The
FA3 helper change alters `_fa3.py`'s hash and source-derived resident/fused native
build keys even though CUDA source is unchanged. The full32 gate must save fresh
source/adapter evidence. Formal runs regenerate native build/header and
mapped-library inventories from the new source. Preserve old cached libraries
used by immutable receipts.

The old attention reference's `capture_000016` is also an input to frozen Q128
diagnostics outside experiments. This is a cleanup dependency: before deleting
the superseded attention-reference output after new publication, retain any
required diagnostic operands/provenance byte-for-byte outside experiments and
record their hashes and original-to-external path mapping. Do not rewrite frozen
helper manifests or present the old timings as a new formal reference. This
handoff has now copied the required closure to
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_original_diagnostic_inputs_20261004/`.
The relocation index contains 467 files and 2,301,547,815 bytes, SHA256
`d54378ec5d1a7d803a2fffa674bbb73acb02c465419d32a22ab506fc073e11ea`.
It retains all32 request16 calls and full K/V/CIS, both methods'64 ownership
files and two standalone work JSONs, bound metadata, source snapshots,
measurement provenance and the request0 history manifest. Request0 tensor
payloads are outside this Q128 replay dependency closure.

Every copy matches its original SHA256; original stat identities and five
frozen consumers are unchanged. `reopen_copied_inputs.py` reopens all64
layer/method cases through the unchanged frozen input contract, with CUDA
hidden and never initialized. Its receipt SHA256 is
`92953e753b5e9cb75755ca21885daa2c5cddeafb58ff0d1817e49b04c593ffd1`.
An independent rehash/closure audit passed at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_original_diagnostic_review_20261004/preservation_review.json`,
SHA256 `46963b93074f9d4390f2ecda8e7365facd372115bdb0853754e00a6e10fef210`.
Frozen defaults still name the original paths; use explicit copied input paths
and the relocation index when reopening them. No frozen helper or manifest was
rewritten. This preservation step removed no original outputs; final removal
of superseded fixed outputs belongs to the fixed publication receipt above.

The six frozen Q128 successor stages with `runtime_source_manifest.json` bind
`operators/nosa/attention/device_only/_fa3.py` to **their own staged copies**, each
at SHA256 `d9612baa1b220136ea4fa40da1ae578b9640af7baf2be6dfe74b6211348d3fe1`.
A production-only helper patch does not mutate those files or their libraries.
They remain rejected diagnostic records; their old source snapshots and frozen
helper manifests must not be refreshed to the new digest. Their audits and the
old `02b5d0...` full32 evidence establish only their recorded implementations.
No further Q128 GPU work is selected.

Execution-dependent replacement scope is narrower than a broad source-manifest
hash change. The owned resident operator/module/full-model paths, dense baseline,
pattern collection and Q1024 cold-union overlap experiment do not pass a reserved
workspace into this helper. In particular, the overlap experiment's resident
call omits `workspace=`, and its offload arms use the separate fused path. Their
valid existing reports keep their original source identities; do not silently
relabel them or schedule blanket replacement solely because `_fa3.py` is included
in a broader source inventory. Any actual call-path change found in the final
patch review expands the affected set before publication.

Verification for this handoff: existing measure/profile/independent-reference/
report/plot CLI help completed with CUDA hidden; command syntax and path inventory
were checked. These are preparation checks, not new measurements.
