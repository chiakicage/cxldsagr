# GR serving workload and reporting contract

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

This component implements the workload/reporting part of the user-requested single-GPU multi-user
serving comparison. It does not validate the model, claim performance, or update research status.

## Interfaces

- `experiments.gr_serving.src.workload.WorkloadConfig(model, num_users, requests,
  history_tokens=8192, candidate_tokens=1024, seed=42, heat_dataset="beauty", heat_field=None,
  sampling="weighted")`.
  The default field resolves to `pv_share` for industrial curves and `interaction_count`
  otherwise; the serialized config always names the resolved field. Both workload and
  measurement CLIs expose `--heat-dataset` and `--heat-field`.
- `build_workload(config, tokenizer=Path|Tokenizer|None)` returns `Workload(requests, manifest)`.
  Requests are a tuple of ordinary GR request dictionaries; `Workload.write(output_dir)` persists
  `requests.jsonl` and `workload.json`, refusing to overwrite an existing workload. Request JSONL
  is streamed one row at a time to avoid constructing a multi-gigabyte joined string.
- `history_tokens` is the entire stable prefix, including instruction; `candidate_tokens` is the
  entire suffix. Generator budgets are adjusted by the tokenizer's actual instruction overhead.
- Request additions: `request_id`, `visit_number`, `is_revisit`, `prefix_sha256`,
  `candidate_sha256`, `input_sha256`. GR `visit_index` remains zero-based. The manifest signs
  the ordered request identities, config, heat-resource hash, and serialized tokenizer hash.
- `report.summarize(rows)` returns schema version, metric definitions, and `groups` containing
  all/first_visit/revisit statistics. `report.write_report(rows, output_dir)` also writes JSON,
  summary/per-request CSV, per-request SVG, and `summary.svg` comparing mean/p95 against user
  populations with separate all-request/revisit panels per model, without plotting dependencies.

Required measurement fields: `run_id`, `model`, `scheme`, `num_users`, `workload_sha256`,
`request_id`, `user_id`, `visit_index`, `input_sha256`, `latency_ms`, `hbm_budget_bytes`,
`dram_budget_bytes`. `phase` if supplied must be `measured`.

Optional fields include `history_tokens`, `candidate_tokens`, `seed`, `prefix_cache_hit` (boolean;
`prefix_hit` is an accepted alias),
`prefix_hit_tier` (`hbm`/`dram`/`miss`, only when it has a meaningful backend definition),
`evicted_users` (list of IDs or count), `cache_hbm_bytes`, `cache_dram_bytes`, `transfer_bytes`,
`prefix_ms`, `extend_ms`, and `cleanup_ms`. CSV preserves all extra fields.

## Workload semantics

Selected cumulative heat interpolation and weighted independent draws come directly from `GR/`.
No coverage-forcing requests are inserted. Fixed synthetic timestamps are metadata only.
All schemes in a model/population/input case reuse one generated manifest. Warmups use independent
state and cannot change visit indices or populate the measured cache. Prefixes are checked by exact
token tuple equality across visits, candidates must change, and all three hashes are saved.
The CPU text generator retains up to `min(num_users, requests)` histories, so revisits do not
rebuild long synthetic histories. This is input-generation memory, outside model cache accounting;
it does not alter token content or prepopulate any measured model cache.

Revisits are independent of hits. The whole-user session design may evict a user between visits;
that request remains a revisit and its prefix reconstruction belongs in request latency. A first
visit cannot hit the whole-user cache after a genuine empty reset. The reporter checks this.

## Grid recommendation and budget scope

Use the same allowed HBM and DRAM cache byte caps for every scheme and both models. Report actual
allocations separately; pure HBM legitimately leaves its allowed DRAM capacity unused. Weights,
temporary computation and allocator peak are separate disclosures. Choose user counts around the
common HBM budget's resident prefix capacity (for example 1, 8, 32 after capacity inspection), and
at least `max(128, 8 * num_users)` requests for reasonable revisit counts. Primary prefix/suffix
16384/1024 stays within the NOSA generator's supported 32768 context. The final grid and sample
count must be recorded as actual run parameters, not implied by this recommendation.

## Publication and validation

No GPU measurement has been performed by this component. CPU tests exercise exact-token generation
with portable real byte tokenizers for both model templates, deterministic traces, workload hashes,
report rejection for mismatched inputs/budgets and contaminated or incomplete traces, revisit misses,
empty subsets, quantiles, and valid standalone SVG output. Temporary outputs use pytest `tmp_path`.

`scripts/run.sh` invokes the root-owned measurement CLI in a system temporary directory and preserves
its failure code with pipefail. Only a successful complete run is published under the new run ID;
failed diagnostics remain outside `experiments/`. Report publication and copying selected artifacts
to tracked `report/` require the experiment's model/numerical/budget/measurement acceptance gates.

Validation performed 2026-10-02: Ruff passed for the two source modules and their tests;
`pytest experiments/gr_serving/tests/test_workload.py experiments/gr_serving/tests/test_gr_serving_report.py -q`
passed 23 tests. `bash -n` and the script/workload/report `--help` paths passed. A separate CPU
generation check using the real local NOSA tokenizer at prefix/suffix 16384/1024 with 8 users,
16 requests produced 7 unique users and 9 revisits and verified every token boundary. This is input
validation, not a model or performance result; no output was published under `experiments/output`.

The standalone `serving.run_multi_user` CLI consumes GR APIs directly, with no experiment imports.
It selects the model backend and persistent runner, reports shapes and request/cache statistics,
and closes sessions on failures. Unlike the formal experiment, it performs no separate warmup;
first-use JIT compilation belongs to its displayed request latency. The CLI/report update passed
Ruff and 28 CPU checks (`serving/tests/test_run_multi_user.py` and
`experiments/gr_serving/tests/test_gr_serving_report.py`); GPU CLI smoke remains root-owned.

## Industrial 10M rerun input contract (2026-10-02)

The requested source is `GR/generated/industrial_10m_pv_share_t4096_seed42/requests_1024.csv`:
`industrial_10M` / `pv_share`, 1024 synthetic users in the sampling population, seed 42,
4096 weighted independent draws, and no revisit cap. The archived curve describes a synthetic
industrial population, not verified production traffic. The observed trace has 751 first visits,
3345 revisits, 507 returning users, 273 unvisited users and a maximum of 129 visits per user.

The CSV SHA256 is `f2a507dd546c33f5b9d9774a50824945e44f3dfce5709052864b20e7a5888452`.
Every new workload adds `access_trace_sha256`, signing the ordered list of dictionaries with
`request_id`, `user_id`, `visit_index`, `previous_request_id`, and `timestamp`. Use canonical
JSON (`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`) and SHA256 of UTF-8 bytes.
Map the CSV's `synthetic_timestamp` to `timestamp`, parse IDs as integers and empty previous IDs
as null. The complete trace signature is
`31d8a1e01b97f53f3d6db1a098085a28050e92d240db8ed9e2ac7ae680c98f97`.
This signature is independent of model tokenizer and history geometry. Existing schema 1,
workload identity signing, manifest request identity fields and saved token-row format are preserved.

CPU scheduling preflight compared every row, including previous-request IDs and timestamps, at
4K/16K/64K +128 for both models against the complete archived CSV: all six matched exactly.
The byte-tokenizer unit tests exercise the archived trace's first eight identities for both model
formats, configurable heat fields, streaming persistence and the existing identity/cap/context gates.
Real local tokenizer preflight separately checks both models' first two 64K +128 inputs. This is
input validation only; full new model execution and numerical acceptance remain root-owned.
Validation completed: 32 workload tests passed; Ruff check and format check passed. The real
64K +128 checks passed using `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B/tokenizer.json` and
`/preset-models/tokenizer.json` (2.81 s / 5.10 s for two requests, respectively).

The then-current user instruction prioritized one complete 64K +128 industrial group with
**64 GiB HBM / 1024 GiB DRAM**. This was subsequently superseded by the small sequential group
below; the industrial attempt stopped before measurement. The following earlier calculation
uses the superseded 4 GiB HBM / 16 GiB DRAM budgets and does not predict the requested new run. Cold/miss
and hit cost estimates use mean observations from the old
`gr_serving_h200_20261002_h64k_01` measurement JSONL; a revisit miss is charged the old cold cost.
These are runtime estimates, not new performance results or a validated cross-user latency model.

| Model | Scheme | Session capacity | Predicted revisit misses / 3345 | Predicted hits | Estimated request hours |
|---|---|---:|---:|---:|---:|
| DeepSeek surrogate | HBM | 5 | 3251 | 94 | 10.021 |
| DeepSeek surrogate | ECHO | 9 | 3180 | 165 | 12.529 |
| DeepSeek surrogate | Serial sparse | 9 | 3180 | 165 | 10.580 |
| DeepSeek surrogate | Dense prefetch | 17 | 3028 | 317 | 9.430 |
| NOSA | HBM | 1 | 3321 | 24 | 2.700 |
| NOSA | Serial sparse | 7 | 3221 | 124 | 4.101 |
| NOSA | Dense prefetch | 7 | 3221 | 124 | 3.355 |
| NOSA | Overlap | 7 | 3221 | 124 | 3.632 |

The combined estimate is 56.35 hours of request execution. Input generation, loading, warmup,
numerical-output I/O, reporting and profiling are excluded. The full trace remains 4096 requests;
no scope reduction or trace truncation follows from this planning estimate.

## Current small sequential group (2026-10-02)

Latest steering replaces the immediate industrial run with history/candidate **65536 +128**,
**16 users / 32 requests**, **4 GiB HBM / 64 GiB DRAM**, and the explicit access order
`0,1,...,15,0,1,...,15`. Both model workloads and all four schemes retain their existing execution
boundaries. This is a controlled two-pass access sequence, not heat-weighted sampling.

`WorkloadConfig.sampling` and `--sampling` accept `weighted` (unchanged default) and `sequential`.
Sequential mode normalizes `config.heat_dataset` and `config.heat_field` to null, rejects any
non-null `max_revisits`, and constructs `InputGenerator` directly with explicit synthetic IDs.
It does not load a heat curve, call `HeatPopulation.from_curve`, or draw random user identities.
The seed still controls synthetic history/candidate content; it does not randomize user order.
GR's sequential scheduler gives request `r` user `r % num_users`, visit `r // num_users`, and
previous request `r-num_users` after the first pass. Timestamps remain synthetic request indices.

The saved schema 1 identity retains `heat_sha256` with value null. `manifest.heat` is exactly:

```json
{
  "source": "explicit_synthetic_ids",
  "synthetic": true,
  "selected_users": 16,
  "selected_weight_sum": 16.0,
  "user_identity": "synthetic IDs 0..N-1",
  "weight_semantics": "equal placeholders required by GR; no heat curve or probabilistic sampling"
}
```

Each `users` entry has `weight=1.0`, `probability=null`, observed visits/revisits and its prefix
hash. The existing request dictionary's `user_heat_weight=1.0` is the same inert GR placeholder;
it is not an observed industrial heat or a random sampling probability. Measurement policy
explicitly describes deterministic cycles and no heat/random user draws. Canonical access
signing and the original full workload identity structure remain unchanged.

Expected outputs are 16 first visits, 16 revisits, 16 returning users and two visits per user.
Every second-pass prefix must equal its first-pass prefix exactly, while its candidate changes.
The first three NOSA profile revisits are consequently request IDs 16/17/18, subject to later
authentication against the complete saved formal workload. A complete measurement still needs
8 cases / 256 measured requests and 192 non-HBM full-candidate-hidden comparisons; CPU generation
does not provide those model results.

The updated workload suite passed **39 CPU tests** and Ruff check. Its sequential tests forbid
curve reads, check all 32 identity records for both model formats, verify stable/changed token
hashes, assert null heat provenance and placeholder semantics, reject caps, and cover CLI output.
Real local-tokenizer checks of all 32 requests at 65536 +128 passed for both models (NOSA 25.36 s,
DeepSeek 42.22 s including writing), using temporary validation inputs outside `experiments/` at
`/tmp/gr-serving-sequential-preflight-rij1jqvh/{nosa,deepseek_v32}/16/`.
Both access signatures are
`c3511ba7f3fb306def87447c0b7d7c545e253f5f113fb98288987612befedda9`.
Workload signatures are `d8569499b36e8a4ee9f1367e760b875494a7f1a545245c47e35535c6970a8040`
(NOSA) and `4885ab864bed148f9f4a54230c3dced1c9a3749ac29ca92e13faf1edb019b9d1` (DeepSeek).
The workload source is frozen at SHA256
`2c16eb30be6888bf19ed07670dbc37b97e1bb18527545c8d6157dc79b6c644de`;
the independent audit agent accepted both complete saved workloads. Root then launched
`gr_serving_h200_20261002_sequential_u16_t32_h64k_01` (exec session `42234`). Source remains
frozen through formal measurement/audit/profile. Workload validation is complete; the new run's
GPU numerical, latency, budget, native-profile and publication gates remain unproven here.
