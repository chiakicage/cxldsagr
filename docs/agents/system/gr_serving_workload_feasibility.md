# GR workload feasibility and cache-pressure redesign

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

2026-10-02. Independent CPU-only planning calculation after the researcher rejected
configured-population scaling and traces without revisit misses as answers to the
intended capacity question. No GPU execution, numerical-source change or new
model-latency measurement was performed. Root owns corrections to prior reports
and research status.

## Why the previous workload does not answer the question

The configured population was a sampling domain, not the number of users actually
exercising the cache. At 64K, the nominal 512-user case touched only six users and
had no revisits. Raising this label or adding distinct first visits cannot fix
the central problem: an evicted history must be requested again before the
capacity difference can affect a revisit.

The earlier independent audit established numerical, budget, LRU, identity and
report consistency. It did not establish that the workload exercised the intended
population/capacity regimes. Passing those engineering gates was insufficient
to declare the requested research experiment complete.

For equal-size whole-user sessions, let C be admitted session capacity and D the
number of distinct other users accessed since a target's preceding access. With
ordinary LRU, the target hits exactly when D < C. Pressure experiments should
cover D < C_HBM, C_HBM <= D < min(C_offload), and D >= max(C_offload). The middle
band isolates HBM-miss/offload-hit behavior; the last band exposes reconstruction
cost after both tiers have lost the target. Mixed offload capacities also create
an intermediate band, e.g. DeepSeek dense prefetch may still hit after ECHO misses.
These are user-session states, not DeepSeek page-slot hit rates.

## Cost of actually covering the original population grid

Calibration uses the persisted `measurements.jsonl` from the named 4K/16K/64K
`gr_serving_h200_20261002_*_01` runs. Only rows with `is_revisit=false` determine
the first-request costs. For each history h, configured population N, model m and
scheme s, the estimate is:

`T_first(h) = sum_N sum_m,s N * mean(observed first-request latency[h,N,m,s])`.

Every request includes actual admission, prefix construction, candidate execution
and cleanup. The same N must be genuinely accessed in all eight independent
model/scheme cases. `sum([1,8,32,64,128,256,512]) = 1001`, so **8008 first requests
per history is a hard request-count lower bound**, or 24,024 across three histories.
The time figures below are extrapolations, not hard wall-time lower bounds.

| Actual N, each of eight cases | 4K cold-request estimate (min) | 16K (min) | 64K (min) |
|---:|---:|---:|---:|
| 1 | 0.05 | 0.20 | 0.85 |
| 8 | 0.41 | 1.69 | 7.02 |
| 32 | 1.61 | 6.72 | 27.69 |
| 64 | 3.15 | 13.15 | 54.73 |
| 128 | 6.39 | 26.56 | 109.42 |
| 256 | 12.66 | 52.80 | 218.93 |
| 512 | 25.26 | 105.24 | 438.44 |
| **Whole grid** | **49.54** | **206.36** | **857.08** |

The combined cold-request estimate is **1112.98 min, or 18.55 hours**. Replacing
each case's mean with its observed first-request minimum/maximum gives sensitivity
totals of 49.03–50.88, 205.10–210.60 and 855.42–859.62 minutes. These are neither
confidence intervals nor guaranteed future bounds; unseen users, cache state and
runtime conditions can change cost.

These estimates omit every revisit, loading, warmup, compilation, input generation,
numerical comparison, profiling and report overhead. Revisit misses add another
request and its reconstruction; they do not eliminate the need for the initial
first request. The old runs' wall time minus summed measured request time was
1.07/2.92/3.52 minutes, but those differences are not guaranteed overhead for a
new design. The complete original grid cannot honestly be advertised as a
roughly 30-minute study at these measured costs.

Detailed per-case calibrations, source/run identities, raw-measurement hashes and
all calculations are in `/tmp/gr_serving_workload_feasibility_20261002.json`.
This is a planning artifact, not an experiment result.

## Two explicit alternatives around 30 minutes

**A. Preserve heat-driven serving requests; narrow the GPU scope.** Use NOSA at
16K first, with actual N=32 and 64, the same four schemes and 4 GiB/16 GiB budgets.
Preserve original weighted draws and the maximum of eight revisits per user.
Predeclare a coverage stopping rule rather than silently stopping at six or 32
requests: continue until every selected user has appeared. Do not insert forced
first visits or run every user to an equal nine-access quota. A finite cap of
9N bounds the possible schedule; the actual coverage stop usually occurs sooner.
This is a coverage-conditioned finite trace, not a steady-state service sample.
Determine its exact request count and sign the complete trace in CPU preflight,
then freeze it before any GPU timing. All schemes receive that same trace and
request count. Separate N/seed runs can use different explicit caps; do not
silently truncate a running model case to fit the clock.

A CPU-only preflight used all seeds 0–4 without filtering. It replayed the saved
Beauty weights with capped weighted draws and evaluated exact LRU reuse distance
for C_HBM=7 and C_offload=31. Every row below contains all N users. The columns
are predicted counts, not GPU-observed events:

| N | Seed | Requests / revisits | HBM hit | HBM miss / offload hit | Both miss | Estimated four-scheme request time (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 137 / 105 | 39 | 66 | 0 | 2.66 |
| 32 | 1 | 191 / 159 | 58 | 101 | 0 | 3.14 |
| 32 | 2 | 122 / 90 | 30 | 60 | 0 | 2.56 |
| 32 | 3 | 125 / 93 | 40 | 53 | 0 | 2.50 |
| 32 | 4 | 191 / 159 | 41 | 118 | 0 | 3.30 |
| 64 | 0 | 363 / 299 | 49 | 142 | 108 | 10.69 |
| 64 | 1 | 317 / 253 | 42 | 150 | 61 | 8.36 |
| 64 | 2 | 409 / 345 | 55 | 169 | 121 | 11.71 |
| 64 | 3 | 310 / 246 | 46 | 120 | 80 | 8.98 |
| 64 | 4 | 398 / 334 | 60 | 150 | 124 | 11.64 |

Using the first two seeds by a fixed index rule would estimate 24.85 minutes of
request execution, leaving roughly five minutes within a 30-minute target for
overheads. Using seed 0 only estimates 13.35 minutes and allows more margin or
prespecified repetitions. Do not choose the seed with the best miss rate or timing.
The estimates charge every modeled miss the measured first-request mean and every
hit an observed hit mean; offload remiss latency was not measured in the old
trace, so this substitution is only a scheduling approximation. A new GPU run
must execute and verify all requests and actual cache transitions. Wider N,
budgets and seeds can remain a clearly labeled CPU cache-state study; synthetic
counts or a timing lookup must never be published as measured model latency.

**B. Preserve both models and all schemes; make the experiment explicitly
mechanistic.** At 16K, use independent traces `target, D distinct real background
users, target with a new candidate`. Execute every request with the real model,
starting each scheme from an empty cache. Select D before timing from the admitted
capacities, not from a favorable seed. One target revisit already tests the
desired state; additional repetitions need independent full traces and a separate
time allowance. No user exceeds the eight-revisit cap.

| Model | Capacities HBM / sparse offload / dense | D values | Actual N values | Estimated four-scheme time for all three bands |
|---|---|---|---|---:|
| NOSA | 7 / 31 / 31 | 6, 7, 31 | 7, 8, 32 | 2.54 min |
| DeepSeek surrogate | 19 / 36 / 68 | 18, 19, 68 | 19, 20, 69 | 16.89 min |

The combined estimate is 19.43 minutes plus overhead, leaving room for a small
prespecified NOSA repetition set. This design changes scheduling semantics:
placing the target revisit at a specified distance is a controlled probe, not
an unmodified heat-driven request stream. Heat can choose background identities
or their weighted order without replacement, but that does not make the whole
trace representative serving traffic. If preserving the original draw process
is mandatory, choose A instead and retain the CPU-predicted distance bands.
The controlled schedule requires an explicit workload/auditor contract; it must
not be presented as passing the current weighted-draw contract unchanged.

Report target-revisit end-to-end latency conditioned on actual hit/miss regime,
alongside the entire trace's time including all background cold construction.
Do not silently preload synthetic sessions, clone their KV, replay a saved hidden
output, or omit background execution and call the result serving latency. For a
budget sweep, use the same allowed HBM/DRAM limits across compared schemes and
recompute all reservations first. D/C_HBM and D/C_offload are useful mechanistic
axes; actual visited N, sample count and byte budgets remain visible.

## Disposition of earlier claims

- Withdraw the original plots as evidence of 1–512 active-user scaling or
  representative multi-user serving. A nominal sampling domain with six observed
  users cannot support the 512-user label as an exercised working set.
- The 4K/64K absence of revisit misses means those traces did not test the intended
  capacity benefit. It does not show that offload lacks a benefit under pressure.
  All-hit traces remain useful only as explicitly paired controls.
- The 16K 13/63 HBM revisit misses and zero offload misses are trace-local facts,
  not robust population-scaling evidence. They motivate a pressure test, but do
  not rescue the whole original grid's adequacy.
- The 20/21 overlap-versus-serial all-request mean comparisons are arithmetic
  facts about the old traces, not 21 independent demonstrations of scaling or a
  general serving advantage. HBM's old all-request lead is equally conditional
  on the highly first-visit-heavy mixture. Re-evaluate both on the new traces.
- Saved exact-output comparisons, source identities, budget accounting and LRU
  checks remain evidence of correctness within their tested boundary. They do
  not turn a rejected workload into a valid experiment for the user's purpose.
- Actual unique-copy and softmax-overlap intervals remain local kernel evidence
  for the recorded inputs. They neither establish active-user scaling nor negate
  measured request-time differences. Any retained experimental presentation must
  have its own valid purpose; do not repackage rejected serving results as a new
  successful study by changing titles. Root decides artifact withdrawal under
  the repository's result-validity rules.

The acceptance gate for the next capacity study must include actual working-set
coverage, prespecified reuse-distance regimes and real measured revisit misses,
in addition to the existing numerical, budget and source gates.
