# Formal prelude before the matched reduced HBM timer

## Scope

The matched GPU0/environment reduced trace still has approximately 96 ns median
gaps. Add the formal four-method warmup package before that reduced lifecycle,
using a new driver and receipt. This diagnoses preparation history as a package;
it does not identify pinned allocation, registration, stream selection or fences.
No production code, `q1_replay_matched.py` or `q1_replay_timer.py` changes.

## Execution boundary

`q1_replay_prelude.py` delegates to the frozen matched wrapper. Its environment
adapter first applies the formal environment, then imports the timer and installs
a model subclass plus source/runtime evidence adapters. The wrapper still checks
for provider imports before environment application and retains all formal
source, backend, participating-runtime and retained-artifact gates.

After the original `prepare_compute_graphs([1024, 1])`, run exactly one warmup
per method in formal order: HBM, ECHO, serial sparse, dense prefetch. Reuse
`profile_layers.select_cache_method`, `restore_extend_prefix` and
`prepare_extend_graph`. Each method selects a fresh cache, forwards H65536,
snapshots, restores with cold offload residency, prepares the Q1 logits graph,
forwards token111090 and releases the snapshot. No prelude graph observer,
output copy or profiler capture is added.

Finally select HBM once. Require an empty resident cache, no shared pools,
sessions, dense helpers or full graphs, five cache-generation advances, and
the original compute-bank object. Do not retain prelude graphs or outputs.
The unchanged timer then performs its original prefix, independent eager and
changed-token checks, fresh 197-node/192-layer graph capture, five plain warmups,
50 clean AB/BA pairs and two profile pairs. All failures propagate through the
timer's existing model cleanup.

## Evidence and acceptance

Use receipt kind `deepseek-q1-replay-prelude-v1`. Bind and archive the new driver
alongside the unchanged matched wrapper, timer and all formal sources. Record
the completed method order and lifecycle gates in runtime identity. Actual
offload binaries/specializations may now fill entries absent from the matched
HBM baseline; every entry must still match the formal record exactly. Each arm
uses its own independently accepted receipt.

Freeze and review the driver before separate GPU check, clean bench and NSYS
profile runs. Use the same GPU0, CPU0-7, request and formal environment. Compare
197 native GPU signatures and 192 layer owners, raw profiler settings, and
all-process clipped activity unions. Report profiler gaps/idle separately from
clean wall time. A baseline cannot run after the prelude in the same process;
use fresh processes, and process-level AB/BA confirmation if an effect appears.
The within-process plain/timed ordering does not balance prelude history.

The CLI is the matched timer CLI with module
`experiments.deepseek_v32_mfu.src.q1_replay_prelude` and fresh run IDs. GPU
execution and numerical acceptance are pending at driver freeze.

`analyze_replay_prelude.py` delegates to the frozen matched analyzer with only
the explicit prelude receipt kind, driver path and completion checks adapted.
It requires the preserved base wrapper, all formal runtime entries, the four
methods, one compute bank, five cache generations and released prelude resources.
It rehashes the accepted matched audit and records corresponding raw-scope and
clean-wall summaries from the two processes. The prelude treatment is not
balanced across processes by the timer's within-process plain/timed AB/BA pairs.
Neither existing analyzer nor any receipt identity is modified.
