# NOSA allocator guard: incremental observable-pool validation

Implemented on 2026-10-04 in `models/nosa/allocation_budget.py` and used by
`models/nosa/serving_resources.py`. Cache allocation size bounds are unchanged.

The prior guard called `gc.get_objects()` and a Python generator containing
`issubclass(type(value), torch.cuda.MemPool)` for every tracked object at every
lease boundary. Existing model/profile state made that scan a substantial CPU
cost. The new guard preserves detection of newly created empty Python MemPools
without repeating a full object scan on every call:

1. The first successful validation scans all observable GC generations.
2. Until collection, new tracked wrappers can only appear in generation zero,
   so subsequent guards inspect that generation.
3. A generation-zero collection requires scanning generations zero and one.
   A generation-one/two collection, a changed frozen-object count, or replaced
   GC functions/MemPool type requires a complete scan.
4. If collection occurs during inspection, validation repeats a complete scan
   before caching the observation. A rejected pool never creates a successful
   cached observation.

On CPython 3.12 with the original GC APIs, complete scans use the interpreter's
C traversal through `gc.get_referrers(MemPool, *all_subclasses)`. Concrete
instance types are visited by CPython's heap-type traversal, so this finds
public Python MemPool wrappers in every ordinary generation without creating
and walking a list of every tracked object's type in Python. Recursive subtype
enumeration calls `type.__subclasses__` directly and uses integer identity keys;
metaclass overrides and hashing do not run. A changed subclass inventory after
the scan triggers another scan. Extension subclasses must obey CPython's
heap-type GC contract, including visiting their concrete type. Other Python
versions/implementations and replaced GC APIs retain the original full-object
fallback. The fast young-generation path is unchanged.

The traversal proof uses CPython 3.12.13 `Objects/typeobject.c:1776`
(`subtype_traverse`) and `Modules/gcmodule.c:1678,1701`
(`gc_referrers_for`, `gc_get_referrers`). Both ordinary GC APIs exclude the frozen
generation. No constructor hooks or GC enable/disable changes are introduced.

Concrete types are deduplicated with integer identity keys before subclass
checks. This does not access an object's `__class__` property or invoke custom
metaclass hashing/equality. The allocator settings, caching-enabled flag,
uncached-allocation environment, capture state, and every reported allocator
segment are still checked on every call.

`validate_allocator(device, allowed_graph_pools=...)` additionally permits only
the private pool ID pairs explicitly owned by the backend's pure computation
graphs. Foreign private pools and expandable segments remain rejected; live
Python MemPool wrappers remain rejected. Allowed graph pool IDs are reported
separately and excluded from the cache allocator geometry key. Graph private
reserved storage is accounted by the graph owner, separately from cache tensor
allocation bounds.

Existing observability limits remain: frozen empty Python pools and empty
private/C++ routed pools cannot be discovered, and inspection is not atomic
against another thread changing allocator state. No global constructor hook or
GC enable/disable change was introduced.

Validation: the allocator unit suite passes 72 tests. Added cases cover exact
owned graph IDs, foreign/expandable rejection, new empty pool subclasses in
all generations, promotion, unfreezing, collection during inspection, and
unrelated objects with an unhashable metaclass. The selected allocator/resource/
owner/fixed-cache regression passed 235 tests before the final extra metaclass
case; the allocator suite was then rerun successfully with that case. The
hybrid complete-scan update adds twelve checks for instance generations and
inheritance depth, a new subtype during inspection, metaclass overrides, and
dispatch after generation-one collection.
The post-update allocator/cache-allocation/resources/owner/drain/fixed regression
passes 281 tests. Nine additional GPU process probes construct actual empty
`torch.cuda.MemPool(no_split=True)` wrappers and two subclass depths in
generations zero, one, and two; every case is rejected by `validate_allocator`.

A controlled CPU diagnostic with 582,578 tracked objects (including 400,000
ordinary retained lists promoted by `gc.collect()`) measured the complete warm
validator at median 0.0283 ms and maximum 0.0911 ms over 100 calls; the former
Python pool scan alone measured median 41.26 ms and maximum 45.98 ms over ten
calls. The CUDA runtime
was uninitialized in that diagnostic. These numbers characterize that guard
fixture, not full-model serving latency or allocator snapshot cost after model
loading. Full serving timings belong to the motivation experiment.

The incremental scan still performs a full scan after a generation-one/two
collection. Full-model diagnostics observe request tails attributable to those
scans; the fast warm fixture above does not remove or characterize that cost.

After the hybrid update, a fresh 582,579-object fixture measured the complete
forced-full validator at median 7.3093 ms and maximum 12.3149 ms over 30 calls,
and the warm validator at median 0.0307 ms and maximum 0.1392 ms over 30 calls.
An independent same-process comparison of the old type aggregation and the new
class-referrer helper measured medians 38.59 ms and 7.09 ms, respectively, over
20 calls each. These are CPU engineering diagnostics with an uninitialized CUDA
runtime, not checkpoint-serving measurements. Full scans still traverse all
ordinary generations and retain their measured cost inside request timing.

## Private snapshot adapter: integrated checkpoint, 2026-10-04

The integrated private C++ adapter takes a fresh all-pool allocator snapshot on
every guard call. It preserves the complete effective settings and deduplicates
only `(pool ID first, pool ID second, is_expandable)` segment predicates, avoiding
Python block dictionaries. Existing guards and allocator geometry keys remain
unchanged. Setup failure has an explicit official-API fallback; errors from an
already selected snapshot provider propagate. No global torch allocator API is
replaced by the implementation.

Post-integration validation passed 220 CPU tests, repository Ruff and format
checks. Five import-order fixes were applied before the production source freeze;
the reviewed C++ and loader bytes did not change. CPU-only compilation, concurrent
cache publication, fresh-process default/register8 reuse, exact settings parity
and strict mapped-library identity checks passed. GPU5 default/register8 parity
also passed with explicit fill nodes in eight graph pools: active, inactive and
released snapshots agree with the official API; owned pools are accepted, foreign
active/inactive pools and an empty Python MemPool are rejected, and geometry keys
remain stable within each configuration. Evidence is recorded in
`/tmp/nosa_allocator_integrated_acceptance_20261004/summary.json` and
`/tmp/nosa_allocator_snapshot_gpu_acceptance_20261004_v2/summary.json`.

An independent CPU reopen audit of production diagnostic
`nosa_guard_production_probe_20261004_01` passed. Saved source bytes match the
current source, the freshly recomputed native build identity matches the saved
plan, warmup/final mapped inventories agree, and the adapter binary matches its
manifest and saved runtime hash. The provider is `private_cpp`, with no fallback;
repository build fingerprint is
`21785fc27ce306846630270eae3af5ea2b31026dbc6bfa42be99c47c410a729e`, and binary SHA256 is
`36bfb148e301b5cb2ad5a2d3b89895c69d6f208c1796929714849e00887f7171`.
Both separately checked full candidate-hidden tensors match the accepted HBM
request-0 hash. These are two explicit output checks, not independent checks of
every timed sample.

The three diagnostics below use the full 32-layer checkpoint, one generated GR
request, H=65536, A=128, chunk=1024 and compute graphs on GPU0 (H200). Each contains
31 complete candidate-extend wall samples on an already built history. The saved
CPU settings agree: CPUs `0-47,96-143`, NUMA bind node 0, one PyTorch intra-op
thread, 96 inter-op threads, and OMP/MKL environment values of 8. This entry point
records CPU settings at summary time; it does not provide separate start/end CPU
observations. Guard/GC instrumentation overhead is included. Setup and prefix
construction are outside these candidate samples; these separate-process
engineering diagnostics are not the complete multi-user serving experiment or
an interleaved performance comparison.

| Diagnostic run | Provider | Median ms | Mean ms | Maximum ms |
| --- | --- | ---: | ---: | ---: |
| `nosa_guard_default_probe_20261004_02` | Official snapshot | 16.714589 | 22.955306 | 59.913105 |
| `nosa_guard_compact_probe_20261004_03` | Isolated compact prototype | 15.166019 | 15.229403 | 16.549604 |
| `nosa_guard_production_probe_20261004_01` | Integrated private adapter | 15.205597 | 15.260605 | 16.283846 |

Production recorded 79 guard calls, with median 0.523768 ms. The maximum,
5183.265645 ms, is the first guard, including cold adapter identity discovery,
build/import and guard checks; it is not a separately measured compiler duration.
All seven generation-0 and one generation-1 collections inside guards occurred
within that first interval, before prefill and candidate sampling. The other 78
guards contained no collection starts. Normal GC continued elsewhere: the
callback's active window recorded 682 generation-0, 62 generation-1 and five
generation-2 collections. One generation-0 collection lasted 0.039780 ms between
the entry/exit guards of candidate sample 16 (zero-based), outside either guard;
that sample took 15.686953 ms. Its collection duration alone does not explain the
sample's full excess latency.

The default probe's periodic long samples are absent from this production sample
set. This observation supports further measurement with the integrated adapter;
it does not establish a tail guarantee or replace formal serving remeasurement.
Full audit, exact hashes, paired GC intervals and recomputed statistics are in
`/tmp/nosa_allocator_integrated_acceptance_20261004/production_probe_audit.json`;
raw probe material remains under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_guard_production_probe_20261004_01/`.
