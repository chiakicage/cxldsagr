# NOSA admission microdiagnostic and cleanup inspection

The [result](result.json), [original diagnostic source](profile_plan.txt), and
[command/file index](evidence.json) record a bounded pure-CPU diagnostic of the
unchanged production planner. H65536/A128/C1024, full32, P65536/NH16777216 and
all four fixed schemes match the request geometry. The resource constructor uses
the existing offline SM90 assumptions. It creates plans but does not allocate
resources or sessions, load weights, or initialize CUDA. CPU0–7, NUMA0 and one
OMP/MKL/OpenBLAS thread differ from the eight-thread serving runs; these numbers
are not a full-request causal attribution.

Unprofiled values below are medians of seven batch means, with 100 calls per
batch and ten warmup calls. All samples are retained; GC stayed enabled.

| Work | HBM, microseconds | Offload range, microseconds |
| --- | ---: | ---: |
| `plan_runtime_session` | 552.8 | 1026.7–1033.8 |
| `session_allocation_layout` | 497.6 | 926.6–928.5 |
| Reconstruct every `AllocationSpec`, using prebuilt kwargs | 233.9 | 441.3–446.5 |
| Validate every existing `AllocationSpec` | 87.9 | 165.2–181.3 |
| All footprint scans internal to layout | 59.0 | 111.4–117.0 |
| One full footprint scan | 18.4 | 37.3–38.0 |
| Construct `SessionPlan` from prebuilt kwargs | 23.2 | 45.5–45.7 |
| Validate an existing `SessionPlan` | 22.2 | 44.0–44.8 |

HBM constructs 121 allocation declarations and scans eight subsets inside layout;
each offload scheme constructs 226 declarations and scans ten subsets. Two more
full scans occur outside layout: the geometry reservation calculation and
`SessionPlan.__post_init__`. The latter also checks all allocation owners.
The separate 200-call cProfile run confirms these counts and records every
function's callers, self time and cumulative time in `result.json`; its inflated
times are not substituted for the unprofiled measurements. Isolated timings
overlap and must not be added as disjoint costs.

The current planner alone takes roughly 1 ms for offload in this CPU diagnostic.
That is the same order as the observed 1.12–1.26 ms increase in offload revisit
admission. The experiment does not time P0 planning or the complete serving
request, and does not assign the whole admission difference to this function.
`create_planned_session` has additional declaration/geometry checks before real
allocation, but only misses call it. This diagnostic does not create a session;
those checks cannot explain revisit-hit admission time.

## Cleanup: concrete added Python work

The two fresh `PrefixSessionPool.audit()` calls remain on either side of
truncate. P0 passed `backend.session_bytes` directly. The current runner passes
`TokenRuntime.session_usage`, adding a runtime session lookup and failure/closed
check, an adapter call, and creation/validation of a `ResourceUsage` per retained
session per audit. The footprint returned by `session_bytes` is still fresh.
The quota methods still execute; moving them into the adapter did not eliminate
their session checks.

The shared audit route now goes through `runtime.shared_usage`, constructs a
`CacheFootprint` and `ResourceUsage`, converts back to a mapping, then constructs
the pool's checked `CacheFootprint`. P0 converted the backend's fresh mapping
once. The extra wrappers are paid for each of the two audits.

Session checks now route through `ResourceLifecycle.check_session` with a
`SessionRegistration` type/identity/generation check. Truncate adds the
`resources.mutation`/`ResourceLifecycle.operation` context, owner and registration
checks, a nonblocking lock, active-kind bookkeeping and finally cleanup. Runtime
cleanup also calls the adapter, passes the owner, and checks that committed
history length still matches after synchronization. Observer callbacks and their
small result dictionary replace local timestamp variables.

At an offload revisit with 16 retained users, the two pool audits create 32 new
session `ResourceUsage` objects and perform 32 runtime-state checks, plus two
shared wrappers. HBM retains one user, so this part runs twice. First-visit pool
sizes increase with admission. These are executed-count differences, not measured
microsecond attributions.

AST source comparison found the resident/offload `stats` and `truncate` bodies,
and shared `storage_tensors`/`shared_bytes` bodies, unchanged from saved P0.
The functions they call now include the lifecycle checks described above.
The old truncate synchronization and the post-truncate synchronization remain;
the new mutation context itself adds no device synchronization. Output validation
moved before the `executed` timestamp, so that small work moved from cleanup into
extend. It does not explain an increase in cleanup.

No fresh storage audit, allocator check, synchronization or validation was
disabled. No runtime-source change or GPU measurement was made for this review.
