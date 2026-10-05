# NOSA layout-cache review and CPU follow-up

The independent static review found no blocking issue in the single-entry
layout/reservation cache. The reviewed adapter source hash is
`3ef7d72a4e00cc366e2983099f9107722555b23887d723a2310bd50948e4e7bb`.
No source was edited and no GPU job was run by this review.

The key contains every actual argument passed to `session_allocation_layout`,
including its Python type, plus provider identity and resource generation. This
prevents `True` and `1` from reusing one entry despite their equality. The cached
tuple holds only immutable allocation declarations and a frozen footprint;
request history identity, resource-plan identity, actual storage and audit
results are absent. A different key replaces the single entry only after layout
and footprint construction succeed. A later `SessionPlan` validation failure
may retain that independently valid pure layout; it retains no failed request.

Every invocation still checks request limits, reconstructs geometry, computes
quota fields and constructs a new `SessionPlan`. Its history/resource identity,
owner scan and full footprint validation execute on warm hits. The live storage
and allocator audit paths are unchanged. Close currently retains the one small
pure declaration; a later resource allocation increments generation, forcing a
miss. It does not retain a session or tensor storage. Clearing that declaration
after successful close would be optional memory housekeeping.

The inspected tests cover ordinary/fixed schemes, fresh request identity,
declaration reuse, one-entry replacement, changed query geometry, every layout
option, generation changes, foreign resource plans and a typed invalid-input
failure that leaves the valid entry intact. The implementation owner reported
CPU validation complete before this follow-up; this reviewer did not rerun that
suite.

The [new CPU result](result.json) and [comparison](comparison.json) retain the
same seven batches of 100 unprofiled calls and the same recorded CPU environment
as the [original diagnostic](../nosa_plan_cpu_profile_01/findings.md). The script
initializes the new cache field on its offline backend. It also measures a cold
miss by clearing that local diagnostic field before each call. Original results
remain unchanged. Within the recorded source set, only the adapter hash differs;
all four reservation byte totals match.

| Scheme | Original repeated plan, µs | New warm hit, µs | New forced cold miss, µs |
| --- | ---: | ---: | ---: |
| HBM | 552.8 | 31.5 | 545.5 |
| Serial sparse | 1027.3 | 54.1 | 994.0 |
| Dense prefetch | 1026.7 | 53.6 | 996.3 |
| Overlap | 1033.8 | 53.8 | 1002.7 |

The separate cProfile run confirms one `SessionPlan` validation and one complete
footprint scan per warm-hit plan, with no layout call or `AllocationSpec`
construction. The complete validation remains about 44 µs for offload and 22 µs
for HBM in its isolated unprofiled measurement.

These are pure-CPU planning measurements with offline resource assumptions,
single-thread settings, no allocated sessions and no CUDA initialization. They
do not measure complete serving latency or establish the GPU/CPU cause of the
earlier request-level changes. New full-trace correctness and formal benchmarks
remain necessary. The [command and file index](evidence.json) preserve the exact
source, output and raw-profile identities. Reopening the diagnostic should use a
fresh directory because it writes its output beside the script.
