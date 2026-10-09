# Retained graph node-type inventory

The formal HBM capture snapshots contain no extra empty, event-wait or
event-record nodes compared with the reduced controls that omit event nodes.
This is a CPU read of retained JSON ledgers, without new GPU work, raw trace
reads or source/native rehashing. It does not identify the reason for the idle
difference.

| Retained run / template | Kernel 0 | Memcpy 1 | Memset 2 | Empty 5 | Event wait 6 | Event record 7 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Formal CUB HBM, original | 181 | 15 | 1 | 0 | 0 | 0 | 197 |
| Formal CUB HBM, reproduced | 181 | 15 | 1 | 0 | 0 | 0 | 197 |
| Event control, no_events | 181 | 15 | 1 | 0 | 0 | 0 | 197 |
| Event control, events | 181 | 15 | 1 | 0 | 0 | 2 | 199 |
| Inspector control, inspect | 181 | 15 | 1 | 0 | 0 | 2 | 199 |
| Replay control, first5_outside | 181 | 15 | 1 | 0 | 0 | 0 | 197 |
| Replay control, first5_inside | 181 | 15 | 1 | 0 | 0 | 0 | 197 |
| Replay-timer control, single | 181 | 15 | 1 | 0 | 0 | 0 | 197 |

The other original/reproduced formal methods also match each other: ECHO has
265/12/13 nodes of types 0/1/2 (290 total), serial sparse has 211/9/1 (221), and
dense prefetch has 190/12/1 (203). All have zero type-5/6/7 entries.

Formal ledgers are the `full_graph_templates.json` files in MFU output runs
`deepseek_h64k_a1_cub_20261008_01_h65536_a1_profile` and
`deepseek_h64k_a1_cub_gap_reproduce_20261008_01`. Reduced ledgers are in each
`result.json` for `q1_event_boundary_profile_20261008_01`,
`q1_inspector_boundary_profile_20261008_01`,
`q1_replay_boundary_profile_20261008_01` and
`q1_replay_timer_profile_20261008_01`. Full paths and counts are recorded in
`/tmp/cxldsagr-checks/bounded_graph_node_inventory_20261008_01.json`.

`GraphInspector.snapshot` enumerates all graph nodes and records the node type;
it accepts 0/1/2/5/6/7 and rejects unsupported types. `FullGraphProfiler` stores
that complete `node_types` map, while `gpu_node_ids` separately filters 0/1/2.
The stored map is the final snapshot inside the captured scope. The artifacts
do not include a complete edge DAG or an independent enumeration after graph
instantiation. Equal type counts do not prove equal dependency edges, scheduling,
JIT/cache state, execution environments or a performance mechanism.
