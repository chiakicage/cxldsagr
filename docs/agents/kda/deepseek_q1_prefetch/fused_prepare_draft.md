# Draft: stateless preparation fusion

The authoritative current formal result is
`deepseek_h64k_a1_isolated_20261008_01`. The derived diagnosis is
`experiments/deepseek_v32_echo_official/output/data/decode_gap_isolated_20261008_01/`;
its `kernel_details.csv` SHA256 is
`22a3521efad8607c77134f82dfd97d3c8e2d5d8e99d7a2b91b4560f7b8210873`.
The measured local ECHO GPU window is 1.426563 ms, including 47.492 us idle.
CPU reread gives these L0-L2 activity sums: packing 15.808 us, identity block
table 2.496 us, official token-table/staging preparation 5.184 us, scheduler
5.088 us, official fused kernel 94.272 us, clean 7.008 us and promotion 22.016 us.
The current exact mean/EMA hint nodes total 14.592 us. These are intrusive node
durations, not clean API latency or projected savings.

The official fused core runs 10.656/34.144/49.472 us locally versus
8.448/8.800/8.416 us in SGLang. The local cold zero-threshold profile records
64 completed prefetches (73,728 bytes) per layer; reservation attempts are not
successful copies. SGLang's captured threshold and initial residency are unknown.
This difference does not justify
replacing the official algorithm or reporting a pure compute regression.

Candidate ranking:

1. Fuse per-call page64 packing, identity table and staging preparation. It
   removes two launches per layer without persistent cache state, changing the
   smallest independently measurable group of repeated adaptation work.
2. Reuse fixed-graph bounds/tables/schedule through an explicitly bound owner.
   Defer until the first candidate is measured; naked-pointer memoization is
   invalid, and setup cost/lifetime must be accounted separately.
3. Incrementally retain packed history. Potentially removes the 8.66 MB copy,
   but changes per-session storage, rollback/restore and capacity planning.
   Defer rather than silently changing the current ephemeral budget.

Main risks: vector copy alignment/padding, cleanup registration before failure,
one-use prepared tokens, no hidden state across changed-input graph replay, and
extra register pressure erasing launch savings. Use aligned vector copies only
under their actual pointer condition and retain byte-equivalent scalar dispatch.
No score arithmetic or official metadata algorithm changes are necessary.

First implement a private native preparation adapter, then a check/bench/profile
driver reusing the saved real-input and promotion fixture helpers. Check raw-byte
preparation independently before running full official calls. Timing must include
the full API and both variants' cleanup, with balanced AB/BA order and cold reset
before every replay. Source snapshots and native assembly/runtime identities
must be retained outside timed regions.

The accepted component experiment `q1_fused_prepare_bench_20261008_02` now
measures 0.976-2.800 us median paired savings across 15 layer/state groups;
both AB/BA strata improve in every group. Cold L0/L1/L2 paired deltas are
-0.976/-1.744/-2.224 us, summing to -4.944 us across separate calls. This modest
component result supports the private full-model gate; it does not establish
an end-to-end speedup. Independent profile/NCU and production integration remain
conditional on that gate.
