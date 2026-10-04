# Dense-prefetch transport bound at fixed P/H

This is a conditional hardware bound, not a new performance measurement or an
excuse to omit further optimization. It constrains literal equality between
E2E MFU and compute-only matrix API MFU for the fixed BF16 dense baseline.

All sixteen dense-prefetch revisits in accepted run
`motivation_c9_triton_20261004_u16_r2_01` record zero resident history records,
655,360 fetched records and 754,974,720 host-to-device bytes. This is ten layers
x H65,536 x 1,152 B/token. The independently audited useful precision-weighted
ideal candidate time is 2.2463935548694125 ms. The matching sampled matrix API
MFU is 30.7097%, with API active time approximately 7.315 ms.

The current device observation reports PCIe generation 5, sixteen lanes for
both current and maximum link configurations. The CPU reports Xeon Platinum
8558P; the topology report lists GPU-to-GPU NVLink and PCIe/NUMA host paths.
Using the optimistic line-coding maximum, before packet/protocol overhead:

```text
B = 32e9 transfers/s * 16 lanes * (128/130) / 8 = 63,015,384,615.38 B/s
T_copy >= 754,974,720 / B = 11.980800 ms
MFU_e2e <= 2.2463935548694125 / 11.980800 = 18.749946%
```

This bound assumes that each recorded byte crosses that one PCIe link inside
the request, with the same BF16 transport, H/P capacities, single-GPU workload
and useful FLOPs/precision. It excludes compressed transport or a separate
coherent host/NVLink path. Protocol costs and remaining work can lower the
ceiling; perfect compute/copy overlap cannot make a request shorter than its
required transport. It therefore does not establish that 18.75% is attainable.

The C9 sampled gather kernel-duration sum is 14.720040 ms, equivalent to
51.2889 GB/s for those bytes. That is an observed activity sum, separate from
this theoretical floor and the formal request mean; their difference is not a
measurement of removable overhead. The hardware observation is contemporaneous
with C10 engineering work, not a retroactive continuous link-state record for
C9. Full future trajectories must confirm their own bytes and hardware state.

Evidence is under `/tmp/deepseek_dense_transport_bound_20261004/`: `pcie.csv`,
`topology.txt`, `cpu.txt`, and `bound.json`. The JSON binds all sixteen formal
rows to the source file hash, records formulas and assumptions, and hashes the
hardware observation files. Independent arithmetic review by graph_profile,
recorded in `/tmp/deepseek_c9_dense_transport_arithmetic.json`, agreed with the
11.980800 ms and 18.749946% values. That audit checked all sixteen source rows
and the arithmetic; it did not independently probe hardware. The original
formal/profile artifacts remain unchanged.
