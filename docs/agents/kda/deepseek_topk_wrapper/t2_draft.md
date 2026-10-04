# T2 evidence and risks

The source proof is the same official dispatch split verified independently for
T1, but T2 keeps `sorted_output=True`. The value vector is therefore already
bit-exactly sorted. Only finite equal-key groups can differ in their index
sequence. A finite equal-key group is sorted iff it has no adjacent descending
ID pair; checking this is sufficient even when many ties exist.

For a row needing repair, the upper packed key is a run's first column rather
than a 32-bit floating key. Since the values are already sorted, increasing run
starts have the same group order as the official value keys. Each group's item
count remains fixed, so repairing IDs leaves every value bit at its original
column. Nonfinite entries use their own column as the group and are masked
after unpacking. For non-power-of-two output widths, padding keys are uint32 max;
when a real key could equal that maximum, the row width is already the full
power of two, so there is no padding collision to truncate.

The no-repair branch avoids the sort's instructions, but its compiled
register/shared-memory requirements can still reduce occupancy. Offline SM90
compilation uses 124 registers and 8,192 B dynamic plus 1,024 B static shared
memory for every 2,048-wide specialization checked. T1 used 122 registers and
16,384 B dynamic shared memory. T2 reduces dynamic shared memory but does not
reduce register pressure; neither comparison establishes a performance benefit.

Root requested three real final-history `[1024,2048]` FP32 values/int32 IDs from
the C10 correctness gate, with source/layer/visible-length metadata. Their
combined tensor payload is 48 MiB. These are correctness-run samples, not C9
performance measurements. Baseline deterministic+sorted IDs cannot reveal
T2's unsorted selector emission order: they measure equal-bit finite run
lengths and bound the fraction of rows that might need repair, not its actual
branch rate. T2's runtime branch frequency requires its own later evidence.

The three captures have 304, 138 and 533 rows with finite equal-bit ties out of
1,024 rows each. Their tie runs are 365 length-two runs; 147 length-two runs;
and 733 length-two plus one length-three run, respectively. All 6,291,456 values
are finite and nonzero. Layer IDs were unavailable, so these are capture IDs
0, 1 and 2, not inferred layer numbers.

The CPU proof compares all value bits and indices against the independent C5
two-pass relation for every width 1..2,048, raw/special/long-finite inputs,
packing boundaries and all real rows with original and reversed tie IDs.
The temporary candidate keeps one implementation without a neighbor-rank path
or autotuning. The later bounded GPU gate passed correctness but regressed
large/tail causal APIs, so root rejected T2 for production. Actual selector
repair frequency was not instrumented; see [the checkpoint](t2_checkpoint.md).
