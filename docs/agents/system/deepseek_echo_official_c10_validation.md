# Official ECHO validation after C10

The three official checkpoint tests passed on frozen C10 source, with no skips
or failures. This refreshes numerical coverage for the shared direct-rotary
change. It is not a new official serving performance result. Subsequent top-k
and cleanup candidates remain separate changes and are not validated by this run.

Run: `/tmp/deepseek_official_c10_validation_20261004_01`.
Driver: `/tmp/deepseek_official_c10_gate_20261004/run.py`, adapted from the
previous gate only to preserve the configured repository PTXAS path while
executing from the frozen C10 working directory. Exec session 61621 and child
PID 2086892 completed with exit 0. Pytest reported 3 passed in 331.93 seconds,
with one dependency deprecation warning and empty stderr.

The source root was `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0`.
Physical GPU 0 mapped to `cuda:0`; PyTorch reported H200/SM90. Both PTXAS paths
were pinned before import, OMP/MKL used eight threads, and production
`torch.compile` was prohibited by the gate. Source/compiler identities were
unchanged before and after. The actual quantizer specializations had loaded
handles and matching PTX/CUBIN artifacts.

- Eager and graph H64K tests each built two independent histories and served
  three candidates at P=65536, NH=16777216, chunk1024 and A128. Each checked
  1,310 attention calls and 2,650,296,320 selected-record occurrences. Selected
  KV records and same-query/same-ordered-selection attention were byte-exact.
  The graph test had complete projection/finish replay coverage, zero eager
  fallback and 5,603,590,144 B observed graph-private reservation.
- The H2304 callback test preserved the ordinary consumer, HBM/ECHO schemes,
  two users, A128/A121/changed-A128, nondefault streams and output ownership.
  All ten phases passed the unchanged numerical policy. Maximum hidden/logits
  relative L2 was 0.001477500726888615 / 0.0017367150660998225; minimum elementwise
  pass fractions were 0.9999901907784599 / 1.0. This does not assert byte equality
  across independent official selector executions.

Evidence SHA-256:

| File | SHA-256 |
| --- | --- |
| `source_before.json` and `source_after.json` | `164841417c9555ffdb69bac4fccc172e93f8b96a85bb9602ee615cc08e046e70` |
| `runtime.json` | `01e38aceb0cfa0baafda129e123594f0aee2981d007b6d8ae685214510defe7a` |
| `numerical_summaries.json` | `aba506d734971427fae6c3c7deed6f036ee7c3edd4517f1168571c56756fd12d` |
| `test_outcomes.json` | `d102cef8a61af16e95f0bdd2b52fde2781e203bd1d02ff836dbefdec77de27dc` |

The driver records its exact command, required node IDs and artifact hashes in
`result.json`. Full official formal/repeat measurements and coordinated public
report replacement remain outstanding.
