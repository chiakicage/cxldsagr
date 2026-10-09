# Initial compute-graph inspection result

Removing initial compute-graph operator inspection does not remove the long
HBM gap. `q1_compute_inspector_profile_20261008_01` keeps all-four-method
warmups, the initial profiler interval and later full-extend inspection.
Only the first compute preparation uses the original uninspected `check`
path. The independent original correctness receipt already covers that path.

| Initial preparation / replay | L0-L2 span us | Busy us | Idle us | Median gap ns |
| --- | ---: | ---: | ---: | ---: |
| Inspected reproduction / warmup | 1073.538 | 1005.186 | 68.352 | 416 |
| Inspected reproduction / measured | 1066.817 | 999.137 | 67.680 | 416 |
| Uninspected / warmup | 1078.817 | 1012.673 | 66.144 | 416 |
| Uninspected / measured | 1056.001 | 988.417 | 67.584 | 416 |

`q1_compute_inspector_audit_20261008_01/windows.json` verifies the native
197-node full replay and 192 layer nodes in all four windows, including
identical names, 17 launch fields, copy sizes and ownership. All-process
clipped activity unions equal the selected-node unions. CPU graph launch
durations in the uninspected run are 512.708/316.750 us.

The initial collection interval still has 12 begin/end capture pairs and
six graph launches. `cudaGraphGetNodes` drops from 306 to 12, and
`cudaGraphNodeGetType` from 2,301 to zero. Each remaining GetNodes follows
EndCapture and precedes Instantiate, once per graph. Thus 294 inspection
GetNodes and all NodeGetType calls were removed; normal graph construction
still queries each graph once. No compute-operator ledger or generic
prefill gap report is claimed for this control.

Independent audit
`/tmp/cxldsagr-checks/compute_inspector_independent_audit_20261008_03.json`
rechecks 1,370 execution sources, unchanged original measurement files plus
the bound wrapper, 6,434 concrete runtime/source files, and all 13 capture
settings/GPU records. The eight saved output tensors match both the check
and inspected reproduction bitwise. The audit JSON and script are copied
into the analysis run directory. Direct-launch metadata differs in run/output
paths, shell depth, `OLDPWD` and `_`; in-process execution/compiler identities
remain equal. Equal in-memory CuTe MLIR digests retain the earlier no-file
rehash boundary.

This controls the entire initial `CaptureGraphOperators`/`InstrumentOperators`
package, not an individual API, and supplies no clean-latency optimization
claim. Initial compute-bank construction is still collected here; reduced
HBM controls construct that bank with collection stopped. That remaining
boundary can be tested while preserving the complete all-four-method prelude.
