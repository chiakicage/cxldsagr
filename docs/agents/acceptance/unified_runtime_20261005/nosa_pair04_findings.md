# Independent audit of the fourth NOSA P0/current pair

Both new benchmark runs passed the complete saved-evidence auditor: original source/native identity, independent numerical receipt and its signed output artifacts, workload, warmup/graph contracts and per-request cache accounting. Each new run has 128 rows, and every previously defined identity/classification/quota/cache-charge/H2D field matches the original P0 reference exactly. The pair and original P0 share the full workload/placement/precision/timing contract. Phase sums equal request wall latency. No new GPU model execution or failure injection was performed by this audit.

The pair driver and both benchmark subprocesses exited 0. The independent CPU audit also exited 0. The first audit script treated every monitor sample as inside a job and stopped on an empty transition sample; its monitor logic was corrected to retain empty samples outside jobs explicitly. No benchmark was retried or removed.

All 124 GPU observations returned zero. Joining each sample to its own UTC job interval gives 60 matching P0 process rows plus one empty startup sample, and 61 matching current process rows. Two additional samples are empty: one between the jobs and one after final completion. There are no unexpected PID/GPU rows. Sample spacing has median 5.103 s and maximum 8.260 s. These process observations do not monitor CPU activity, device clocks or utilization, and cannot exclude short activity between samples.

| Metric | Original three-run median difference | New adjacent pair difference |
|---|---:|---:|
| Complete trace | +406.132 ms (+0.19664%) | +388.599 ms (+0.18777%) |
| HBM first candidate extend | +0.110724 ms | +0.109374 ms |
| Dense revisit request | +0.194520 ms | +0.745889 ms |
| Sync sparse revisit request | +0.447442 ms | +0.307955 ms |
| Async sparse revisit request | −0.019303 ms | −0.437987 ms |

The new P0 and current trace totals are 206951.116 and 207339.714 ms. They sit respectively about 412.685 and 395.152 ms above their original three-run medians. The paired difference is similar to the earlier aggregate difference, while the separate baseline/current drift demonstrates why these observations do not isolate a causal whole-trace effect. The original three-run ranges remain 223.016 and 409.043 ms; all original runs are retained.

Cleanup increases in all eight paired method/visit means by 28.648–129.511 µs, with positive differences for 126 of 128 matched requests. The original three-run comparison had positive cleanup medians for all 128 requests and all eight groups. This is a repeated residual overhead observation and must remain visible in the publication. Dense revisit request latency and HBM first-candidate extend remain higher in the new pair; the other phase values and per-request differences are preserved in `nosa_pair04_independent_review.json`.

The separate 16-session mock diagnostic measured +38.522 µs across two pool audits and +40.205 µs in cleanup, associated with fresh runtime checks and usage-object construction/validation. This demonstrates added generic contract work in that fixture. It does not model production storage enumeration, allocator/lifecycle audit, mutation lease, or CUDA synchronization, and it does not explain every production cleanup difference. The one-session fixture also has a different scale from the 16-session fixture. Preserve required validations and report the measured overhead; do not label the runtime cost zero.

No concrete output/accounting/contract or sampled timing-integrity blocker was found. Phase timestamps alone cannot prove absence of every unexpected device synchronization. Existing source review identified no added device synchronization; the already planned main profile remains responsible for checking execution, synchronization and API reasonableness. This audit neither schedules another measurement campaign nor treats the incomplete profile/publication work as complete.

The JSON preserves full auditor returns, input hashes, exact source identities, all phase means, original three-run ranges, all 128 paired request differences, and the UTC monitor join. The audit script is retained as `nosa_pair04_audit.py.txt`. CPU work is released for the three DeepSeek benchmarks.
