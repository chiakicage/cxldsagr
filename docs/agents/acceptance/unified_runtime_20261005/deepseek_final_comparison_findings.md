# DeepSeek final three-version comparison: independent findings

The complete comparison command exited 0 and audited all seven original inputs: historical published C10, three frozen P0 benchmarks, and three final benchmarks. An independent pass reopened all 896 original measurement rows, checked exact receipt bytes/identities, compared all final check/benchmark source manifests, reconstructed per-run phase means and matched-request differences, and joined every final monitor sample to its own run interval and process tree. This is saved-evidence acceptance, not another model execution or injected-failure test.

All six P0/final runs have identical per-request input/prefix/candidate/workload identities, visit/hit classifications, fixed quotas, reservation/cache charges and complete candidate cache diagnostics, including H2D/D2H and layer records. Phase sums equal request wall latency. The original P0 mean/range/MAD noise records agree with independent arithmetic. Allocator/device occupancy is observed separately; cache equality does not establish a physical device-memory peak.

| Selected metric | P0 median of run means (ms) | Final median (ms) | Change (ms) | P0 range (ms) | Final range (ms) |
|---|---:|---:|---:|---:|---:|
| HBM first candidate extend | 10.600519 | 10.718264 | +0.117745 | 0.016074 | 0.031974 |
| ECHO revisit request | 24.128328 | 24.530059 | +0.401731 | 0.103185 | 0.080999 |
| Serial sparse revisit request | 17.762562 | 18.083835 | +0.321273 | 0.271310 | 0.044168 |
| Dense-prefetch revisit request | 32.123054 | 32.633588 | +0.510534 | 0.372216 | 0.600213 |

The first three increases exceed both groups' repetition ranges, and all four have nonoverlapping P0/final run ranges. Dense-prefetch's median increase does not exceed the final range; all raw runs and request values remain visible. ECHO has higher matched medians for 16/16 revisit requests, serial sparse for 15/16, and dense-prefetch for 15/16. Every final sample exceeds every P0 sample for 7, 12 and 8 of those request IDs, respectively. These are observed residual latency increases, with three repeats and no claim of statistical confidence or causality.

Admission increases in all eight method/visit means by 0.189032–0.490996 ms; all eight P0/final run ranges are nonoverlapping, and six median increases exceed both ranges. Cleanup also increases in all eight means by 0.036084–0.194572 ms. All cleanup run ranges are nonoverlapping; seven median increases exceed both ranges. Cleanup matched medians increase for 128/128 requests, with every final sample above every P0 sample for 124/128. The full JSON retains every phase, range, MAD and matched-request value. Required runtime/resource validation remains in place; this audit proposes no optimization or further measurement campaign.

Complete-trace medians are 179493.801 ms for P0 and 179330.893 ms for final, a change of −162.908 ms (−0.09076%). Their ranges are 110.319 and 249.918 ms and overlap. Whole-trace aggregation includes much larger history construction phases; this aggregate decrease cannot be used to claim zero regression in candidate or revisit work. First-visit request means have mixed signs and remain subject to the observed variation.

The final benchmarks use full source snapshot `1319a540b7f78da9c363dd09c0f1568b450df3df5e4d3e8b60155a58c348b3dc`. Reopening each saved manifest shows that only `experiments/deepseek_v32_motivation/src/profile.py` differs from the final check snapshot `1fc1cbaf7a97579f7094879938330a9933a82109909d2701a177e1653f0dc9f8`. That profile-only file is excluded from the unchanged validated execution identity. Final receipt file SHA is `a25cc5d08891fd6811db232de0b862831b46702c1b2f3124b2921b611574c1b3`; its distinct canonical self-digest is `e7ade48c00d4085b3b410bbb26d27cecd1791fb33bdf795c45065fe1d9910332`.

The historical published C10 source is `11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f`; P0 uses `d31a0ff6bde97acab24bb2f2c648e9d91ae46ba18f6c074b5f572cf49aaaab7a`. Historical published values keep their original placement and combined validation/timing context and are descriptive controls only.

Each final run has 65 recorded monitor samples: 63 owned compute-process rows and two empty samples. All 195 queries return zero; there are no foreign or wrongly assigned process/GPU rows. The independently checked UTC windows and PIDs are retained in the JSON. Median spacing is about 5.100 s; maximum gaps are 28.285, 28.460 and 29.235 s. The long gaps must not be hidden by the nominal five-second polling target. These discrete observations cannot exclude short unsampled activity or characterize CPU contention, clocks or utilization.

No concrete numerical/accounting/contract or sampled process-integrity blocker was found. The separately planned main profile still needs to inspect execution, synchronization and API reasonableness. This comparison does not complete that profile or final publication. Report assets are prepared without replacing READMEs or retiring old inputs. Original three-run groups remain preserved.
