# Independent exact recall

Each labeled run completed 31 independent reset samples for each of three states and three layers, after the run-specific warmup count listed below. The matched independent acceptance covers 18 checks. Each input is captured int32 `[128,2048]` exact top-k IDs; H=65,536, A=128, chunk=1,024, P=NH=65,664, BF16 records of width 576 (1,152 B).

- Production append02: bench `cache_recall_bench_20261006_04`, check `cache_recall_check_20261006_04`, identity `56f9478139ba8f581e29b93201fa2e867dfe9e5a70f4f9d6adab7219adb0ea35`; 2 warmups per case.

Hardware: NVIDIA H200, GPU UUID `80ff95c3-176e-fd8a-728f-9c5577c4a779`; CPU affinity [24, 25, 26, 27, 28, 29, 30, 31]. PyTorch 2.12.1+cu130, CUDA 13.0, TVM FFI 0.1.13.post3.

![Per-layer exact recall medians and interquartile ranges](latency.svg)

Points are medians; error bars span Q1–Q3, calculated by linear interpolation over all 31 retained samples (`statistics.quantiles(method="inclusive")`). IQR is Q3−Q1, not a confidence interval. Separate labeled runs are not paired samples. Full values and recall counters are in [latency.csv](latency.csv).

| Run label | State | Layer | Wall median [Q1, Q3] ms | Enqueue median [Q1, Q3] ms |
| --- | --- | ---: | ---: | ---: |
| Production append02 | certified_resident | 0 | 0.069001 [0.068158, 0.069919] | 0.059305 [0.058700, 0.060247] |
| Production append02 | certified_resident | 1 | 0.065852 [0.065348, 0.066544] | 0.058271 [0.057668, 0.058970] |
| Production append02 | certified_resident | 2 | 0.066244 [0.065547, 0.067005] | 0.059292 [0.058634, 0.060096] |
| Production append02 | restored_all_hit | 0 | 0.164676 [0.161930, 0.167761] | 0.158148 [0.155411, 0.160989] |
| Production append02 | restored_all_hit | 1 | 0.164513 [0.162346, 0.168662] | 0.157939 [0.155702, 0.162014] |
| Production append02 | restored_all_hit | 2 | 0.165667 [0.162248, 0.168518] | 0.158882 [0.155707, 0.161766] |
| Production append02 | cold_sparse_miss | 0 | 0.352424 [0.350275, 0.355171] | 0.157149 [0.154935, 0.161033] |
| Production append02 | cold_sparse_miss | 1 | 0.212160 [0.207861, 0.214236] | 0.159685 [0.156224, 0.164527] |
| Production append02 | cold_sparse_miss | 2 | 0.330120 [0.328387, 0.336151] | 0.157537 [0.155301, 0.164031] |

Certified resident rebuilds history by production append and retains its CPU proof. Restored all-hit restores a production snapshot; GPU mappings remain resident but the CPU proof is uncertified. Cold sparse miss additionally releases history GPU IDs. Every sample resets independently.

Timing includes production `_ensure_from_topk` and device synchronization. Prefix construction, reset/snapshot, begin/append, host drain, counter reset/read, commit and numerical checks are outside the timer. The report does not add separately fenced layers into a grouped request latency. Wall minus enqueue is not pure GPU compute, and no IO time is subtracted. Transfer bytes come from successful-record counters, not a hardware transfer profile.

The entry consumes captured selections without executing or changing indexer arithmetic/top-k. It covers one session per layer with P=H+A: allocation ties concern free slots only, with no live eviction. It is not a full cache/indexer or model measurement and has no model gate.

[Provenance](provenance.json) binds each check, benchmark, input, measured source archive, native identity and observer. [Report helper hashes](report_helper_sources.json) bind separately archived reporting sources. Observer samples are discrete and do not establish continuous GPU isolation or exclude unrelated CPU work.

The check observer retained 6 clean discrete samples. The benchmark observer retained 5 samples, including one terminal exit race for PID 4189485 at 2026-10-06 01:31:36.483134 UTC. Two preceding observations identified the same child and start_ticks 68861815; the GPU UUID and memory record matched the immediately preceding observation, and the next sample had no process. The maximum benchmark sample gap was 2.272866 s. No foreign GPU process or monitor error was observed. [Independent reconciliation](observer_reconciliation.json) retains the original nonempty exit-race entry; identity at the race sample is inferred because its /proc record was already absent. This is not a zero-race or continuous-isolation claim.
