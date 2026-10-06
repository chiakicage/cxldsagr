# SGLang HBM-only performance-only results: first3_hbm_perfonly_20261006_01

The user explicitly requested performance measurements despite known numerical inconsistency. These results cover the completed 32-request SGLang HBM-only timing trace; numerical acceptance has not passed. The original failed numerical diagnostic is retained with its file hash and failure counts.

Per-request HTTP serialization through complete response read and JSON decode. Output validation, logging and saving are excluded. The main server starts with fresh radix and allocator state after a separate three-request warmup server exits. Before client launch, GET /get_model_info and one POST /slow_down with forward_sleep_time=null initialize the normal tokenizer receive loop, watchdog and signal handlers, retaining the default no-delay state. This control-plane setup is outside request timing and performs no inference or cache operation. JIT disk caches persist; model first-use costs remain in measured requests, while tokenizer-loop startup does not. JIT artifact inventories are captured after each server closes, outside request timing; files present do not establish which artifacts executed.

| Visits | Requests | Mean ms | Median ms | p95 ms | Sum ms | History hits | Cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first_visit | 16 | 1238.838 | 1216.973 | 1311.721 | 19821.409 | 0 | 0 |
| revisit | 16 | 1222.878 | 1222.715 | 1229.755 | 19566.044 | 0 | 0 |
| all | 32 | 1230.858 | 1220.167 | 1233.626 | 39387.453 | 0 | 0 |

Linear interpolation at (n-1)*0.95 over sorted observed durations.

- first_visit: 0 tokens: 16 requests; 0.807208 timed requests/s.
- revisit: 0 tokens: 16 requests; 0.817743 timed requests/s.
- all: 0 tokens: 32 requests; 0.812441 timed requests/s.

Main-phase maximum observed owned compute-process memory: 13532.000 MiB, across 97 samples; 0 samples have incomplete values.

## Scope and limits

- The workload uses original FP8 checkpoint layers 0, 1 and 2 sequentially, including embedding, final norm and one final-token LM head. It uses neither AWQ nor C10 replay and does not measure the complete 61-layer model.
- Each request computes all 128 candidate hidden states and samples one token; no subsequent decode forward is executed. Bench requests do not return or save full hidden states or log probabilities.
- The official indexer retains Hadamard after RoPE. The retained resident-repeat and offload comparisons show known numerical inconsistency. This report does not establish numerical acceptance or equivalence to the project's no-Hadamard implementation.
- Official radix caching retains complete H+A prompts in HBM, including candidates. There is no host KV pool. It does not implement the project's candidate-discard and session-LRU lifecycle.
- This report measures the resident_reference case as SGLang HBM-only. Its 131072-token device pool differs from ECHO's 65664 device slots and 16777216 host-token quota; this is not an equal-capacity or equal-byte-budget performance comparison. The earlier resident check runs remain numerical diagnostics and are not used as latency samples.
- The resident device and scheduler capacities are both 131072 tokens, with 64 additional allocator padding slots and no host KV pool. Token capacity does not specify total HBM bytes. Cache hits and misses are the observed outcomes of the native radix cache under this capacity.
- GPU memory observations sum identified owned compute-process memory reported by nvidia-smi at discrete samples over each server phase, including initialization. They are not PyTorch allocated/reserved bytes, whole-device used memory, or a continuous peak, and cannot certify an HBM byte budget.
- Reported percentiles describe these 16 first visits and 16 revisits only. Timed request throughput divides request count by summed HTTP durations, excluding inter-request validation/logging, warmup, startup and shutdown.

## Evidence

Numerical diagnostic: `/mnt/ssd-wlcb/chenkaiqi/cxldsagr/3rdparty/ECHO/reproduction/cxldsagr/output/acceptance/diagnostic_first3_triplet_20261006_01.json`. Numerical acceptance: **not passed**.

- resident_repeat_vs_resident_reference: 16/32 comparisons failed.
- offload_vs_resident_reference: 19/32 comparisons failed.

- Official ECHO commit: `bc1b75c1000010d0ac6f032ebaac283255c050b1`
- Workload SHA256: `7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`
- Model export manifest SHA256: `41a81cf78cd3c5dfd9d32800473aac0b71f485c2148092eb57f2d0d26400fc17`
- Native build receipt SHA256: `757dc7b8424c468ea09239578364951b1dcb22418251d78545c5cfd2af4c7b45`
- Execution binding SHA256: `1984693f2f543dae3e2f806e5e3e1aed6c8306756e196bf328865277af1d8968`
- GPU UUID: `5c6220cc-a8ef-ca97-91bb-cb61a278faf8`
- Recorded environment SHA256: `520180ffc1cf0b0bd7c0eb00786800093e65290c372af314676b76b614d24099`
- Report signature: `539fd4f8ec6ce1edc36f57d0a6c181d97e469755cb5921e1dd127ab60f011181`
- Retained failed numerical diagnostic SHA256: `412d66af8d5aea0d3ff16ae012549a6517a976cebcca09249fe7004ebf2190cf`

The JSON report preserves the exact execution binding, public environment values, model and runtime/native identities, failed numerical diagnostic, and hashes of source evidence. `summary.csv` contains grouped statistics; `requests.csv` contains the 32 observed requests.

Source run: `/mnt/ssd-wlcb/chenkaiqi/cxldsagr/3rdparty/ECHO/reproduction/cxldsagr/output/data/first3_hbm_perfonly_20261006_01`.
