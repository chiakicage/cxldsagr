# Cache management evidence: `cache_unified_20261005_02`

Static allocations and complete motivation traces have separate provenance. No static maximum below is a measured physical-capacity limit.

## DeepSeek static P/NH plans

| Plan | P | NH | Users | Extra headroom GiB | HBM reservation GiB | DRAM reservation GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cache_deepseek_nh_20261005_02` | 32,768 | 29,818,880 | 455 | 0 | 45.583292 | 320.003471 |
| `cache_deepseek_p_base_20261005_02` | 10,296,447 | 1,048,576 | 16 | 0 | 116.486652 | 20.000122 |
| `cache_deepseek_p_base_headroom_20261005_02` | 8,983,043 | 1,048,576 | 16 | 14.5 | 116.486669 | 20.000122 |
| `cache_deepseek_p_nhmax_20261005_02` | 6,451,364 | 29,818,880 | 455 | 0 | 116.486671 | 320.003471 |
| `cache_deepseek_p_nhmax_headroom_20261005_02` | 5,137,960 | 29,818,880 | 455 | 14.5 | 116.486672 | 320.003471 |

## NOSA static allocation declarations

Graph allocations are excluded here. Shared graph storage and private reservation are already included in the matched request observations; do not add them again.

| Scheme | Quota users | Shared HBM bound GiB | One-session HBM bound GiB | One-session host payload GiB | One-session host reservation GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| hbm | 1 | 0.000980 | 2.120296 | 0.000000 | 0.000000 |
| serial_sparse | 256 | 2.007358 | 0.167775 | 2.000000 | 4.000000 |
| dense_prefetch | 256 | 2.007329 | 0.167775 | 2.000000 | 4.000000 |
| overlap | 256 | 2.007358 | 0.167775 | 2.000000 | 4.000000 |

## Complete-request memory observations

Allocator peaks include model weights and ordinary activations. Device-used values are maxima of after-request samples, not continuous process peaks. Cache charges and reservations are reported independently in memory.csv; charges include graph private reservations and are not a tensor-payload sum. These traces do not fill the offload NH quota and do not validate any offline maximum above.

Memory after model loading is recorded separately in summary.json. It is not a tensor-only weight inventory. Ordinary activation peaks were not measured independently and are null; subtracting cache reservations from process peaks does not establish them.

| Model | Scheme | Retained users | Allocated peak GiB | Reserved peak GiB | Max sampled device used GiB | Cache HBM GiB | Cache DRAM GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nosa | hbm | 1 | 18.839923 | 24.078125 | 24.831116 | 8.602579 | 0.000000 |
| nosa | dense_prefetch | 16 | 19.989842 | 25.330078 | 26.147522 | 9.727347 | 32.000000 |
| nosa | serial_sparse | 16 | 19.989294 | 25.318359 | 26.137756 | 9.727373 | 32.000000 |
| nosa | overlap | 16 | 19.989294 | 25.318359 | 26.137756 | 9.727373 | 32.000000 |
| deepseek | hbm | 1 | 14.517269 | 23.634766 | 24.376038 | 6.624029 | 0.000000 |
| deepseek | echo | 16 | 16.357162 | 24.296875 | 25.667053 | 8.462240 | 320.001038 |
| deepseek | serial_sparse | 16 | 16.358688 | 24.296875 | 25.667053 | 8.462240 | 320.001038 |
| deepseek | dense_prefetch | 16 | 16.358688 | 24.304688 | 25.674866 | 8.462240 | 320.001038 |

nosa observation source: `refactor_final_nosa_bench_20261005_01`.

deepseek observation source: `refactor_final_deepseek_bench_20261005_01`.

Both traces use fixed token quotas, not equal HBM/DRAM byte budgets. NOSA's NH is a lazy per-session host quota; DeepSeek allocates a global arena. The workloads also differ: full 32-layer NOSA hidden outputs versus a ten-block DeepSeek source-input replay with a last-token LM head. The table does not rank cross-model efficiency.
