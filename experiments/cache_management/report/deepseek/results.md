# Cache management evidence: `cache_unified_dma_20261006_01`

Static allocations and complete motivation traces have separate provenance. No static maximum below is a measured physical-capacity limit.

## DeepSeek static P/NH plans

| Plan | P | NH | Users | Extra headroom GiB | HBM reservation GiB | DRAM reservation GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cache_deepseek_nh_20261006_01` | 32,768 | 29,818,880 | 455 | 0 | 45.583292 | 320.003471 |
| `cache_deepseek_p_base_20261006_01` | 10,296,447 | 1,048,576 | 16 | 0 | 116.486652 | 20.000122 |
| `cache_deepseek_p_base_headroom_20261006_01` | 8,983,043 | 1,048,576 | 16 | 14.5 | 116.486669 | 20.000122 |
| `cache_deepseek_p_nhmax_20261006_01` | 6,451,364 | 29,818,880 | 455 | 0 | 116.486671 | 320.003471 |
| `cache_deepseek_p_nhmax_headroom_20261006_01` | 5,137,960 | 29,818,880 | 455 | 14.5 | 116.486672 | 320.003471 |

## Complete-request memory observations

Allocator peaks include model weights and ordinary activations. Device-used values are maxima of after-request samples, not continuous process peaks. Cache charges and reservations are reported independently in memory.csv; charges include graph private reservations and are not a tensor-payload sum. These traces do not fill the offload NH quota and do not validate any offline maximum above.

Memory after model loading is recorded separately in summary.json. It is not a tensor-only weight inventory. Ordinary activation peaks were not measured independently and are null; subtracting cache reservations from process peaks does not establish them.

| Model | Scheme | Retained users | Allocated peak GiB | Reserved peak GiB | Max sampled device used GiB | Cache HBM GiB | Cache DRAM GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek | hbm | 1 | 14.517354 | 23.634766 | 24.376038 | 6.624072 | 0.000000 |
| deepseek | echo | 16 | 16.357248 | 24.296875 | 25.665100 | 8.462284 | 320.001038 |
| deepseek | serial_sparse | 16 | 16.358774 | 24.296875 | 25.665100 | 8.462284 | 320.001038 |
| deepseek | dense_prefetch | 16 | 16.358774 | 24.294922 | 25.665100 | 8.462284 | 320.001038 |

deepseek observation source: `deepseek_dma_c10_bench_20261006_01`.

The trace uses fixed token quotas, not HBM/DRAM byte budgets. DeepSeek allocates a global host arena. The ten-block source-input replay includes a last-token LM head; it does not represent the full 61-layer model.
