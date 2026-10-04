# C5 accepted component checkpoint

Run ID: `deepseek_topk_wrapper_c5_20261004_01`. Engineering probe only; this does
not replace a full-request motivation experiment. GPU 2 was exclusive during
correctness and timing, which ran sequentially and both exited with status 0.
Device: NVIDIA H200 / SM90, PyTorch 2.12.1+cu130, CUDA 13.0.

The candidate calls the same registered official FlashInfer radix_topk operation
with sorted=True, deterministic=True, SMALL tie mode and the same row-state cache.
It retains the original int32 IDs and applies one lazy in-place nonfinite mask.
Official selection and both official ordering kernels remain unchanged. The mask
writes only -1 indices for exponent-all-ones values and preserves all value bits.

Correctness: 18 selection tests passed, including every K from 1 through 2048,
strict public-wrapper bitwise comparison, large tie overflow, all-invalid rows,
finite extrema, signed zero, noncontiguous input, empty rows, a nondefault stream,
and CUDA graph capture/replay. CPU ABI checks passed before GPU execution. Probe
strictly compared all six input patterns before timing each case.

Timing is complete API wall time including per-call GPU synchronization. Each
method used 10 warmups and 40 measured calls in alternating order. Profiles used
10 further calls and report summed actual CUDA kernel activity, not wall latency.
The measured wrapper reduces 11 kernels to 4: three unchanged official kernels
plus one mask. The int32-only intermediate used 9 kernels and was slower than the
fused candidate in all six measured cases.

| Q / N | Input | Public wall ms | C5 wall ms | Public GPU us | C5 GPU us |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024 / 1024 | causal | 0.113000 | 0.095148 | 87.8339 | 66.7645 |
| 1024 / 32768 | causal | 0.361101 | 0.329610 | 329.9266 | 295.6331 |
| 1024 / 65536 | causal | 0.466106 | 0.434206 | 433.7817 | 399.2956 |
| 128 / 65664 | causal | 0.103387 | 0.090159 | 69.8049 | 57.7422 |
| 128 / 65664 | ties | 0.127016 | 0.115328 | 94.6498 | 82.4509 |
| 128 / 65664 | invalid | 0.159191 | 0.147401 | 126.8076 | 114.6004 |

For Q1024/K2048, peak allocated delta fell from 50,331,648 to 16,777,216 bytes;
for Q128/K2048, from 6,291,456 to 2,097,152 bytes. These are warmed component
PyTorch allocated deltas, including retained output, not reserved memory or
process/device peak capacity evidence. No persistent cache storage was added.

The earlier C3 cold profile attributed 1920 official launches / 185.431568 ms and
5120 wrapper launches / 22.862042 ms across 640 calls. These are GPU activity
observations from that older run, not a predicted or measured C5 full-model gain.

Artifacts: `/tmp/deepseek_c5_topk_probe.py`,
`/tmp/deepseek_topk_wrapper_c5_20261004_01.json`, and adjacent per-case Chrome traces.
All are engineering probes, kept outside experiment deliverables.

Measured source identities:

- `operators/deepseek_v32/indexer/selection.py`: `0e68ecb4e16e40e9dda1541693ce3e00e41ccd4a05ea2e708d04dd2545a4dcd8`
- `operators/deepseek_v32/indexer/tests/test_selection.py`: `a3e915cf675d4127c859e2e4365c63c071c1755180a8e50a36ba1aaaad1dae37`

C5 is frozen for root integration. Full-model correctness/performance acceptance
and affected result replacement remain outstanding at the parent task scope.
