# Current DMA formal runs

All commands run from repository root with `/tmp/deepseek_mfu_env.sh`. GPU3 is
serialized under `/tmp/deepseek_observed_run.py`, CPU24-31/NUMA0. Background
analysis uses CPU32-39; heavy analysis is paused during formal bench.

Current completed check:
`deepseek_dma_c10_check_20261006_01`,144 observer samples, no foreign GPU process,
maximum observation gap30.606560s. All four methods x32 requests passed. Receipt:
`/tmp/cxldsagr-checks/deepseek_v32_motivation/data/deepseek_dma_c10_check_20261006_01/receipt.json`.

Current live benchmark: exec session26726, run `deepseek_dma_c10_bench_20261006_01`.
It uses H65536,A128,chunk1024,P65536,NH16777216,16users,2rounds,seed42,graphs.

After the benchmark observer and child fully exit, run:

```bash
source /tmp/deepseek_mfu_env.sh
CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=24-31 --membind=0 \
  .venv/bin/python /tmp/deepseek_observed_run.py \
  --output /tmp/deepseek_dma_c10_profile_20261006_01_driver -- \
  bash experiments/deepseek_v32_motivation/scripts/profile.sh \
  --run-id deepseek_dma_c10_profile_20261006_01 \
  --reference-run experiments/deepseek_v32_motivation/output/data/deepseek_dma_c10_bench_20261006_01
```

Profile executes4x(3warmups+17trace requests)=80requests and captures4graph setups
plus8selected requests. It must pass numerical and DMA byte/address/activity
checks before replacing the mapped-gather publication. No official ECHO rerun.

All84changed Python files pass Ruff check. The only pending format-only edit is
line wrapping in motivation `src/graph_attribution.py`; apply after all source-
frozen GPU jobs end, then refresh relevant report helper identities as needed.
