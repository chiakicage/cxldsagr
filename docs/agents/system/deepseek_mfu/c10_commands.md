# Paused C10 DMA follow-up

This independent C10 task remains paused by the user. The real-three-layer
full-graph MFU supplement is complete and does not resume these commands.
Current C10 status is in `experiments/deepseek_v32_motivation/README.md`.
The historical command below used `/tmp/deepseek_mfu_env.sh`, GPU3,
`/tmp/deepseek_observed_run.py`, and CPU24-31/NUMA0. Verify its inputs and
environment again before any future authorized execution.

Recorded completed check:
`deepseek_dma_c10_check_20261006_01`,144 observer samples, no foreign GPU process,
maximum observation gap30.606560s. All four methods x32 requests passed. Receipt:
`/tmp/cxldsagr-checks/deepseek_v32_motivation/data/deepseek_dma_c10_check_20261006_01/receipt.json`.

Recorded benchmark: `deepseek_dma_c10_bench_20261006_01`; it is no longer a live job.
It uses H65536,A128,chunk1024,P65536,NH16777216,16users,2rounds,seed42,graphs.

Historical profile command, not executed by the current MFU supplement:

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
