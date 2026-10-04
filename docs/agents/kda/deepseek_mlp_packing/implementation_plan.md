# Dense MLP packing executable plan

- [x] Inspect current MLP/FP8/SiLU calls and historical attribution boundary.
- [x] Trace official SM90 output stride from API validation through TMA stores.
- [x] Record contract/draft before temporary implementation.
- [x] Prepare `/tmp/deepseek_mlp_packing_probe_20261004/probe.py`, using the
  current production quantizer/DeepGEMM/FlashInfer adapters with no source edits.
- [x] CPU prepare: record shape/pointer/stride rules and source hashes, compile
  Python and verify CUDA remains uninitialized.
- [x] Obtain final independent source/harness review before GPU execution.
- [x] After a GPU grant, screen baseline and three controls on small aligned
  N/K and Q tile boundaries, plus Q121/128/1024 checkpoint MLPs. Compare all
  bytes and full down output; prove guard/other-half preservation after each GEMM.
- [x] Verify non-default-stream predecessor/consumer ordering, unchanged source
  tensors, retained outputs, and graph replay with changed input values.
- [x] Obtain independent audit of the completed original screen result.
- [x] Review the CPU-prepared complete benchmark driver independently,
  preserving the original probe's exact bytes.
- [x] Screen the extended four-mode/two-boundary graph matrix under the new
  driver's own source identity after root grants CUDA work.
- [x] After screen acceptance and an exclusive timing grant, use fresh output
  IDs for eager complete gate/up/SiLU and full-MLP measurements. Alternate paths,
  warm all compilation, and use final device synchronization. Keep inspection
  outside samples; record allocations, kernel profiles and complete samples.
- [x] Compare graph replay separately with changed-input correctness; include
  graph-private reserved memory and each graph's actual source identity.
- [x] Select a candidate based on complete cost and exactness. Before integration,
  design narrow fallback dispatch and retain/update gate/up API attribution.
- [x] Preserve original source reconstruction and implement the selected path;
  first focused CPU checks passed. Follow [integration_plan.md](integration_plan.md).
- [x] Complete independent production review and scheduled GPU correctness:
  37 production tests and nine checkpoint-weight cases, including 27 changed-input
  graph replays; saved records independently audited.
- [ ] Root-owned combined checkpoint, graph lifetime, full serving performance
  and matching operator-attribution acceptance precede report replacement.

The completed initial commands and source IDs are recorded in
[checkpoint.md](checkpoint.md). The initial probe's `bench` completed the
separately granted eager measurement only. The complete graph measurement used
the separate driver described in [benchmark_plan.md](benchmark_plan.md).

The initial CPU preparation command was:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/probe.py prepare \
  --output /tmp/deepseek_mlp_packing_probe_20261004/prepare_03.json
```

The granted correctness screen completed with the following command:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 \
.venv/bin/python /tmp/deepseek_mlp_packing_probe_20261004/probe.py screen \
  --checkpoint /preset-models \
  --output /tmp/deepseek_mlp_packing_probe_20261004/screen_01.json
```

Use new paths for all later runs. No C7a+hint formal result is an MLP-packing
result. Temporary artifacts are engineering evidence and do not belong in
experiment/report/test directories.
