# ECHO prefetch hint executable plan

- [x] Inspect current expression and accepted historical profile.
- [x] Fix exactness and timing contract before implementation.
- [x] Prepare a temporary Triton mask/count plus unchanged torch.sum prototype.
- [x] CPU syntax/import checks without initializing CUDA.
- [x] Exclusive GPU screen against checked expression: all primary N sizes,
  Q=1/2/3/4/128/1024, padded strides, adversarial values; compare scratch bytes,
  scalar bytes, unchanged inputs and offset tail.
- [x] Only after exactness, alternate complete checked/candidate API timings
  with warmup, final synchronization, output audit and source hashes.
- [x] Integrate if measured faster; otherwise reject without production edits.
- [x] Run affected GPU/model/serving checks (176 combined checks passed).
- [ ] Complete and independently audit the new quiet full formal workload.

All component work completed after the C6 dense profile released CUDA.
The full trace `motivation_c7_hint_20261004_u16_r2_01` is running; other CUDA
work is held until it finishes. Results and final source identities are in
[the checkpoint](checkpoint.md).

Initial temporary source: `/tmp/deepseek_prefetch_hint_probe_20261004/probe.py`, SHA-256
`8ecf64b60a827bdefecaccf6c3d935f7bcd1c5c5fb2e9bcf374b90bab27d7efe`.
Ruff passed and prepare mode confirmed CUDA remains uninitialized.

Executed screen after extending the boundary cases (final source identity is
recorded in the checkpoint):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python /tmp/deepseek_prefetch_hint_probe_20261004/probe.py screen --output /tmp/deepseek_prefetch_hint_probe_20261004/screen_01.json
```

The 64 history shapes use the four rows consumed by the hint; Q128 and Q1024
checks also verify actual tail slices. This is component input geometry, not
an entire indexer or full-model measurement. Separate `bench` and production
benchmark invocations followed the exactness gate with fresh output paths.
