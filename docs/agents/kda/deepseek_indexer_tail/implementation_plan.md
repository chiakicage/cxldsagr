# C4 executable plan

- [x] Inspect official DeepGEMM wrapper and installed FlashInfer length APIs.
- [x] Implement sliced PyTorch suffix-mask candidate in echo.logits.
- [x] Add GPU integration tests for Q/N/start and arbitrary padded row stride.
- [x] Wait for root's GPU window; run indexer/selection correctness suites.
- [x] Measure full-mask versus sliced-mask API wall and CUDA work at Q1024,
  N1024/32768/65536 and Q128,N65664, including strided backing.
- [x] Assess one-launch Triton after the sliced baseline: defer it; final
  contiguous-stride tail work is 4.95–10.60 us, and integrated profiling should
  set its priority against remaining top-k/common launch costs.
- [x] Freeze winner, record source identity and evidence, return to root for
  complete-request/full-model validation and current-source profile acceptance.

Commands (run only after GPU window grant):

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q operators/deepseek_v32/indexer/tests/test_echo_indexer.py operators/deepseek_v32/indexer/tests/test_selection.py
```

Engineering probe scripts/results stay under /tmp. KDA checkpoint records device,
dependency/source identity, warmups/repetitions, correctness and measurement
boundary. This component alone cannot prove end-to-end MFU convergence.
