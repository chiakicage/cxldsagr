# C5 executable plan

- [x] Read installed official API, C++ dispatch/sorting and C3 stage kernel data.
- [x] Implement direct official int32 adapter and lazy in-place nonfinite mask.
- [x] Add CPU adapter contract checks and CUDA strict differential fixtures.
- [x] Prepare /tmp probes for public wrapper, int32-only and fused wrapper.
- [x] After root grants GPU window, run existing all-K/tie/extreme/padding tests,
  strict public-API bitwise differentials, nondefault-stream and capture/replay.
- [x] Benchmark same GPU, Q128/1024 and N1024/32768/65536/65664; include all-invalid,
  short-prefix, signed-zero/tied inputs. Record actual GPU work/launches and API
  wall, and explain the remaining official kernel cost.
- [x] Freeze measured winner and update checkpoint before root's full acceptance.

CPU only while C4 quiet run is active:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q operators/deepseek_v32/indexer/tests/test_selection.py -k 'rejects or adapter'
```

GPU (only with window grant):

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q operators/deepseek_v32/indexer/tests/test_selection.py
```
