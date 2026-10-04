# T1 source investigation

2026-10-04: CPU-only review selected a combined exact output-ordering kernel.
No production source changed and no T1 GPU work ran. graph_validate independently
confirmed the selector/ordering split and signed-zero semantics from source;
runtime equality remains pending.

| Source | SHA-256 |
|---|---|
| `operators/deepseek_v32/indexer/selection.py` | `0e68ecb4e16e40e9dda1541693ce3e00e41ccd4a05ea2e708d04dd2545a4dcd8` |
| Installed `flashinfer/topk.py` | `9320d14b938c5d5b5f966675d72eb19de72c91d6861f350cfd29aa25d0b01b50` |
| Installed `flashinfer/data/include/flashinfer/topk.cuh` | `1cc299ccd00bb6b6ae5bf376b7dc14c7032063468daa684ac36485d6352bbb80` |
| Installed `flashinfer/data/include/flashinfer/topk_common.cuh` | `4ed389d1535ab2e1b759eddca9fac3c46346b47711be2b72bbb1b99b21524f2d` |
| `models/deepseek_v32/echo_attention.py` | `cddf8aef94f2c3f3520bc8152aea6eaed4411ad9b5d6ad26c0303d2facd1766e` |
| `operators/deepseek_v32/indexer/csrc/echo_resident.cuh` | `670f6b3807d642bd10653dcd36d848c9b2f9e6f643b5b037b5aec2d5993b2490` |
| `3rdparty/FlashMLA/csrc/kernels/sm90/prefill/sparse/phase1.cuh` | `f4bc7937b19b4828d4b25a9a078fbdebd1427f88cbc229b2fc4a776b88dda848` |

Installed paths are under `.venv/lib/python3.12/site-packages/`. Relevant source:
`topk.cuh` 3161 (index sort), 3321 (stable value sort), 3413 (deterministic
selection), 3711–3733 (dispatch); `topk_common.cuh` 26–44 (FP32 transform);
`echo_attention.py` 289–319 (selection/append); FlashMLA `phase1.cuh` 470–523
(tiled selected-axis loads). The current API is FP32 only, even though vendor
traits support BF16 and FP16. All K=1..2048 and dynamic Q/N must keep that API.

KernelWiki source page `pr-flashinfer-2661` at
`/root/.codex/skills/kernel-wiki/sources/prs/flashinfer/PR-2661.md` supplied
source-reported background only. Its 2026-04-27 cutoff and partial excerpt do
not establish the installed API or performance; installed source above does.

Resident selection is deferred as a separate candidate. Its physical output
must retain input position order; its union may be computed in another order
only if exact counters, priority/clock effects, traps and leases stay equivalent.
The existing CTA-local bitmap already reduces global contention. Avoid replacing
it with unmeasured per-element atomics or coupling it to pre-append selection.

CPU preparation passed for candidate SHA-256
`7cb5c3cf0e54e44d6254aad11084d422f4998a312d5b75e59ff97368fb0732ec` and
screen SHA-256
`0d740bac803129fa2f37f63bbc6673d8f731a23c045a7e50cdabc5d632e6b4af`.
The exhaustive two-pass-versus-packed CPU proof covers 2,048 widths and three
patterns (6,144 cases), plus exact ABI flags/scratch/output and fourteen invalid
metadata rejections across baseline/candidate. Both CUDA initialization checks
were false. Result `/tmp/deepseek_topk_order_t1_20261004/cpu_contract_v3.json`
has SHA-256 `2757cc8f6843d90eb59f39ad33764edcbb429ab42f7a6507fb7f191469033edd`.

Explicit-target offline compilation passed for widths 1/32/128/256/512/1024/2048,
without loading a CUDA module or initializing CUDA. The target is SM90, four
warps, bundled Triton PTXAS. Output is
`/tmp/deepseek_topk_order_t1_20261004/offline_compile.json`, SHA-256
`0f07dd4a850c0f7b7d4fe9d2a31248947b6344e72ba519a4fede3c09f3bf03a4`.
The 2048-wide CUBIN reports 122 registers, zero stack/local storage and 1,024
bytes static shared; Triton's launch metadata requests 16,384 bytes dynamic
shared. No FP arithmetic/FTZ or local load/store instructions were found in its
PTX. These are compilation/resource observations, not executed correctness or
performance. Resource dump SHA-256 is
`194bde2d78f627497fa59cbe02ffa12fc8ad2d86b16cc30de2610b1f834ca602`.
Ruff check/format passed for those three temporary Python files. The later
runtime wrapper collected actual loaded artifacts after execution. GPU
correctness passed; all six complete API timing cases regressed. T1 was rejected
and GPU0 released. The final evidence is in [t1_checkpoint.md](t1_checkpoint.md).
