# Q1 hint baseline: NCU launcher identity resolution

Status, 2026-10-08: baseline check, clean benchmark and both targeted NCU
collections completed under the third launcher identity. This is component
diagnosis, not candidate acceptance or formal model-result publication. The
production hint APIs, official ECHO and attention implementations are unchanged.

## Two failed collections and the launcher fixes

The driver remains SHA256
`c9553ab4244f8be823ae5a59644eec0b66ea7ebf59b0e48bb7d7e7f7fb16ddbf`.
The launcher changed as follows; each change received a new independent check
and clean benchmark before its next collection.

| Launcher SHA256 | Collection | Observed outcome |
| --- | --- | --- |
| `fd36eb8baf7fbe25c9f9ba640bb44caf7c9295a7828a22cc898e504bf643bcd1` | `q1_hint_baseline_full_20261008_01` | Strict identity gate failed before the marked profile call. |
| `6a780b4f0f89bde59fa5c0922a607bfdfbf1ea2ccc34eba6e45efb039fd26f85` | `q1_hint_baseline_full_20261008_02` | Child identity/completion gates passed, but NCU collected no kernels. |
| `08b57c51de8f2a18c9d7a7c9810559ac3c7472013fb61c6f9f1ef38ccdb51b21` | `q1_hint_baseline_{full,source}_20261008_03` | Both reports contain one selected reduction kernel and passed child completion. |

For `_01`, an independent diagnostic captured the identity at the original
receipt gate and terminated before `profile()`. Recursive comparison found
exactly one difference: `source.environment.LD_LIBRARY_PATH`. NCU prepended
`/opt/nvidia/nsight-compute/2026.1.1/target/linux-desktop-glibc_2_11_3-x64/.:`
to the accepted `/usr/local/cuda/compat:/usr/local/cuda/lib64:` value. All
source, input, runtime and loaded-library hashes matched. Warmup/preparation
had run; no marked profile sample or NCU report was produced.

The launcher now captures the outer variable before invoking NCU and launches
Python through `/usr/bin/env` with that exact value. It uses `env -u` only when
the variable was originally unset, preserving the distinction from an empty
value. Quoted array arguments preserve the value without shell re-evaluation.
Other NCU instrumentation variables and the strict driver receipt gate remain
intact. Bash syntax/help checks passed; source review found no remaining
quoting or environment-preservation issue.

For `_02`, restoration fixed the identity mismatch and the child completed its
post-profile audits. However, `--target-processes application-only` excluded
the Python execution reached through `env`. NCU explicitly reported no kernels
and requested `--target-processes all`; no `.ncu-rep` existed. The final launcher
enables that process coverage while retaining `--profile-from-start off`, the
single `q1_hint_baseline_layer_0/` NVTX range, `reduce_kernel` filter and
`--launch-count 1`. The receipt comparison was never relaxed.

Both failed runs were copied and rehashed under `/tmp/cxldsagr-checks/` before
deleting only their corresponding experiment `output/data`, `output/log` and
`output/profile` directories. Archive indexes are:

- `q1_hint_baseline_failed_full_20261008_01_archive/index.json`, SHA256
  `ee334e49ceeefb7b299ae9e4f4cf7b4d2c6e3cd2ac76f12249d7a5aefcd45fcc`.
- `q1_hint_baseline_failed_full_20261008_02_archive/index.json`, SHA256
  `06bfe97980a13d77bfd560cfaee6f3ca3836fdb0714e6087d00c1aaf0f989192`.

Their original source identities and valid check/benchmark records were
preserved, not relabeled as the third launcher.

## Accepted component baseline and successful collection

The current receipt is
`/tmp/cxldsagr-checks/q1-hint/q1_hint_baseline_check_20261008_03/receipt.json`,
SHA256 `0e6fc9ff8e03a1110c7c5f666d3f68e761aac1663ece134b7706b71896cba8b4`.
Independent CPU reread verified six saved GPU oracle/output comparisons, all
16 offset slots, the 30 receipt artifacts and archived input identities.
The runtime matches check `_02`; only the launcher source hash changed.
This real-input baseline check does not cover candidate promotion cases.

`q1_hint_baseline_bench_20261008_03` retains 600 unique samples, 100 per layer
and execution mode. The independently recomputed medians are:

| Layer | Complete graph replay, us | Complete eager call, us |
| --- | ---: | ---: |
| L0 | 16.128 | 73.312 |
| L1 | 15.904 | 72.608 |
| L2 | 16.256 | 72.864 |

CUDA events surround one mean-plus-unchanged-EMA call or replay. Offset reset
and its pre-event synchronization are excluded. Each stratum has 20 warmups,
following three eager preparation calls and one graph capture per layer.
Inputs are the saved extra eager checkpoint score corpus, with logical
`[1,65537]` slices and accepted top-k values; they are not formal timed-forward
inputs. These are one-process component measurements, not model latency or a
paired candidate comparison.

The full and source reports use separate processes, kernel replay,
`cache-control all` and `clock-control base`. CPU parsing with `ncu_report`
confirmed one range and one `reduce_kernel` action in each. Retained details
identify the FP32 sum template and L0 NVTX range. Both completion files match
the independent receipt and exact source/runtime identity; archived sources,
headers, Triton artifacts and inputs were checked.

| Report | SHA256 |
| --- | --- |
| `output/profile/q1_hint_baseline_full_20261008_03/full.ncu-rep` | `3516df7144bc9752925b43609f1c5901f6c93d55ba17f3ddbfb6a235fdda1be2` |
| `output/profile/q1_hint_baseline_source_20261008_03/source.ncu-rep` | `f1f540776bcad1e11c56860a116cfde7922bf4a7894799621f01ec6062cf9439` |

The full collector preserves a warning that six CTC RX/TX byte metrics were
unavailable. Successful collection does not imply all requested metrics exist.
The installed wheel has no verified source-line mapping; root's initial report
inspection found no mapping. Counter and dynamic-SASS interpretation belongs
to the separate raw-report review. NCU duration under cache flushing/base clock
must not replace the clean complete-call measurements above.

CPU evidence under `/tmp/cxldsagr-checks/`:

- `q1_hint_baseline_check_03_independent.json`, SHA256
  `261090777638fe7c79cfd2f78afd12d74dadc04fc57a10d928891c7a192a6d33`.
- `q1_hint_baseline_bench_03_independent.json`, SHA256
  `6dd7b60906e68dbc79e9f21e95a9b19af73603559b634fb37ecf4724cffb8568`.
- `q1_hint_ncu_03_identity_review.json`, SHA256
  `e73497389e717c0acb20fb94a932d29560ed956207a1d0fee71cfcfcdbfd8f72`.

Each JSON has its companion review script. These reviews launched no GPU work
and made no changes to the frozen baseline source.
