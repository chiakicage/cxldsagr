# Aligned-host sector diagnostic

Run `nosa_q128_aligned_host_ncu_gpu0_20261004_01` completed with exit0 on
2026-10-04. The unchanged independent-warps binary was tested on GPU0/NUMA0
with the same request16/layer16 Q128/H65536 operands, selection and ownership.
The [frozen plan](aligned_host_diagnostic_plan.md) defines the two buffer arms.
This result establishes a host-access alignment effect, not API latency,
overlap acceptance or a production allocator change.

## Controlled result

Both reports contain one selected NVTX `fused_main` action. Each collection
used two strict application-replay invocations. Including the standalone
validation for each arm, all six GPU invocations passed their four checked
calls:24 calls total. Two additional CPU audits passed. All collection stderr
files are empty.

| Counter | Original buffers | Aligned buffers |
|---|---:|---:|
| Actual mapped pointer modulo32 | 16 | 0 |
| Instructions at each host LDG | 14,368 | 14,368 |
| True-thread instructions at each host LDG | 459,776 | 459,776 |
| Theoretical / ideal sectors at each host LDG | 258,624 / 229,888 | 229,888 / 229,888 |
| Sectors per host LDG warp instruction | 18 / 16 ideal | 16 / 16 ideal |
| Combined TEX sysmem read-miss sectors | 517,248 | 459,776 |
| Logical payload bytes | 14,712,832 | 14,712,832 |

The host LDGs are offsets `0xf750` and `0xf770`; dependent HBM stores at
`0xf7c0` and `0xf800` remain16 sectors per warp with zero excess in both
arms. The same host instruction counts and exact outputs are preserved.
The total sector decrease is57,472 sectors, or1,839,104 bytes:11.11% of original
sector traffic. Equivalently, the original12.5% amplification above payload
disappears. These counters do not measure a whole-bus physical transfer total
or establish a calibrated link bandwidth.

Function bases are `0x79398c484900` and `0x7fcc34484900`, taken from
`launch__function_pcs.value(0)`. Both profiled target receipts in each arm
record the predicted alignment. Aligned views have16-byte storage offsets;
CPU and mapped aliases are explicitly recorded. The raw units remain separate:
instruction metrics use `inst`, source sector metrics use `sectors`, and
TEX uses `sector`. No intrusive duration is interpreted as performance.

## Capacity and lifetime

Original K/V own two32 MiB allocator blocks. The aligned arm retains those
originals and two padded owners that each round to64 MiB:192 MiB of distinct
known rounded backing, including128 MiB for the aligned owners alone.
Storage wrappers are deduplicated. This diagnostic therefore cannot be promoted
as a capacity-safe runtime cache allocation.

All host payload hashes, padding guards, mapped offsets, suffix state, unique
bytes, queue checks and GPU cleanup pass. After confirmed workspace/device
completion, original tensor, aligned view, owner and storage weakrefs are dead.
Raw active host counters remain recorded without treating them as physical
live capacity; allocator-owned bytes and known owner buckets are separate.

## Identity and next step

Run directory:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_aligned_host_diagnostic_20261004/output/nosa_q128_aligned_host_ncu_gpu0_20261004_01/`.
Analysis:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_aligned_host_analysis_20261004/`.

| Artifact | SHA256 |
|---|---|
| Original report | `82b2422695ea4147aa1571b864a50b6b884aec009f6a8c5a2f1e1ed09a4bd74c` |
| Aligned report | `c7606ab12382aec5e42e81e2d47bce5637fa2c4b77989bb0caba089c27c5f451` |
| Frozen helper manifest | `a6825a8c5646ee849f671500e5cebe384631d948f98e18bc9a59170a68234341` |
| Application audit | `b0baf30541b0b9dca8a327de6dc70f7e977e99a66a78e6a6703e23450e1396aa` |
| Comparison | `405f48aef658f9f88f8f53d95345e68b77ced1d04f06eee068600e62994b189b` |
| Independent receipt consistency review | `725726aec35831ee09feaf7065747b07f16c82c379bc465019b5217ce341e790` |

Root independently loaded both reports and matched every selected PC counter
and TEX total; `root_raw_counter_check.json` retains that extraction.

The subsequent internal screen is recorded below. No native candidate or aligned
allocation has been integrated.

## Internal overlap screen

Run `nosa_q128_aligned_internal_gpu0_20261004_01` completed its two exact controls
and rejected the first aligned halves2 internal sample. Both page-envelope and
nonempty-stripe ratios were0.8408366534, below0.90. Copy, softmax and intersection
unions were321.280,290.528 and270.144 microseconds. Root independently rebuilt
these unions from raw integer globaltimer intervals; page and stripe unions
were identical. The driver stopped after this sample, with no later internal
samples or complete-API latency run.

All three calls passed exact saved-output and full524,288-element FP32 checks,
unique14,712,832-byte payload, suffix, queue, native-identity and cleanup checks.
Aligned calls retained192 MiB of known rounded pinned backing; all tracked
owners/views/storage wrappers were released after confirmed completion. The
failure is the measured overlap gate, not numerical or lifecycle failure.
This single sample does not establish a statistical improvement over the prior
unaligned sample.

Screen record SHA256:
`324dfe01ba1d72ffc1c7204318c4998b427b18187fbe31131f9e39b8ed70fc4a`.
Independent application receipt audit SHA256:
`64cbbe22a98b1d305d0cd9e0749b82dc4c794ab23033cf7b3b55a65747e43a09`.
Independent raw interval review SHA256:
`f443e85d472a77dc7c7f31095440c882702225e087cc0ca3c605754d229dd7c3`.
The latter reconstructs all449 page envelopes,3,592 nonempty stripes and3,030
softmax windows. These records are under the sibling external screen/audit/review
directories named by the run.

Root next selected isolated preparation of the
[four-pair ILP proposal](independent_warps_ilp_proposal.md), using original host
allocation and a halves2-only register-budget increase. Alignment's sector
effect is established; its tested implementation does not satisfy overlap
acceptance or justify a production allocator change.
