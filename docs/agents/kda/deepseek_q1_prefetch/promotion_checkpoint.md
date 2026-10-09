# Promotion candidate checkpoint

Status: promoted into production and covered by final real-three-layer matrix
`deepseek_h64k_a1_official_20261008_03`; see [the current checkpoint](checkpoint.md).
The isolated candidate
uses one 64-thread validation CTA followed by 64 128-thread record-copy CTAs.
Both kernels are included in clean timing. No scratch or API change is required.
Prepare/clear and every original device assertion expression are unchanged.

Baseline SHA256: `fcc2bf553ede3838ecaebd35383cc1c60102858958ade63352680b798c4cbe32`.
Candidate SHA256: `45e4375996072f877f308f332a0c69947b8498fe766d10f0b72a2cb92d739c5c`.

Independent acceptance:
`/tmp/cxldsagr-checks/deepseek_q1_promotion/data/deepseek_q1_promotion_check_20261008_01/`.
84 custom cases passed exact CPU and baseline comparisons, including random raw
uint16 BF16 payloads, 2-byte/mixed base
offsets, count 0/1/17/64, saturated attempts through uint32 max, host zero,
consecutive/random slots and empty/mixed/full selected ownership. The unchanged
22 adapter tests pass after substituting only the module provider. Four isolated
negative cases still raise device asserts: duplicate slot, invalid host, stale
owner mapping and occupied journal. No skips.

Additional acceptance under `deepseek_q1_promotion_graph_check_20261008_01`
passed 72 graph replays and 24 nondefault-stream calls, all record/map/journal/
statistic bytes exact for both baseline and candidate. These are engineering
checks, not performance experiments. The first build command failed because
`ninja` was outside PATH; the explicit `.venv/bin` PATH corrected the environment
before any check output was created. The implementation does not retry builds.

Valid-state race proof: the validation kernel finishes before publication begins.
It proves unique slots/new host IDs and valid old ownership. New hosts have
staging tags; old hosts have real-slot mappings, so these host sets are disjoint.
Two distinct slots cannot share a valid old owner. Each copy CTA writes only its
own record, slot owner, new/old host entries and journal entry; its block barrier
precedes mapping publication. Stats are written only by validation. The existing
exclusive lease and later same-stream consumption remain required. Failed CUDA
operations still propagate and poison the caller; no recovery path was added.

Clean benchmark: `deepseek_q1_promotion_bench_20261008_01`, GPU1 (NVIDIA M403,
SM90, 143771 MiB), CPU8-15, Torch 2.12.1+cu130, FP8 arithmetic absent and BF16
records copied bitwise. Input KV is saved real layer-2 `q1_inputs_20261008_01`,
N=65,537/P=65,600, 64 records of width576. Each case has 20 warmups and 100
alternating pairs. External event nodes are captured immediately before/after
the complete adapter call, excluding metadata reset and host replay dispatch;
there is no cache flush. The two candidate kernel phases are both measured.

| Slot layout | Selected occupied slots | Baseline median us | Candidate median us | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| consecutive | 0 | 33.664 | 12.864 | 100/100 |
| consecutive | 32 | 34.400 | 13.248 | 100/100 |
| consecutive | 64 | 34.432 | 13.152 | 100/100 |
| random | 0 | 34.352 | 13.296 | 100/100 |
| random | 32 | 34.496 | 13.456 | 100/100 |
| random | 64 | 34.720 | 13.568 | 100/100 |

Full samples, p90, source/native hashes and identity-bound receipts are indexed
by the benchmark output's `summary.json`. The patch is
`docs/agents/kda/deepseek_q1_prefetch/promotion_candidate.patch`.
The profile mode exports one ProfilerAPI/NVTX-scoped direct call and supports
baseline/candidate, consecutive/random slots and 0/32/64 occupied selected slots.
The isolated results above do not establish full-model latency. Root subsequently
completed NCU, applied this exact candidate source and issued final matrix `_03`;
its full-model evidence is recorded separately in the current checkpoint.


Root NCU audit: `q1_promotion_ncu_20261008_02`, GPU3/CPUs24-31, real
layer-2 BF16 payload, N65,537/P65,600 and first 64 free slots. Separate full/source
captures for each kernel use kernel replay, all-cache flush and base clocks.
Baseline has grid1/block256, 68.896 us, SM throughput 0.0389% and DRAM-read
throughput 0.0272% of peak. Source sampling maps 37 long-scoreboard samples to
the serial record-copy loop at baseline.cu:79. Candidate validation takes
10.784 us with 1 CTA; copy/publication takes 5.440 us with 64 CTAs. These are
invasive per-kernel measurements, not clean complete-call latency. Together
with the 600/600 clean paired wins, this supports promotion of the unchanged
private source into production. The first NCU attempt matched the wrong
kernel-name base, produced no captures and was removed; `_02` explicitly
matches demangled names. No failed profile is used as evidence.
