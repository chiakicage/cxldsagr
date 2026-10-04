# Four-pair fetch ILP proposal

Status: CPU-only proposal on 2026-10-04. Not selected for implementation. The
aligned-host diagnostic comes first. No CUDA source edit, compilation, or GPU
execution is authorized by this note.

## Bounded question

The current independent-warp path loads one K/V `uint4` pair and immediately
stores it before advancing by 32 elements. Current NCU source counters identify
host-load-dependent stores as a plausible limit. Consumer pipeline waits also
carry substantial `long_scoreboard` samples, so the global stall total is not a
host-load-only measurement. Host sector amplification is being tested separately
by the aligned-host diagnostic.

If another candidate is warranted after that diagnostic, test whether keeping
four independent K/V pairs in registers allows more host loads to be issued
before the first dependent store. Compare candidates with the same host-pointer
alignment and the same captured ownership/selection state.

## Exact source scope

The full stripe has 8 tokens × 16 `uint4` elements per token = 128 elements for
each of K and V. A 32-lane warp therefore has exactly four K/V pairs per lane.
The addresses remain those of the existing `item = stripe * 128 + lane + 32*j`
mapping, for `j = 0..3`.

A narrow implementation can use a warp-uniform full-stripe branch: issue the
four K loads and four V loads into fixed register temporaries, then write the
four K and four V results to the existing HBM offsets. Keep the current loop for
partial and empty stripes. This preserves the established partial-page behavior
without forming or loading an out-of-range host pointer. Fixed-size arrays are
acceptable only if compiled code keeps their payload in registers.

Do not change task claiming, three fetch workers per CTA, queue order, page or
stripe ownership, byte attribution, masks, sentinel/suffix handling, or the
post-store fence → full-warp join → leader acq_rel ready increment. Warp 0 keeps
its TMA role. The existing ready-acquire and async-proxy fence remain unchanged.

## Register budget

For the current halves2 specialization, the measured compiled allocation is
61,440 registers per 256-thread CTA. Both producer and consumer groups contain
128 threads:

| Producer / consumer limits | Dynamic demand | Margin in current CTA pool |
|---|---:|---:|
| 24 / 240 | 33,792 | 27,648 |
| 80 / 240 | 40,960 | 20,480 |

Thus 80/240 fits the current measured CTA pool. The extra payload alone needs
32 scalar registers per lane, compared with eight for one K/V pair. Addressing,
predicates, queue metadata and compiler allocation still need measurement; 80
is a proposed budget, not proof that the implementation will avoid new spills.

The producer limit applies to the entire producer warpgroup, including TMA
warp 0. Scope it to halves2. A global 80/240 change would make halves1 require
128×80 + 256×240 = 71,680 registers, exceeding its 64,512-register CTA pool.
The halves1 control must retain 24/240.

For any new build, recompute the existing guard from actual function attributes:

```text
CTA pool = round_up(numRegs * 32, 256) * (block_threads / 32)
required = 128 * producer_limit + consumer_threads * 240
```

Require `CTA pool >= required`. Do not substitute the entire SM register limit.
Recheck actual cooperative occupancy, configured shared memory and grid bounds.

## Source and compiled acceptance

- Freeze a new isolated candidate and bind source, include closure, build flags,
  library identity, resource report and disassembly. Do not modify the accepted
  historical controls or the rejected independent-warp evidence.
- Confirm halves1 source behavior and selected compiled body remain unchanged.
  For halves2, verify the producer instruction establishes the intended 80
  register limit while the consumer remains at 240.
- Inspect the full-stripe control-flow path: eight `.cv` host vector loads must
  issue before the first dependent HBM payload store; each original address is
  loaded once and each destination is written once. Source ordering alone is
  insufficient. The existing volatile asm memory clobber helps express the
  intended order but does not replace disassembly review.
- Verify the four-pair payload stays in registers. Record any new local loads,
  stores or stack growth, including outlined functions. The baseline already
  has stack accesses; do not describe the entire kernel as spill-free.
- Verify the unchanged publication chain, empty-stripe completion and terminal
  claims. Do not attribute a gain solely to ILP if the changed producer budget
  also changes TMA or metadata spill behavior. A budget-only 80/240 control is
  the smallest additional comparison if that causal attribution is needed.

## Conditional executable validation

If root selects implementation, first run the established partial-page checks
and exact/FP32, payload, unique-read, queue, library/resource and cleanup gates.
Then run fresh representative internal page-envelope and nonempty stripe-copy
checks, with every sample meeting the existing 0.90 gate before advancing.
Full API and broader workload gates remain separate. Intrusive NCU duration is
not an acceptance metric. A same-alignment baseline and fresh run IDs are
required; neither the old rejection nor the alignment-only diagnostic validates
this proposed implementation.

The next decision belongs to root after the aligned-host result. This proposal
does not select a candidate, predict speedup, or reopen acceptance.
