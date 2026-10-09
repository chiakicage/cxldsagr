# Private bounded-free preparation

## Contract and baseline

Root approved this candidate after personally reviewing current baseline NCU
`q1_prepare_baseline_ncu_20261008_01`. Two 12-CTA radix OneSweep passes dominate
the current kernel cost. This candidate tests a different algorithm from the
rejected DeviceTopK attempt: select existing free slots without sorting ages.
GPU0 and CPUs 0-7 are assigned. Production and published reports remain frozen.

Future normal dispatch requires sole session, exclusive operation lease,
ordinary Q1, official paged eligibility (`N>=32768`, initialized history equal
to query start), and `P-H>=64`. Then live records L satisfy L<=H and at least
64 slots are free. This phase is a standalone private component; it does not
change model/cache dispatch or claim integration acceptance.

## Draft

Build a private extension that includes the current unchanged ECHO translation
unit. This provides both the original baseline prepare API and the existing
`free_append::prepare_masks` / `select_slots` algorithms. Record all local,
CUTLASS and CCCL source dependencies plus the immutable compiled artifact.

Candidate phases:

1. Reuse `prepare_masks` verbatim: scan every priority, trap on invalid age,
   and write free masks into existing P-int64 scratch.
2. Reuse `select_slots` verbatim with count 64: select ascending free slots,
   check selected free bitmap/reverse owner, trap if fewer than 64 exist.
   Temporarily use the first 64 journal entries as int64 output slots.
3. One final kernel converts those slots into the int32 slot list, writes
   MISSING to its unused tail, clears the complete journal, counter and
   statistics, and writes request metadata `[1, 0]` into now-dead mask scratch.
   Each chosen journal entry is read and cleared by the same thread.

No records, mappings, priority or clock are modified. Slot zero remains
excluded. A private one-use token explicitly carries `prepared_limit=64`
and requires a declared consumer limit of 64 or less at consumption; a general
consumer cannot silently use a partially prepared list. No failure fallback.

Risks: the single-CTA selector may scan many masks when only trailing slots
are free; reset/publication adds a third launch; included baseline must retain
the exact original sort and flags; malformed states must fail before final
publication. All active scratch and CUDA stream ownership remain explicit.

## Executable plan

1. Save private source under
   `experiments/deepseek_v32_mfu/output/data/q1_free_prepare_candidate_20261008_01/source/`.
   Add a reusable MFU harness for separate check, bench and profile modes.
2. Independently compare complete baseline output against CPU stable argsort,
   candidate first 64 slots against both, full journal/counter/stats/metadata
   reset, and unchanged priority/bitmap/reverse-map storage. Verify the private
   token's one-use identity and bounded-consumer checks.
3. Cover saved real cold inputs, partial prefix/random residency, only the last
   64 slots free, exactly 64 scattered free slots, tight P-H=64, dirty scratch,
   graph replay with changed valid state and nondefault streams. Invalid
   priorities, insufficient free capacity, selected bitmap mismatch and reverse
   owner mismatch run in separate child processes and must terminate with a
   CUDA failure. Never run timing before the independent check passes.
4. Clean timing uses 100 alternating AB/BA pairs per state. Baseline and
   candidate have separate CUDA graphs and buffers; every state restore is
   outside the event interval. Time the entire preparation, including all
   candidate phases and baseline radix preparation. Keep all samples.
5. If component timing wins, collect an independent NSYS call profile and NCU
   full/source evidence on the exact same sources/inputs. Validate actual
   kernel coverage and inspect the reports through `ncu_report`.
6. Send root the component source/patch, receipt, samples and profile evidence
   before any production integration. Shared cache integration will require
   both model rule sets and affected-experiment acceptance separately.
