# Q1 bounded FIFO preparation

## Contract

Reduce the ten-node full-slot FIFO preparation in the published H64K/A1 ECHO
path while keeping the official fused decode kernel unchanged. Select the
first 64 legal FIFO slots, exclude slot zero, validate every pool priority,
reset the full allocation journal/counter/statistics and prepare the same
single-request metadata. Do not modify records, maps, priorities or clocks.
The ordinary pending query is not yet stored in main KV during preparation;
the official kernel continues to exclude it from history prefetch.

This is a separate private candidate from promotion validation. Production
and published artifacts remain unchanged. The general prefill preparation
continues to require its original number of slots. Any eventual production
dispatch must explicitly carry the 64-slot preparation bound and only allow
consumers whose maximum consumption does not exceed that bound.

## Draft and choice

The CUDA toolkit contains official CCCL `cub::DeviceTopK::MinKeys`. It returns
an unsorted nondeterministic output. Encoding `(age + 1, slot - 1)` as a
64-bit integer makes every key unique; selecting the smallest 64 then sorting
only those 64 restores the current exact stable FIFO prefix, even when fewer
than 64 predictions arrive. Merely accepting unsorted results could evict a
newer owner while an older/free slot remains and is not acceptable.

The first candidate reuses the current full priority-validation/key-generation
kernel, runs official DeviceTopK on all slots, then block-sorts the selected
64 while clearing the full journal and writing request metadata. Remaining
unprepared slot-list entries are MISSING; the bound must prevent consumption.
It requires no new persistent storage. The CUB temporary workspace is queried
and recorded separately from the baseline radix-sort workspace. Toolkit CCCL
headers are fingerprinted along with compiler, wrapper, native DSO and inputs.

Risks are AIR TopK launch overhead, shared-memory block sort overhead, output
ordering, partial prediction counts, and workspace growth. No cold-only
residency assumption or failure fallback is permitted. Equal-priority slot
reordering is allowed by model rules, but this first candidate deliberately
preserves exact ordering to isolate preparation overhead.

## Executable plan

1. On GPU3 / CPUs24-31, compile a private source containing the exact existing
   preparation functions and the new bounded entry through immutable cache.
   Baseline extraction is verified against the current source and saved.
2. Independently validate all outputs against CPU stable argsort for empty,
   partial, full, tied, ascending, descending, random and skewed priorities;
   include pools 65, 193, 1024, 65,600 and larger, several timestamps, graph
   replay with changed priorities and nondefault streams. Check that the
   full journal and metadata reset match and priority remains byte-identical.
   Invalid priorities below -1 or above the clock must trigger device assert.
3. Verify prefix equality for every prediction count 0-64, and feed the
   selected slots into existing official promotion and lifecycle checks.
   No benchmark is authorized until these checks pass.
4. Run clean independent paired AB/BA graph timing for full preparation,
   restoring identical priorities/counters/journal outside every timed sample.
   Include all candidate key generation, DeviceTopK, sort/reset and metadata.
5. If faster, collect separate full/source or node-level profile to explain
   launch and kernel costs. Promote only after later model-level acceptance,
   measurement and publication; otherwise reject and remove private failed
   performance outputs, retaining the decision in KDA notes.

## Check status

The revised check `q1_prepare64_check_20261008_02` passed108 states,42 changed
graph replays,3 nondefault-stream calls and22 official-adapter checks. Both
baseline/candidate invalid-age cases preserve the CUDA launch-failure trap.
The initial checker expected a different trap-message spelling; the revised
checker compares the exact baseline/candidate CUDA error class.

The inherited agent provider failed after check completion. Parent continued
from its completed receipt. The first parent bench command (`_02`) specified
a nonexistent input filename and failed before any sample/output creation;
the corrected `_03` command binds `model_priorities.pt` to the existing receipt.

## Rejection
The completed clean benchmark `q1_prepare64_bench_20261008_03` rejects this
DeviceTopK candidate. Across five residency/priority patterns, none of500
paired samples improves. It reduces temporary workspace545,791→17,407 B but
increases complete preparation latency. Existing radix sorting already ignores
slot bits and only sorts the significant age bits, so this attempt does not
eliminate the expected full64-bit sort. The production full FIFO preparation
remains unchanged. No model trial or promotion is justified.
| Pattern | Baseline us | DeviceTopK us | Wins |
| --- | ---: | ---: | ---: |
| model_cold | 27.632000 | 38.575999 | 0/100 |
| partial | 30.688001 | 38.752001 | 0/100 |
| full_ties | 26.736001 | 38.336001 | 0/100 |
| random | 30.176001 | 39.487999 | 0/100 |
| skewed | 28.608000 | 38.463999 | 0/100 |

Evidence identity before withdrawing the rejected outputs:

- `experiments/deepseek_v32_echo_official/output/data/q1_prepare64_bench_20261008_03/result.json` SHA256 `cb615ba35500b7053e653c4171ae823b4c538cbd04a44a74628f0b7252e6a581`
- `experiments/deepseek_v32_echo_official/output/data/q1_prepare64_candidate_20261008_01/source/prepare64.cu` SHA256 `59f8c9a9dd2da903e53c3c370761a9d857b9e7882e364d5eac24d4914c336605`

The rejected experimental output/candidate trees and dedicated prototype
harness were removed after recording this decision; temporary engineering
check files are not published results.
