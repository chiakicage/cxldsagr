# Host allocator observations after the four-method prelude

The allocation-only control reports zero PyTorch-owned pinned bytes after the
normal measured graph construction. The four-method prelude needs its own
observations before retained pinned allocation is considered an explanation
for its longer gaps.

`q1_replay_prelude_host_stats.py` calls the unchanged `q1_replay_prelude.main`.
It preserves that driver's model, compute bank, four warmups, cache switches,
graph preparation, normal cache flushes and timer. Its only new runtime work is
`torch.cuda.host_memory_stats()` at the timer's two existing runtime callbacks:
after measured graph construction and five warmups, then after check or
samples. Both callbacks are outside collection; neither immediately follows
the host-cache flush.

Use kind `deepseek-q1-replay-prelude-host-stats-v1`. The process-local prelude
kind and final receipt use this new kind; the source contract records the
original prelude kind and hashes the observer, prelude, matched wrapper and
timer. Existing frozen source files are unchanged.

Keep variable counters outside execution identity. Final result saving writes
`prelude_host_allocator_observations.json` once and attaches its path, hash and
size outside identity. The check receipt includes that artifact. The matched
runtime archive inventory never receives the variable observations.

Report allocator-owned rounded bytes and counters without subtracting the
uncertain active counters to infer exact cached bytes. Zero allocator-owned
bytes does not cover memory outside this allocator; nonzero bytes does not
identify individual owners or establish a fence mechanism. Root owns GPU
checks, timing and profiling after CPU/static review and source freeze.
