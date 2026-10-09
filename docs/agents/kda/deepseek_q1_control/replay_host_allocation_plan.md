# Pinned allocation history before the matched HBM timer

The accepted four-method prelude produces 416 ns gaps; the HBM x4 control
retains 96 ns gaps. The next control exercises pinned allocation history
without creating offload pools, switching cache methods, launching offload
kernels or requesting extra streams. It does not assume a visibility/fence
mechanism.

`q1_replay_host_allocation.py` reuses the frozen matched environment and timer.
After the original one compute bank, perform three lifetimes. Each holds three
simultaneous `allocate_host_tensor((65600, 576), dtype=bfloat16, pin_memory=True)`
results, then drops their logical-view and root-backing owners. Each request
has 75,571,200 logical bytes and 134,217,728 storage bytes. Weak references
verify owner release without retaining tensors or invoking garbage collection.
Nine requests may reuse the same allocator blocks; they are not nine claimed
CUDA allocations. CUDA pinned memory can be GPU-accessible without an explicit
`cudaHostGetDevicePointer` call.

Require the original empty HBM cache objects, zero cache-generation advances,
one unchanged compute bank, and no shared pools/helpers or full graphs at hook
exit. Do not add a cache flush. The original prefix, fresh graph capture,
five warmups, numerical checks, clean timing and profiling remain unchanged,
including Torch's normal graph-entry host-cache flush.

Use kind `deepseek-q1-replay-host-allocation-v1`. Bind the new driver and exact
`cache/host_allocation.py` hash in the source contract. The returned runtime
identity contains only static completed-state gates. Record host statistics
before allocation, with three owners live and after dropping owners in each
lifetime, then at the two existing runtime callbacks. Those callbacks are
after graph construction plus five warmups, and after the check or samples;
they are not immediate observations of the graph-entry flush.

Write the nine observation points once, at final result saving, to
`host_allocation_observations.json`. Add its path/hash/bytes outside result
identity and include it in the final check receipt's artifact inventory.
Do not insert variable statistics, addresses or diagnostic hashes into exact
runtime identity or the matched wrapper's runtime artifact inventory. Report
rounded owned bytes and allocation/free counters without deriving a cached
byte count from the known uncertain active counters.

Root runs separate check, clean bench and profile after source freeze and
independent CPU review. Preserve exact participating formal runtime subsets,
197 native signatures, 192 layer owners and raw clipped activity unions.
Keep intrusive gaps separate from clean wall time and use fresh-process order
confirmation for any effect. This task adds the driver and evidence plumbing;
it does not execute GPU work or change production allocation policy.

`experiments.deepseek_v32_mfu.src.analyze_replay_host_allocation` reuses the
matched analyzer's raw checks and adds contract/completion and nine-point
observation validation for the check receipt, bench and profile. Optional
`--matched-audit-dir`, `--four-method-audit-dir` and `--hbm-prelude-audit-dir`
select accepted reference audits; the defaults are their first accepted runs.
Raw counters remain separate for every mode, with their original evidence
hashes. Source hashes are frozen before root runs the raw analysis.
