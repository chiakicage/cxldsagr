# Formal versus reduced HBM preparation lifecycle

This is a read-only source and retained-SQLite investigation. No source or GPU
execution changed. The offload-prelude difference is real; a CUDA visibility or
fence explanation is not established.

## Confirmed lifecycle difference

Both callers construct `DeepSeekEchoModel` without an `offload` argument:
`experiments/deepseek_v32_mfu/src/profile_layers.py:700` and
`experiments/deepseek_v32_mfu/src/q1_replay_timer.py:126`. The constructor defaults
to `offload=False` (`models/deepseek_v32/model.py:80`), initializes empty shared
pool/session dictionaries at line 178, and allocates shared pools only when
offload is true at line 199. Passing `host_arena_tokens=65600` alone does not
allocate a host arena.

The formal run prepares its compute bank at `profile_layers.py:824`, then runs
the complete default forward path for all methods at lines 831-839. The order
is HBM, ECHO, serial sparse, dense prefetch
(`experiments/deepseek_v32_mfu/src/run_contract.py:13`). With the measured
`warmups=1`, each method gets a fresh cache, full 65536-token prefix, snapshot,
extend-graph preparation, and one Q1 forward before the measured HBM phase.
Consequently three offload pools have been created and used before formal HBM
capture. The reduced timer prepares one HBM bank/prefix at
`q1_replay_timer.py:139-142` and captures its first HBM full graph at lines
166-173. It never changes to an offload method.

Formal measured HBM returns to a fresh resident cache through
`gap_profile.py:217` (or `profile_layers.py:451` for operator profiling).
`model.py:395-455` synchronizes, closes previous full graphs, releases shared
sessions and pools, drops old block caches, and constructs resident runners.
`profile_layers.py:44-52` asserts the shared-pool dictionary is empty for HBM.
Therefore this is past offload allocation/use, not an active offload cache
remaining in the measured HBM model.

## Host allocation and registration boundary

`SharedSparseTokenPool` allocates one pinned host backing per layer at
`cache/sparse_token_pool.py:213-216`. For this shape each logical backing is
65600 x 576 x 2 = 75,571,200 B. `cache/host_allocation.py:31-54` requests the
whole power-of-two bin, 134,217,728 B per layer, using
`torch.empty(..., device="cpu", pin_memory=True)`. Each three-layer pool thus
owns 384 MiB of these bins while live. Three offload method switches make nine
large allocation requests; allocator reuse prevents inferring nine independent
underlying CUDA host allocations or their remaining resident bytes.

Mapped-host pointers are actually used in the prelude:
`operators/deepseek_v32/indexer/csrc/echo_indexer.cu:71` and
`operators/deepseek_v32/indexer/csrc/official_decode.cu:105` call
`cudaHostGetDevicePointer`. Other transport goes through the same pinned host
backing. The local allocator wrapper does not explicitly call HostRegister or
HostUnregister; its backend is PyTorch's host allocator.

Pool close drains and drops its tensors (`cache/sparse_token_pool.py:882-905`).
That local close operation does not flush PyTorch's host allocator. Installed
`torch/include/ATen/core/CachingHostAllocator.h:171-185,345-404` states and
implements putting freed blocks on cached free lists, while
`empty_cache` at line 454 releases available cached allocations.
`torch/include/ATen/cuda/CachingHostAllocator.h:12-20,40-44` explicitly describes
retaining freed pinned blocks to avoid `cudaFreeHost` synchronization and a
separate cache-empty operation.

The backend graph context adds a flush that the initial local-code review
missed. Installed `torch/cuda/graphs.py:246-259` calls
`torch.cuda.synchronize()`, `torch.cuda.empty_cache()` and
`torch._C._host_emptyCache()` in `torch.cuda.graph.__enter__`, before
`capture_begin`. Both `models/deepseek_v32/execution/compute_graphs.py:246,261`
and `execution/extend_graph.py:254` use this context. Consequently the fresh
measured HBM capture attempts to release available cached pinned blocks after
the offload pools have been closed. The prelude's freed backing cannot simply
be assumed to remain cached through measurement.

`CachingHostAllocator.h:454-470` processes completed events and frees available
blocks from the default pool and graph pools already eligible for release.
This does not free live owners or establish that every pending/private-pool
block is releasable. There is no retained evidence of free pinned pages
surviving at measurement, nor of exact pinned bytes, the actual HostAlloc
versus HostRegister calls, or a kernel-completion fence effect. No pinned
allocator statistics or full prelude allocation trace were saved. Reduced HBM
also performs CPU/GPU transfers and may have backend staging allocations.
The complete-prelude control remains valid, but retaining freed pinned backing
is a weaker explanation after accounting for the graph-entry flush.

## Available host allocator observations

Installed Torch 2.12.1 exposes `torch.cuda.host_memory_stats()` and
`host_memory_stats_as_nested_dict()` (`torch/cuda/memory.py:384-443`). Public
statistics return an empty dictionary before CUDA initialization. Their
documented core fields have `current`, `peak`, `allocated` and `freed` suffixes:

- `allocations`: blocks owned by the allocator, active plus cached.
- `allocated_bytes`: rounded bytes of all owned blocks, active plus cached.
  This is the closest host-allocator quantity to reserved bytes; it does not
  mean only bytes held by live tensors.
- `active_requests` and `active_bytes`: blocks/rounded bytes handed out and
  not yet returned to the reusable pool after tracked dependencies complete.
- `num_host_alloc` / `num_host_free`: pool-growth/free block counters;
  `host_alloc_time` / `host_free_time` expose `total`, `max`, `min`, `count`,
  and `avg` in microseconds. They do not identify which CUDA allocation API ran.

There is no documented unrounded `requested_bytes` or separate host
`reserved_bytes` metric. Keep logical tensor bytes, backing storage bytes and
allocator totals separate. Here each logical 75,571,200 B pool backing requests
the entire 134,217,728 B bin, so its storage/requested-bin capacity is already
rounded. `getStats()` does not process pending events to refresh availability
(`CachingHostAllocator.h:486-493`); peaks sum per-bucket peaks. These counters
alone are not an ownership or completed-IO proof.

The installed generic header also warrants checking active-counter behavior
before deriving cached bytes by subtraction: the no-event `free()` branch at
lines 384-388 adds a free-list block without decrementing active statistics,
and `process_events_for_specific_size(-1)` uses `decrease(size)` at line 727.
This is a source-level inconsistency with the documented active-byte meaning;
the compiled allocator was not exercised here. Initially retain reported
`allocated_bytes.current`, allocation/free counters and explicit owner/storage
evidence separately rather than asserting an exact active/cached split.

Public cache release is `torch.accelerator.empty_host_cache()`
(`torch/accelerator/memory.py:35-42`). The available lower-level functions are
`torch._C._accelerator_emptyHostCache()` and `torch._C._host_emptyCache()`;
the latter is already used by CUDA graph entry. C++ exposes
`at::getHostAllocator(at::kCUDA)->get_stats()` / `empty_cache()`. Statistics-only
resets are `torch.cuda.reset_accumulated_host_memory_stats()` and
`reset_peak_host_memory_stats()`. No reset, flush, allocation or stats query
was executed for this source inspection; callable names were checked after
importing Torch, and CUDA remained uninitialized.

The formal SQLite environment contains neither `PYTORCH_CUDA_ALLOC_CONF` nor
`PYTORCH_ALLOC_CONF`. Installed `c10/cuda/CUDAAllocatorConfig.h:132-151,210-219`
defaults `pinned_use_cuda_host_register` to false, register threads to one,
and reserved host segment size to zero. That selects the ordinary
`cudaHostAlloc` policy rather than the explicit host-registration option;
it is not a trace of the actual allocation calls. If a later control needs
effective settings, the existing allocator snapshot metadata exposes
`pinned_use_cuda_host_register`, `pinned_num_register_threads` and the settings
string (`cache/allocator/csrc/allocator_snapshot.cpp:15-22`); do not infer
runtime configuration solely from an absent environment variable.

## Streams and peer contexts

Both paths use the same `torch.cuda.Stream(device=...)` constructors without
priority or flag overrides: projection side streams in
`models/deepseek_v32/projections.py:98`, compute-bank capture stream in
`execution/compute_graphs.py:168`, and full-graph stream in
`execution/extend_graph.py:235`. Formal offload additionally requests pool copy
streams (`cache/sparse_token_pool.py:263`) and a dense helper copy stream
(`models/deepseek_v32/cache/prefetch.py:107`). PyTorch obtains streams from a
process-wide round-robin pool (`torch/include/c10/cuda/CUDAStream.h:12-40`), so
the longer prelude changes stream-allocation history and concrete stream IDs.

Retained trace metadata narrows this difference. Formal CUB `capture_4.sqlite`
has active kernel streams 7 and 168; reduced timer `capture.2.sqlite` has 7 and
158. Both have priority 0, a null stream with flag 3 and a nonblocking stream
with flag 2. These are CUPTI enum values, verified at
`/usr/local/cuda/include/cupti_activity.h:1771-1786`. Each selected trace records
one normal CUDA context on logical device 0, parent context 0, and no green
context. The metadata is saved in
`/tmp/cxldsagr-checks/q1_formal_reduced_stream_metadata.json`. Thus differing
active-stream priorities/blocking flags are not supported by these traces.

Both model calls use `devices=[0]`; their historical CUDA_VISIBLE_DEVICES
selects a single physical GPU (formal 0, reduced 1). No explicit peer-access
enable or device-map-host flag call exists in the inspected local model,
operator, and allocation paths. One recorded context during selected capture
does not audit every native library's context or peer history outside capture.

## What prior controls cover

`q1_compute_inspector.py:41` and `q1_compute_collection.py:45` explicitly retain
the unchanged all-four-method prelude. They change only initial compute-bank
inspection/collection and leave approximately 67.6 us idle / 416 ns median
gaps. Formal reproduction also retains that prelude. They do not rule out
allocation or offload-use history. Reduced event, replay-order, and outside
timer controls remain HBM-only and do not test presence versus absence of the
offload prelude. Their low gaps also had a different physical GPU and some
different JIT binaries, as already recorded in the matched-environment audit.

If the currently planned matched GPU0/environment reduced run remains low,
the next bounded control can vary only whether the same public four-method
warmup sequence precedes the fresh measured HBM cache/capture. Alternatively,
omit only the offload prelude before the formal first HBM capture. Retain the
same source/native payloads, request, compute bank, profiler settings, output
checks, graph signatures, and fresh-process order balance; check that changing
warmup coverage has not silently changed the loaded-runtime manifest.

That control isolates the offload prelude as a package. A positive effect would
still not distinguish pinned registration, GPU allocator/graph history,
stream-pool selection, or offload kernel use. A later allocation-only control
would be needed before attributing the approximately 320 ns difference to
system-memory visibility. Do not alter production fences, cache cleanup, or
warmup policy based on the current evidence.
