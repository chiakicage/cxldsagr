# C7 dense prefetch task contract

Status: C7a passed independent source review, GPU correctness, exclusive
component timing and combined checkpoint/lifecycle validation with the exact
ECHO hint. Formal full-workload performance remains pending. C6 performance
acceptance uses its frozen source copy.

Optimize fixed P/NH DeepSeek PoolHistoryPrefetch without changing whole-history
semantics: H <= P, every initialized historical record is requested, only actual
misses are copied into existing per-layer pool slots, candidate tails are not
read/written by prefetch, and no full-layer staging buffer is added. Preserve
stable FIFO events, exact maps, owner/page identity, map_generation, dense metrics,
consumer selection metrics, wait(ticket), same-caller-stream and drain/rollback/
release contracts. The generic budget-mode double staging path is outside C7.

C7a target: replace checked whole-range/miss/allocation metadata with native
preparation while retaining the existing generic transport. C7b is a separately
measured candidate, if needed: bounded copy CTAs may improve overlap with the
current layer's matrix work, but no scheduling or latency benefit is assumed.

Hardware: Hopper SM90, current measured GPU NVIDIA H200 with 132 SMs. Record
width/dtype remain caller supplied; current workload uses BF16 x576. Correctness
precedes component timing, which precedes full-model/trajectory promotion. Root
owns report publication and affected-experiment replacement. Engineering probes
remain in /tmp and do not replace experiment results.
