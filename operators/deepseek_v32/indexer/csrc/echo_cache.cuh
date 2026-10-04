#pragma once
#include "echo_helpers.cuh"

namespace echo_native {
// The layer lease excludes every other user of these maps until this kernel
// completes. A published map is metadata only, not a record-readiness signal.
// Slot rank is unique for the entire invocation, so no writer can reuse a
// physical slot while another warp is still copying its record.
__device__ __forceinline__ void claim_and_copy_warp(
    bool candidate, int host_id, uint32_t limit, int host_capacity,
    const int* sorted_slots, int64_t* allocation_log, uint32_t* counter,
    int64_t* stats, int* h2d, int64_t* d2h,
    const __nv_bfloat16* host, __nv_bfloat16* records,
    int* shared_host_ids, int* shared_slots) {
  constexpr uint32_t full_warp = 0xffffffffu;
  const int lane = threadIdx.x % 32;
  bool miss = false;
  if (candidate && host_id >= 0 && host_id < host_capacity)
    miss = atomicCAS(h2d + host_id, INT32_MAX, -1) == INT32_MAX;
  const uint32_t mask = __ballot_sync(full_warp, miss);
  if (!mask) return;
  const uint32_t count = __popc(mask);
  const int leader = __ffs(mask) - 1;
  uint32_t base = 0, granted = 0;
  if (lane == leader) {
    base = atomicAdd(counter, count);
    granted = base < limit ? min(limit - base, count) : 0;
    atomicAdd(reinterpret_cast<unsigned long long*>(stats + 2), count - granted);
  }
  base = __shfl_sync(full_warp, base, leader);
  granted = __shfl_sync(full_warp, granted, leader);
  const uint32_t lower = lane == 0 ? 0u : ((1u << lane) - 1);
  const uint32_t rank = __popc(mask & lower);
  if (miss) {
    if (rank < granted) {
      const int slot = sorted_slots[base + rank];
      const int64_t old_id = d2h[slot];
      // A plain store could erase a later claim for an evicted token. Clear
      // only this physical slot's ownership in the live global map.
      if (old_id >= 0 && old_id < host_capacity &&
          atomicCAS(h2d + old_id, slot, INT32_MAX) == slot)
        atomicAdd(reinterpret_cast<unsigned long long*>(stats + 1), 1ull);
      d2h[slot] = host_id;
      atomicExch(h2d + host_id, slot);
      allocation_log[slot] = host_id;
      shared_host_ids[rank] = host_id;
      shared_slots[rank] = slot;
    } else {
      atomicCAS(h2d + host_id, -1, INT32_MAX);
    }
  }
  __syncwarp(full_warp);
  #pragma unroll
  for (uint32_t i = 0; i < 32; ++i) {
    if (i >= granted) break;
    if (i + 1 < granted)
      prefetch_item_warp<1152>(host + size_t(shared_host_ids[i + 1]) * 576);
    transfer_mla_record_warp(host + size_t(shared_host_ids[i]) * 576,
                             records + size_t(shared_slots[i]) * 576);
  }
  __syncwarp(full_warp);
  if (lane == leader)
    atomicAdd(reinterpret_cast<unsigned long long*>(stats), granted);
}
}  // namespace echo_native
