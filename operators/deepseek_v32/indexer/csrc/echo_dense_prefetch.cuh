#pragma once

namespace echo_native {

// Clearing and publication are distinct launches. A permutation of the same
// session's previous slots must never race a new h2d write with an old clear.
__global__ void dense_history_clear_kernel(
    int* h2d, int64_t* d2h, int64_t* priority, bool* free_bitmap,
    unsigned long long* evictions, int host_capacity, int slots,
    int history, int host_start) {
  uint32_t evicted = 0;
  for (int slot = blockIdx.x * blockDim.x + threadIdx.x + 1;
       slot <= slots; slot += gridDim.x * blockDim.x) {
    const int64_t old = d2h[slot];
    const bool incoming = old >= host_start && old < int64_t(host_start) + history;
    if (slot > history && !incoming) continue;
    if (old != INT32_MAX) {
      resident_guard(old >= 0 && old < host_capacity && h2d[old] == slot);
      h2d[old] = INT32_MAX;
      // Relocating an incoming record is not an eviction of another history.
      evicted += !incoming;
    }
    d2h[slot] = INT32_MAX;
    priority[slot] = -1;
    free_bitmap[slot] = true;
  }
  const uint32_t block_evicted = resident_block_sum(evicted);
  if (threadIdx.x == 0)
    atomicAdd(evictions, static_cast<unsigned long long>(block_evicted));
}

void dense_history_clear(
    TensorView h2d, TensorView d2h, TensorView priority, TensorView free_bitmap,
    TensorView evictions, int64_t history, int64_t host_start) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int blocks = std::max<int64_t>(1, std::min<int64_t>(128, (priority.size(0)-1+255)/256));
  dense_history_clear_kernel<<<blocks,256,0,stream>>>(
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
      static_cast<unsigned long long*>(evictions.data_ptr()), h2d.size(0),
      priority.size(0)-1, history, host_start);
  checked(cudaGetLastError());
}

__global__ void dense_history_publish_kernel(
    const int* pages, int* h2d, int64_t* d2h, int64_t* priority,
    bool* free_bitmap, int64_t* clock, int host_capacity, int history,
    int host_start, int64_t timestamp) {
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < history; logical += gridDim.x * blockDim.x) {
    const int global = host_start + logical;
    const int slot = logical + 1;
    resident_guard(resident_global(logical, pages, host_capacity) == global);
    resident_guard(h2d[global] == INT32_MAX && d2h[slot] == INT32_MAX);
    h2d[global] = slot;
    d2h[slot] = global;
    priority[slot] = timestamp;
    free_bitmap[slot] = false;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    priority[0] = INT32_MAX;
    free_bitmap[0] = false;
    *clock = timestamp + 1;
  }
}

void dense_history_publish(
    TensorView pages, TensorView h2d, TensorView d2h, TensorView priority,
    TensorView free_bitmap, TensorView clock, int64_t history,
    int64_t host_start, int64_t timestamp) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int blocks = std::max<int64_t>(1, std::min<int64_t>(128, (history+255)/256));
  dense_history_publish_kernel<<<blocks,256,0,stream>>>(
      static_cast<int*>(pages.data_ptr()), static_cast<int*>(h2d.data_ptr()),
      static_cast<int64_t*>(d2h.data_ptr()), static_cast<int64_t*>(priority.data_ptr()),
      static_cast<bool*>(free_bitmap.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
      h2d.size(0), history, host_start, timestamp);
  checked(cudaGetLastError());
}

} // namespace echo_native
