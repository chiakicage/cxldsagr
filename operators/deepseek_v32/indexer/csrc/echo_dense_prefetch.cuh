#pragma once

namespace echo_native {

__global__ void dense_miss_count_clear(uint32_t* count) {
  *count = 0;
}

__global__ void dense_history_classify_kernel(
    const int* pages, const int* h2d, const int64_t* d2h, int host_capacity,
    int64_t* priority, int64_t* clock, int* flags, uint32_t* miss_count,
    int history, int slots, int64_t timestamp) {
  uint32_t missed = 0;
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < slots; logical += gridDim.x * blockDim.x) {
    int miss = 0;
    if (logical < history) {
      const int global = resident_global(logical, pages, host_capacity);
      const int slot = h2d[global];
      if (slot == INT32_MAX) {
        miss = 1;
        ++missed;
      } else {
        resident_guard(slot > 0 && slot <= slots && d2h[slot] == global);
        priority[slot] = timestamp;
      }
    }
    flags[logical] = miss;
  }
  const uint32_t block_missed = resident_block_sum(missed);
  if (threadIdx.x == 0) {
    atomicAdd(miss_count, block_missed);
    if (blockIdx.x == 0) {
      priority[0] = INT32_MAX;
      *clock = timestamp + 1;
    }
  }
}

void dense_history_classify(
    TensorView pages, TensorView h2d, TensorView d2h, TensorView priority,
    TensorView clock, TensorView flags, TensorView miss_count,
    int64_t history, int64_t timestamp) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  dense_miss_count_clear<<<1,1,0,stream>>>(static_cast<uint32_t*>(miss_count.data_ptr()));
  checked(cudaGetLastError());
  dense_history_classify_kernel<<<std::max<int64_t>(1,std::min<int64_t>(128,(priority.size(0)-1+255)/256)),256,0,stream>>>(
      static_cast<int*>(pages.data_ptr()), static_cast<int*>(h2d.data_ptr()),
      static_cast<int64_t*>(d2h.data_ptr()), h2d.size(0),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
      static_cast<int*>(flags.data_ptr()), static_cast<uint32_t*>(miss_count.data_ptr()),
      history, priority.size(0)-1, timestamp);
  checked(cudaGetLastError());
}

// Dense tickets reserve maps on the caller before asynchronous copies. The
// execution owner must join ticket readiness before touching the target layer.
// ID outputs own their storage; shared prefix and temporary sort cannot escape.
__global__ void dense_history_reserve_kernel(
    const int64_t* prefix, const int64_t* victims, const int* pages, int* h2d,
    int64_t* d2h, int64_t* priority, bool* free_bitmap, int64_t* clock,
    unsigned long long* evictions, int64_t* misses, int64_t* chosen,
    int host_capacity, int history, int slots, int count, int64_t timestamp) {
  uint32_t evicted = 0;
  if (count) {
    for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
         logical < history; logical += gridDim.x * blockDim.x) {
      const int64_t end = prefix[logical], begin = logical ? prefix[logical-1] : 0;
      if (begin == end) continue;
      resident_guard(end == begin + 1 && begin >= 0 && begin < count);
      const int global = resident_global(logical, pages, host_capacity);
      const int64_t slot = victims[begin] + 1;
      resident_guard(slot > 0 && slot <= slots && h2d[global] == INT32_MAX);
      resident_guard(priority[slot] < timestamp);
      const int64_t old = d2h[slot];
      if (old != INT32_MAX) {
        resident_guard(old >= 0 && old < host_capacity && h2d[old] == slot);
        h2d[old] = INT32_MAX;
        ++evicted;
      }
      // Every new global ID was a miss, so it cannot be another victim's old
      // ID. These unique per-rank map updates therefore do not race each other.
      d2h[slot] = global;
      h2d[global] = slot;
      priority[slot] = timestamp + 1;
      free_bitmap[slot] = false;
      misses[begin] = global;
      chosen[begin] = slot;
    }
  }
  const uint32_t block_evicted = resident_block_sum(evicted);
  if (threadIdx.x == 0) {
    atomicAdd(evictions, static_cast<unsigned long long>(block_evicted));
    if (blockIdx.x == 0) {
      priority[0] = INT32_MAX;
      *clock = timestamp + 2;
    }
  }
}

void dense_history_reserve(
    TensorView prefix, TensorView victims, TensorView pages, TensorView h2d,
    TensorView d2h, TensorView priority, TensorView free_bitmap, TensorView clock,
    TensorView evictions, TensorView misses, TensorView chosen,
    int64_t history, int64_t timestamp) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int blocks = misses.size(0) ? std::max<int64_t>(1,std::min<int64_t>(128,(history+255)/256)) : 1;
  dense_history_reserve_kernel<<<blocks,256,0,stream>>>(
      static_cast<int64_t*>(prefix.data_ptr()), static_cast<int64_t*>(victims.data_ptr()),
      static_cast<int*>(pages.data_ptr()), static_cast<int*>(h2d.data_ptr()),
      static_cast<int64_t*>(d2h.data_ptr()), static_cast<int64_t*>(priority.data_ptr()),
      static_cast<bool*>(free_bitmap.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
      static_cast<unsigned long long*>(evictions.data_ptr()), static_cast<int64_t*>(misses.data_ptr()),
      static_cast<int64_t*>(chosen.data_ptr()), h2d.size(0), history, priority.size(0)-1,
      misses.size(0), timestamp);
  checked(cudaGetLastError());
}

} // namespace echo_native
