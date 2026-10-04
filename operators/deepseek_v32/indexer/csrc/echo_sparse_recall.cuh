#pragma once

namespace echo_native {

__global__ void sparse_union_clear(uint32_t* bitmap, int64_t words,
                                   uint32_t* count, uint32_t* miss_count) {
  for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
       i < words; i += int64_t(gridDim.x)*blockDim.x)
    bitmap[i] = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *count = 0;
    *miss_count = 0;
  }
}

// The logical bitmap is private to one exclusive operation and cleared before
// every union. Candidate bits count toward selection, but never translate to host.
template <bool LocalBitmap>
__global__ void sparse_union_kernel(const int* indices, int64_t elements,
                                    uint32_t* bitmap, int words, int written) {
  extern __shared__ uint32_t local_bitmap[];
  if constexpr (LocalBitmap) {
    for (int word = threadIdx.x; word < words; word += blockDim.x)
      local_bitmap[word] = 0;
    __syncthreads();
  }
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += int64_t(gridDim.x) * blockDim.x) {
    const int logical = indices[i];
    if (logical < 0) continue;
    resident_guard(logical < written);
    resident_claim_bit(LocalBitmap ? local_bitmap : bitmap, logical);
  }
  if constexpr (LocalBitmap) {
    __syncthreads();
    for (int word = threadIdx.x; word < words; word += blockDim.x)
      if (local_bitmap[word]) atomicOr(bitmap + word, local_bitmap[word]);
  }
}

__global__ void sparse_classify_kernel(
    const uint32_t* bitmap, const int* pages, const int* h2d, const int64_t* d2h,
    int host_capacity, int64_t* priority, int64_t* clock, int* flags, uint32_t* count,
    uint32_t* miss_count, unsigned long long* totals,
    int slots, int history, int written, int64_t timestamp) {
  uint32_t selected = 0, missed = 0;
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < max(slots, written); logical += gridDim.x * blockDim.x) {
    int miss = 0;
    if (logical < written && (bitmap[logical / 32] & (uint32_t(1) << (logical % 32)))) {
      ++selected;
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
    }
    if (logical < slots) flags[logical] = miss;
  }
  const uint32_t block_selected = resident_block_sum(selected);
  __syncthreads();
  const uint32_t block_missed = resident_block_sum(missed);
  if (threadIdx.x == 0) {
    const uint32_t old = atomicAdd(count, block_selected);
    atomicAdd(miss_count, block_missed);
    atomicAdd(totals, static_cast<unsigned long long>(block_selected));
    atomicAdd(totals + 1, static_cast<unsigned long long>(block_selected - block_missed));
    atomicMax(totals + 2, static_cast<unsigned long long>(old + block_selected));
    if (blockIdx.x == 0) {
      priority[0] = INT32_MAX;
      *clock = timestamp + 1;
    }
  }
}

void sparse_selection_classify(
    TensorView indices, TensorView pages, TensorView h2d, TensorView d2h,
    TensorView priority, TensorView clock, TensorView bitmap, TensorView count, TensorView miss_count,
    TensorView flags, TensorView totals, int64_t history, int64_t written,
    int64_t timestamp) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int64_t elements = indices.size(0) * indices.size(1);
  sparse_union_clear<<<std::min<int64_t>(128,(bitmap.size(0)+255)/256),256,0,stream>>>(
      static_cast<uint32_t*>(bitmap.data_ptr()), bitmap.size(0),
      static_cast<uint32_t*>(count.data_ptr()), static_cast<uint32_t*>(miss_count.data_ptr()));
  checked(cudaGetLastError());
  const int blocks = std::max<int64_t>(1,std::min<int64_t>(128,(elements+255)/256));
  const auto launch = [&]<bool LocalBitmap>() {
    sparse_union_kernel<LocalBitmap><<<blocks,256,LocalBitmap ? bitmap.size(0)*sizeof(uint32_t) : 0,stream>>>(
        static_cast<int*>(indices.data_ptr()), elements,
        static_cast<uint32_t*>(bitmap.data_ptr()), bitmap.size(0), written);
  };
  if (elements >= (1 << 20) && bitmap.size(0) <= 4096)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
  checked(cudaGetLastError());
  const int classify_blocks = std::max<int64_t>(1,std::min<int64_t>(128,(std::max<int64_t>(priority.size(0)-1,written)+255)/256));
  sparse_classify_kernel<<<classify_blocks,256,0,stream>>>(
      static_cast<uint32_t*>(bitmap.data_ptr()), static_cast<int*>(pages.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()), h2d.size(0),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
      static_cast<int*>(flags.data_ptr()),
      static_cast<uint32_t*>(count.data_ptr()), static_cast<uint32_t*>(miss_count.data_ptr()),
      static_cast<unsigned long long*>(totals.data_ptr()), priority.size(0)-1,
      history, written, timestamp);
  checked(cudaGetLastError());
}

__global__ void sparse_compact_kernel(
    const int64_t* prefix, const int* pages, int* h2d, int64_t* d2h,
    int64_t* priority, bool* free_bitmap, unsigned long long* evictions,
    int64_t* misses, int64_t* chosen, int host_capacity, int history, int slots, int count,
    int64_t timestamp) {
  uint32_t evicted = 0;
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < history; logical += gridDim.x * blockDim.x) {
    const int64_t end = prefix[logical], begin = logical ? prefix[logical-1] : 0;
    if (begin == end) continue;
    resident_guard(end == begin + 1 && begin >= 0 && begin < count);
    const int global = resident_global(logical, pages, host_capacity);
    resident_guard(h2d[global] == INT32_MAX);
    const int64_t slot = chosen[begin] + 1;
    resident_guard(slot > 0 && slot <= slots);
    misses[begin] = global;
    chosen[begin] = slot;
    resident_guard(priority[slot] < timestamp);
    const int64_t old = d2h[slot];
    if (old != INT32_MAX) {
      resident_guard(old >= 0 && old < host_capacity && h2d[old] == slot);
      h2d[old] = INT32_MAX;
      ++evicted;
    }
    // Tombstone before overwrite, including recoverable host-side failures
    // between the gather and publication launches.
    d2h[slot] = INT32_MAX;
    priority[slot] = -1;
    free_bitmap[slot] = true;
  }
  const uint32_t block_evicted = resident_block_sum(evicted);
  if (threadIdx.x == 0)
    atomicAdd(evictions, static_cast<unsigned long long>(block_evicted));
}

void sparse_selection_compact(
    TensorView prefix, TensorView pages, TensorView h2d, TensorView d2h,
    TensorView priority, TensorView free_bitmap, TensorView evictions,
    TensorView misses, TensorView chosen, int64_t history, int64_t timestamp) {
  const int gpu = h2d.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  sparse_compact_kernel<<<std::max<int64_t>(1,std::min<int64_t>(128,(history+255)/256)),256,0,stream>>>(
      static_cast<int64_t*>(prefix.data_ptr()), static_cast<int*>(pages.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
      static_cast<unsigned long long*>(evictions.data_ptr()), static_cast<int64_t*>(misses.data_ptr()),
      static_cast<int64_t*>(chosen.data_ptr()), h2d.size(0), history, priority.size(0)-1, misses.size(0), timestamp);
  checked(cudaGetLastError());
}

// Victims were tombstoned before the generic H2D gather. Publish only after
// that gather on the same stream; no other lease can observe partial records.
__global__ void sparse_publish_kernel(
    const int64_t* misses, const int64_t* chosen, int count, int* h2d, int64_t* d2h,
    int64_t* priority, bool* free_bitmap, int64_t* clock, int host_capacity, int slots,
    int64_t timestamp) {
  for (int rank = blockIdx.x * blockDim.x + threadIdx.x;
       rank < count; rank += gridDim.x * blockDim.x) {
    const int64_t global = misses[rank], slot = chosen[rank];
    resident_guard(global >= 0 && global < host_capacity && slot > 0 && slot <= slots);
    resident_guard(h2d[global] == INT32_MAX && d2h[slot] == INT32_MAX && free_bitmap[slot]);
    h2d[global] = slot;
    d2h[slot] = global;
    priority[slot] = timestamp + 1;
    free_bitmap[slot] = false;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    priority[0] = INT32_MAX;
    *clock = timestamp + 2;
  }
}

void sparse_selection_publish(
    TensorView misses, TensorView chosen, TensorView h2d, TensorView d2h,
    TensorView priority, TensorView free_bitmap, TensorView clock, int64_t timestamp) {
  const int gpu = h2d.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  sparse_publish_kernel<<<std::max<int64_t>(1,std::min<int64_t>(128,(misses.size(0)+255)/256)),256,0,stream>>>(
      static_cast<int64_t*>(misses.data_ptr()), static_cast<int64_t*>(chosen.data_ptr()),
      misses.size(0), static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
      static_cast<int64_t*>(clock.data_ptr()), h2d.size(0), priority.size(0)-1, timestamp);
  checked(cudaGetLastError());
}

__global__ void sparse_map_kernel(
    const int* indices, int* physical, int64_t elements, const int* pages, const int* h2d,
    int host_capacity, int slots, int written, int history) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += int64_t(gridDim.x) * blockDim.x) {
    const int logical = indices[i];
    int slot = -1;
    if (logical >= 0) {
      resident_guard(logical < written);
      if (logical >= history) {
        slot = slots + 1 + logical - history;
      } else {
        slot = h2d[resident_global(logical, pages, host_capacity)];
        resident_guard(slot > 0 && slot <= slots);
      }
    }
    physical[i] = slot;
  }
}

void sparse_selection_map(TensorView indices, TensorView physical, TensorView pages,
                           TensorView h2d, TensorView priority,
                           int64_t history, int64_t written) {
  const int gpu = h2d.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int64_t elements = indices.size(0)*indices.size(1);
  sparse_map_kernel<<<std::max<int64_t>(1,std::min<int64_t>(128,(elements+255)/256)),256,0,stream>>>(
      static_cast<int*>(indices.data_ptr()), static_cast<int*>(physical.data_ptr()), elements,
      static_cast<int*>(pages.data_ptr()), static_cast<int*>(h2d.data_ptr()), h2d.size(0),
      priority.size(0)-1, written, history);
  checked(cudaGetLastError());
}

} // namespace echo_native
