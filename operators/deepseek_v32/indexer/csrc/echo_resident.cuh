#pragma once

namespace echo_native {

// These kernels consume an exclusive pool lease and the host residency proof.
// Violating that private contract must fail instead of publishing invalid IDs.
__device__ __forceinline__ void resident_guard(bool valid) {
  if (!valid) asm volatile("trap;");
}

__device__ __forceinline__ uint32_t resident_block_sum(uint32_t value) {
  __shared__ uint32_t warp_totals[8];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int offset = 16; offset; offset /= 2)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  if (lane == 0) warp_totals[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warp_totals[lane] : 0;
  if (warp == 0)
    for (int offset = 16; offset; offset /= 2)
      value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ int resident_global(
    int logical, const int* pages, int host_capacity) {
  const int64_t global = int64_t(pages[logical / 64]) * 64 + logical % 64;
  resident_guard(global >= 0 && global < host_capacity);
  return int(global);
}

__global__ void resident_union_clear(uint32_t* bitmap, int64_t words, uint32_t* count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += int64_t(gridDim.x) * blockDim.x)
    bitmap[i] = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) *count = 0;
}

__device__ __forceinline__ bool resident_claim_bit(uint32_t* bitmap, int slot) {
  const int lane = threadIdx.x % 32;
  const uint32_t bit = uint32_t(1) << (slot % 32);
  const uint32_t group = __match_any_sync(__activemask(), slot / 32);
  if (group == (uint32_t(1) << lane))
    return !(atomicOr(bitmap + slot / 32, bit) & bit);
  // Adjacent selected IDs often share a bitmap word. Publish all of this
  // subgroup's bits in one atomic, then nominate one lane per newly set bit.
  const int leader = __ffs(group) - 1;
  const uint32_t bits = __reduce_or_sync(group, bit);
  uint32_t old = 0;
  if (lane == leader) old = atomicOr(bitmap + slot / 32, bits);
  old = __shfl_sync(group, old, leader);
  const uint32_t same_slot = __match_any_sync(group, slot);
  return !(old & bit) && lane == __ffs(same_slot) - 1;
}

template <bool LocalBitmap>
__global__ void resident_selection_kernel(
    const int* indices, int* physical, int64_t elements, const int* pages,
    const int* h2d, int host_capacity, int64_t* priority, int64_t* clock,
    uint32_t* bitmap, uint32_t* count, unsigned long long* totals,
    int written, int history, bool transient, int slots, int bitmap_words,
    int64_t timestamp) {
  extern __shared__ uint32_t local_bitmap[];
  if constexpr (LocalBitmap) {
    for (int word = threadIdx.x; word < bitmap_words; word += blockDim.x)
      local_bitmap[word] = 0;
    __syncthreads();
  }
  uint32_t unique = 0;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += int64_t(gridDim.x) * blockDim.x) {
    const int logical = indices[i];
    if (logical < 0) {
      physical[i] = -1;
      continue;
    }
    resident_guard(logical < written);
    int slot;
    const bool suffix = transient && logical >= history;
    if (suffix) {
      slot = slots + 1 + logical - history;
    } else {
      slot = h2d[resident_global(logical, pages, host_capacity)];
      resident_guard(slot > 0 && slot <= slots);
    }
    physical[i] = slot;
    if constexpr (LocalBitmap) {
      resident_claim_bit(local_bitmap, slot);
    } else {
      if (resident_claim_bit(bitmap, slot)) {
        ++unique;
        if (!suffix)
          atomicExch(reinterpret_cast<unsigned long long*>(priority + slot),
                     static_cast<unsigned long long>(timestamp));
      }
    }
  }
  if constexpr (LocalBitmap) {
    __syncthreads();
    for (int word = threadIdx.x; word < bitmap_words; word += blockDim.x) {
      const uint32_t bits = local_bitmap[word];
      if (!bits) continue;
      uint32_t claimed = bits & ~atomicOr(bitmap + word, bits);
      unique += __popc(claimed);
      // One CTA wins each physical bit. Only that winner refreshes priority;
      // candidate bits participate in the union but own no history metadata.
      while (claimed) {
        const int bit = __ffs(claimed) - 1;
        const int slot = word * 32 + bit;
        if (slot <= slots) priority[slot] = timestamp;
        claimed &= claimed - 1;
      }
    }
  }
  const uint32_t block_unique = resident_block_sum(unique);
  if (threadIdx.x == 0) {
    const uint32_t old = atomicAdd(count, block_unique);
    atomicAdd(totals, static_cast<unsigned long long>(block_unique));
    atomicAdd(totals + 1, static_cast<unsigned long long>(block_unique));
    // All contributions are nonnegative. The final nonzero endpoint equals
    // the exact union, so no separate finalization launch is necessary.
    atomicMax(totals + 2, static_cast<unsigned long long>(old + block_unique));
    if (blockIdx.x == 0) {
      priority[0] = INT32_MAX;
      *clock = timestamp + 2;
    }
  }
}

void resident_selection(
    TensorView indices, TensorView physical, TensorView pages, TensorView h2d,
    TensorView priority, TensorView clock, TensorView bitmap, TensorView count,
    TensorView totals, int64_t written, int64_t history, bool transient,
    int64_t candidate_slots, int64_t timestamp) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int64_t elements = indices.size(0) * indices.size(1);
  resident_union_clear<<<std::min<int64_t>(128, (bitmap.size(0)+255)/256),256,0,stream>>>(
      static_cast<uint32_t*>(bitmap.data_ptr()), bitmap.size(0),
      static_cast<uint32_t*>(count.data_ptr()));
  checked(cudaGetLastError());
  const int blocks = std::max<int64_t>(1, std::min<int64_t>(128, (elements+255)/256));
  const auto launch = [&]<bool LocalBitmap>() {
    const size_t shared_bytes = LocalBitmap ? bitmap.size(0) * sizeof(uint32_t) : 0;
    resident_selection_kernel<LocalBitmap><<<blocks,256,shared_bytes,stream>>>(
        static_cast<int*>(indices.data_ptr()), static_cast<int*>(physical.data_ptr()), elements,
        static_cast<int*>(pages.data_ptr()), static_cast<int*>(h2d.data_ptr()), h2d.size(0),
        static_cast<int64_t*>(priority.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
        static_cast<uint32_t*>(bitmap.data_ptr()), static_cast<uint32_t*>(count.data_ptr()),
        static_cast<unsigned long long*>(totals.data_ptr()), written, history, transient,
        priority.size(0)-1, bitmap.size(0), timestamp);
  };
  if (elements >= (1 << 20) && bitmap.size(0) <= 4096)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
  checked(cudaGetLastError());
}

template <typename Word>
__global__ void planned_append_kernel(
    const char* source, char* records, int count, int64_t row_bytes,
    const int* pages, const int64_t* chosen, int* h2d, int64_t* d2h,
    int host_capacity, int slots, int64_t* priority, bool* free_bitmap,
    int64_t* clock, unsigned long long* evictions, int start, int64_t timestamp) {
  const int lane = threadIdx.x % 32;
  uint32_t evicted = 0;
  for (int row = blockIdx.x * 8 + threadIdx.x / 32; row < count; row += gridDim.x * 8) {
    int slot = 0;
    if (lane == 0) {
      slot = int(chosen[row]);
      resident_guard(slot > 0 && slot <= slots);
      const int global = resident_global(start + row, pages, host_capacity);
      const int64_t old = d2h[slot];
      resident_guard(h2d[global] == INT32_MAX);
      if (old != INT32_MAX) {
        resident_guard(old >= 0 && old < host_capacity && h2d[old] == slot);
        h2d[old] = INT32_MAX;
        ++evicted;
      }
      h2d[global] = slot;
      d2h[slot] = global;
      priority[slot] = timestamp;
      free_bitmap[slot] = false;
    }
    slot = __shfl_sync(0xffffffffu, slot, 0);
    const auto* src = reinterpret_cast<const Word*>(source + int64_t(row) * row_bytes);
    auto* dst = reinterpret_cast<Word*>(records + int64_t(slot) * row_bytes);
    for (int64_t column = lane; column < row_bytes / sizeof(Word); column += 32)
      dst[column] = src[column];
  }
  const uint32_t block_evicted = resident_block_sum(evicted);
  if (threadIdx.x == 0) {
    atomicAdd(evictions, static_cast<unsigned long long>(block_evicted));
    if (blockIdx.x == 0) {
      priority[0] = INT32_MAX;
      *clock = timestamp + 1;
    }
  }
}

void planned_append(
    TensorView source, TensorView records, TensorView pages, TensorView chosen,
    TensorView h2d, TensorView d2h, TensorView priority, TensorView free_bitmap,
    TensorView clock, TensorView evictions, int64_t start, int64_t row_bytes,
    int64_t timestamp) {
  const int gpu = records.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int blocks = std::min<int64_t>(128, (source.size(0)+7)/8);
  const auto launch = [&]<typename Word>() {
    planned_append_kernel<Word><<<blocks,256,0,stream>>>(
        static_cast<char*>(source.data_ptr()), static_cast<char*>(records.data_ptr()),
        source.size(0), row_bytes, static_cast<int*>(pages.data_ptr()),
        static_cast<int64_t*>(chosen.data_ptr()), static_cast<int*>(h2d.data_ptr()),
        static_cast<int64_t*>(d2h.data_ptr()), h2d.size(0), priority.size(0)-1,
        static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
        static_cast<int64_t*>(clock.data_ptr()), static_cast<unsigned long long*>(evictions.data_ptr()),
        start, timestamp);
  };
  const auto alignment = reinterpret_cast<uintptr_t>(source.data_ptr()) |
                         reinterpret_cast<uintptr_t>(records.data_ptr()) | row_bytes;
  if (!(alignment % 16)) launch.template operator()<uint4>();
  else if (!(alignment % 4)) launch.template operator()<uint32_t>();
  else launch.template operator()<uint8_t>();
  checked(cudaGetLastError());
}

__global__ void protect_resident_history_kernel(
    const int* pages, const int* h2d, int host_capacity, int64_t* priority,
    int64_t* clock, int history, int slots, int64_t timestamp) {
  for (int64_t logical = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       logical < history; logical += int64_t(gridDim.x) * blockDim.x) {
    const int slot = h2d[resident_global(logical, pages, host_capacity)];
    resident_guard(slot > 0 && slot <= slots);
    priority[slot] = timestamp;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    priority[0] = INT32_MAX;
    *clock = timestamp + 2;
  }
}

void protect_resident_history(TensorView pages, TensorView h2d, TensorView priority,
                              TensorView clock, int64_t history, int64_t timestamp) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  protect_resident_history_kernel<<<std::max<int64_t>(1,std::min<int64_t>(128,(history+255)/256)),256,0,stream>>>(
      static_cast<int*>(pages.data_ptr()), static_cast<int*>(h2d.data_ptr()), h2d.size(0),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
      history, priority.size(0)-1, timestamp);
  checked(cudaGetLastError());
}

} // namespace echo_native
