#pragma once

#include <cstdint>
#include <climits>

// Sole-session bounded recall has enough free slots; no occupied slot is selected.
namespace echo_native::free_recall {
using tvm::ffi::TensorView;
constexpr uint32_t FULL = 0xffffffffu;
constexpr int THREADS = 256;
constexpr int HEADER_WORDS = 8;
// Header: exact M, free count, threshold age+1, # below threshold,
// emitted misses, emitted below-threshold slots, attempted threshold ties.
__device__ void guard(bool value) { if (!value) asm volatile("trap;"); }
void checked(cudaError_t error) { TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error); }

__device__ uint64_t sum_block(uint64_t value) {
  __shared__ uint64_t values[8];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int delta = 16; delta; delta /= 2) value += __shfl_down_sync(FULL, value, delta);
  if (lane == 0) values[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? values[lane] : 0;
  if (warp == 0)
    for (int delta = 16; delta; delta /= 2) value += __shfl_down_sync(FULL, value, delta);
  return value;
}

__global__ void count_flags_and_free(const int* flags, const int64_t* priority,
                                      uint64_t* scratch, int slots, int history,
                                      int64_t timestamp) {
  uint32_t misses = 0, free = 0;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < slots;
       index += gridDim.x * blockDim.x) {
    const int64_t age = priority[index + 1];
    guard(age >= -1 && age <= timestamp);
    free += age == -1;
    if (index < history) { guard(flags[index] == 0 || flags[index] == 1); misses += flags[index]; }
  }
  const uint64_t sum = sum_block((uint64_t(misses) << 32) | free);
  if (threadIdx.x == 0) scratch[HEADER_WORDS + blockIdx.x] = sum;
}

__global__ void prepare_free_compaction(uint64_t* scratch, int count_blocks, int slots) {
  const int tid = threadIdx.x;
  const uint64_t part = tid < count_blocks ? scratch[HEADER_WORDS + tid] : 0;
  const uint64_t counts = sum_block(part);
  if (tid == 0) {
    const uint32_t total = uint32_t(counts >> 32), free_count = uint32_t(counts);
    guard(total <= uint32_t(slots) && free_count >= total);
    scratch[0] = total; scratch[1] = free_count;
    scratch[2] = scratch[3] = scratch[4] = scratch[5] = scratch[6] = 0;
  }
}

__device__ uint32_t warp_append(bool selected, unsigned long long* counter) {
  const uint32_t mask = __ballot_sync(FULL, selected);
  if (!mask) return UINT32_MAX;
  const int lane = threadIdx.x % 32, leader = __ffs(mask) - 1;
  uint32_t base = 0;
  if (lane == leader) base = uint32_t(atomicAdd(counter, uint64_t(__popc(mask))));
  base = __shfl_sync(FULL, base, leader);
  const uint32_t lower = lane ? (uint32_t(1) << lane) - 1 : 0;
  return selected ? base + __popc(mask & lower) : UINT32_MAX;
}

__global__ void compact_and_tombstone(
    const int* flags, const int* pages, int* h2d, int64_t* d2h,
    int64_t* priority, bool* free_bitmap, unsigned long long* evictions,
    int64_t* misses, int64_t* chosen, uint64_t* scratch,
    int host_capacity, int slots, int history, int64_t timestamp) {
  const uint32_t total = scratch[0];
  if (!total) return;
  // The prepared threshold is always zero (age -1), so only free slots qualify.
  const uint32_t threshold = scratch[2], below = scratch[3], ties = total - below;
  uint32_t evicted = 0;
  const int stride = gridDim.x * blockDim.x;
  // Round to a full block so every warp lane reaches each ballot.
  const int64_t stop = ((int64_t(slots) + stride - 1) / stride) * stride;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < stop; index += stride) {
    const bool actual = index < slots;
    const bool missing = actual && index < history && flags[index] != 0;
    const uint32_t miss_rank = warp_append(missing, reinterpret_cast<unsigned long long*>(scratch + 4));
    if (missing) {
      guard(miss_rank < total);
      const int64_t global = int64_t(pages[index / 64]) * 64 + index % 64;
      guard(global >= 0 && global < host_capacity && h2d[global] == INT32_MAX);
      misses[miss_rank] = global;
    }
    const int64_t age = actual ? priority[index + 1] : timestamp;
    const uint32_t key = age < timestamp ? uint32_t(age + 1) : UINT32_MAX;
    const uint32_t low_rank = warp_append(actual && key < threshold,
                                         reinterpret_cast<unsigned long long*>(scratch + 5));
    const uint32_t tie_rank = warp_append(actual && key == threshold,
                                         reinterpret_cast<unsigned long long*>(scratch + 6));
    const bool take_low = low_rank != UINT32_MAX;
    const bool take_tie = tie_rank < ties;
    if (take_low || take_tie) {
      const uint32_t rank = take_low ? low_rank : below + tie_rank;
      guard(rank < total && age < timestamp);
      const int slot = index + 1;
      chosen[rank] = slot;
      const int64_t old = d2h[slot];
      if (old != INT32_MAX) {
        guard(old >= 0 && old < host_capacity && h2d[old] == slot);
        h2d[old] = INT32_MAX;
        ++evicted;
      }
      d2h[slot] = INT32_MAX;
      priority[slot] = -1;
      free_bitmap[slot] = true;
    }
  }
  const uint64_t sum = sum_block(evicted);
  if (threadIdx.x == 0) atomicAdd(evictions, sum);
}

void allocate(TensorView flags, TensorView prefix, TensorView pages, TensorView h2d,
              TensorView d2h, TensorView priority, TensorView free_bitmap, TensorView evictions,
              TensorView misses, TensorView chosen, TensorView workspace,
              int64_t history, int64_t timestamp) {
  const int gpu = h2d.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int slots = priority.size(0) - 1;
  const int blocks = std::min<int64_t>(128, (int64_t(slots) + THREADS - 1) / THREADS);
  TVM_FFI_ICHECK(timestamp >= 0 && timestamp < INT32_MAX - 1);
  TVM_FFI_ICHECK(workspace.size(0) >= int64_t(HEADER_WORDS + blocks) * 8);
  auto scratch = static_cast<uint64_t*>(workspace.data_ptr());
  count_flags_and_free<<<blocks, THREADS, 0, stream>>>(static_cast<int*>(flags.data_ptr()),
      static_cast<int64_t*>(priority.data_ptr()), scratch, slots, history, timestamp);
  checked(cudaGetLastError());
  prepare_free_compaction<<<1, THREADS, 0, stream>>>(scratch, blocks, slots);
  checked(cudaGetLastError());
  compact_and_tombstone<<<blocks, THREADS, 0, stream>>>(
      static_cast<int*>(flags.data_ptr()), static_cast<int*>(pages.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
      static_cast<unsigned long long*>(evictions.data_ptr()),
      static_cast<int64_t*>(misses.data_ptr()), static_cast<int64_t*>(chosen.data_ptr()),
      scratch, h2d.size(0), slots, history, timestamp);
  checked(cudaGetLastError());
}
}  // namespace echo_native::free_recall
