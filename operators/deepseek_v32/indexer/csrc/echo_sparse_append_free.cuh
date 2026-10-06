#pragma once

// Included after echo_resident.cuh: reuse its guard and unchanged append kernel.
namespace echo_native::free_append {
using tvm::ffi::TensorView;

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr uint32_t FULL = 0xffffffffu;

__global__ void prepare_masks(const int64_t* priority, uint32_t* masks,
                              int slots, int64_t timestamp) {
  // Full-block iterations keep every lane in each ballot, including the tail.
  const int lane = threadIdx.x % 32;
  const int stride = gridDim.x * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < slots; base += stride) {
    const int64_t index = base + threadIdx.x;
    const bool actual = index < slots;
    const int64_t age = actual ? priority[index + 1] : timestamp;
    resident_guard(!actual || (age >= -1 && age <= timestamp));
    const uint32_t mask = __ballot_sync(FULL, actual && age == -1);
    if (lane == 0 && actual) masks[index / 32] = mask;
  }
}

__global__ void select_slots(const uint32_t* masks, const bool* free_bitmap,
                             const int64_t* d2h, int64_t* chosen,
                             int slots, int count) {
  __shared__ int warp_counts[WARPS];
  __shared__ int warp_selected[WARPS];
  __shared__ uint32_t selected_masks[THREADS];
  __shared__ int selected_bases[THREADS];
  __shared__ int selected_ranks[THREADS];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint32_t lower = lane ? (uint32_t(1) << lane) - 1 : 0;
  const int mask_count = (int64_t(slots) + 31) / 32;
  int emitted = 0;
  for (int base = 0; base < mask_count && emitted < count; base += THREADS) {
    const int index = base + threadIdx.x;
    const uint32_t mask = index < mask_count ? masks[index] : 0;
    const int available = __popc(mask);
    int prefix = available;
    for (int delta = 1; delta < 32; delta *= 2) {
      const int previous = __shfl_up_sync(FULL, prefix, delta);
      if (lane >= delta) prefix += previous;
    }
    if (lane == 31) warp_counts[warp] = prefix;
    __syncthreads();
    int warp_base = 0, tile_count = 0;
    for (int w = 0; w < WARPS; ++w) {
      if (w < warp) warp_base += warp_counts[w];
      tile_count += warp_counts[w];
    }
    const int rank = emitted + warp_base + prefix - available;
    const bool take = available != 0 && rank < count;
    const uint32_t selected = __ballot_sync(FULL, take);
    if (lane == 0) warp_selected[warp] = __popc(selected);
    __syncthreads();
    int selected_base = 0, selected_count = 0;
    for (int w = 0; w < WARPS; ++w) {
      if (w < warp) selected_base += warp_selected[w];
      selected_count += warp_selected[w];
    }
    if (take) {
      const int output = selected_base + __popc(selected & lower);
      selected_masks[output] = mask;
      selected_bases[output] = index * 32;
      selected_ranks[output] = rank;
    }
    __syncthreads();
    // One warp expands one contributing mask. This writes adjacent selected
    // ranks cooperatively even at count=P, and skips entirely occupied masks.
    for (int item = warp; item < selected_count; item += WARPS) {
      const uint32_t bits = selected_masks[item];
      const int output = selected_ranks[item] + __popc(bits & lower);
      if ((bits & (uint32_t(1) << lane)) && output < count) {
        const int slot = selected_bases[item] + lane + 1;
        resident_guard(slot > 0 && slot <= slots);
        resident_guard(free_bitmap[slot] && d2h[slot] == INT32_MAX);
        chosen[output] = slot;
      }
    }
    emitted += tile_count;
    // All users finish before shared lists/counts are overwritten next tile.
    __syncthreads();
  }
  resident_guard(emitted >= count);
}

void append(TensorView source, TensorView records, TensorView pages,
            TensorView h2d, TensorView d2h, TensorView priority,
            TensorView free_bitmap, TensorView clock, TensorView evictions,
            TensorView masks, TensorView chosen, int64_t start,
            int64_t row_bytes, int64_t timestamp) {
  const int gpu = h2d.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int64_t slots = priority.size(0) - 1;
  TVM_FFI_ICHECK(slots > 0 && slots < INT32_MAX);
  TVM_FFI_ICHECK(timestamp >= 0 && timestamp < INT32_MAX - 1);
  TVM_FFI_ICHECK(source.size(0) > 0 && source.size(0) <= slots);
  TVM_FFI_ICHECK(start >= 0 && start + source.size(0) <= slots);
  TVM_FFI_ICHECK(chosen.size(0) >= source.size(0));
  // Python retains the old P-int64 keys/sorted-keys shape and nonalias checks.
  TVM_FFI_ICHECK(masks.size(0) >= slots);
  auto mask_pointer = static_cast<uint32_t*>(masks.data_ptr());
  const int blocks = std::min<int64_t>(128, (slots + THREADS - 1) / THREADS);
  prepare_masks<<<blocks, THREADS, 0, stream>>>(
      static_cast<int64_t*>(priority.data_ptr()), mask_pointer, int(slots), timestamp);
  checked(cudaGetLastError());
  select_slots<<<1, THREADS, 0, stream>>>(mask_pointer,
      static_cast<bool*>(free_bitmap.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int64_t*>(chosen.data_ptr()), int(slots), int(source.size(0)));
  checked(cudaGetLastError());
  // Both metadata kernels must finish on this stream before unchanged copying
  // and publication. Asynchronous traps obey the existing let-crash boundary.
  planned_append(source, records, pages, chosen, h2d, d2h, priority, free_bitmap,
                 clock, evictions, start, row_bytes, timestamp);
}
}  // namespace echo_native::free_append
