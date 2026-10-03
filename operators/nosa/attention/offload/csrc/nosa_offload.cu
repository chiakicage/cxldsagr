// GPU initialization, suffix staging and first-use planning for NOSA offload.
// All host loads are owned by nosa_offload_fused.cu.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace nosa_offload {
using tvm::ffi::TensorView;

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Initialize {
  uint16_t *keys, *values;
  const uint16_t *suffix_keys, *suffix_values;
  int *first_use, *ready, *queue;
  int64_t *tile_bytes, *total_bytes;
  int64_t key_row, key_head, value_row, value_head;
  int64_t prefix, heads, page_zero_vectors, suffix_vectors;
  int64_t page_capacity, tile_capacity, work_items;
  bool vector_keys, vector_values;
};

__device__ __forceinline__ void copy_eight(uint16_t* destination,
                                         const uint16_t* source, bool vectorized) {
  if (vectorized) {
    *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
  } else {
    // Validation requires only unit innermost stride. Misaligned pointers and
    // arbitrary nonnegative row/head strides remain valid suffix inputs.
#pragma unroll
    for (int column = 0; column < 8; ++column) destination[column] = source[column];
  }
}

__global__ void nosa_initialize_stage_kernel(Initialize p) {
  for (int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       item < p.work_items; item += int64_t(gridDim.x) * blockDim.x) {
    if (item < p.page_zero_vectors) {
      // Only the historical part of page 0 is cleared. The suffix begins at
      // prefix, so page-zero clearing and suffix copies have disjoint writers.
      reinterpret_cast<uint4*>(p.keys)[item] = make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4*>(p.values)[item] = make_uint4(0, 0, 0, 0);
    }
    if (item < p.suffix_vectors) {
      int64_t query = item / (p.heads * 16);
      int64_t head = (item / 16) % p.heads;
      int64_t column = (item % 16) * 8;
      int64_t destination = (p.prefix * p.heads * 16 + item) * 8;
      copy_eight(p.keys + destination,
                 p.suffix_keys + query * p.key_row + head * p.key_head + column,
                 p.vector_keys);
      copy_eight(p.values + destination,
                 p.suffix_values + query * p.value_row + head * p.value_head + column,
                 p.vector_values);
    }
    if (item < p.page_capacity) {
      p.first_use[item] = INT_MAX;
      p.ready[item] = 0;
    }
    if (item < p.tile_capacity) p.tile_bytes[item] = 0;
    if (item < 2) p.queue[item] = 0;
    if (item == 0) *p.total_bytes = 0;
  }
}

struct Selection {
  const void* ids;
  const bool* valid;
  int64_t row, head, slot, mask_row, mask_head, mask_slot;
  bool ids64;
};

__global__ void nosa_first_use_plan_kernel(Selection selection, int* first_use,
                                         int64_t count, int heads, int budget,
                                         int64_t prefix, int tile_size) {
  int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= count) return;
  int slot = item % budget;
  int head = (item / budget) % heads;
  int query = item / (int64_t(heads) * budget);
  int64_t offset = query * selection.row + head * selection.head + slot * selection.slot;
  if (selection.valid && !selection.valid[query * selection.mask_row +
                                          head * selection.mask_head +
                                          slot * selection.mask_slot])
    return;
  int64_t block = selection.ids64 ? static_cast<const int64_t*>(selection.ids)[offset]
                                  : static_cast<const int32_t*>(selection.ids)[offset];
  // Every prefix token precedes every query. Suffix/future/padded blocks are
  // never fetched from the host; the current suffix already occupies HBM.
  if (block < 0 || block >= (prefix + 63) / 64) return;
  // FA3's odd-page padding physically reads page 0 with membership 0. If
  // page 0 is selected later, fetch it before the first attention to avoid a
  // concurrent padded read/write. Unselected page 0 is initialized before plan.
  atomicMin(first_use + block * heads + head, block == 0 ? 0 : query / tile_size);
}

void plan(TensorView ids, TensorView mask, TensorView first_use,
          int64_t queries, int64_t prefix, int64_t tile_size) {
  if (queries == 0 || prefix == 0) return;
  Selection selection{};
  selection.ids = ids.data_ptr();
  selection.valid = mask.numel() ? static_cast<const bool*>(mask.data_ptr()) : nullptr;
  selection.row = ids.size(0) == 1 ? 0 : ids.stride(0);
  selection.head = ids.size(1) == 1 ? 0 : ids.stride(1);
  selection.slot = ids.stride(2);
  if (selection.valid) {
    selection.mask_row = mask.size(0) == 1 ? 0 : mask.stride(0);
    selection.mask_head = mask.size(1) == 1 ? 0 : mask.stride(1);
    selection.mask_slot = mask.stride(2);
  }
  selection.ids64 = ids.dtype().bits == 64;
  int heads = first_use.size(1), budget = ids.size(2);
  int64_t count = queries * heads * budget;
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, first_use.device().device_id));
  nosa_first_use_plan_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      selection, static_cast<int*>(first_use.data_ptr()), count, heads, budget, prefix,
      int(tile_size));
  check_cuda(cudaGetLastError());
}

void prepare(TensorView keys, TensorView values,
             TensorView suffix_keys, TensorView suffix_values,
             TensorView first_use, TensorView ready, TensorView queue,
             TensorView tile_bytes, TensorView total_bytes,
             TensorView ids, TensorView mask, int64_t prefix, int64_t tile_size) {
  Initialize p{};
  p.keys = static_cast<uint16_t*>(keys.data_ptr());
  p.values = static_cast<uint16_t*>(values.data_ptr());
  p.suffix_keys = static_cast<const uint16_t*>(suffix_keys.data_ptr());
  p.suffix_values = static_cast<const uint16_t*>(suffix_values.data_ptr());
  p.first_use = static_cast<int*>(first_use.data_ptr());
  p.ready = static_cast<int*>(ready.data_ptr());
  p.queue = static_cast<int*>(queue.data_ptr());
  p.tile_bytes = static_cast<int64_t*>(tile_bytes.data_ptr());
  p.total_bytes = static_cast<int64_t*>(total_bytes.data_ptr());
  p.key_row = suffix_keys.stride(0);
  p.key_head = suffix_keys.stride(1);
  p.value_row = suffix_values.stride(0);
  p.value_head = suffix_values.stride(1);
  p.prefix = prefix;
  p.heads = first_use.size(1);
  p.page_zero_vectors = std::min(prefix, int64_t(64)) * p.heads * 16;
  p.suffix_vectors = suffix_keys.size(0) * p.heads * 16;
  p.page_capacity = first_use.numel();
  p.tile_capacity = tile_bytes.numel();
  p.work_items = std::max({p.page_zero_vectors, p.suffix_vectors, p.page_capacity,
                           p.tile_capacity, int64_t(2)});
  p.vector_keys = reinterpret_cast<uintptr_t>(p.suffix_keys) % 16 == 0 &&
                  p.key_row % 8 == 0 && p.key_head % 8 == 0;
  p.vector_values = reinterpret_cast<uintptr_t>(p.suffix_values) % 16 == 0 &&
                    p.value_row % 8 == 0 && p.value_head % 8 == 0;
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, first_use.device().device_id));
  int blocks = int(std::min((p.work_items + 255) / 256, int64_t(4096)));
  nosa_initialize_stage_kernel<<<blocks, 256, 0, stream>>>(p);
  check_cuda(cudaGetLastError());
  // A distinct launch supplies the global dependency between initialization
  // and atomicMin. Empty calls still execute initialization and reset counters.
  plan(ids, mask, first_use, suffix_keys.size(0), prefix, tile_size);
}

}  // namespace nosa_offload

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, nosa_offload::plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, nosa_offload::prepare);
