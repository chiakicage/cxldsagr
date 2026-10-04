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
  const int64_t* cache_tags;
  int64_t cache_owner, tag_count;
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
      const int head = (item / 16) % p.heads;
      if (!p.cache_tags || p.prefix < 64 || p.cache_tags[head] != p.cache_owner) {
        reinterpret_cast<uint4*>(p.keys)[item] = make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(p.values)[item] = make_uint4(0, 0, 0, 0);
      }
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
      p.ready[item] = p.cache_tags && item < p.tag_count &&
                             item / p.heads < p.prefix / 64 &&
                             p.cache_tags[item] == p.cache_owner ? 8 : 0;
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

void prepare_impl(TensorView keys, TensorView values,
             TensorView suffix_keys, TensorView suffix_values,
             TensorView first_use, TensorView ready, TensorView queue,
             TensorView tile_bytes, TensorView total_bytes,
             TensorView ids, TensorView mask, int64_t prefix, int64_t tile_size,
             const int64_t* cache_tags, int64_t cache_owner, int64_t tag_count) {
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
  p.cache_tags = cache_tags;
  p.cache_owner = cache_owner;
  p.tag_count = tag_count;
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

void prepare(TensorView keys, TensorView values,
             TensorView suffix_keys, TensorView suffix_values,
             TensorView first_use, TensorView ready, TensorView queue,
             TensorView tile_bytes, TensorView total_bytes,
             TensorView ids, TensorView mask, int64_t prefix, int64_t tile_size) {
  prepare_impl(keys, values, suffix_keys, suffix_values, first_use, ready, queue,
               tile_bytes, total_bytes, ids, mask, prefix, tile_size, nullptr, 0, 0);
}

void prepare_cached(TensorView keys, TensorView values,
             TensorView suffix_keys, TensorView suffix_values,
             TensorView first_use, TensorView ready, TensorView queue,
             TensorView tile_bytes, TensorView total_bytes,
             TensorView ids, TensorView mask, int64_t prefix, int64_t tile_size,
             TensorView tags, int64_t owner) {
  prepare_impl(keys, values, suffix_keys, suffix_values, first_use, ready, queue,
               tile_bytes, total_bytes, ids, mask, prefix, tile_size,
               static_cast<const int64_t*>(tags.data_ptr()), owner, tags.numel());
}

__global__ void publish_cache_kernel(int64_t* tags, const int* first_use,
                                    int64_t count, int heads, int64_t owner,
                                    int64_t prefix, int64_t end) {
  const int64_t slot = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= count) return;
  const int64_t block = slot / heads;
  // Suffix writes invalidate previous owners, including a partial prefix page.
  if (block >= prefix / 64 && block < (end + 63) / 64) tags[slot] = 0;
  else if (block < prefix / 64 && first_use[slot] != INT_MAX) tags[slot] = owner;
  else if (block == 0 && prefix && tags[slot] != owner) tags[slot] = 0;
}

void publish_cache(TensorView tags, TensorView first_use, int64_t owner,
                   int64_t prefix, int64_t end) {
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, tags.device().device_id));
  publish_cache_kernel<<<(tags.numel() + 255) / 256, 256, 0, stream>>>(
      static_cast<int64_t*>(tags.data_ptr()), static_cast<const int*>(first_use.data_ptr()),
      tags.numel(), tags.size(1), owner, prefix, end);
  check_cuda(cudaGetLastError());
}

__device__ __forceinline__ uint4 host_load(const uint4* address) {
  uint4 value;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address) : "memory");
  return value;
}

__global__ void prefetch_cached_kernel(uint4* keys, uint4* values,
    const uint4* host_keys, const uint4* host_values, int64_t* tags,
    int64_t owner, int64_t prefix, int heads, unsigned long long* bytes) {
  const int block = blockIdx.x / heads, head = blockIdx.x % heads;
  const int tokens = min(int64_t(64), prefix - int64_t(block) * 64);
  if (tokens == 64 && tags[blockIdx.x] == owner) return;
  for (int item = threadIdx.x; item < tokens * 16; item += blockDim.x) {
    const int64_t offset = ((int64_t(block) * 64 + item / 16) * heads + head) * 16 + item % 16;
    keys[offset] = host_load(host_keys + offset);
    values[offset] = host_load(host_values + offset);
  }
  __threadfence();
  __syncthreads();
  if (!threadIdx.x) {
    tags[blockIdx.x] = tokens == 64 ? owner : 0;
    atomicAdd(bytes, static_cast<unsigned long long>(tokens) * 128 * 2 * sizeof(uint16_t));
  }
}

void prefetch_cached(TensorView keys, TensorView values,
    TensorView host_keys, TensorView host_values, TensorView tags,
    TensorView bytes, int64_t owner, int64_t prefix) {
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, keys.device().device_id));
  check_cuda(cudaMemsetAsync(bytes.data_ptr(), 0, sizeof(int64_t), stream));
  if (!prefix) return;
  void *mapped_keys = nullptr, *mapped_values = nullptr;
  check_cuda(cudaHostGetDevicePointer(&mapped_keys, host_keys.data_ptr(), 0));
  check_cuda(cudaHostGetDevicePointer(&mapped_values, host_values.data_ptr(), 0));
  prefetch_cached_kernel<<<((prefix + 63) / 64) * tags.size(1), 128, 0, stream>>>(
      static_cast<uint4*>(keys.data_ptr()), static_cast<uint4*>(values.data_ptr()),
      static_cast<const uint4*>(mapped_keys), static_cast<const uint4*>(mapped_values),
      static_cast<int64_t*>(tags.data_ptr()), owner, prefix, tags.size(1),
      static_cast<unsigned long long*>(bytes.data_ptr()));
  check_cuda(cudaGetLastError());
}

}  // namespace nosa_offload

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, nosa_offload::plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, nosa_offload::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_cached, nosa_offload::prepare_cached);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(publish_cache, nosa_offload::publish_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefetch_cached, nosa_offload::prefetch_cached);
