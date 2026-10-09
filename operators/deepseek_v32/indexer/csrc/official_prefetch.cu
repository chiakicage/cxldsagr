// Local pool publication for the unchanged official ECHO Q1 staging kernel.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cassert>
#include <cstdint>

namespace official_prefetch {
using tvm::ffi::TensorView;
constexpr int kCap = 64;
constexpr int kWidth = 576;
constexpr int kMissing = INT32_MAX;

void checked(cudaError_t error) {
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

cudaStream_t stream_for(TensorView tensor) {
  const int gpu = tensor.device().device_id;
  checked(cudaSetDevice(gpu));
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
}

__global__ void prepare_kernel(const int* pages, int* token_ids, int* stage_ids,
                               uint32_t* counter, const int* context_lens,
                               int history, int columns, int host_capacity) {
  assert(context_lens[0] == history + 1);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) *counter = 0;
  if (i < kCap) stage_ids[i] = -1;
  if (i < columns) {
    int host = 0;
    if (i < history) {
      const int page = pages[i / 64];
      assert(page >= 0 && page < host_capacity / 64);
      host = page * 64 + i % 64;
    }
    token_ids[i] = host;
  }
}

__global__ void validate_promotion_kernel(const int* stage_ids,
                               const uint32_t* counter, const int* h2d,
                               const int64_t* d2h, const int* sorted_slots,
                               const int64_t* allocations, int64_t* stats,
                               int pool_rows, int host_capacity) {
  __shared__ int count;
  __shared__ int ids[kCap], slots[kCap];
  __shared__ int64_t old_ids[kCap];
  if (threadIdx.x == 0) count = min(*counter, uint32_t(kCap));
  __syncthreads();
  const int lane = threadIdx.x;
  if (lane < count) {
    const int host = stage_ids[lane], slot = sorted_slots[lane];
    assert(host >= 0 && host < host_capacity);
    assert(slot > 0 && slot < pool_rows);
    assert(h2d[host] == pool_rows + lane);
    const int64_t old = d2h[slot];
    assert(old == kMissing || (old >= 0 && old < host_capacity));
    assert(old == kMissing || h2d[old] == slot);
    assert(allocations[slot] == kMissing);
    ids[lane] = host;
    slots[lane] = slot;
    old_ids[lane] = old;
  }
  __syncthreads();
  // Distribute the triangular uniqueness checks across the block. Every
  // active pair is checked before any record or persistent map is modified.
  for (int pair = lane; pair < kCap * kCap; pair += blockDim.x) {
    const int left = pair / kCap, right = pair % kCap;
    if (left < count && right < left) {
      assert(ids[left] != ids[right]);
      assert(slots[left] != slots[right]);
    }
  }
  const int evicted = __syncthreads_count(lane < count && old_ids[lane] != kMissing);
  if (lane == 0) {
    stats[0] = count;
    stats[1] = evicted;
    stats[2] = int64_t(*counter) - count;
  }
}

__global__ void copy_publish_kernel(const int* stage_ids, const uint16_t* stage,
                               const uint32_t* counter, uint16_t* records,
                               int* h2d, int64_t* d2h, const int* sorted_slots,
                               int64_t* allocations) {
  const int item = blockIdx.x;
  if (item >= min(*counter, uint32_t(kCap))) return;
  const int slot = sorted_slots[item], host = stage_ids[item];
  // Validation is complete before this kernel begins. Valid staged hosts and
  // old owners are disjoint; the checked unique slots make all writes disjoint.
  for (int element = threadIdx.x; element < kWidth; element += blockDim.x) {
    records[int64_t(slot) * kWidth + element] = stage[item * kWidth + element];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int64_t old = d2h[slot];
    if (old != kMissing) h2d[old] = kMissing;
    d2h[slot] = host;
    h2d[host] = slot;
    allocations[slot] = host;
  }
}

__global__ void clear_kernel(const int* stage_ids, int* h2d,
                             const uint32_t* counter, int pool_rows,
                             int host_capacity) {
  const int i = threadIdx.x;
  if (i < min(*counter, uint32_t(kCap))) {
    const int host = stage_ids[i];
    if (host >= 0 && host < host_capacity)
      atomicCAS(h2d + host, pool_rows + i, kMissing);
  }
}

void prepare(TensorView pages, TensorView token_ids, TensorView stage_ids,
             TensorView counter, TensorView context_lens, int64_t history,
             int64_t host_capacity) {
  const int columns = token_ids.size(1);
  auto stream = stream_for(token_ids);
  prepare_kernel<<<(columns + 255) / 256, 256, 0, stream>>>(
      static_cast<int*>(pages.data_ptr()), static_cast<int*>(token_ids.data_ptr()),
      static_cast<int*>(stage_ids.data_ptr()), static_cast<uint32_t*>(counter.data_ptr()),
      static_cast<int*>(context_lens.data_ptr()), history, columns, host_capacity);
  checked(cudaGetLastError());
}

void promote(TensorView stage_ids, TensorView stage, TensorView counter,
             TensorView records, TensorView h2d, TensorView d2h,
             TensorView sorted_slots, TensorView allocations, TensorView stats) {
  auto stream = stream_for(records);
  validate_promotion_kernel<<<1, 256, 0, stream>>>(
      static_cast<int*>(stage_ids.data_ptr()), static_cast<uint32_t*>(counter.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int*>(sorted_slots.data_ptr()), static_cast<int64_t*>(allocations.data_ptr()),
      static_cast<int64_t*>(stats.data_ptr()), records.size(0), h2d.size(0));
  checked(cudaGetLastError());
  copy_publish_kernel<<<kCap, 128, 0, stream>>>(
      static_cast<int*>(stage_ids.data_ptr()), static_cast<uint16_t*>(stage.data_ptr()),
      static_cast<uint32_t*>(counter.data_ptr()), static_cast<uint16_t*>(records.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int*>(sorted_slots.data_ptr()), static_cast<int64_t*>(allocations.data_ptr()));
  checked(cudaGetLastError());
}

void clear(TensorView stage_ids, TensorView h2d, TensorView counter, int64_t pool_rows) {
  auto stream = stream_for(h2d);
  clear_kernel<<<1, kCap, 0, stream>>>(
      static_cast<int*>(stage_ids.data_ptr()), static_cast<int*>(h2d.data_ptr()),
      static_cast<uint32_t*>(counter.data_ptr()), pool_rows, h2d.size(0));
  checked(cudaGetLastError());
}
template <bool kVectorKeys>
__global__ void prepare_keys_kernel(const uint32_t* keys, const uint32_t* scales,
                               const int* pages, const int* context,
                               uint32_t* packed, int* blocks, int* token_ids,
                               int* stage_ids, uint32_t* counter,
                               int columns, int history, int host_capacity) {
  // Keep the production assertion, including its execution before reads.
  assert(context[0] == history + 1);
  const int page = blockIdx.x;
  const int lane = threadIdx.x;
  if constexpr (kVectorKeys) {
    for (int vector = lane; vector < 512; vector += blockDim.x) {
      const int token = page * 64 + vector / 8;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (token < columns) {
        value = reinterpret_cast<const uint4*>(keys + int64_t(page) * 2048)[vector];
      }
      reinterpret_cast<uint4*>(packed + int64_t(page) * 2112)[vector] = value;
    }
  } else {
    for (int word = lane; word < 2048; word += blockDim.x) {
      const int token = page * 64 + word / 32;
      packed[int64_t(page) * 2112 + word] =
          token < columns ? keys[int64_t(page) * 2048 + word] : 0;
    }
  }
  if (lane < 64) {
    const int token = page * 64 + lane;
    packed[int64_t(page) * 2112 + 2048 + lane] =
        token < columns ? scales[token] : 0;
    if (token < columns) {
      int host = 0;
      if (token < history) {
        const int host_page = pages[page];
        assert(host_page >= 0 && host_page < host_capacity / 64);
        host = host_page * 64 + lane;
      }
      token_ids[token] = host;
    }
    if (page == 0) stage_ids[lane] = -1;
  }
  if (lane == 0) {
    blocks[page] = page;
    if (page == 0) *counter = 0;
  }
}

void prepare_keys(TensorView keys, TensorView scales, TensorView pages,
             TensorView context, TensorView packed, TensorView blocks,
             TensorView token_ids, TensorView stage_ids, TensorView counter,
             int64_t history, int64_t host_capacity) {
  const int gpu = keys.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int64_t columns = keys.size(0);
  TVM_FFI_ICHECK(columns > 0 && columns < INT32_MAX);
  const auto key_pointer = reinterpret_cast<uintptr_t>(keys.data_ptr());
  TVM_FFI_ICHECK(key_pointer % sizeof(uint32_t) == 0);
  const int grid = (columns + 63) / 64;
  auto kernel = key_pointer % sizeof(uint4) == 0 ? prepare_keys_kernel<true> : prepare_keys_kernel<false>;
  kernel<<<grid, 128, 0, stream>>>(
      static_cast<const uint32_t*>(keys.data_ptr()),
      static_cast<const uint32_t*>(scales.data_ptr()),
      static_cast<const int*>(pages.data_ptr()),
      static_cast<const int*>(context.data_ptr()),
      static_cast<uint32_t*>(packed.data_ptr()), static_cast<int*>(blocks.data_ptr()),
      static_cast<int*>(token_ids.data_ptr()), static_cast<int*>(stage_ids.data_ptr()),
      static_cast<uint32_t*>(counter.data_ptr()), columns, history, host_capacity);
  checked(cudaGetLastError());
}
}  // namespace official_prefetch

TVM_FFI_DLL_EXPORT_TYPED_FUNC(official_prefetch_prepare, official_prefetch::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(official_prefetch_promote, official_prefetch::promote);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(official_prefetch_clear, official_prefetch::clear);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(official_prefetch_prepare_keys, official_prefetch::prepare_keys);
