// Standalone adapter for ECHO's MIT-licensed DeepGEMM kernels; see echo_logits.cuh.
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <algorithm>
#include <limits>
#include "echo_logits.cuh"

namespace echo_native {
using tvm::ffi::TensorView;

void checked(cudaError_t error) {
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

CUtensorMap tensor_map(void* pointer, CUtensorMapDataType dtype,
                       uint64_t width, uint64_t height, uint64_t item_bytes,
                       uint32_t tile_width, uint32_t tile_height, bool swizzle) {
  CUtensorMap descriptor;
  uint64_t dimensions[] = {width, height};
  uint64_t strides[] = {width * item_bytes};
  uint32_t box[] = {tile_width, tile_height};
  uint32_t element_strides[] = {1, 1};
  auto result = cuTensorMapEncodeTiled(
      &descriptor, dtype, 2, pointer, dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << int(result);
  return descriptor;
}

__global__ void clean(float* output, int rows, int columns, int stride, int query_start) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < int64_t(rows) * columns; i += int64_t(gridDim.x) * blockDim.x) {
    int row = i / columns, column = i % columns;
    if (column > query_start + row) output[int64_t(row) * stride + column] = -CUDART_INF_F;
  }
}

void forward(TensorView q, TensorView k, TensorView weights, TensorView scales,
             TensorView starts, TensorView ends, TensorView output,
             TensorView page_table, TensorView extend_lengths, TensorView query_requests,
             TensorView host, TensorView device, TensorView host_to_device,
             TensorView device_to_host, TensorView slots, TensorView allocations,
             TensorView counter, TensorView stats, TensorView offsets, int64_t query_start,
             int64_t history_length, int64_t max_prefetch) {
  const int rows = q.size(0), columns = k.size(0), stride = output.size(1);
  const int gpu = q.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  auto tq = tensor_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       128, uint64_t(rows)*64, 1, 128, 128, true);
  auto tk = tensor_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       128, columns, 1, 128, 128, true);
  auto tw = tensor_map(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       64, rows, 4, 64, 2, false);
  auto ts = tensor_map(scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       scales.size(0), 1, 4, 128, 1, false);
  int sms;
  checked(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu));
  int grid = std::min(sms, (rows+1)/2);
  auto begin = static_cast<uint32_t*>(starts.data_ptr());
  auto end = static_cast<uint32_t*>(ends.data_ptr());
  auto logits = static_cast<float*>(output.data_ptr());
  void* mapped = nullptr;
  checked(cudaHostGetDevicePointer(&mapped, host.data_ptr(), 0));
  constexpr int stages = 96;
  constexpr int fixed = 16384 + 3*16384 + 3*16384 + 3*512 + 3*512 + 16*8 + 4;
  constexpr int shared_bytes = fixed + stages*(1024+16);
  auto kernel = sm90_fp8_mqa_logits_fuse_prefetch<64,128,576,2,128,3,3,stages,128,256,256,16384>;
  checked(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<grid,640,shared_bytes,stream>>>(
      rows, columns, stride, page_table.size(0), begin, end, logits,
      static_cast<int*>(page_table.data_ptr()), static_cast<int*>(extend_lengths.data_ptr()),
      static_cast<int*>(query_requests.data_ptr()),
      static_cast<__nv_bfloat16*>(device.data_ptr()), static_cast<__nv_bfloat16*>(mapped),
      static_cast<int64_t*>(allocations.data_ptr()), static_cast<int*>(slots.data_ptr()),
      static_cast<int64_t*>(device_to_host.data_ptr()), static_cast<int*>(host_to_device.data_ptr()),
      static_cast<uint32_t*>(counter.data_ptr()), static_cast<int64_t*>(stats.data_ptr()),
      static_cast<float*>(offsets.data_ptr()),
      max_prefetch, history_length, host.size(0), tq, tk, ts, tw);
  checked(cudaGetLastError());
  clean<<<std::min<int64_t>(1024, (int64_t(rows)*columns+255)/256),256,0,stream>>>(
      logits, rows, columns, stride, query_start);
  checked(cudaGetLastError());
}

__global__ void prefetch_ids_kernel(const int64_t* ids, int count, int host_capacity,
                                    uint32_t limit, const int* slots,
                                    int64_t* allocations, uint32_t* counter, int64_t* stats,
                                    int* h2d, int64_t* d2h,
                                    const __nv_bfloat16* host, __nv_bfloat16* records) {
  __shared__ int shared_ids[256], shared_slots[256];
  const int warp_base = threadIdx.x / 32 * 32;
  for (int base = blockIdx.x * blockDim.x; base < count;
       base += gridDim.x * blockDim.x) {
    const int index = base + threadIdx.x;
    const int64_t id = index < count ? ids[index] : -1;
    claim_and_copy_warp(id >= 0 && id < host_capacity, int(id), limit, host_capacity,
                        slots, allocations, counter, stats, h2d, d2h, host, records,
                        shared_ids + warp_base, shared_slots + warp_base);
  }
}

void prefetch_ids(TensorView ids, TensorView host, TensorView records,
                  TensorView h2d, TensorView d2h, TensorView slots,
                  TensorView allocations, TensorView counter, TensorView stats,
                  int64_t limit) {
  const int gpu = records.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  void* mapped = nullptr;
  checked(cudaHostGetDevicePointer(&mapped, host.data_ptr(), 0));
  if (!ids.size(0)) return;
  prefetch_ids_kernel<<<std::min<int64_t>(128, (ids.size(0)+255)/256),256,0,stream>>>(
      static_cast<int64_t*>(ids.data_ptr()), ids.size(0), host.size(0), limit,
      static_cast<int*>(slots.data_ptr()), static_cast<int64_t*>(allocations.data_ptr()),
      static_cast<uint32_t*>(counter.data_ptr()), static_cast<int64_t*>(stats.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<__nv_bfloat16*>(mapped), static_cast<__nv_bfloat16*>(records.data_ptr()));
  checked(cudaGetLastError());
}

__global__ void finalize_kernel(int64_t* priority, bool* free_bitmap,
                                 const int64_t* allocations, const int64_t* clock,
                                 int physical_slots) {
  for (int slot = blockIdx.x * blockDim.x + threadIdx.x + 1;
       slot < physical_slots; slot += gridDim.x * blockDim.x) {
    if (allocations[slot] >= 0 && allocations[slot] < INT32_MAX) {
      priority[slot] = *clock;
      free_bitmap[slot] = false;
    }
  }
}
__global__ void advance_clock_kernel(int64_t* priority, int64_t* clock) {
  priority[0] = INT32_MAX;
  *clock += 1;
}
void finalize_prefetch(TensorView priority, TensorView free_bitmap,
                        TensorView allocations, TensorView clock) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  // This boundary is intentional: every CTA reads the same clock before the
  // scalar is advanced, including the empty-allocation case in official code.
  finalize_kernel<<<std::min<int64_t>(128, (priority.size(0)+255)/256),256,0,stream>>>(
      static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
      static_cast<int64_t*>(allocations.data_ptr()), static_cast<int64_t*>(clock.data_ptr()),
      priority.size(0));
  checked(cudaGetLastError());
  advance_clock_kernel<<<1,1,0,stream>>>(static_cast<int64_t*>(priority.data_ptr()),
                                        static_cast<int64_t*>(clock.data_ptr()));
  checked(cudaGetLastError());
}

__global__ void protect_kernel(const int64_t* ids, int count, const int* h2d,
                               int host_capacity, int64_t* priority,
                               int physical_slots, const int64_t* clock) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    const int64_t id = ids[i];
    if (id < 0 || id >= host_capacity) continue;
    const int slot = h2d[id];
    if (slot > 0 && slot < physical_slots)
      atomicExch(reinterpret_cast<unsigned long long*>(priority + slot), *clock);
  }
}
void protect(TensorView ids, TensorView h2d, TensorView priority, TensorView clock) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  if (ids.size(0)) {
    protect_kernel<<<std::min<int64_t>(128, (ids.size(0)+255)/256),256,0,stream>>>(
        static_cast<int64_t*>(ids.data_ptr()), ids.size(0), static_cast<int*>(h2d.data_ptr()),
        h2d.size(0), static_cast<int64_t*>(priority.data_ptr()), priority.size(0),
        static_cast<int64_t*>(clock.data_ptr()));
    checked(cudaGetLastError());
  }
  advance_clock_kernel<<<1,1,0,stream>>>(static_cast<int64_t*>(priority.data_ptr()),
                                        static_cast<int64_t*>(clock.data_ptr()));
  checked(cudaGetLastError());
}

__global__ void release_ids_kernel(const int64_t* ids, int count, int* h2d,
                                   int host_capacity, int64_t* d2h, int64_t* priority,
                                   bool* free_bitmap, int physical_slots) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    const int64_t id = ids[i];
    if (id < 0 || id >= host_capacity) continue;
    const int slot = atomicAdd(h2d + id, 0);
    if (slot <= 0 || slot >= physical_slots) continue;
    if (atomicCAS(h2d + id, slot, INT32_MAX) != slot) continue;
    if (atomicCAS(reinterpret_cast<unsigned long long*>(d2h + slot),
                  static_cast<unsigned long long>(id), INT32_MAX) == id) {
      priority[slot] = -1;
      free_bitmap[slot] = true;
    }
  }
}
void release_ids(TensorView ids, TensorView h2d, TensorView d2h,
                  TensorView priority, TensorView free_bitmap) {
  const int gpu = priority.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  if (!ids.size(0)) return;
  release_ids_kernel<<<std::min<int64_t>(128, (ids.size(0)+255)/256),256,0,stream>>>(
      static_cast<int64_t*>(ids.data_ptr()), ids.size(0), static_cast<int*>(h2d.data_ptr()),
      h2d.size(0), static_cast<int64_t*>(d2h.data_ptr()),
      static_cast<int64_t*>(priority.data_ptr()), static_cast<bool*>(free_bitmap.data_ptr()),
      priority.size(0));
  checked(cudaGetLastError());
}

__global__ void mark_misses_kernel(const int64_t* ids, int count, const int* h2d,
                                   int host_capacity, int64_t* output) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    const int64_t id = ids[i];
    output[i] = id >= 0 && id < host_capacity && h2d[id] == INT32_MAX ? id : INT32_MAX;
  }
}
void mark_misses(TensorView ids, TensorView h2d, TensorView output) {
  const int gpu = ids.device().device_id;
  checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  if (!ids.size(0)) return;
  mark_misses_kernel<<<std::min<int64_t>(128, (ids.size(0)+255)/256),256,0,stream>>>(
      static_cast<int64_t*>(ids.data_ptr()), ids.size(0), static_cast<int*>(h2d.data_ptr()),
      h2d.size(0), static_cast<int64_t*>(output.data_ptr()));
  checked(cudaGetLastError());
}
} // namespace echo_native
#include "echo_resident.cuh"
#include "echo_sparse_recall.cuh"
#include "echo_sparse_free.cuh"
#include "echo_sparse_append_free.cuh"
#include "echo_sparse_prepare_free.cuh"
#include "echo_dense_prefetch.cuh"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_logits, echo_native::forward);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_prefetch_ids, echo_native::prefetch_ids);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_finalize_prefetch, echo_native::finalize_prefetch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_protect, echo_native::protect);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_release_ids, echo_native::release_ids);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_mark_misses, echo_native::mark_misses);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_resident_selection, echo_native::resident_selection);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_empty_event, echo_native::empty_event);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_planned_append, echo_native::planned_append);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_protect_resident_history, echo_native::protect_resident_history);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_classify, echo_native::sparse_selection_classify);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_compact, echo_native::sparse_selection_compact);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_workspace, echo_native::sparse_selection_workspace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_prepare_prefetch, echo_native::prepare_prefetch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_prepare_prefetch_free, echo_native::free_prepare::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_append, echo_native::sparse_append);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_append_free, echo_native::free_append::append);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_allocate, echo_native::sparse_selection_allocate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_allocate_free, echo_native::free_recall::allocate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_publish, echo_native::sparse_selection_publish);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_publish_bounded, echo_native::sparse_selection_publish_bounded);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_sparse_selection_map, echo_native::sparse_selection_map);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_dense_history_clear, echo_native::dense_history_clear);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_dense_history_publish, echo_native::dense_history_publish);
