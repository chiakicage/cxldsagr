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
             TensorView counter, TensorView offsets, int64_t query_start,
             int64_t max_prefetch, bool prefetch) {
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
  if (prefetch) {
    void* mapped = nullptr;
    checked(cudaHostGetDevicePointer(&mapped, host.data_ptr(), 0));
    constexpr int stages = 96;
    constexpr int fixed = 16384 + 3*16384 + 3*16384 + 3*512 + 3*512 + 16*8 + 4;
    constexpr int shared_bytes = fixed + stages*(1024+16);
    auto kernel = sm90_fp8_mqa_logits_fuse_prefetch<64,128,576,2,128,3,3,stages,128,256,256,16384>;
    checked(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    kernel<<<grid,640,shared_bytes,stream>>>(
        rows, columns, stride, page_table.size(1), begin, end, logits,
        static_cast<int*>(page_table.data_ptr()), static_cast<int*>(extend_lengths.data_ptr()),
        static_cast<int*>(query_requests.data_ptr()),
        static_cast<__nv_bfloat16*>(device.data_ptr()), static_cast<__nv_bfloat16*>(mapped),
        static_cast<int*>(allocations.data_ptr()), static_cast<int*>(slots.data_ptr()),
        static_cast<int64_t*>(device_to_host.data_ptr()), static_cast<int*>(host_to_device.data_ptr()),
        static_cast<uint32_t*>(counter.data_ptr()), static_cast<float*>(offsets.data_ptr()),
        max_prefetch, query_start, tq, tk, ts, tw);
  } else {
    constexpr int shared_bytes = 3*16384 + 3*16384 + 3*512 + 3*512 + 12*8;
    auto kernel = sm90_fp8_mqa_logits<64,128,2,128,3,3,128,256>;
    checked(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    kernel<<<grid,384,shared_bytes,stream>>>(rows, columns, stride, begin, end, logits, tq, tk, ts, tw);
  }
  checked(cudaGetLastError());
  clean<<<std::min<int64_t>(1024, (int64_t(rows)*columns+255)/256),256,0,stream>>>(
      logits, rows, columns, stride, query_start);
  checked(cudaGetLastError());
}
} // namespace echo_native
TVM_FFI_DLL_EXPORT_TYPED_FUNC(echo_logits, echo_native::forward);
