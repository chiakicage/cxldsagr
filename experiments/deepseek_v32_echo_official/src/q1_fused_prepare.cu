// Private candidate: unchanged page64 bytes plus local official staging setup.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cassert>
#include <cstdint>

namespace q1_fused_prepare {
using tvm::ffi::TensorView;

void checked(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

template <bool kVectorKeys>
__global__ void prepare_kernel(const uint32_t* keys, const uint32_t* scales,
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

void prepare(TensorView keys, TensorView scales, TensorView pages,
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
  auto kernel = key_pointer % sizeof(uint4) == 0 ? prepare_kernel<true> : prepare_kernel<false>;
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
}  // namespace q1_fused_prepare

TVM_FFI_DLL_EXPORT_TYPED_FUNC(q1_fused_prepare, q1_fused_prepare::prepare);
