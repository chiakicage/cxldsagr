// Exact postprocessing of the unchanged FlashInfer SMALL-selected set.
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cstdint>

namespace q1_topk {
using tvm::ffi::TensorView;

template<int Threads, int Items>
__global__ __launch_bounds__(Threads)
void cub_sort_mask_kernel(float* values, int* indices, int count) {
    using Sort = cub::BlockRadixSort<uint64_t, Threads, Items>;
    __shared__ typename Sort::TempStorage storage;
    uint64_t keys[Items];
    const int row = blockIdx.x;
    const int base = row * count;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int position = threadIdx.x * Items + i;
        uint64_t key = UINT64_MAX;
        if (position < count) {
            const uint32_t bits = __float_as_uint(values[base + position]);
            const uint32_t ordered = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
            const uint32_t index = static_cast<uint32_t>(indices[base + position]);
            key = (static_cast<uint64_t>(~ordered) << 32) | index;
        }
        keys[i] = key;
    }
    Sort(storage).Sort(keys);
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int position = threadIdx.x * Items + i;
        if (position < count) {
            const uint32_t ordered = ~static_cast<uint32_t>(keys[i] >> 32);
            const uint32_t bits = ordered ^ ((ordered & 0x80000000u) ? 0x80000000u : 0xffffffffu);
            values[base + position] = __uint_as_float(bits);
            indices[base + position] = (bits & 0x7fffffffu) >= 0x7f800000u
                ? -1 : static_cast<int>(static_cast<uint32_t>(keys[i]));
        }
    }
}

void sort_mask(TensorView values, TensorView indices) {
    TVM_FFI_ICHECK(values.ndim() == 2 && indices.ndim() == 2);
    TVM_FFI_ICHECK(values.size(0) == indices.size(0) && values.size(1) == indices.size(1));
    TVM_FFI_ICHECK(values.dtype().code == kDLFloat && values.dtype().bits == 32);
    TVM_FFI_ICHECK(indices.dtype().code == kDLInt && indices.dtype().bits == 32);
    TVM_FFI_ICHECK(values.device().device_type == kDLCUDA);
    TVM_FFI_ICHECK(indices.device().device_type == values.device().device_type);
    TVM_FFI_ICHECK(indices.device().device_id == values.device().device_id);
    TVM_FFI_ICHECK(values.stride(1) == 1 && indices.stride(1) == 1);
    TVM_FFI_ICHECK(values.stride(0) == values.size(1) && indices.stride(0) == indices.size(1));
    TVM_FFI_ICHECK(values.size(1) > 0 && values.size(1) <= 2048);
    const int device = values.device().device_id;
    auto error = cudaSetDevice(device);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    cub_sort_mask_kernel<256, 8><<<values.size(0), 256, 0, stream>>>(
        static_cast<float*>(values.data_ptr()), static_cast<int*>(indices.data_ptr()), values.size(1));
    error = cudaGetLastError();
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}
}  // namespace q1_topk

TVM_FFI_DLL_EXPORT_TYPED_FUNC(q1_topk_sort_mask, q1_topk::sort_mask);
