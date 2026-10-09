// Fixed-shape finite mean: preserve the accepted Torch FP32 reduction tree.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cstdint>

namespace q1_hint_exact {
using tvm::ffi::TensorView;

__device__ __forceinline__ float finite_value(float value, int& count) {
    const bool finite = (__float_as_uint(value) & 0x7fffffffu) < 0x7f800000u;
    count += static_cast<int>(finite);
    return finite ? value : 0.0f;
}

__global__ __launch_bounds__(512)
void mean_kernel(const float* scores, float* offset) {
    const int t = threadIdx.x;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int count = 0;
    // Ascending per-accumulator order is part of the numerical contract.
#pragma unroll 4
    for (int j = 0; j < 32; ++j) {
        const float4 x = reinterpret_cast<const float4*>(scores)[t + 512 * j];
        a0 = __fadd_rn(a0, finite_value(x.x, count));
        a1 = __fadd_rn(a1, finite_value(x.y, count));
        a2 = __fadd_rn(a2, finite_value(x.z, count));
        a3 = __fadd_rn(a3, finite_value(x.w, count));
    }
    if (t == 0) {
        a0 = __fadd_rn(a0, finite_value(scores[65536], count));
    }
    float value = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
    __shared__ float sums[512];
    __shared__ int counts[512];
    sums[t] = value;
    counts[t] = count;
#pragma unroll
    for (int shift = 256; shift >= 32; shift >>= 1) {
        __syncthreads();
        if (t < shift) {
            value = __fadd_rn(value, sums[t + shift]);
            count += counts[t + shift];
            sums[t] = value;
            counts[t] = count;
        }
    }
    __syncthreads();
    if (t < 32) {
#pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1) {
            value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, shift));
            count += __shfl_down_sync(0xffffffffu, count, shift);
        }
        if (t == 0) {
            offset[0] = __fdiv_rn(value, static_cast<float>(count > 0 ? count : 1));
        }
    }
}

void mean(TensorView scores, TensorView offset) {
    TVM_FFI_ICHECK(scores.ndim() == 2 && scores.size(0) == 1 && scores.size(1) == 65537);
    TVM_FFI_ICHECK(scores.stride(1) == 1);
    TVM_FFI_ICHECK(offset.ndim() == 1 && offset.size(0) == 16 && offset.stride(0) == 1);
    TVM_FFI_ICHECK(scores.dtype().code == kDLFloat && scores.dtype().bits == 32 && scores.dtype().lanes == 1);
    TVM_FFI_ICHECK(offset.dtype().code == kDLFloat && offset.dtype().bits == 32 && offset.dtype().lanes == 1);
    TVM_FFI_ICHECK(scores.device().device_type == kDLCUDA);
    TVM_FFI_ICHECK(offset.device().device_type == kDLCUDA && offset.device().device_id == scores.device().device_id);
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0);
    const int device = scores.device().device_id;
    auto error = cudaSetDevice(device);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
    mean_kernel<<<1, 512, 0, stream>>>(static_cast<const float*>(scores.data_ptr()), static_cast<float*>(offset.data_ptr()));
    error = cudaGetLastError();
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}
}  // namespace q1_hint_exact

TVM_FFI_DLL_EXPORT_TYPED_FUNC(q1_hint_mean, q1_hint_exact::mean);
