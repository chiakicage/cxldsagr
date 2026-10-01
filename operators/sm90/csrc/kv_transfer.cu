// Generic fixed-width token records; GPU reads mapped pinned host memory.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cstdint>
#include <stdexcept>

namespace {
using tvm::ffi::TensorView;

__global__ void gather_records(const unsigned char* host, unsigned char* device,
                              const int64_t* host_ids, const int64_t* device_ids,
                              int64_t count, int64_t host_count, int64_t device_count,
                              int64_t bytes) {
  int64_t row = blockIdx.x;
  if (row >= count) return;
  int64_t h = host_ids[row], d = device_ids[row];
  if (h < 0 || h >= host_count || d < 0 || d >= device_count) {
    asm("trap;");
    return;
  }
  const auto* source = host + h * bytes;
  auto* destination = device + d * bytes;
  // Contiguous tensors can still have an unaligned storage offset. Both row
  // addresses, as well as the record width, must support uint4 accesses.
  if (bytes % 16 == 0 && reinterpret_cast<uintptr_t>(source) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(destination) % 16 == 0) {
    auto src = reinterpret_cast<const uint4*>(source);
    auto dst = reinterpret_cast<uint4*>(destination);
    for (int64_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) dst[i] = src[i];
  } else {
    for (int64_t i = threadIdx.x; i < bytes; i += blockDim.x)
      destination[i] = source[i];
  }
}

void gather(TensorView host, TensorView device, TensorView host_ids, TensorView device_ids) {
  void* mapped = nullptr;
  auto err = cudaHostGetDevicePointer(&mapped, host.data_ptr(), 0);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device.device().device_id));
  int64_t bytes = host.size(1) * host.dtype().bits * host.dtype().lanes / 8;
  gather_records<<<host_ids.numel(), 128, 0, stream>>>(
      static_cast<const unsigned char*>(mapped),
      static_cast<unsigned char*>(device.data_ptr()),
      static_cast<const int64_t*>(host_ids.data_ptr()),
      static_cast<const int64_t*>(device_ids.data_ptr()), host_ids.numel(),
      host.size(0), device.size(0), bytes);
  err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather, gather);
