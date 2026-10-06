// Generic fixed-width token records: sparse SM gathers or contiguous H2D DMA.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace {
using tvm::ffi::TensorView;

template <typename CountT>
__global__ void gather_records(const unsigned char* host, unsigned char* device,
                              const int64_t* host_ids, const int64_t* device_ids,
                              int64_t count, int64_t host_count, int64_t device_count,
                              int64_t bytes, const CountT* valid_count) {
  const int64_t active = valid_count ? *valid_count : count;
  if (active < 0 || active > count) {
    asm("trap;");
    return;
  }
  // A bounded grid leaves SM capacity for the consumer's independent stream.
  // Serial callers retain the original one-CTA-per-record launch by default.
  for (int64_t row = blockIdx.x; row < active; row += gridDim.x) {
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
}

template <typename CountT>
void launch_gather(TensorView host, TensorView device, TensorView host_ids,
                   TensorView device_ids, const CountT* valid_count, int64_t max_ctas) {
  void* mapped = nullptr;
  auto err = cudaHostGetDevicePointer(&mapped, host.data_ptr(), 0);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device.device().device_id));
  int64_t bytes = host.size(1) * host.dtype().bits * host.dtype().lanes / 8;
  const int64_t blocks = max_ctas ? std::min(host_ids.numel(), max_ctas) : host_ids.numel();
  gather_records<CountT><<<blocks, 128, 0, stream>>>(
      static_cast<const unsigned char*>(mapped),
      static_cast<unsigned char*>(device.data_ptr()),
      static_cast<const int64_t*>(host_ids.data_ptr()),
      static_cast<const int64_t*>(device_ids.data_ptr()), host_ids.numel(),
      host.size(0), device.size(0), bytes, valid_count);
  err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

void gather(TensorView host, TensorView device, TensorView host_ids,
            TensorView device_ids, int64_t max_ctas) {
  launch_gather<int64_t>(host, device, host_ids, device_ids, nullptr, max_ctas);
}

void gather_counted_u32(TensorView host, TensorView device, TensorView host_ids,
                        TensorView device_ids, TensorView valid_count, int64_t max_ctas) {
  launch_gather(host, device, host_ids, device_ids,
                static_cast<const uint32_t*>(valid_count.data_ptr()), max_ctas);
}

void gather_counted(TensorView host, TensorView device, TensorView host_ids,
                    TensorView device_ids, TensorView valid_count, int64_t max_ctas) {
  launch_gather(host, device, host_ids, device_ids,
                static_cast<const int64_t*>(valid_count.data_ptr()), max_ctas);
}

void copy_contiguous(TensorView host, TensorView device, int64_t host_start,
                     int64_t device_start, int64_t count) {
  // Python validates dtype, rank, contiguity and pinned source ownership. Keep
  // span bounds checked here as well before computing either pointer offset.
  if (host_start < 0 || device_start < 0 || count < 0 ||
      host_start > host.size(0) || count > host.size(0) - host_start ||
      device_start > device.size(0) || count > device.size(0) - device_start) {
    throw std::invalid_argument("contiguous record span exceeds its allocation");
  }
  if (count == 0) return;
  const int64_t bytes = host.size(1) * host.dtype().bits * host.dtype().lanes / 8;
  const auto* source = static_cast<const unsigned char*>(host.data_ptr()) + host_start * bytes;
  auto* destination = static_cast<unsigned char*>(device.data_ptr()) + device_start * bytes;
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device.device().device_id));
  const auto err = cudaMemcpyAsync(destination, source, count * bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather, gather);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather_counted, gather_counted);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather_counted_u32, gather_counted_u32);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_contiguous, copy_contiguous);
