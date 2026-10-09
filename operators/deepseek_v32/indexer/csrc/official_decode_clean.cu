// Upstream JIT builds these kernels separately: their dynamic shared-memory
// declarations have different types, so preserve separate translation units.
#include <cuda_runtime.h>
#include <tvm/ffi/error.h>
#include <deep_gemm/impls/smxx_clean_logits.cuh>

namespace official_echo {
void clean(cudaStream_t stream, uint32_t max_context_len, uint64_t stride,
           const uint32_t* context_lens, float* output) {
  deep_gemm::smxx_clean_logits<1, 8192, 8><<<CXL_ECHO_NUM_SMS, 256, 8192 * 4, stream>>>(
      1, max_context_len, stride, nullptr, context_lens, output);
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
}  // namespace official_echo
