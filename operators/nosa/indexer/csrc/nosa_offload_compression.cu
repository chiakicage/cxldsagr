// Reuse the resident arithmetic without changing its implementation or launches.
// This also fingerprints nosa_prepare.cu in the Python extension cache key.
#include "nosa_prepare.cu"

namespace nosa_offload_compression {
using tvm::ffi::TensorView;
using T = __nv_bfloat16;

__global__ void append_windows(nosa_prepare::Params p) {
  __shared__ float sums[4 * 128];
  int local = blockIdx.x, head = blockIdx.y;
  // p.keys and p.ck are shifted to the first new window. Reusing key_mean
  // preserves its sequential, warp and cross-warp FP32 addition order.
  nosa_prepare::key_mean<T, 128>(p, local, head, sums);
  if (threadIdx.x < 32) {
    int window = p.old_c + local;
    float value = nosa_prepare::cis_mean<T>(p, window, head);
    if (threadIdx.x == 0)
      static_cast<T*>(p.cc)[int64_t(window) * p.heads + head] =
          nosa_prepare::rounded<T>(value);
  }
}

__global__ void append_pools(nosa_prepare::Params p) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= (p.new_s - p.old_s) * p.heads) return;
  int block = p.old_s + item / p.heads, head = item % p.heads;
  auto cis = static_cast<T const*>(p.cc);
  float pooled = -CUDART_INF_F;
  // Match the native order and fmax semantics, including finite-input
  // overflow that creates NaN compressed records beside finite records.
#pragma unroll
  for (int within = 0; within < 4; ++within)
    pooled = nosa_prepare::maximum(
        pooled, nosa_prepare::to_float(cis[int64_t(block * 4 + within) * p.heads + head]));
  if (block > 0)
    pooled = nosa_prepare::maximum(
        pooled, nosa_prepare::to_float(cis[int64_t(block * 4 - 1) * p.heads + head]));
  static_cast<T*>(p.pool)[int64_t(block) * p.heads + head] =
      nosa_prepare::rounded<T>(pooled);
}

void append_compressed(TensorView keys, TensorView cis, TensorView ck,
                       TensorView cc, TensorView pool, int64_t old_c, int64_t old_s) {
  nosa_prepare::Params p{};
  p.keys = nosa_prepare::data(keys);
  p.raw_cis = nosa_prepare::data(cis);
  p.heads = keys.size(1);
  p.dim = 128;
  p.ck = static_cast<T*>(nosa_prepare::data(ck)) + old_c * p.heads * 128;
  p.cc = nosa_prepare::data(cc);
  p.pool = nosa_prepare::data(pool);
  p.ks0 = keys.stride(0);
  p.ks1 = keys.stride(1);
  p.cs0 = cis.stride(0);
  p.cs1 = cis.stride(1);
  p.old_c = old_c;
  p.new_c = std::max<int64_t>(0, cis.size(0) / 16 - 1);
  p.old_s = old_s;
  p.new_s = std::max<int64_t>(0, (cis.size(0) - 16) / 64);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, keys.device().device_id));
  if (p.old_c < p.new_c) {
    append_windows<<<dim3(p.new_c - p.old_c, p.heads), nosa_prepare::Threads, 0, stream>>>(p);
    auto error = cudaGetLastError();
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  }
  if (p.old_s < p.new_s) {
    int items = (p.new_s - p.old_s) * p.heads;
    append_pools<<<(items + 127) / 128, 128, 0, stream>>>(p);
    auto error = cudaGetLastError();
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  }
}
}  // namespace nosa_offload_compression

TVM_FFI_DLL_EXPORT_TYPED_FUNC(append_compressed, nosa_offload_compression::append_compressed);
