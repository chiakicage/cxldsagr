// Source-only ABI bridge. Algorithm bodies remain in the pinned upstream headers.
#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

namespace official_echo {
using tvm::ffi::TensorView;

void clean(cudaStream_t stream, uint32_t max_context_len, uint64_t stride,
           const uint32_t* context_lens, float* output);

void checked(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

CUtensorMap map_2d(void* pointer, CUtensorMapDataType dtype, uint64_t inner,
                  uint64_t outer, uint64_t outer_stride_bytes,
                  uint32_t tile_inner, uint32_t tile_outer, bool swizzle) {
  CUtensorMap result;
  uint64_t dimensions[] = {inner, outer};
  uint64_t strides[] = {outer_stride_bytes};
  uint32_t tiles[] = {tile_inner, tile_outer};
  uint32_t steps[] = {1, 1};
  auto status = cuTensorMapEncodeTiled(
      &result, dtype, 2, pointer, dimensions, strides, tiles, steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS) << "cuTensorMapEncodeTiled: " << int(status);
  return result;
}

CUtensorMap map_kv(void* pointer, uint64_t pages) {
  CUtensorMap result;
  uint64_t dimensions[] = {128, 64, pages};
  uint64_t strides[] = {128, 64 * 132};
  uint32_t tiles[] = {128, 64, 1};
  uint32_t steps[] = {1, 1, 1};
  auto status = cuTensorMapEncodeTiled(
      &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, pointer, dimensions, strides,
      tiles, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS) << "cuTensorMapEncodeTiled: " << int(status);
  return result;
}

cudaStream_t stream_for(TensorView tensor) {
  const int gpu = tensor.device().device_id;
  checked(cudaSetDevice(gpu));
  int sms;
  checked(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu));
  TVM_FFI_ICHECK(sms == CXL_ECHO_NUM_SMS) << "SM-count-specialized metadata mismatch";
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
}

void metadata(TensorView context_lens, TensorView output) {
  auto stream = stream_for(context_lens);
  deep_gemm::smxx_paged_mqa_logits_metadata<32, 256, CXL_ECHO_NUM_SMS>
      <<<1, 32, 32 * sizeof(int), stream>>>(
          1, static_cast<uint32_t*>(context_lens.data_ptr()),
          static_cast<uint32_t*>(output.data_ptr()));
  checked(cudaGetLastError());
}

constexpr int align(int value, int alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Exact Q1 specialization of the upstream launcher's stage and smem formula.
constexpr int kSwizzle = 1024;
constexpr int kQPipe = 3 * (64 * 128 + align(64 * 4, kSwizzle)) + align(3 * 16, kSwizzle);
constexpr int kKVPipe = 3 * (64 * 128 + align(64 * 4, kSwizzle)) + align(3 * 16, kSwizzle);
constexpr int kFixed = kQPipe + 4 * kKVPipe + 4 * 2 * 8 + 4;
constexpr int kStatic = align(128 * sizeof(int), kSwizzle);
constexpr int kCapacity = 232448;
constexpr int prefetch_stages() {
  int stages = (kCapacity - kStatic - kFixed) / (256 * sizeof(float) + 16);
  while (kFixed + stages * 256 * sizeof(float) + align(stages * 16, kSwizzle) + kStatic >
         kCapacity) --stages;
  return stages;
}
constexpr int kPFStages = prefetch_stages();
constexpr int kShared = kFixed + kPFStages * 256 * sizeof(float) + align(kPFStages * 16, kSwizzle);
static_assert(kPFStages == 83 && kShared == 230468);

void forward(TensorView q, TensorView packed, TensorView weights,
             TensorView context_lens, TensorView block_table, TensorView schedule,
             TensorView output, TensorView page_table, TensorView device_pool,
             TensorView host_pool, TensorView prefetch_locs, TensorView prefetch_kv,
             TensorView h2d, TensorView counter, TensorView threshold,
             int64_t max_context_len) {
  auto stream = stream_for(q);
  auto tq = map_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 128, 64, 128, 128, 64, true);
  auto tk = map_kv(packed.data_ptr(), packed.size(0));
  auto ts = map_2d(static_cast<uint8_t*>(packed.data_ptr()) + 64 * 128,
                   CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 64, packed.size(0),
                   64 * 132, 64, 1, false);
  auto tw = map_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                   64, 1, 64 * 4, 64, 1, false);
  void* mapped_host = nullptr;
  checked(cudaHostGetDevicePointer(&mapped_host, host_pool.data_ptr(), 0));
  auto kernel = deep_gemm::sm90_fp8_paged_mqa_logits_fused_v2<
      1, 64, 128, 576, 64, 3, 3, kPFStages, 256, 128, 512, 128>;
  checked(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared));
  kernel<<<CXL_ECHO_NUM_SMS, 768, kShared, stream>>>(
      1, output.size(1), block_table.size(1), page_table.size(1),
      static_cast<uint32_t*>(context_lens.data_ptr()), static_cast<float*>(output.data_ptr()),
      static_cast<uint32_t*>(block_table.data_ptr()), static_cast<uint32_t*>(schedule.data_ptr()),
      static_cast<uint32_t*>(page_table.data_ptr()),
      static_cast<__nv_bfloat16*>(device_pool.data_ptr()), static_cast<__nv_bfloat16*>(mapped_host),
      static_cast<int*>(prefetch_locs.data_ptr()), static_cast<__nv_bfloat16*>(prefetch_kv.data_ptr()),
      static_cast<int*>(h2d.data_ptr()), static_cast<uint32_t*>(counter.data_ptr()),
      static_cast<float*>(threshold.data_ptr()), device_pool.size(0) - 1,
      tq, tk, ts, tw);
  checked(cudaGetLastError());
  clean(stream, max_context_len, output.size(1),
        static_cast<uint32_t*>(context_lens.data_ptr()), static_cast<float*>(output.data_ptr()));
}
}  // namespace official_echo

TVM_FFI_DLL_EXPORT_TYPED_FUNC(official_decode_metadata, official_echo::metadata);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(official_decode_forward, official_echo::forward);
