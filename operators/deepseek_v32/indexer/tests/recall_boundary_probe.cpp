// CPU boundary oracle: include exact bridge source, replacing only CUDA pointer
// inspection. Stage providers are native C++ stubs and never launch CUDA.
#include <cuda_runtime_api.h>
static int pointer_checks = 0;
static cudaError_t probe_pointer(cudaPointerAttributes* out, const void*) {
  ++pointer_checks;
  out->type = cudaMemoryTypeHost;
  return cudaSuccess;
}
#define cudaPointerGetAttributes probe_pointer
#include "../csrc/echo_recall_dispatch.cpp"
#undef cudaPointerGetAttributes
#include <tvm/ffi/container/array.h>
#include <array>
#include <vector>

namespace {
int scenario = 0, trace = 0;
tvm::ffi::Error original_error("RuntimeError", "native stage sentinel", "");
void stage(int value) {
  trace = trace * 10 + value;
  if ((scenario == 1 && value == 1) || (scenario == 3 && value == 2) ||
      (scenario == 5 && value == 3)) throw original_error;
}
}
#define NATIVE_STUB(name, value) \
extern "C" TVM_FFI_DLL_EXPORT int __tvm_ffi_##name(void*, const TVMFFIAny*, int32_t, TVMFFIAny*) { \
  TVM_FFI_SAFE_CALL_BEGIN(); stage(value); TVM_FFI_SAFE_CALL_END(); \
}
NATIVE_STUB(echo_sparse_selection_allocate, 1)
NATIVE_STUB(echo_sparse_selection_allocate_free, 1)
NATIVE_STUB(gather_counted_u32, 2)
NATIVE_STUB(gather_counted, 4)
NATIVE_STUB(echo_sparse_selection_publish_bounded, 3)

tvm::ffi::Array<int64_t> probe(Module self, int mode, bool free_only) {
  scenario = mode;
  trace = pointer_checks = 0;
  std::array<std::array<int64_t, 2>, 16> shape{};
  std::array<DLTensor, 16> tensor{};
  // Positions 0..10 allocation tensors; 11 host, 12 records, 13 count,
  // 14 clock and 15 recalled. Underlying bytes are never accessed by stubs.
  for (int i = 0; i < 16; ++i) {
    shape[i] = {4, 1};
    tensor[i] = {nullptr, {kDLCUDA, 0}, 1, {kDLInt, 64, 1}, shape[i].data(), nullptr, 0};
  }
  tensor[3].dtype = {kDLInt, 32, 1};
  shape[4][0] = shape[5][0] = shape[6][0] = 5;
  tensor[6].dtype = {kDLBool, 8, 1};
  shape[11] = {8, mode == 2 ? 3 : 7};
  shape[12] = {5, 7};
  tensor[11].device = {kDLCPU, 0};
  tensor[11].ndim = tensor[12].ndim = 2;
  for (int i : {13, 14, 15}) shape[i][0] = 1;
  if (mode == 4) shape[14][0] = 2;
  tensor[13].dtype = mode == 6 ? DLDataType{kDLInt, 64, 1} : DLDataType{kDLUInt, 32, 1};
  int64_t same_error = 0, failed = 0;
  auto function = make_bridge(self, self, free_only);
  try {
    function(TensorView(&tensor[0]), TensorView(&tensor[1]), TensorView(&tensor[2]),
             TensorView(&tensor[3]), TensorView(&tensor[4]), TensorView(&tensor[5]),
             TensorView(&tensor[6]), TensorView(&tensor[7]), TensorView(&tensor[8]),
             TensorView(&tensor[9]), TensorView(&tensor[10]), 4, 100,
             TensorView(&tensor[11]), TensorView(&tensor[12]), TensorView(&tensor[13]),
             TensorView(&tensor[14]), TensorView(&tensor[15]));
  } catch (const tvm::ffi::Error& error) {
    same_error = error.same_as(original_error);
    failed = 1;
  }
  return {trace, pointer_checks, failed, same_error};
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(probe, probe);
