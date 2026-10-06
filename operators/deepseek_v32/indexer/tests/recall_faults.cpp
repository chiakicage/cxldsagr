// Test-only native forwarding shim. It calls original DSOs and injects host
// exceptions at exact stage boundaries; it contains no CUDA kernels/callbacks.
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <memory>
#include <array>
using tvm::ffi::Any;
using tvm::ffi::AnyView;
using tvm::ffi::Function;
using tvm::ffi::Module;

struct Context {
  Function allocate, allocate_free, gather, gather_u32, publish;
  int fail_stage, trace = 0;
  tvm::ffi::Error error;
  Context(Module echo, Module common, int stage)
      : allocate(echo->GetFunction("echo_sparse_selection_allocate").value()),
        allocate_free(echo->GetFunction("echo_sparse_selection_allocate_free").value()),
        gather(common->GetFunction("gather_counted").value()),
        gather_u32(common->GetFunction("gather_counted_u32").value()),
        publish(echo->GetFunction("echo_sparse_selection_publish_bounded").value()),
        fail_stage(stage), error("RuntimeError", "injected native stage " + std::to_string(stage), "") {}
};
static std::unique_ptr<Context> context;

void configure(Module echo, Module common, int stage) {
  TVM_FFI_ICHECK(stage >= 0 && stage <= 3);
  context = std::make_unique<Context>(echo, common, stage);
}
void invoke(int stage, const Function& original, const TVMFFIAny* args, int32_t size, TVMFFIAny* result) {
  context->trace = context->trace * 10 + stage;
  if (context->fail_stage == stage && stage != 2) throw context->error;
  original->CallPacked(reinterpret_cast<const AnyView*>(args), size, reinterpret_cast<Any*>(result));
  // Copy failure is injected AFTER original gather enqueues the real copy.
  if (context->fail_stage == stage && stage == 2) throw context->error;
}
#define FORWARD(name, member, stage) \
extern "C" TVM_FFI_DLL_EXPORT int __tvm_ffi_##name(void*, const TVMFFIAny* args, int32_t size, TVMFFIAny* result) { \
  TVM_FFI_SAFE_CALL_BEGIN(); TVM_FFI_ICHECK(context != nullptr); \
  invoke(stage, context->member, args, size, result); TVM_FFI_SAFE_CALL_END(); \
}
FORWARD(echo_sparse_selection_allocate, allocate, 1)
FORWARD(echo_sparse_selection_allocate_free, allocate_free, 1)
FORWARD(gather_counted, gather, 2)
FORWARD(gather_counted_u32, gather_u32, 2)
FORWARD(echo_sparse_selection_publish_bounded, publish, 3)
int trace() { return context->trace; }
bool cpu_error_identity(Function bridge) {
  TVM_FFI_ICHECK(context->fail_stage == 1);
  try {
    // Allocation injection precedes argument interpretation or CUDA execution.
    bridge(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  } catch (const tvm::ffi::Error& error) {
    return error.same_as(context->error);
  }
  return false;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure, configure);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trace, trace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cpu_error_identity, cpu_error_identity);
