// Host dispatch only. The original ECHO/common DSOs own every CUDA kernel.
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <limits>

using tvm::ffi::Any;
using tvm::ffi::AnyView;
using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::ffi::PackedArgs;
using tvm::ffi::TensorView;

namespace {
void require(bool ok, const char* message) {
  if (!ok) TVM_FFI_THROW(ValueError) << message;
}
bool dtype(TensorView value, int code, int bits) {
  auto d = value.dtype();
  return d.code == code && d.bits == bits && d.lanes == 1;
}
bool same_device(TensorView lhs, TensorView rhs) {
  return lhs.device().device_type == rhs.device().device_type &&
         lhs.device().device_id == rhs.device().device_id;
}
bool vector(TensorView value, int64_t size, int code, int bits, TensorView reference) {
  return value.ndim() == 1 && value.shape()[0] == size && dtype(value, code, bits) &&
         same_device(value, reference) && value.IsContiguous();
}

void validate_copy(TensorView host, TensorView records, TensorView misses,
                   TensorView chosen, TensorView count) {
  require(host.device().device_type == kDLCPU, "host records must use pinned CPU memory");
  cudaPointerAttributes attributes{};
  auto status = cudaPointerGetAttributes(&attributes, host.data_ptr());
  if (status != cudaSuccess) TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(status);
  require(attributes.type == cudaMemoryTypeHost, "host records must use pinned CPU memory");
  auto hd = host.dtype(), rd = records.dtype();
  require(records.device().device_type == kDLCUDA && hd.code == rd.code &&
          hd.bits == rd.bits && hd.lanes == rd.lanes,
          "device records must be CUDA and match host dtype");
  require(host.ndim() == 2 && records.ndim() == 2 && host.shape()[1] == records.shape()[1],
          "host/device records must have the same row width");
  require(host.IsContiguous() && records.IsContiguous(), "records must be contiguous");
  require(dtype(misses, kDLInt, 64) && dtype(chosen, kDLInt, 64), "record IDs must be int64");
  require(same_device(misses, records) && same_device(chosen, records),
          "record IDs must be on the cache GPU");
  require(misses.ndim() == 1 && chosen.ndim() == 1 && misses.shape()[0] == chosen.shape()[0],
          "host/device IDs must be equal length vectors");
  require(misses.IsContiguous() && chosen.IsContiguous(), "record IDs must be contiguous");
  require(same_device(count, records) && (dtype(count, kDLInt, 64) || dtype(count, kDLUInt, 32)) &&
          count.numel() == 1 && count.IsContiguous(),
          "valid_count must be a contiguous int64/uint32 scalar on the cache GPU");
}

void validate_publish(TensorView misses, TensorView chosen, TensorView h2d,
                      TensorView d2h, TensorView priority, TensorView free,
                      TensorView clock, int64_t timestamp, TensorView count, TensorView recalled) {
  require(timestamp >= 0 && timestamp < std::numeric_limits<int64_t>::max() - 1,
          "timestamp is outside the native clock domain");
  require(vector(clock, 1, kDLInt, 64, priority), "clock must be contiguous int64[1]");
  require(vector(misses, misses.numel(), kDLInt, 64, priority), "misses must be contiguous int64 IDs");
  require(vector(chosen, misses.numel(), kDLInt, 64, priority), "chosen must match misses");
  require(vector(h2d, h2d.numel(), kDLInt, 32, priority), "host_to_device must be contiguous int32");
  require(vector(d2h, priority.numel(), kDLInt, 64, priority), "device_to_host must match priority");
  require(vector(priority, priority.numel(), kDLInt, 64, priority), "priority must be contiguous int64");
  require(vector(free, priority.numel(), kDLBool, 8, priority), "free_bitmap must match priority");
  require(vector(count, 1, kDLUInt, 32, priority), "valid_count must be contiguous uint32[1]");
  require(vector(recalled, 1, kDLInt, 64, priority), "recalled_totals must be contiguous int64[1]");
  require(misses.numel() == priority.numel() - 1,
          "bounded publication IDs must cover every physical slot");
}

Function native_function(Module module, const char* name) {
  require(std::strcmp(module->kind(), "library") == 0, "bridge requires native library modules");
  auto found = module->GetFunction(name);
  require(found.has_value(), "native library lacks a required symbol");
  Function function = *found;
  require(function->cpp_call != nullptr, "native library function lacks a C++ call target");
  return function;
}
}  // namespace

Function make_bridge(Module echo, Module common, bool free_only) {
  auto allocate = native_function(echo, free_only ? "echo_sparse_selection_allocate_free"
                                                : "echo_sparse_selection_allocate");
  auto gather_u32 = native_function(common, "gather_counted_u32");
  auto gather_i64 = native_function(common, "gather_counted");
  auto publish = native_function(echo, "echo_sparse_selection_publish_bounded");
  return Function::FromPacked([allocate, gather_u32, gather_i64, publish](PackedArgs args, Any* result) {
    require(args.size() == 18, "allocate/copy/publish requires 18 arguments");
    // The first thirteen arguments have already passed production allocation
    // validation, including whole-storage alias checks and scratch reservation.
    Any ignored;
    allocate->CallPacked(args.data(), 13, &ignored);
    // These conversions and checks deliberately occur AFTER allocation.
    auto misses = args[8].cast<TensorView>(), chosen = args[9].cast<TensorView>();
    auto host = args[13].cast<TensorView>(), records = args[14].cast<TensorView>();
    auto count = args[15].cast<TensorView>();
    validate_copy(host, records, misses, chosen, count);
    if (misses.numel()) {
      auto& gather = dtype(count, kDLUInt, 32) ? gather_u32 : gather_i64;
      gather(host, records, misses, chosen, count, 128);
    }
    // Publication-only metadata is checked AFTER gather, exactly as before.
    auto h2d = args[3].cast<TensorView>(), d2h = args[4].cast<TensorView>();
    auto priority = args[5].cast<TensorView>(), free = args[6].cast<TensorView>();
    auto clock = args[16].cast<TensorView>(), recalled = args[17].cast<TensorView>();
    auto timestamp = args[12].cast<int64_t>();
    validate_publish(misses, chosen, h2d, d2h, priority, free, clock, timestamp, count, recalled);
    publish(misses, chosen, h2d, d2h, priority, free, clock, timestamp, count, recalled);
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_bridge, make_bridge);
