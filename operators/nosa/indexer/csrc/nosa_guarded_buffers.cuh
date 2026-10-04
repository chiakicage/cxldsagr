#pragma once

#include <cstdint>
#include <initializer_list>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>

namespace nosa_guarded_buffers {
using tvm::ffi::TensorView;

inline char* data(TensorView value) {
  return static_cast<char*>(value.data_ptr()) + value.byte_offset();
}

inline bool same_device(TensorView left, TensorView right) {
  return left.device().device_type == right.device().device_type &&
         left.device().device_id == right.device().device_id;
}

inline bool overlaps(TensorView left, TensorView right) {
  if (!same_device(left, right)) return false;
  auto bytes = [](TensorView value) {
    int64_t span = 1;
    for (int d = 0; d < value.ndim(); ++d) {
      TVM_FFI_ICHECK(value.stride(d) >= 0);
      if (value.size(d) == 0) return int64_t(0);
      span += (value.size(d) - 1) * value.stride(d);
    }
    return span * value.dtype().bits / 8;
  };
  auto l = reinterpret_cast<uintptr_t>(data(left));
  auto r = reinterpret_cast<uintptr_t>(data(right));
  auto lb = bytes(left), rb = bytes(right);
  return lb > 0 && rb > 0 && l < r + rb && r < l + lb;
}

inline void disjoint(std::initializer_list<TensorView> inputs,
                     std::initializer_list<TensorView> outputs) {
  for (auto left = outputs.begin(); left != outputs.end(); ++left) {
    for (auto input : inputs) {
      TVM_FFI_ICHECK(!overlaps(*left, input))
          << "writable indexer buffers must not overlap inputs or other outputs";
    }
    for (auto right = left + 1; right != outputs.end(); ++right) {
      TVM_FFI_ICHECK(!overlaps(*left, *right))
          << "writable indexer buffers must not overlap inputs or other outputs";
    }
  }
}

inline bool const* finite_pointer(TensorView finite, TensorView input) {
  TVM_FFI_ICHECK(finite.device().device_type == kDLCUDA && same_device(finite, input) &&
                finite.ndim() == 0 && finite.IsContiguous() &&
                finite.dtype().code == kDLBool && finite.dtype().bits == 8 &&
                finite.dtype().lanes == 1);
  return reinterpret_cast<bool const*>(data(finite));
}
}  // namespace nosa_guarded_buffers
