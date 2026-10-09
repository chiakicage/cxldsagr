#pragma once

// Included after echo_sparse_append_free.cuh; reuse its guarded free selection.

namespace echo_native::free_prepare {
using tvm::ffi::TensorView;
constexpr int kLimit = 64;

__global__ void reset_publish(int* slots_out, int64_t* journal, uint32_t* counter,
                              int64_t* stats, int* metadata, int slots) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < slots;
       index += gridDim.x * blockDim.x) {
    // Each selected journal entry is consumed and cleared by its sole writer.
    const int slot = index < kLimit ? int(journal[index]) : INT32_MAX;
    slots_out[index] = slot;
    journal[index] = INT32_MAX;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    journal[slots] = INT32_MAX;
    *counter = 0;
    stats[0] = stats[1] = stats[2] = 0;
    metadata[0] = 1;
    metadata[1] = 0;
  }
}

void prepare(TensorView priority, TensorView bitmap, TensorView reverse,
             TensorView slots_out, TensorView journal, TensorView counter,
             TensorView stats, TensorView scratch, int64_t timestamp) {
  const int gpu = priority.device().device_id;
  echo_native::checked(cudaSetDevice(gpu));
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, gpu));
  const int64_t slots64 = priority.size(0) - 1;
  TVM_FFI_ICHECK(slots64 >= kLimit && slots64 < INT32_MAX);
  TVM_FFI_ICHECK(timestamp >= 0 && timestamp < INT32_MAX - 1);
  const int slots = int(slots64);
  const int blocks = std::min<int64_t>(128, (slots64 + 255) / 256);
  auto masks = static_cast<uint32_t*>(scratch.data_ptr());
  echo_native::free_append::prepare_masks<<<blocks, 256, 0, stream>>>(
      static_cast<int64_t*>(priority.data_ptr()), masks, slots, timestamp);
  echo_native::checked(cudaGetLastError());
  echo_native::free_append::select_slots<<<1, 256, 0, stream>>>(
      masks, static_cast<bool*>(bitmap.data_ptr()), static_cast<int64_t*>(reverse.data_ptr()),
      static_cast<int64_t*>(journal.data_ptr()), slots, kLimit);
  echo_native::checked(cudaGetLastError());
  reset_publish<<<blocks, 256, 0, stream>>>(
      static_cast<int*>(slots_out.data_ptr()), static_cast<int64_t*>(journal.data_ptr()),
      static_cast<uint32_t*>(counter.data_ptr()), static_cast<int64_t*>(stats.data_ptr()),
      static_cast<int*>(scratch.data_ptr()), slots);
  echo_native::checked(cudaGetLastError());
}
}  // namespace echo_native::free_prepare
