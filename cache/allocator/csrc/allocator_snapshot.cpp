#include <pybind11/pybind11.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cstdint>
#include <set>
#include <tuple>

namespace py = pybind11;

py::dict compact_snapshot() {
  // Take the same fresh all-pool snapshot as torch's Python adapter. Only
  // deduplicate metadata whose predicates are identical for every segment.
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot({0, 0}, false);
  auto const& config = snapshot.config_metadata;
  py::dict settings;
  settings["PYTORCH_CUDA_ALLOC_CONF"] = config.last_allocator_settings;
  settings["max_split_size"] = static_cast<int64_t>(config.max_split_size);
  settings["garbage_collection_threshold"] = config.garbage_collection_threshold;
  settings["expandable_segments"] = config.expandable_segments;
  settings["pinned_num_register_threads"] = static_cast<int64_t>(config.pinned_num_register_threads);
  settings["release_lock_on_cudamalloc"] = config.release_lock_on_malloc;
  settings["pinned_use_cuda_host_register"] = config.pinned_use_host_register;
  settings["graph_capture_record_stream_reuse"] = config.graph_capture_record_stream_reuse;
  py::dict divisions;
  unsigned int key = 1;
  for (auto value : config.roundup_power2_divisions) {
    divisions[py::str(std::to_string(key))] = static_cast<int64_t>(value);
    key *= 2;
  }
  settings["roundup_power2_divisions"] = divisions;
  std::set<std::tuple<uint64_t, uint64_t, bool>> unique;
  for (auto const& segment : snapshot.segments) {
    unique.emplace(segment.owner_private_pool_id.first,
                   segment.owner_private_pool_id.second, segment.is_expandable);
  }
  py::list segments;
  for (auto const& [first, second, expandable] : unique) {
    py::dict segment;
    segment["segment_pool_id"] = py::make_tuple(first, second);
    segment["is_expandable"] = expandable;
    segments.append(segment);
  }
  py::dict result;
  result["allocator_settings"] = settings;
  result["segments"] = segments;
  result["physical_segment_count"] = snapshot.segments.size();
  return result;
}

PYBIND11_MODULE(_cxldsagr_nosa_allocator_snapshot, module) {
  module.def("snapshot", &compact_snapshot);
}
