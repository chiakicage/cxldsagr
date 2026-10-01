// NOSA compressed-QK scoring for Hopper.
// QK WGMMA and online-softmax reference: EzKernelKit commit
// 995d5a47fc35cef2c3d536133c8c87e8a6e46acd,
// csrc/sparse_attn/sm90/fwd.cu. That source computes QK within attention;
// this owned adaptation uses a split normalizer pass followed by a second QK
// pass for normalized GQA scores and five-window pooling. The normalizer
// overlaps QK(next) with softmax(current). Layouts and pooling are local;
// no EzKernelKit headers or libraries are used.
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cmath>
#include <cstdint>
#include "nosa_selection_prefix.cuh"
#include "nosa_selection_cutoff.cuh"

namespace nosa_scores {
using namespace cute;
using tvm::ffi::TensorView;
constexpr int kColumns = 128;
// Keep scaled logits and maxima in natural units. Converting a large common
// offset to base two before subtraction loses small, meaningful differences.
// Only the centered difference is converted for the approximate exponent.
__device__ __forceinline__ float exp_approx(float x) {
  float result;
  float exponent = x * 1.4426950408889634f;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
}
using Barrier = cutlass::arch::ClusterTransactionBarrier;
template <class S, class TMA, class QS, class QTMA> struct TensorMaps {
  S shape;
  TMA tma;
  QS qshape;
  QTMA qtma;
};

struct Params {
  const void *q, *k, *positions;
  void* out;
  float* normalizers;
  int splits;
  int rows, heads, count, blocks;
  int64_t query_start;
  int64_t qr, qh, qg, kc, kh;
  bool contiguous, pos64;
  int device;
  unsigned const* ranking = nullptr;
  int64_t* ids = nullptr;
  bool* valid = nullptr;
};

template <typename T> struct Traits {
  static constexpr int M = 64, N = kColumns, D = 128;
  using QLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},
                                       Shape<Int<M>, Int<D>>{}));
  using KLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},
                                       Shape<Int<N>, Int<D>>{}));
  using Atom = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
      GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
      GMMA::MMA_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
  using MMA = decltype(make_tiled_mma(Atom{}, Layout<Shape<_1, _1, _1>>{}));
  template <int Stages = 2> struct Shared {
    alignas(128) T q[cosize_v<QLayout>];
    alignas(128) T k[Stages][cosize_v<KLayout>];
    T rounded[4][N];
    int64_t positions[4];
    Barrier ready[Stages], q_ready;
  };
};

template <typename T, typename Shared, typename Acc>
__device__ __forceinline__ void issue(Shared& smem, int buffer, Acc& acc) {
  typename Traits<T>::MMA mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto a = thr.partition_fragment_A(make_tensor(make_smem_ptr(smem.q), typename Traits<T>::QLayout{}));
  auto b = thr.partition_fragment_B(make_tensor(make_smem_ptr(smem.k[buffer]), typename Traits<T>::KLayout{}));
  warpgroup_fence_operand(acc);
  warpgroup_arrive();
  mma.accumulate_ = GMMA::ScaleOut::Zero;
  CUTE_UNROLL
  for (int kk = 0; kk < size<2>(a); ++kk) {
    cute::gemm(mma, a(_, _, kk), b(_, _, kk), acc);
    mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  warpgroup_fence_operand(acc);
}

template <bool Masked, bool Record = false, typename Acc>
__device__ __forceinline__ void normalize_tile(Acc& score, int tile, int count,
    int64_t position, float (&maximum)[2], float (&denominator)[2], __half2* tile_summaries = nullptr) {
  constexpr float scale = 0.08838834764831845f;  // 1 / sqrt(128)
  int lane = threadIdx.x % 32;
  CUTE_UNROLL
  for (int row = 0; row < 2; ++row) {
    float block_max = -INFINITY;
    CUTE_UNROLL
    for (int i = row * 2; i < size(score); i += 4) {
      int col = tile * kColumns + 8 * (i / 4) + (lane % 4) * 2;
      if constexpr (Masked) {
        score(i) = col < count && col * 16 + 31 <= position ? __fmul_rn(score(i), scale) : -INFINITY;
        score(i + 1) = col + 1 < count && (col + 1) * 16 + 31 <= position ? __fmul_rn(score(i + 1), scale) : -INFINITY;
      } else {
        score(i) = __fmul_rn(score(i), scale);
        score(i + 1) = __fmul_rn(score(i + 1), scale);
      }
      block_max = fmaxf(block_max, fmaxf(score(i), score(i + 1)));
    }
    block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, 1));
    block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, 2));
    if constexpr (Record) {
      // Lane three owns aligned column 127 for this per-head row. Capture
      // its natural-unit RN logit before exponentiation. Both summaries are
      // rounded toward +infinity, preserving the later upper-bound proof.
      // One packed word retains the four-tile conflict-free shared layout.
      if (lane % 4 == 3) {
        float halo = score(size(score) - 3 + row * 2);
        tile_summaries[(lane / 4 + row * 8) * 36 + tile] =
            __halves2half2(__float2half_ru(block_max), __float2half_ru(halo));
      }
    }
    float next_max = fmaxf(maximum[row], block_max);
    float safe = next_max == -INFINITY ? 0.0f : next_max;
    float local_sum = 0.0f;
    CUTE_UNROLL
    for (int i = row * 2; i < size(score); i += 4) {
      local_sum += exp_approx(score(i) - safe) + exp_approx(score(i + 1) - safe);
    }
    // Each of the four lanes keeps its partial denominator until the first
    // pass finishes; only maxima need a cross-lane reduction every tile.
    denominator[row] = denominator[row] * exp_approx(maximum[row] - safe) + local_sum;
    maximum[row] = next_max;
  }
}

__device__ __forceinline__ float mask_pool(float value, int block, int64_t qblock) {
  bool causal = block <= qblock;
  bool mandatory = block == 0 || (causal && qblock <= block + 16);
  value = causal ? (mandatory ? INFINITY : value) : -INFINITY;
  return value == 0.0f ? 0.0f : value;
}

template <typename T, typename Shared, typename Map>
__device__ __forceinline__ void load_k(Shared& smem, int buffer, int start, const Map& map) {
  if (threadIdx.x == 0) {
    auto global = domain_offset(make_coord(start, _0{}), map.tma.get_tma_tensor(map.shape)(_, _, blockIdx.y));
    auto src = flat_divide(global, Tile<Int<kColumns>, _128>{})(_, _, _0{}, _0{});
    auto dst = make_tensor(make_smem_ptr(smem.k[buffer]), typename Traits<T>::KLayout{});
    auto slice = map.tma.get_slice(_0{});
    cute::copy(map.tma.with(reinterpret_cast<Barrier::ValueType&>(smem.ready[buffer])),
               slice.partition_S(src), slice.partition_D(dst));
    smem.ready[buffer].arrive_and_expect_tx(kColumns * 128 * sizeof(T));
  }
}

template <typename T, typename Shared, typename Map>
__device__ __forceinline__ void load_query(const Params& p, Shared& smem, const Map& map) {
  int tid = threadIdx.x;
  if (tid < 4) {
    int row = blockIdx.x * 4 + tid;
    smem.positions[tid] = row >= p.rows ? -1 : p.contiguous ? p.query_start + row :
      p.pos64 ? static_cast<const int64_t*>(p.positions)[row] :
                static_cast<const int32_t*>(p.positions)[row];
  }
  if (tid == 0) {
    auto global = map.qtma.get_tma_tensor(map.qshape)(_, _, blockIdx.y);
    auto src = flat_divide(global, Tile<_64, _128>{})(_, _, blockIdx.x, _0{});
    auto dst = make_tensor(make_smem_ptr(smem.q), typename Traits<T>::QLayout{});
    auto slice = map.qtma.get_slice(_0{});
    cute::copy(map.qtma.with(reinterpret_cast<Barrier::ValueType&>(smem.q_ready)),
               slice.partition_S(src), slice.partition_D(dst));
    smem.q_ready.arrive_and_expect_tx(64 * 128 * sizeof(T));
  }
}

template <typename T, typename Map>
__global__ __launch_bounds__(128) void normalizer_kernel(__grid_constant__ const Params p, __grid_constant__ const Map map) {
  using Tr = Traits<T>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& smem = *reinterpret_cast<typename Tr::template Shared<>*>(storage);
  int tid = threadIdx.x;
  if (tid == 0) {
    smem.ready[0].init(1); smem.ready[1].init(1); smem.q_ready.init(1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  load_query<T>(p, smem, map);
  smem.q_ready.wait(0);
  __syncthreads();
  int tiles = (p.count + kColumns - 1) / kColumns;
  int per_split = (tiles + p.splits - 1) / p.splits;
  int begin = blockIdx.z * per_split, end = min(begin + per_split, tiles);
  auto acc0 = partition_fragment_C(typename Tr::MMA{}, Shape<_64, Int<kColumns>>{});
  auto acc1 = partition_fragment_C(typename Tr::MMA{}, Shape<_64, Int<kColumns>>{});
  float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {0.0f, 0.0f};
  int64_t position = smem.positions[tid / 32];
  if (begin < end) {
    load_k<T>(smem, begin & 1, begin * kColumns, map);
    smem.ready[begin & 1].wait(0);
    if (begin & 1) issue<T>(smem, 1, acc1); else issue<T>(smem, 0, acc0);
    if (begin + 1 < end) load_k<T>(smem, (begin + 1) & 1, (begin + 1) * kColumns, map);
    for (int tile = begin; tile < end; ++tile) {
      if (tile + 1 < end) {
        smem.ready[(tile + 1) & 1].wait(((tile + 1 - begin) / 2) & 1);
        if ((tile + 1) & 1) issue<T>(smem, 1, acc1); else issue<T>(smem, 0, acc0);
        warpgroup_wait<1>();
      } else {
        warpgroup_wait<0>();
      }
      if (tile + 2 < end) load_k<T>(smem, (tile + 2) & 1, (tile + 2) * kColumns, map);
      bool full = (tile + 1) * kColumns <= p.count && ((tile + 1) * kColumns - 1) * 16 + 31 <= position;
      if (full) {
        if (tile & 1) normalize_tile<false>(acc1, tile, p.count, position, maximum, denominator);
        else normalize_tile<false>(acc0, tile, p.count, position, maximum, denominator);
      } else {
        if (tile & 1) normalize_tile<true>(acc1, tile, p.count, position, maximum, denominator);
        else normalize_tile<true>(acc0, tile, p.count, position, maximum, denominator);
      }
    }
  }
  int row = blockIdx.x * 4 + tid / 32;
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    float sum = denominator[r];
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if (tid % 4 == 0 && row < p.rows) {
      int group = (tid % 32) / 4 + r * 8;
      int64_t index = ((int64_t(blockIdx.z) * p.rows + row) * p.heads + blockIdx.y) * 16 + group;
      p.normalizers[index * 2] = maximum[r];
      p.normalizers[index * 2 + 1] = sum;
    }
  }
}

template <typename T, typename O, bool Pool, bool Masked, typename Shared, typename Acc>
__device__ __forceinline__ void write_result(const Params& p, Shared& smem, Acc& acc,
    int start, int64_t position, const float (&maximum)[2], const float (&inv)[2]) {
  int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  int row = blockIdx.x * 4 + warp, head = blockIdx.y;
  constexpr int BlocksPerTile = (kColumns - 1) / 4;
  constexpr float scale = 0.08838834764831845f;  // 1 / sqrt(128)
  // The normalizer stores the rounded FP32 scaled logit. Reproduce that
  // rounding before subtraction; an FFMA here keeps extra low bits and can
  // bias even a uniform softmax when logits are large. Likewise, retain max
  // and inverse separately: folding log(sum) into a large max loses precision.
  CUTE_UNROLL
  for (int i = 0; i < size(acc); i += 4) {
    int local = 8 * (i / 4) + (lane % 4) * 2, col = start + local;
    float a = 0.0f, b = 0.0f;
    if (!Masked || (col >= 0 && col < p.count && col * 16 + 31 <= position))
      a = exp_approx(__fmul_rn(acc(i), scale) - maximum[0]) * inv[0]
        + exp_approx(__fmul_rn(acc(i + 2), scale) - maximum[1]) * inv[1];
    if (!Masked || (col + 1 >= 0 && col + 1 < p.count && (col + 1) * 16 + 31 <= position))
      b = exp_approx(__fmul_rn(acc(i + 1), scale) - maximum[0]) * inv[0]
        + exp_approx(__fmul_rn(acc(i + 3), scale) - maximum[1]) * inv[1];
    CUTE_UNROLL
    for (int offset = 4; offset <= 16; offset *= 2) {
      a += __shfl_xor_sync(0xffffffff, a, offset);
      b += __shfl_xor_sync(0xffffffff, b, offset);
    }
    if (lane < 4) {
      if constexpr (Pool) {
        smem.rounded[warp][local] = T(a);
        smem.rounded[warp][local + 1] = T(b);
      } else if (row < p.rows) {
        auto output = static_cast<T*>(p.out) + (int64_t(row) * p.heads + head) * p.count;
        if (col < p.count) output[col] = T(a);
        if (col + 1 < p.count) output[col + 1] = T(b);
      }
    }
  }
  if constexpr (Pool) {
    __syncwarp();
    if (lane < BlocksPerTile) {
      int block = blockIdx.z * BlocksPerTile + lane;
      float value = -INFINITY;
      CUTE_UNROLL
      for (int j = 0; j < 5; ++j) {
        int c = start + lane * 4 + j;
        if (c >= 0 && c < p.count) value = fmaxf(value, float(smem.rounded[warp][lane * 4 + j]));
      }
      if (row < p.rows && block < p.blocks)
        static_cast<O*>(p.out)[(int64_t(row) * p.heads + head) * p.blocks + block] =
          O(mask_pool(value, block, position / 64));
    }
  }
}

template <typename T, typename O, bool Pool, typename Map>
__global__ __launch_bounds__(128) void scores_kernel(__grid_constant__ const Params p, __grid_constant__ const Map map) {
  using Tr = Traits<T>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& smem = *reinterpret_cast<typename Tr::template Shared<1>*>(storage);
  int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  int row = blockIdx.x * 4 + warp, head = blockIdx.y;
  constexpr int BlocksPerTile = (kColumns - 1) / 4;
  int start = Pool ? int(blockIdx.z) * BlocksPerTile * 4 - 1 : int(blockIdx.z) * kColumns;
  if (tid == 0) {
    smem.ready[0].init(1); smem.q_ready.init(1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  load_query<T>(p, smem, map);
  load_k<T>(smem, 0, start, map);
  // Overlap normalizer loads with the Q/K TMA transfers.
  float maximum[2] = {-INFINITY, -INFINITY}, inv[2];
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    int group = lane / 4 + r * 8;
    float m = -INFINITY, sum = 0.0f;
    if (row < p.rows && p.splits == 1) {
      int64_t index = (int64_t(row) * p.heads + head) * 16 + group;
      m = p.normalizers[index * 2];
      sum = p.normalizers[index * 2 + 1];
    } else if (row < p.rows) {
      for (int split = 0; split < p.splits; ++split) {
        int64_t index = ((int64_t(split) * p.rows + row) * p.heads + head) * 16 + group;
        float other = p.normalizers[index * 2], s = p.normalizers[index * 2 + 1];
        float next = fmaxf(m, other), safe = next == -INFINITY ? 0.0f : next;
        sum = sum * exp_approx(m - safe) + s * exp_approx(other - safe);
        m = next;
      }
    }
    maximum[r] = m == -INFINITY ? 0.0f : m;
    inv[r] = 1.0f / (sum > 0.0f ? sum : 1.0f);
  }
  smem.q_ready.wait(0);
  smem.ready[0].wait(0);
  __syncthreads();
  auto acc = partition_fragment_C(typename Tr::MMA{}, Shape<_64, Int<kColumns>>{});
  issue<T>(smem, 0, acc);
  warpgroup_wait<0>();
  int64_t position = smem.positions[warp];
  bool full = start >= 0 && start + kColumns <= p.count && (start + kColumns - 1) * 16 + 31 <= position;
  if (full) write_result<T, O, Pool, false>(p, smem, acc, start, position, maximum, inv);
  else write_result<T, O, Pool, true>(p, smem, acc, start, position, maximum, inv);
}

// A context ID is unique even after context destruction. Cache attributes
// for the actual loaded kernel, device, and shared-memory requirement.
template <typename Kernel>
bool configure(Kernel kernel, int shared, int device, bool register_guard = false) {
  static auto get_context_id = [] {
    void* entry = nullptr;
#if CUDART_VERSION >= 12050
    auto error = cudaGetDriverEntryPointByVersion("cuCtxGetId", &entry, 12000,
                                                  cudaEnableDefault, nullptr);
#else
    auto error = cudaGetDriverEntryPoint("cuCtxGetId", &entry, cudaEnableDefault, nullptr);
#endif
    TVM_FFI_ICHECK(error == cudaSuccess && entry) << cudaGetErrorString(error);
    return reinterpret_cast<decltype(&cuCtxGetId)>(entry);
  }();
  unsigned long long context = 0;
  TVM_FFI_ICHECK(get_context_id(nullptr, &context) == CUDA_SUCCESS);
  struct Entry {
    unsigned long long context;
    const void* kernel;
    int shared, device;
    bool register_guard, feasible;
  };
  static thread_local Entry entries[16] = {};
  static thread_local unsigned next = 0;
  auto pointer = reinterpret_cast<const void*>(kernel);
  for (const auto& entry : entries)
    if (entry.context == context && entry.kernel == pointer && entry.shared == shared &&
        entry.device == device && entry.register_guard == register_guard)
      return entry.feasible;
  bool feasible = true;
  if (register_guard) {
    cudaFuncAttributes attributes{};
    auto error = cudaFuncGetAttributes(&attributes, pointer);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    // The 640-thread launch must provide at least 96 initial registers/thread
    // before four consumer WGs request 112 and the producer WG releases to 24.
    feasible = attributes.numRegs >= 96;
  }
  if (feasible) {
    auto error = cudaFuncSetAttribute(pointer, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    error = cudaFuncSetAttribute(pointer, cudaFuncAttributePreferredSharedMemoryCarveout,
                                cudaSharedmemCarveoutMaxShared);
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    if (register_guard) {
      int resident = 0;
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 640, shared);
      TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
      feasible = resident >= 1;
    }
  }
  entries[next++ % 16] = {context, pointer, shared, device, register_guard, feasible};
  return feasible;
}

template <typename T, typename O, bool Pool>
void launch(const Params& p, cudaStream_t stream) {
  auto shape = make_shape(p.count, 128, p.heads);
  auto tma = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<const T*>(p.k)),
                  make_layout(shape, make_stride(p.kc, _1{}, p.kh))),
      typename Traits<T>::KLayout{});
  auto qshape = make_shape(make_shape(_16{}, p.rows), _128{}, p.heads);
  auto qtma = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<const T*>(p.q)),
                  make_layout(qshape, make_stride(make_stride(p.qg, p.qr), _1{}, p.qh))),
      typename Traits<T>::QLayout{}, Tile<_64, _128>{}, _1{});
  TensorMaps<decltype(shape), decltype(tma), decltype(qshape), decltype(qtma)> map{shape, tma, qshape, qtma};
  auto normalizer = normalizer_kernel<T, decltype(map)>;
  auto kernel = scores_kernel<T, O, Pool, decltype(map)>;
  int normalizer_shared = sizeof(typename Traits<T>::template Shared<>);
  int score_shared = sizeof(typename Traits<T>::template Shared<1>);
  configure(normalizer, normalizer_shared, p.device);
  configure(kernel, score_shared, p.device);
  normalizer<<<dim3((p.rows + 3) / 4, p.heads, p.splits), 128, normalizer_shared, stream>>>(p, map);
  int tiles = Pool ? (p.blocks + (kColumns - 1) / 4 - 1) / ((kColumns - 1) / 4) : (p.count + kColumns - 1) / kColumns;
  kernel<<<dim3((p.rows + 3) / 4, p.heads, tiles), 128, score_shared, stream>>>(p, map);
  auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

#include "nosa_scores_fused.cuh"
#include "nosa_scores_pruned.cuh"

void scores_out(TensorView q, TensorView k, TensorView positions, TensorView out, TensorView normalizers,
                int64_t query_start, int64_t blocks, bool contiguous, bool pool_output) {
  auto same_device = [](TensorView a, TensorView b) {
    return a.device().device_type == b.device().device_type &&
           a.device().device_id == b.device().device_id;
  };
  auto data = [](TensorView tensor) {
    return static_cast<char*>(tensor.data_ptr()) + tensor.byte_offset();
  };
  TVM_FFI_ICHECK(q.ndim() == 4 && k.ndim() == 3);
  TVM_FFI_ICHECK(out.ndim() == 3 || (pool_output && out.ndim() == 2));
  TVM_FFI_ICHECK(q.size(2) == 16 && q.size(3) == 128 && k.size(2) == 128);
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA && same_device(k, q) && same_device(out, q));
  TVM_FFI_ICHECK(q.dtype() == k.dtype() && q.dtype().bits == 16 && q.dtype().lanes == 1);
  TVM_FFI_ICHECK(q.dtype().code == kDLBfloat || q.dtype().code == kDLFloat);
  TVM_FFI_ICHECK(q.stride(3) == 1 && k.stride(2) == 1 && out.IsContiguous());
  TVM_FFI_ICHECK(q.size(0) <= 262144 && k.size(0) > 0 && k.size(0) <= 16383);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(data(q)) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(data(k)) % 16 == 0);
  TVM_FFI_ICHECK(q.size(1) == k.size(1));
  if (out.ndim() == 3) {
    TVM_FFI_ICHECK(out.size(0) == q.size(0) && out.size(1) == q.size(1));
    TVM_FFI_ICHECK(out.size(2) == (pool_output ? blocks : k.size(0)));
  } else {
    // The shared indexer scratch flattens query and KV-head for top-k.
    TVM_FFI_ICHECK(out.size(0) == q.size(0) * q.size(1) && out.size(1) == blocks);
  }
  TVM_FFI_ICHECK(out.dtype() == q.dtype() ||
                (pool_output && out.dtype().code == kDLFloat && out.dtype().bits == 32));
  for (int d = 0; d < 3; ++d) TVM_FFI_ICHECK(q.stride(d) > 0 && q.stride(d) % 8 == 0);
  for (int d = 0; d < 2; ++d) TVM_FFI_ICHECK(k.stride(d) > 0 && k.stride(d) % 8 == 0);
  if (!contiguous) {
    TVM_FFI_ICHECK(same_device(positions, q) && positions.ndim() == 1 && positions.size(0) == q.size(0));
    TVM_FFI_ICHECK(positions.dtype().code == kDLInt && (positions.dtype().bits == 32 || positions.dtype().bits == 64));
    TVM_FFI_ICHECK(positions.stride(0) == 1);
  }
  TVM_FFI_ICHECK(normalizers.ndim() == 5 && normalizers.dtype().code == kDLFloat && normalizers.dtype().bits == 32);
  TVM_FFI_ICHECK(normalizers.IsContiguous() && same_device(normalizers, q));
  TVM_FFI_ICHECK(normalizers.size(1) == q.size(0) && normalizers.size(2) == q.size(1) &&
                normalizers.size(3) == 16 && normalizers.size(4) == 2);
  TVM_FFI_ICHECK(normalizers.size(0) > 0 && normalizers.size(0) <= 16);
  Params p{data(q), data(k), data(positions), data(out), static_cast<float*>(static_cast<void*>(data(normalizers))), int(normalizers.size(0)),
    int(q.size(0)), int(q.size(1)), int(k.size(0)), int(blocks), query_start,
    q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), contiguous, positions.dtype().bits == 64, q.device().device_id};
  if (!p.rows) return;
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q.device().device_id));
  // A sufficiently large grid amortizes the shared K schedule.
  // The dispatch is shape-based; values and cached selections do not affect it.
  if (pool_output && p.rows >= 1024 && ((p.rows + 15) / 16) * p.heads >= 128) {
    if (q.dtype().code == kDLBfloat) {
      if (out.dtype().bits == 32) fused_scores::launch<cutlass::bfloat16_t, float, true>(p, stream);
      else fused_scores::launch<cutlass::bfloat16_t, cutlass::bfloat16_t, true>(p, stream);
    } else {
      if (out.dtype().bits == 32) fused_scores::launch<cutlass::half_t, float, true>(p, stream);
      else fused_scores::launch<cutlass::half_t, cutlass::half_t, true>(p, stream);
    }
    return;
  }
  if (q.dtype().code == kDLBfloat) {
    if (!pool_output) launch<cutlass::bfloat16_t, cutlass::bfloat16_t, false>(p, stream);
    else if (out.dtype().bits == 32) launch<cutlass::bfloat16_t, float, true>(p, stream);
    else launch<cutlass::bfloat16_t, cutlass::bfloat16_t, true>(p, stream);
  } else {
    if (!pool_output) launch<cutlass::half_t, cutlass::half_t, false>(p, stream);
    else if (out.dtype().bits == 32) launch<cutlass::half_t, float, true>(p, stream);
    else launch<cutlass::half_t, cutlass::half_t, true>(p, stream);
  }
}
}  // namespace nosa_scores

TVM_FFI_DLL_EXPORT_TYPED_FUNC(scores_out, nosa_scores::scores_out);


// Ranking must be prepared from the guarded BF16 CIS prefix before this call.
bool fused_select_out(tvm::ffi::TensorView q, tvm::ffi::TensorView k,
                      tvm::ffi::TensorView ranking, tvm::ffi::TensorView ids,
                      tvm::ffi::TensorView valid, int64_t query_start, int64_t blocks) {
  auto data=[](tvm::ffi::TensorView t) { return static_cast<char*>(t.data_ptr())+t.byte_offset(); };
  auto same_device=[&](tvm::ffi::TensorView t) {
    return t.device().device_type==kDLCUDA && t.device().device_id==q.device().device_id;
  };
  TVM_FFI_ICHECK(q.device().device_type==kDLCUDA && same_device(k) && same_device(ranking) && same_device(ids) && same_device(valid));
  TVM_FFI_ICHECK(q.ndim()==4 && k.ndim()==3 && q.size(1)==k.size(1));
  TVM_FFI_ICHECK(q.dtype().code==kDLBfloat && q.dtype().bits==16 && q.dtype()==k.dtype());
  TVM_FFI_ICHECK(q.size(2)==16 && q.size(3)==128 && k.size(2)==128);
  TVM_FFI_ICHECK(q.size(0)>=1024 && q.size(0)<=1088 && k.size(0)>0 && k.size(0)<=16383);
  TVM_FFI_ICHECK(((q.size(0)+15)/16)*q.size(1)>=128);
  TVM_FFI_ICHECK(query_start/64>=64 && (query_start+q.size(0)-1)/64-query_start/64<=16);
  TVM_FFI_ICHECK(blocks>64 && blocks<=1056 && query_start/64<=blocks-1);
  TVM_FFI_ICHECK(q.stride(3)==1 && k.stride(2)==1);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(data(q))%16==0 && reinterpret_cast<uintptr_t>(data(k))%16==0);
  for(int d=0;d<3;++d) TVM_FFI_ICHECK(q.stride(d)>0 && q.stride(d)%8==0);
  for(int d=0;d<2;++d) TVM_FFI_ICHECK(k.stride(d)>0 && k.stride(d)%8==0);
  TVM_FFI_ICHECK(ranking.ndim()==2 && ranking.size(0)==q.size(1) && ranking.size(1)==64 && ranking.IsContiguous());
  TVM_FFI_ICHECK(ranking.dtype().code==kDLInt && ranking.dtype().bits==32);
  TVM_FFI_ICHECK(ids.ndim()==3 && ids.size(0)==q.size(0) && ids.size(1)==q.size(1) && ids.size(2)==64 && ids.IsContiguous());
  TVM_FFI_ICHECK(ids.dtype().code==kDLInt && ids.dtype().bits==64);
  TVM_FFI_ICHECK(valid.ndim()==3 && valid.size(0)==q.size(0) && valid.size(1)==q.size(1) && valid.size(2)==64 && valid.IsContiguous() && valid.dtype().bits==8);
  nosa_scores::Params p{data(q),data(k),nullptr,nullptr,nullptr,1,
    int(q.size(0)),int(q.size(1)),int(k.size(0)),int(blocks),query_start,
    q.stride(0),q.stride(1),q.stride(2),k.stride(0),k.stride(1),true,true,q.device().device_id,
    reinterpret_cast<unsigned const*>(data(ranking)),reinterpret_cast<int64_t*>(data(ids)),reinterpret_cast<bool*>(data(valid))};
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,q.device().device_id));
  // Enable pruning only for the measured 64K-prefix + 1K-query geometry.
  // Shorter prefixes can retain too many tiles; they keep the fused schedule.
  if (p.rows == 1024 && p.heads == 2 && p.count == 4159 && p.blocks == 1040 &&
      p.query_start == 65536 &&
      nosa_scores::pruned_scores::launch<cutlass::bfloat16_t,cutlass::bfloat16_t,true,true>(p,stream))
    return true;
  return nosa_scores::fused_scores::launch<cutlass::bfloat16_t,cutlass::bfloat16_t,true,true>(p,stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_select_out,fused_select_out);
