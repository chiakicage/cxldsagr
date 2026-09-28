// Resident NOSA attention, implemented locally using CUTLASS/CuTe primitives.
// Scheduling reference: EzKernelKit 995d5a47fc35cef2c3d536133c8c87e8a6e46acd,
// csrc/sparse_attn/sm90/fwd.cu. No EzKernelKit headers/runtime.
// The per-query path uses 16 GQA heads as the WGMMA N dimension.
// Longer batches share selected K/V across four independent queries; see
// nosa_attention_grouped.cuh. Both paths pipeline TMA with WGMMA.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <climits>
#include <cmath>
#include <cstdint>

namespace nosa_attention {
using namespace cute;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using tvm::ffi::TensorView;

struct Params {
  void const *q, *ids, *bias;
  int64_t qr, qh;
  bool const *valid;
  void *out;
  int *group_fallback;
  int tokens, queries, heads, kv_heads, count, query_start;
  int bias_type;
  bool ids64;
  int64_t ir, ih, ib, mr, mh, mb, br, bh;
};

__device__ __forceinline__ int selected_block(Params const &p, int slot) {
  int64_t offset = blockIdx.x * p.ir + blockIdx.y * p.ih + slot * p.ib;
  int64_t id = p.ids64 ? static_cast<int64_t const *>(p.ids)[offset]
                       : static_cast<int32_t const *>(p.ids)[offset];
  bool valid = !p.valid || p.valid[blockIdx.x * p.mr + blockIdx.y * p.mh + slot * p.mb];
  // Invalid selections are loaded from the TMA OOB region, which returns zero.
  return valid && id >= 0 && id < (p.tokens + 63) / 64 && id * 64 <= p.query_start + blockIdx.x
             ? int(id) : (p.tokens + 63) / 64;
}

__device__ __forceinline__ float bias_at(Params const &p, int token) {
  int64_t at = int64_t(token) * p.br + blockIdx.y * p.bh;
  if (p.bias_type == 0) return static_cast<float const *>(p.bias)[at];
  if (p.bias_type == 1) return float(static_cast<cutlass::bfloat16_t const *>(p.bias)[at]);
  return float(static_cast<cutlass::half_t const *>(p.bias)[at]);
}

template <typename T> struct Traits {
  using QLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, Shape<_16, _128>{}));
  using KVLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, Shape<_64, _128>{}));
  using PLayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<T>{}, Shape<_64, _16>{}));
  using PTLayout = decltype(composition(PLayout{}, Layout<Shape<_16, _64>, Stride<_64, _1>>{}));
  using VTLayout = decltype(composition(KVLayout{}, Layout<Shape<_128, _64>, Stride<_64, _1>>{}));
  using QKOp = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
      GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
      GMMA::MMA_64x16x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
  using PVOp = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
      GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>,
      GMMA::MMA_64x16x16_F32F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>>;
  using QKMma = decltype(make_tiled_mma(QKOp{}, Layout<Shape<_1, _1, _1>>{}));
  using PVMma = decltype(make_tiled_mma(PVOp{}, Layout<Shape<_1, _1, _1>>{}, Tile<_128, _16, _16>{}));
  struct Shared {
    alignas(128) T q[cosize_v<QLayout>];
    alignas(128) T k[cosize_v<KVLayout>];
    alignas(128) T v[cosize_v<KVLayout>];
    alignas(128) T probability[cosize_v<PLayout>];
    float exchange[4][4][4];
    int blocks[64], count;
    Barrier q_ready, k_ready, v_ready, k_free, v_free;
  };
};

template <class MMA, class A, class B, class C>
__device__ __forceinline__ void mma(bool clear, MMA op, A const &a, B const &b, C &c, int thread) {
  auto slice = op.get_slice(thread);
  auto af = slice.partition_fragment_A(a);
  auto bf = slice.partition_fragment_B(b);
  warpgroup_fence_operand(c);
  warpgroup_arrive();
  op.accumulate_ = clear ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTE_UNROLL
  for (int k = 0; k < size<2>(af); ++k) {
    cute::gemm(op, af(_, _, k), bf(_, _, k), c);
    op.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_fence_operand(c);
}

template <class TMA, class Src, class Dst>
__device__ __forceinline__ void tma_load(TMA const &tma, Src src, Dst dst, Barrier &bar, int bytes) {
  auto slice = tma.get_slice(_0{});
  cute::copy(tma.with(reinterpret_cast<Barrier::ValueType &>(bar)),
             slice.partition_S(src), slice.partition_D(dst));
  bar.arrive_and_expect_tx(bytes);
}

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

template <bool Max>
__device__ __forceinline__ void reduce_heads(float (&x)[4], float (&shared)[4][4][4], int tid) {
  int lane = tid % 32, warp = tid / 32;
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    CUTE_UNROLL
    for (int distance = 4; distance <= 16; distance *= 2) {
      float other = __shfl_xor_sync(0xffffffff, x[i], distance);
      x[i] = Max ? fmaxf(x[i], other) : x[i] + other;
    }
  }
  if (lane < 4) *reinterpret_cast<float4 *>(shared[warp][lane]) = *reinterpret_cast<float4 *>(x);
  consumer_sync();
  if (lane < 4) {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      float r = shared[0][lane][i];
      CUTE_UNROLL
      for (int w = 1; w < 4; ++w) r = Max ? fmaxf(r, shared[w][lane][i]) : r + shared[w][lane][i];
      x[i] = r;
    }
  }
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) x[i] = __shfl_sync(0xffffffff, x[i], lane % 4);
}

template <class QS, class QT, class KS, class KT, class VS, class VT>
struct Maps { QS qs; QT qt; KS ks; KT kt; VS vs; VT vt; };

template <typename T, class MapsT>
__global__ __launch_bounds__(256, 4) void attention_kernel(
    __grid_constant__ Params const p, __grid_constant__ MapsT const maps) {
  if (p.group_fallback && !p.group_fallback[(blockIdx.x / 4) * p.kv_heads + blockIdx.y]) return;
  using Tr = Traits<T>;
  extern __shared__ char storage[];
  auto &s = *reinterpret_cast<typename Tr::Shared *>(storage);
  int tid = threadIdx.x % 128, lane = tid % 32, warp = tid / 32;
  auto sq = make_tensor(make_smem_ptr(s.q), typename Tr::QLayout{});
  auto sk = make_tensor(make_smem_ptr(s.k), typename Tr::KVLayout{});
  auto sv = make_tensor(make_smem_ptr(s.v), typename Tr::KVLayout{});
  if (threadIdx.x == 0) {
    s.q_ready.init(1); s.k_ready.init(1); s.v_ready.init(1);
    s.k_free.init(128); s.v_free.init(128);
    cutlass::arch::fence_barrier_init();
  }
  if (threadIdx.x < 32) {
    // Short causal prefixes often occupy only a few of the 64 selection
    // slots. Warp ballots compact them once, preserving selection order.
    int count = 0;
    for (int base = 0; base < p.count; base += 32) {
      int block = base + lane < p.count ? selected_block(p, base + lane) : (p.tokens + 63) / 64;
      bool valid = block < (p.tokens + 63) / 64;
      unsigned bits = __ballot_sync(0xffffffff, valid);
      int rank = __popc(bits & ((1u << lane) - 1));
      if (valid) s.blocks[count + rank] = block;
      count += __popc(bits);
    }
    if (lane == 0) s.count = count;
  }
  __syncthreads();
  if (!s.count) {
    auto out = static_cast<T *>(p.out) + (int64_t(blockIdx.x) * p.heads + blockIdx.y * 16) * 128;
    for (int i = threadIdx.x; i < 16 * 128; i += 256) out[i] = T(0.f);
    return;
  }
  if (threadIdx.x >= 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (tid == 0) {
      auto gk = maps.kt.get_tma_tensor(maps.ks)(_, _, blockIdx.y);
      auto gv = maps.vt.get_tma_tensor(maps.vs)(_, _, blockIdx.y);
      for (int slot = 0; slot < s.count; ++slot) {
        int block = s.blocks[slot];
        if (slot) s.k_free.wait((slot - 1) & 1);
        auto ktile = flat_divide(gk, Tile<_64, _128>{})(_, _, block, _0{});
        tma_load(maps.kt, ktile, sk, s.k_ready, 64 * 128 * 2);
        if (slot) s.v_free.wait((slot - 1) & 1);
        auto vtile = flat_divide(gv, Tile<_64, _128>{})(_, _, block, _0{});
        tma_load(maps.vt, vtile, sv, s.v_ready, 64 * 128 * 2);
      }
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<104>();
  auto sp = make_tensor(make_smem_ptr(s.probability), typename Tr::PLayout{});
  auto spt = make_tensor(make_smem_ptr(s.probability), typename Tr::PTLayout{});
  auto svt = make_tensor(make_smem_ptr(s.v), typename Tr::VTLayout{});
  auto rp = partition_fragment_C(typename Tr::QKMma{}, Shape<_64, _16>{});
  auto ro = partition_fragment_C(typename Tr::PVMma{}, Shape<_128, _16>{});
  auto ps = make_tensor<T>(partition_shape_C(typename Tr::QKMma{}, Shape<_64, _16>{}));
  float maximum[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float denominator[4] = {};
  clear(ro);
  if (tid == 0) {
    auto gq = maps.qt.get_tma_tensor(maps.qs)(_, _, blockIdx.x);
    auto tile = flat_divide(gq, Tile<_16, _128>{})(_, _, blockIdx.y, _0{});
    tma_load(maps.qt, tile, sq, s.q_ready, 16 * 128 * 2);
  }
  s.q_ready.wait(0);
  s.k_ready.wait(0);
  mma(true, typename Tr::QKMma{}, sk, sq, rp, tid);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  for (int slot = 0; slot < s.count; ++slot) {
    if (slot + 1 < s.count) s.k_free.arrive();
    int block = s.blocks[slot];
    int first_token = block * 64 + warp * 16 + lane / 4;
    // Masks and CIS belong to the logical token, shared across the 16 heads.
    CUTE_UNROLL
    for (int n = 0; n < 2; ++n) {
      int token = first_token + n * 8;
      bool valid = token < p.tokens && token <= p.query_start + blockIdx.x;
      float bias = valid && p.bias ? bias_at(p, token) : 0.f;
      CUTE_UNROLL
      for (int h = 0; h < 4; ++h) {
        auto c = make_coord(h % 2, n, h / 2);
        rp(c, 0, 0) = valid ? (rp(c, 0, 0) * 0.08838834764831845f + bias) * 1.4426950408889634f : -INFINITY;
      }
    }
    float m[4], alpha[4];
    CUTE_UNROLL
    for (int h = 0; h < 4; ++h)
      m[h] = fmaxf(rp(make_coord(h % 2, 0, h / 2), 0, 0), rp(make_coord(h % 2, 1, h / 2), 0, 0));
    reduce_heads<true>(m, s.exchange, tid);
    CUTE_UNROLL
    for (int h = 0; h < 4; ++h) {
      m[h] = fmaxf(m[h], maximum[h]);
      float safe = m[h] == -INFINITY ? 0.f : m[h];
      alpha[h] = exp2f(maximum[h] - safe);
      CUTE_UNROLL
      for (int n = 0; n < 2; ++n) {
        auto c = make_coord(h % 2, n, h / 2);
        ro(c, 0, 0) *= alpha[h]; ro(c, 1, 0) *= alpha[h];
        rp(c, 0, 0) = exp2f(rp(c, 0, 0) - safe);
        ps(c, 0, 0) = T(rp(c, 0, 0));
      }
    }
    auto copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, T>{}, typename Tr::QKMma{});
    auto thread_copy = copy.get_slice(tid);
    cute::copy(copy, thread_copy.retile_S(ps), thread_copy.partition_D(sp));
    s.v_ready.wait(slot & 1);
    // TMA bounds cover the physical cache; zero future values in the final
    // causal tile as well, so even poisoned future padding cannot enter PV.
    if ((block + 1) * 64 > p.query_start + blockIdx.x + 1) {
      CUTE_UNROLL
      for (int offset = tid * 8; offset < 64 * 128; offset += 128 * 8) {
        int n = offset / 128, d = offset % 128;
        if (block * 64 + n > p.query_start + blockIdx.x)
          *reinterpret_cast<uint4 *>(&sv(n, d)) = make_uint4(0, 0, 0, 0);
      }
      consumer_sync();
    }
    cutlass::arch::fence_view_async_shared();
    mma(false, typename Tr::PVMma{}, svt, spt, ro, tid);
    warpgroup_commit_batch();
    CUTE_UNROLL
    for (int h = 0; h < 4; ++h) {
      denominator[h] = denominator[h] * alpha[h] + rp(make_coord(h % 2, 0, h / 2), 0, 0)
                                                       + rp(make_coord(h % 2, 1, h / 2), 0, 0);
      maximum[h] = m[h];
    }
    if (slot + 1 < s.count) {
      s.k_ready.wait((slot + 1) & 1);
      mma(true, typename Tr::QKMma{}, sk, sq, rp, tid);
      warpgroup_commit_batch();
      warpgroup_wait<1>();
      s.v_free.arrive();
      warpgroup_wait<0>();
    } else {
      warpgroup_wait<0>();
    }
  }
  reduce_heads<false>(denominator, s.exchange, tid);
  auto out = static_cast<T *>(p.out);
  CUTE_UNROLL
  for (int tile = 0; tile < 2; ++tile) {
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
      int h = i / 4 * 2 + i % 2;
      int head = blockIdx.y * 16 + lane % 4 * 2 + h / 2 * 8 + h % 2;
      int dim = warp * 16 + lane / 4 + (i / 2 % 2) * 8 + tile * 64;
      float inverse = denominator[h] > 0.f ? 1.f / denominator[h] : 1.f;
      out[(int64_t(blockIdx.x) * p.heads + head) * 128 + dim] = T(ro(i, tile, 0) * inverse);
    }
  }
}

#include "nosa_attention_grouped.cuh"

template <typename T>
void launch(Params const &p, TensorView q, TensorView k, TensorView v, cudaStream_t stream) {
  using Tr = Traits<T>;
  auto qs = make_shape(p.heads, 128, p.queries);
  auto ks = make_shape(p.tokens, 128, p.kv_heads);
  auto vs = ks;
  auto qt = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<T const *>(q.data_ptr())), make_layout(qs, make_stride(q.stride(1), _1{}, q.stride(0)))), typename Tr::QLayout{});
  auto kt = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<T const *>(k.data_ptr())), make_layout(ks, make_stride(k.stride(0), _1{}, k.stride(1)))), typename Tr::KVLayout{});
  auto vt = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<T const *>(v.data_ptr())), make_layout(vs, make_stride(v.stride(0), _1{}, v.stride(1)))), typename Tr::KVLayout{});
  Maps<decltype(qs), decltype(qt), decltype(ks), decltype(kt), decltype(vs), decltype(vt)> maps{qs, qt, ks, kt, vs, vt};
  if (p.queries >= 4) {
    auto kernel = grouped_attention_kernel<T, decltype(maps)>;
    cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename GroupedTraits<T>::Shared));
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
    kernel<<<dim3((p.queries + 3) / 4, p.kv_heads), 256, sizeof(typename GroupedTraits<T>::Shared), stream>>>(p, maps);
    error = cudaGetLastError();
    TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  }
  auto kernel = attention_kernel<T, decltype(maps)>;
  cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename Tr::Shared));
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  kernel<<<dim3(p.queries, p.kv_heads), 256, sizeof(typename Tr::Shared), stream>>>(p, maps);
  error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

void forward(TensorView q, TensorView k, TensorView v, TensorView ids, TensorView mask,
             TensorView bias, TensorView out, TensorView group_fallback, int64_t query_start) {
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(2) == 128 && q.size(1) == k.size(1) * 16);
  TVM_FFI_ICHECK(q.dtype().bits == 16 && q.device().device_type == kDLCUDA);
  if (!q.size(0)) return;
  Params p{};
  p.q = q.data_ptr(); p.qr = q.stride(0); p.qh = q.stride(1);
  p.ids = ids.data_ptr(); p.ids64 = ids.dtype().bits == 64;
  p.valid = mask.numel() ? static_cast<bool const *>(mask.data_ptr()) : nullptr;
  p.bias = bias.numel() ? bias.data_ptr() : nullptr;
  p.bias_type = bias.dtype().bits == 32 ? 0 : bias.dtype().code == kDLBfloat ? 1 : 2;
  p.group_fallback = q.size(0) >= 4 ? static_cast<int *>(group_fallback.data_ptr()) : nullptr;
  p.out = out.data_ptr(); p.tokens = k.size(0); p.queries = q.size(0);
  p.heads = q.size(1); p.kv_heads = k.size(1); p.count = ids.size(2); p.query_start = query_start;
  p.ir = ids.size(0) == 1 ? 0 : ids.stride(0); p.ih = ids.size(1) == 1 ? 0 : ids.stride(1); p.ib = ids.stride(2);
  if (p.valid) { p.mr = mask.size(0) == 1 ? 0 : mask.stride(0); p.mh = mask.size(1) == 1 ? 0 : mask.stride(1); p.mb = mask.stride(2); }
  if (p.bias) { p.br = bias.stride(0); p.bh = bias.stride(1); }
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q.device().device_id));
  if (q.dtype().code == kDLBfloat) launch<cutlass::bfloat16_t>(p, q, k, v, stream);
  else launch<cutlass::half_t>(p, q, k, v, stream);
}
}  // namespace nosa_attention

TVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, nosa_attention::forward);
