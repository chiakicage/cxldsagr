__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Four independent NOSA query rows share the union of their selected blocks.
// This is a local CuTe implementation of the query-package scheduling idea in
// EzKernelKit's experimental fwd_v2.cu; selection membership remains per query.
template <typename T> struct GroupedTraits {
  using QLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, Shape<_64, _128>{}));
  using KVLayout = QLayout;
  using VTLayout = decltype(composition(KVLayout{}, Layout<Shape<_128, _64>, Stride<_64, _1>>{}));
  using QKOp = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
      GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
      GMMA::MMA_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
  using PVOp = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
      GMMA::MMA_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>,
      GMMA::MMA_64x128x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>>;
  using QKMma = decltype(make_tiled_mma(QKOp{}));
  using PVMma = decltype(make_tiled_mma(PVOp{}));
  struct Shared {
    alignas(128) T q[cosize_v<QLayout>], k[2][cosize_v<KVLayout>], v[2][cosize_v<KVLayout>];
    float bias[2][64];
    int sorted[4][64], blocks[256], membership[256], count;
    Barrier k_ready[2], v_ready[2], k_free[2], v_free[2], bias_ready[2], bias_free[2];
  };
};

__device__ __forceinline__ int grouped_block(Params const &p, int row, int slot) {
  if (row >= p.queries || slot >= p.count) return INT_MAX;
  int64_t offset = int64_t(row) * p.ir + blockIdx.y * p.ih + slot * p.ib;
  int64_t id = p.ids64 ? static_cast<int64_t const *>(p.ids)[offset]
                       : static_cast<int32_t const *>(p.ids)[offset];
  bool valid = !p.valid || p.valid[int64_t(row) * p.mr + blockIdx.y * p.mh + slot * p.mb];
  return valid && id >= 0 && id < (p.tokens + 63) / 64 && id * 64 <= p.query_start + row ? int(id) : INT_MAX;
}

template <class MMA, class A, class B, class C>
__device__ __forceinline__ void mma_rs(MMA op, A &a, B const &b, C &c, int thread) {
  auto bf = op.get_slice(thread).partition_fragment_B(b);
  warpgroup_fence_operand(a);
  warpgroup_fence_operand(c);
  warpgroup_arrive();
  op.accumulate_ = GMMA::ScaleOut::One;
  CUTE_UNROLL
  for (int k = 0; k < size<2>(a); ++k) cute::gemm(op, a(_, _, k), bf(_, _, k), c);
  warpgroup_fence_operand(c);
  warpgroup_fence_operand(a);
}

template <typename T, class MapsT>
__global__ __launch_bounds__(256, 2) void grouped_attention_kernel(
    __grid_constant__ Params const p, __grid_constant__ MapsT const maps) {
  using Tr = GroupedTraits<T>;
  extern __shared__ char storage[];
  auto &s = *reinterpret_cast<typename Tr::Shared *>(storage);
  int tid = threadIdx.x % 128, lane = tid % 32, warp = tid / 32;
  int query_base = blockIdx.x * 4;
  auto sq = make_tensor(make_smem_ptr(s.q), typename Tr::QLayout{});
  if (threadIdx.x == 0) {
    p.group_fallback[blockIdx.x * p.kv_heads + blockIdx.y] = 0;
    CUTE_UNROLL
    for (int buffer = 0; buffer < 2; ++buffer) {
      s.k_ready[buffer].init(1); s.v_ready[buffer].init(1);
      s.k_free[buffer].init(128); s.v_free[buffer].init(128);
      s.bias_ready[buffer].init(1); s.bias_free[buffer].init(128);
    }
    cutlass::arch::fence_barrier_init();
  }
  // Sort each independent selection in registers; invalid slots sort last.
  // The normal sorted-indexer case and arbitrary public selection order have
  // identical semantics and require no device-to-host inspection.
  if (threadIdx.x < 128) {
    int a = grouped_block(p, query_base + warp, lane);
    int b = grouped_block(p, query_base + warp, lane + 32);
    CUTE_UNROLL
    for (int size = 2; size <= 64; size *= 2) {
      CUTE_UNROLL
      for (int distance = size / 2; distance; distance /= 2) {
        int aa = distance == 32 ? b : __shfl_xor_sync(0xffffffff, a, distance);
        int bb = distance == 32 ? a : __shfl_xor_sync(0xffffffff, b, distance);
        bool lo_a = bool(lane & distance) == bool(lane & size);
        bool lo_b = bool((lane + 32) & distance) == bool((lane + 32) & size);
        a = lo_a ? min(a, aa) : max(a, aa);
        b = lo_b ? min(b, bb) : max(b, bb);
      }
    }
    s.sorted[warp][lane] = a;
    s.sorted[warp][lane + 32] = b;
  }
  CUTE_UNROLL
  for (int offset = threadIdx.x * 8; offset < 64 * 128; offset += 256 * 8) {
    int row = offset / 128, dim = offset % 128;
    int query = query_base + row / 16;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (query < p.queries) {
      auto ptr = static_cast<T const *>(p.q) + int64_t(query) * p.qr + (blockIdx.y * 16 + row % 16) * p.qh + dim;
      value = *reinterpret_cast<uint4 const *>(ptr);
    }
    *reinterpret_cast<uint4 *>(&sq(row, dim)) = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int cursor[4] = {}, count = 0;
    while (true) {
      int cur[4];
      CUTE_UNROLL
      for(int i=0;i<4;++i) cur[i] = cursor[i] < 64 ? s.sorted[i][cursor[i]] : INT_MAX;
      int block = min(min(cur[0],cur[1]),min(cur[2],cur[3]));
      if(block == INT_MAX) break;
      int members = 0;
      CUTE_UNROLL
      for(int i=0;i<4;++i) if(cur[i] == block) { members |= 1 << i; ++cursor[i]; }
      s.blocks[count] = block; s.membership[count] = members; ++count;
    }
    s.count = count;
  }
  __syncthreads();
  // Union packing is profitable only when adjacent queries reuse K/V.
  // Sparse rows with little overlap use the original per-query kernel.
  if (s.count > p.count * 2) {
    if (threadIdx.x == 0) p.group_fallback[blockIdx.x * p.kv_heads + blockIdx.y] = 1;
    return;
  }
  cutlass::arch::fence_view_async_shared();
  if (threadIdx.x >= 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (tid < 32) {
      auto gk = maps.kt.get_tma_tensor(maps.ks)(_, _, blockIdx.y);
      auto gv = maps.vt.get_tma_tensor(maps.vs)(_, _, blockIdx.y);
      for (int slot = 0; slot < s.count; ++slot) {
        int block = s.blocks[slot], buffer = slot & 1, phase = (slot / 2 - 1) & 1;
        auto sk = make_tensor(make_smem_ptr(s.k[buffer]), typename Tr::KVLayout{});
        auto sv = make_tensor(make_smem_ptr(s.v[buffer]), typename Tr::KVLayout{});
        if (tid == 0) {
          if (slot >= 2) s.k_free[buffer].wait(phase);
          tma_load(maps.kt, flat_divide(gk, Tile<_64, _128>{})(_, _, block, _0{}), sk, s.k_ready[buffer], 16384);
        }
        if (slot >= 2) s.bias_free[buffer].wait(phase);
        CUTE_UNROLL
        for(int offset=0;offset<64;offset+=32) {
          int token = block * 64 + tid + offset;
          s.bias[buffer][tid + offset] = p.bias && token < p.tokens ? bias_at(p, token) : 0.f;
        }
        __syncwarp();
        if (tid == 0) {
          s.bias_ready[buffer].arrive();
          if (slot >= 2) s.v_free[buffer].wait(phase);
          tma_load(maps.vt, flat_divide(gv, Tile<_64, _128>{})(_, _, block, _0{}), sv, s.v_ready[buffer], 16384);
        }
        __syncwarp();
      }
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<192>();
  auto rp = partition_fragment_C(typename Tr::QKMma{}, Shape<_64, _64>{});
  auto ro = partition_fragment_C(typename Tr::PVMma{}, Shape<_64, _128>{});
  auto ps = make_tensor<T>(partition_shape_A(typename Tr::PVMma{}, Shape<_64, _64>{}));
  float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {};
  int value_shift = pv_scale_exponent<T>(s.count);
  float value_rescale = __uint_as_float(uint32_t(127 + value_shift) << 23);
  clear(ro);
  if (s.count) {
    auto sk = make_tensor(make_smem_ptr(s.k[0]), typename Tr::KVLayout{});
    s.k_ready[0].wait(0);
    mma(true, typename Tr::QKMma{}, sq, sk, rp, tid);
    warpgroup_commit_batch(); warpgroup_wait<0>();
  }
  for (int slot = 0; slot < s.count; ++slot) {
    int buffer = slot & 1, phase = (slot / 2) & 1;
    auto sv = make_tensor(make_smem_ptr(s.v[buffer]), typename Tr::KVLayout{});
    auto svt = make_tensor(make_smem_ptr(s.v[buffer]), typename Tr::VTLayout{});
    if (slot + 2 < s.count) s.k_free[buffer].arrive();
    int block = s.blocks[slot], query = query_base + warp;
    bool member = (s.membership[slot] >> warp) & 1;
    float biases[16];
    unsigned valid_tokens = 0;
    s.bias_ready[buffer].wait(phase);
    CUTE_UNROLL
    for (int i = 0; i < 16; ++i) {
      int offset = (i / 2) * 8 + lane % 4 * 2 + i % 2;
      int token = block * 64 + offset;
      bool valid = member && token < p.tokens && token <= p.query_start + query;
      valid_tokens |= unsigned(valid) << i;
      biases[i] = s.bias[buffer][offset];
    }
    if (slot + 2 < s.count) s.bias_free[buffer].arrive();
    CUTE_UNROLL
    for (int h = 0; h < 2; ++h) {
      float m = -INFINITY;
      CUTE_UNROLL
      for (int i = h * 2; i < 32; i += 4) {
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j) {
          bool valid = valid_tokens & (1u << (i / 4 * 2 + j));
          float bias = biases[i / 4 * 2 + j];
          rp(i + j) = valid ? __fadd_rn(__fmul_rn(rp(i + j), 0.08838834764831845f), bias) : -INFINITY;
          m = fmaxf(m, rp(i + j));
        }
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      m = fmaxf(m, maximum[h]);
      float safe = m == -INFINITY ? 0.f : m;
      float alpha = fast_exp2((maximum[h] - safe) * 1.4426950408889634f), sum = 0.f;
      if (alpha != 1.f) {
        CUTE_UNROLL
        for (int i = h * 2; i < 64; i += 4) { ro(i) *= alpha; ro(i + 1) *= alpha; }
      }
      CUTE_UNROLL
      for (int i = h * 2; i < 32; i += 4) {
        float a = fast_exp2((rp(i) - safe) * 1.4426950408889634f), b = fast_exp2((rp(i + 1) - safe) * 1.4426950408889634f);
        ps(i) = T(a); ps(i + 1) = T(b); sum += a + b;
      }
      denominator[h] = denominator[h] * alpha + sum;
      maximum[h] = m;
    }
    s.v_ready[buffer].wait(phase);
    int last = min(p.queries - 1, query_base + 3) + p.query_start;
    if ((block + 1) * 64 > last + 1) {
      CUTE_UNROLL
      for (int offset = tid * 8; offset < 64 * 128; offset += 128 * 8) {
        int n = offset / 128, d = offset % 128;
        if (block * 64 + n > last) *reinterpret_cast<uint4 *>(&sv(n, d)) = make_uint4(0, 0, 0, 0);
      }
      consumer_sync(); cutlass::arch::fence_view_async_shared();
    }
    // A shared PV tile cannot suppress NaN/Inf with a zero probability.
    // Physical-block flags are computed once before the grouped launch.
    // Conservatively repair when nonfinite V might be masked by some rows.
    int all_members = (1 << min(4, p.queries - query_base)) - 1;
    int scan_from = s.membership[slot] == all_members ? max(0, p.query_start + query_base + 1 - block * 64) : 0;
    int scan_to = min(64, last + 1 - block * 64);
    if (scan_from < scan_to && p.block_nonfinite[block * p.kv_heads + blockIdx.y] && tid == 0)
      p.group_fallback[blockIdx.x * p.kv_heads + blockIdx.y] = 1;
    if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
      scale_bf16_v(s.v[buffer], tid, value_shift);
      consumer_sync();
      cutlass::arch::fence_view_async_shared();
    }
    mma_rs(typename Tr::PVMma{}, ps, svt, ro, tid);
    warpgroup_commit_batch();
    if (slot + 1 < s.count) {
      auto sk = make_tensor(make_smem_ptr(s.k[(slot + 1) & 1]), typename Tr::KVLayout{});
      s.k_ready[(slot + 1) & 1].wait(((slot + 1) / 2) & 1);
      mma(true, typename Tr::QKMma{}, sq, sk, rp, tid);
      warpgroup_commit_batch(); warpgroup_wait<1>();
      if (slot + 2 < s.count) s.v_free[buffer].arrive();
      warpgroup_wait<0>();
    } else warpgroup_wait<0>();
  }
  int query = query_base + warp;
  CUTE_UNROLL
  for (int h = 0; h < 2; ++h) {
    float sum = denominator[h];
    sum += __shfl_xor_sync(0xffffffff, sum, 1); sum += __shfl_xor_sync(0xffffffff, sum, 2);
    float inv = sum > 0.f ? value_rescale / sum : value_rescale;
    if (query < p.queries) {
      int head = blockIdx.y * 16 + lane / 4 + h * 8;
      CUTE_UNROLL
      for (int i = h * 2; i < 64; i += 4) {
        int dim = i / 4 * 8 + lane % 4 * 2;
        auto out = static_cast<T *>(p.out) + (int64_t(query) * p.heads + head) * 128 + dim;
        out[0] = bounded_pv_output<T>(ro(i), inv); out[1] = bounded_pv_output<T>(ro(i + 1), inv);
      }
    }
  }
}
