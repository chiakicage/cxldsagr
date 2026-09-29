// Large-query scoring keeps the normalizer in registers across both QK passes.
// Four consumer warpgroups share each K transfer. A producer warpgroup
// releases registers before its elected lane fills the three TMA buffers.
// Consumers release each buffer only after their WGMMA completes.
// The smaller-query and unpooled paths retain the split-kernel schedule.
namespace fused_scores {
constexpr int kQueryRows = 16;
constexpr int kThreads = kQueryRows * 32;
constexpr int kStages = 3;
template<typename T, bool Selection> struct SelectionStorage {};
template<typename T> struct SelectionStorage<T, true> {
  alignas(16) T pooled[kQueryRows][1056];
  unsigned promotions[kQueryRows][33], promoted_ids[kQueryRows][64];
};
template <typename T> struct Traits {
  static constexpr int M = kQueryRows * 16, N = kColumns, D = 128;
  using QLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},
                                       Shape<Int<M>, Int<D>>{}));
  using KLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},
                                       Shape<Int<N>, Int<D>>{}));
  using Atom = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
      GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
      GMMA::MMA_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>>;
  using MMA = decltype(make_tiled_mma(Atom{}, Layout<Shape<_1, _1, _1>>{}));
  template <int Stages = 2, bool Selection = false> struct Shared : SelectionStorage<T, Selection> {
    alignas(128) T q[cosize_v<QLayout>];
    alignas(128) T k[Stages][cosize_v<KLayout>];
    T rounded[kQueryRows][N];
    int64_t positions[kQueryRows];
    Barrier ready[Stages], released[Stages], q_ready, cohort_done[2];
  };
};

template <typename T, typename Shared, typename Acc>
__device__ __forceinline__ void issue(Shared& smem, int buffer, Acc& acc) {
  typename Traits<T>::MMA mma;
  auto thr = mma.get_slice(threadIdx.x % 128);
  auto qa = local_tile(make_tensor(make_smem_ptr(smem.q), typename Traits<T>::QLayout{}), Tile<_64, _128>{}, make_coord(threadIdx.x / 128, _0{}));
  auto a = thr.partition_fragment_A(qa);
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

template <typename T, typename Shared, typename Map>
__device__ __forceinline__ void load_k(Shared& smem, int buffer, int start, const Map& map) {
  if (threadIdx.x == 0 || threadIdx.x == kThreads) {
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
  if (tid < kQueryRows) {
    int row = blockIdx.x * kQueryRows + tid;
    smem.positions[tid] = row >= p.rows ? -1 : p.contiguous ? p.query_start + row :
      p.pos64 ? static_cast<const int64_t*>(p.positions)[row] :
                static_cast<const int32_t*>(p.positions)[row];
  }
  if (tid == 0) {
    auto global = map.qtma.get_tma_tensor(map.qshape)(_, _, blockIdx.y);
    auto src = flat_divide(global, Tile<Int<kQueryRows * 16>, _128>{})(_, _, blockIdx.x, _0{});
    auto dst = make_tensor(make_smem_ptr(smem.q), typename Traits<T>::QLayout{});
    auto slice = map.qtma.get_slice(_0{});
    cute::copy(map.qtma.with(reinterpret_cast<Barrier::ValueType&>(smem.q_ready)),
               slice.partition_S(src), slice.partition_D(dst));
    smem.q_ready.arrive_and_expect_tx(kQueryRows * 16 * 128 * sizeof(T));
  }
}

template <typename T, typename O, bool Pool, bool Masked, bool Selection, typename Shared, typename Acc>
__device__ __forceinline__ void write_result(const Params& p, Shared& smem, Acc& acc,
    int start, int64_t position, const float (&maximum)[2], const float (&inv)[2]) {
  int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  int row = blockIdx.x * kQueryRows + warp, head = blockIdx.y;
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
      int block = (start + 1) / 4 + lane;
      float value = -INFINITY;
      CUTE_UNROLL
      for (int j = 0; j < 5; ++j) {
        int c = start + lane * 4 + j;
        if (c >= 0 && c < p.count) value = fmaxf(value, float(smem.rounded[warp][lane * 4 + j]));
      }
      if (row < p.rows && block < p.blocks) {
        if constexpr (Selection) smem.pooled[warp][block] = T(mask_pool(value, block, position / 64));
        else static_cast<O*>(p.out)[(int64_t(row) * p.heads + head) * p.blocks + block] =
          O(mask_pool(value, block, position / 64));
      }
    }
  }
}

template <typename T, typename O, bool Pool, bool Selection, typename Map>
__global__ __launch_bounds__(kThreads + 128, 1) void fused_scores_kernel(__grid_constant__ const Params p, __grid_constant__ const Map map) {
  using Tr = Traits<T>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& smem = *reinterpret_cast<typename Tr::template Shared<kStages, Selection>*>(storage);
  int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  if (tid == 0) {
    CUTE_UNROLL
    for (int stage = 0; stage < kStages; ++stage) {
      smem.ready[stage].init(1);
      smem.released[stage].init(kThreads);
    }
    smem.q_ready.init(1);
    smem.cohort_done[0].init(kThreads / 2);
    smem.cohort_done[1].init(kThreads / 2);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  fused_scores::load_query<T>(p, smem, map);
  __syncthreads();
  constexpr int BlocksPerTile = (kColumns - 1) / 4;
  if (tid >= kThreads) {
    // A full producer warpgroup must participate in register deallocation.
    // Only its elected lane performs the transfers; other lanes can retire.
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (tid == kThreads) {
      int iteration = 0;
      for (; iteration * kColumns < p.count; ++iteration) {
        int buffer = iteration % kStages;
        if (iteration >= kStages) smem.released[buffer].wait(((iteration / kStages) - 1) & 1);
        fused_scores::load_k<T>(smem, buffer, iteration * kColumns, map);
      }
      for (int tile = 0; Pool ? tile * BlocksPerTile < p.blocks : tile * kColumns < p.count; ++tile, ++iteration) {
        int buffer = iteration % kStages;
        if (iteration >= kStages) smem.released[buffer].wait(((iteration / kStages) - 1) & 1);
        int start = Pool ? tile * BlocksPerTile * 4 - 1 : tile * kColumns;
        fused_scores::load_k<T>(smem, buffer, start, map);
      }
    }
    return;
  }
  // The launch allocates 640 * 96 registers. Consumers plus producer need
  // 512 * 112 + 128 * 24 = 60,416, below the 61,440-register CTA allocation.
  // Keeping the request within that pool also avoids a setmaxnreg deadlock.
  cutlass::arch::warpgroup_reg_alloc<112>();
  smem.q_ready.wait(0);
  auto acc = partition_fragment_C(typename Tr::MMA{}, Shape<_64, Int<kColumns>>{});
  float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {0.0f, 0.0f};
  int64_t position = smem.positions[warp];
  int iteration = 0;
  for (; iteration * kColumns < p.count; ++iteration) {
    // Two cohorts alternate QK while the other cohort normalizes. Each
    // cohort observes its peer's previous completion before advancing.
    if (tid >= kThreads / 2) smem.cohort_done[0].wait(iteration & 1);
    else if (iteration > 0) smem.cohort_done[1].wait((iteration - 1) & 1);
    int buffer = iteration % kStages;
    smem.ready[buffer].wait((iteration / kStages) & 1);
    fused_scores::issue<T>(smem, buffer, acc);
    warpgroup_wait<0>();
    smem.released[buffer].arrive();
    smem.cohort_done[tid / (kThreads / 2)].arrive();
    bool full = (iteration + 1) * kColumns <= p.count && ((iteration + 1) * kColumns - 1) * 16 + 31 <= position;
    if (full) normalize_tile<false>(acc, iteration, p.count, position, maximum, denominator);
    else normalize_tile<true>(acc, iteration, p.count, position, maximum, denominator);
  }
  float inv[2];
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) {
    float sum = denominator[r];
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    inv[r] = 1.0f / (sum > 0.0f ? sum : 1.0f);
    if (maximum[r] == -INFINITY) maximum[r] = 0.0f;
  }
  for (int tile = 0; Pool ? tile * BlocksPerTile < p.blocks : tile * kColumns < p.count; ++tile, ++iteration) {
    int start = Pool ? tile * BlocksPerTile * 4 - 1 : tile * kColumns;
    int buffer = iteration % kStages;
    smem.ready[buffer].wait((iteration / kStages) & 1);
    fused_scores::issue<T>(smem, buffer, acc);
    warpgroup_wait<0>();
    smem.released[buffer].arrive();
    bool full = start >= 0 && start + kColumns <= p.count && (start + kColumns - 1) * 16 + 31 <= position;
    if (full) fused_scores::write_result<T, O, Pool, false, Selection>(p, smem, acc, start, position, maximum, inv);
    else fused_scores::write_result<T, O, Pool, true, Selection>(p, smem, acc, start, position, maximum, inv);
  }
  if constexpr (Selection) {
    __syncwarp();
    int row = blockIdx.x * kQueryRows + warp;
    if (row < p.rows) {
      int64_t offset = (int64_t(row) * p.heads + blockIdx.y) * 64;
      nosa_selection_fused::select_prefix_row(
          reinterpret_cast<uint16_t const*>(smem.pooled[warp]), p.blocks, position / 64,
          p.ranking + blockIdx.y * 64, p.ids + offset, p.valid ? p.valid + offset : nullptr,
          smem.promotions[warp], smem.promoted_ids[warp]);
    }
  }
}

template <typename T, typename O, bool Pool, bool Selection = false>
bool launch(const Params& p, cudaStream_t stream) {
  auto shape = make_shape(p.count, 128, p.heads);
  auto tma = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<const T*>(p.k)),
                  make_layout(shape, make_stride(p.kc, _1{}, p.kh))),
      typename Traits<T>::KLayout{});
  auto qshape = make_shape(make_shape(_16{}, p.rows), _128{}, p.heads);
  auto qtma = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<const T*>(p.q)),
                  make_layout(qshape, make_stride(make_stride(p.qg, p.qr), _1{}, p.qh))),
      typename Traits<T>::QLayout{}, Tile<Int<kQueryRows * 16>, _128>{}, _1{});
  TensorMaps<decltype(shape), decltype(tma), decltype(qshape), decltype(qtma)> map{shape, tma, qshape, qtma};
  auto fused = fused_scores_kernel<T, O, Pool, Selection, decltype(map)>;
  int shared = sizeof(typename Traits<T>::template Shared<kStages, Selection>);
  if (!configure(fused, shared, p.device, true)) {
    // A future compiler may choose a smaller initial register allocation.
    // Use the original schedule instead of risking a setmaxnreg deadlock.
    if constexpr (Selection) {
      return false;
    } else {
      nosa_scores::launch<T, O, Pool>(p, stream);
      return true;
    }
  }
  fused<<<dim3((p.rows + kQueryRows - 1) / kQueryRows, p.heads), kThreads + 128, shared, stream>>>(p, map);
  auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  return true;
}
}  // namespace fused_scores
