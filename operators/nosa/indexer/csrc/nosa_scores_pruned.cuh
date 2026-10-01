// Certified tile pruning: upward-rounded summaries and pooled QA share storage.
// Final normalizers live in shared memory outside the accumulator scopes.
// Four consumer warpgroups share each K transfer. A producer warpgroup
// releases registers before its elected lane fills three TMA buffers.
// Consumers release each buffer only after their WGMMA completes.
// After the shared seed/cutoff phase, two eight-query cohorts use independent
// tail lists and K buffers so that a short cohort can finish without carrying
// the other cohort's retained tiles through its WGMMA and GQA work.
// The smaller-query and unpooled paths retain the split-kernel schedule.
namespace pruned_scores {
// This kernel is dispatched only for the guarded 64K+1K geometry.
constexpr int kRows = 1024, kHeads = 2, kCount = 4159, kBlocks = 1040;
constexpr int64_t kQueryStart = 65536;
constexpr int kQueryRows = 16;
constexpr int kThreads = kQueryRows * 32;
constexpr int kStages = 3;
template<typename T, bool Selection> struct SelectionStorage {};
template<typename T> struct SelectionStorage<T, true> {
  union {
    alignas(16) T pooled[kQueryRows][1056];
    __half2 tile_summary[kQueryRows][16][36];
  };
  unsigned promotions[kQueryRows][33], promoted_ids[kQueryRows][64];
  float maximum[kQueryRows][16], inverse[kQueryRows][16];
  union {
    unsigned certifiable[kQueryRows];
    unsigned seed_cut[kQueryRows];
  };
  unsigned bounds[kQueryRows][35];
  unsigned char keep[kQueryRows][35];
  int order[35], selected_count;
  Barrier seed_ready, tail_ready;
  int cohort_tail_order[2][35], cohort_tail_count[2];
  Barrier cohort_tail_ready[2], cohort_tail_released[2];
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
  // Keep Q descriptors local to this issue, including across scoring phases.
  int query_group;
  asm volatile("mov.u32 %0, %1;" : "=r"(query_group) : "r"(threadIdx.x / 128));
  auto qa = local_tile(make_tensor(make_smem_ptr(smem.q), typename Traits<T>::QLayout{}), Tile<_64, _128>{}, make_coord(query_group, _0{}));
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
__device__ __forceinline__ void load_tail_k(Shared& smem, int cohort, int start, const Map& map) {
  auto global = domain_offset(make_coord(start, _0{}), map.tma.get_tma_tensor(map.shape)(_, _, blockIdx.y));
  auto src = flat_divide(global, Tile<Int<kColumns>, _128>{})(_, _, _0{}, _0{});
  auto dst = make_tensor(make_smem_ptr(smem.k[cohort]), typename Traits<T>::KLayout{});
  auto slice = map.tma.get_slice(_0{});
  cute::copy(map.tma.with(reinterpret_cast<Barrier::ValueType&>(smem.cohort_tail_ready[cohort])),
             slice.partition_S(src), slice.partition_D(dst));
  smem.cohort_tail_ready[cohort].arrive_and_expect_tx(kColumns * 128 * sizeof(T));
}

template <typename T, typename Shared, typename Map>
__device__ __forceinline__ void load_query(const Params& p, Shared& smem, const Map& map) {
  int tid = threadIdx.x;
  if (tid < kQueryRows) {
    int row = blockIdx.x * kQueryRows + tid;
    smem.positions[tid] = row >= kRows ? -1 : p.contiguous ? kQueryStart + row :
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
  constexpr int BlocksPerTile = kColumns / 4;
  constexpr float scale = 0.08838834764831845f;  // 1 / sqrt(128)
  // Keep the tile-dependent column base within this result invocation.
  // This blocks cross-call hoisting of fifteen lane-plus-column offsets.
  int column_base;
  asm volatile("mov.u32 %0, %1;" : "=r"(column_base)
               : "r"(start + (lane % 4) * 2));
  // The normalizer stores the rounded FP32 scaled logit. Reproduce that
  // rounding before subtraction; an FFMA here keeps extra low bits and can
  // bias even a uniform softmax when logits are large. Likewise, retain max
  // and inverse separately: folding log(sum) into a large max loses precision.
  CUTE_UNROLL
  for (int i = 0; i < size(acc); i += 4) {
    int local = 8 * (i / 4) + (lane % 4) * 2, col = column_base + 8 * (i / 4);
    float a = 0.0f, b = 0.0f;
    if (!Masked || (col >= 0 && col < kCount && col * 16 + 31 <= position))
      a = exp_approx(__fmul_rn(acc(i), scale) - maximum[0]) * inv[0]
        + exp_approx(__fmul_rn(acc(i + 2), scale) - maximum[1]) * inv[1];
    if (!Masked || (col + 1 >= 0 && col + 1 < kCount && (col + 1) * 16 + 31 <= position))
      b = exp_approx(__fmul_rn(acc(i + 1), scale) - maximum[0]) * inv[0]
        + exp_approx(__fmul_rn(acc(i + 3), scale) - maximum[1]) * inv[1];
    CUTE_UNROLL
    for (int offset = 4; offset <= 16; offset *= 2) {
      a += __shfl_xor_sync(0xffffffff, a, offset);
      b += __shfl_xor_sync(0xffffffff, b, offset);
    }
    if (lane < 4) {
      if constexpr (Pool) {
        if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
          uint32_t packed;
          // PTX packs the first source high and the second source low.
          asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(b), "f"(a));
          *reinterpret_cast<uint32_t*>(&smem.rounded[warp][local]) = packed;
        } else {
          smem.rounded[warp][local] = T(a);
          smem.rounded[warp][local + 1] = T(b);
        }
      } else if (row < kRows) {
        auto output = static_cast<T*>(p.out) + (int64_t(row) * kHeads + head) * kCount;
        if (col < kCount) output[col] = T(a);
        if (col + 1 < kCount) output[col + 1] = T(b);
      }
    }
  }
  if constexpr (Pool) {
    __syncwarp();
    if (lane < BlocksPerTile) {
      int block = start / 4 + lane;
      // Aligned QK shares the first-pass column mapping. A five-value pool
      // window needs the previous tile's final BF16 score only for lane 0.
      float value = -INFINITY;
      if (lane > 0 && start + lane * 4 - 1 < kCount)
        value = fmaxf(-INFINITY, float(smem.rounded[warp][lane * 4 - 1]));
      // A preceding tile may already have pushed its halo into this block.
      if (lane == 0 && block < kBlocks) value = fmaxf(-INFINITY, float(smem.pooled[warp][block]));
      CUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        int c = start + lane * 4 + j;
        if (c < kCount) value = fmaxf(value, float(smem.rounded[warp][lane * 4 + j]));
      }
      if (row < kRows && block < kBlocks) {
        if constexpr (Selection) smem.pooled[warp][block] = T(mask_pool(value, block, position / 64));
        else static_cast<O*>(p.out)[(int64_t(row) * kHeads + head) * kBlocks + block] =
          O(mask_pool(value, block, position / 64));
      }
    }
    // Only lane 0 updates the next tile's first block. These max updates
    // make arbitrary seed/tail order equivalent to a full five-window pool.
    int next = start / 4 + BlocksPerTile;
    if (lane == 0 && row < kRows && next < kBlocks && start + kColumns - 1 < kCount) {
      float halo = mask_pool(float(smem.rounded[warp][kColumns - 1]), next, position / 64);
      smem.pooled[warp][next] = T(fmaxf(float(smem.pooled[warp][next]), halo));
    }
    __syncwarp();
  }
}

constexpr int kSeedTiles = 4;

template<int K,int J> __device__ __forceinline__ void sort64(unsigned (&values)[2]) {
  unsigned next[2];
#pragma unroll
  for(int i=0;i<2;++i) {
    unsigned other;
    if constexpr(J<32) other=__shfl_xor_sync(0xffffffff,values[i],J);
    else other=values[i^1];
    int index=threadIdx.x%32+32*i;
    bool lower=((index&K)==0)!=((index&J)==0);
    next[i]=lower ? min(values[i],other) : max(values[i],other);
  }
  values[0]=next[0];values[1]=next[1];
  if constexpr(J>1) sort64<K,J/2>(values);
  else if constexpr(K<64) sort64<K*2,K>(values);
}

// Directed centering and conversion dominate the actual RN exponent input.
// 16 upward FP32 steps cover the documented ex2.approx 2-ULP error on both
// endpoints and power-of-two boundaries; the normal floor covers FTZ.
__device__ __forceinline__ float exp_upper(float maximum_tile,float maximum) {
  float exponent=__fmul_ru(__fsub_ru(maximum_tile,maximum),1.4426950408889634f);
  if(!isfinite(exponent)) return INFINITY;
  float value;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(exponent));
  if(!isfinite(value)) return INFINITY;
  unsigned bits=max(__float_as_uint(value),0x00800000u);
  return __uint_as_float(min(bits+16u,0x7f800000u));
}
__device__ __forceinline__ unsigned upper_key(float bound) {
  if(!isfinite(bound) || bound<0) return 0xffff;
  unsigned bits=__float_as_uint(bound);
  return nosa_selection_fused::key16((bits+0xffffu)>>16);
}

template<typename T,typename Shared>
__device__ __forceinline__ void build_bounds(const Params& p,Shared& smem,
    int tiles,int64_t position) {
  int warp=threadIdx.x/32,lane=threadIdx.x%32;
  bool active=blockIdx.x*kQueryRows+warp<kRows;
  for(int base=0;base<tiles;base+=4) {
    int tile=base+lane%4;
    float bound=0;
#pragma unroll
    for(int r=0;r<2;++r) {
      int group=lane/4+r*8;
      // Every block window lies inside this tile except its first halo,
      // which needs only column 127 of the preceding tile. FP16 summaries
      // round upward and decode exactly; nonfinite/overflow stays unknown.
      float current=tile*128<kCount ? __low2float(smem.tile_summary[warp][group][tile]) : -INFINITY;
      float maximum_tile=isfinite(current) ? current : INFINITY;
      if(tile>0) {
        float halo=tile*128-1<kCount ? __high2float(smem.tile_summary[warp][group][tile-1]) : -INFINITY;
        maximum_tile=isfinite(halo) ? fmaxf(maximum_tile,halo) : INFINITY;
      }
      float inverse=smem.inverse[warp][group];
      float term=isfinite(inverse) && inverse>=0 ?
        __fmul_ru(exp_upper(maximum_tile,smem.maximum[warp][group]),inverse) : INFINITY;
      bound=__fadd_ru(bound,term);
    }
#pragma unroll
    for(int offset=4;offset<=16;offset*=2)
      bound=__fadd_ru(bound,__shfl_xor_sync(0xffffffff,bound,offset));
    if(lane<4 && tile<tiles) {
      bool eligible=tile*32<=position/64-17 && tile*32+31>=1;
      smem.bounds[warp][tile]=active && !smem.certifiable[warp] ? 0xffff :
        (active && eligible ? upper_key(bound) : 0);
    }
  }
}

template<typename T,typename Shared>
__device__ __forceinline__ void score_tile(const Params& p,Shared& smem,
    int tile,int iteration,int64_t position) {
  auto acc=partition_fragment_C(typename Traits<T>::MMA{},Shape<_64,Int<kColumns>>{});
  int start=tile*128,buffer=iteration%kStages;
  smem.ready[buffer].wait((iteration/kStages)&1);
  pruned_scores::issue<T>(smem,buffer,acc);
  warpgroup_wait<0>();smem.released[buffer].arrive();
  float maximum[2],inv[2];
#pragma unroll
  for(int r=0;r<2;++r) {
    int group=(threadIdx.x%32)/4+r*8;
    maximum[r]=smem.maximum[threadIdx.x/32][group];
    inv[r]=smem.inverse[threadIdx.x/32][group];
  }
  bool full=start>=0 && start+kColumns<=kCount && (start+kColumns-1)*16+31<=position;
  if(full) pruned_scores::write_result<T,T,true,false,true>(p,smem,acc,start,position,maximum,inv);
  else pruned_scores::write_result<T,T,true,true,true>(p,smem,acc,start,position,maximum,inv);
}

template<typename T,typename Shared>
__device__ __forceinline__ void score_tail_tile(const Params& p,Shared& smem,
    int tile,int iteration,int64_t position) {
  constexpr int CohortThreads=kThreads/2;
  int cohort=threadIdx.x/CohortThreads;
  auto acc=partition_fragment_C(typename Traits<T>::MMA{},Shape<_64,Int<kColumns>>{});
  int start=tile*128;
  smem.cohort_tail_ready[cohort].wait(iteration&1);
  // Preserve the exact buffer while bounding descriptor live ranges to
  // this tile; otherwise ptxas hoists all eight Q/K descriptor pairs.
  int tile_buffer;
  asm volatile("mov.u32 %0, %1;" : "=r"(tile_buffer) : "r"(cohort));
  pruned_scores::issue<T>(smem,tile_buffer,acc);
  warpgroup_wait<0>();smem.cohort_tail_released[cohort].arrive();
  // The row bound also covers its outgoing column-127 halo.
  // Every WG still completes the collective MMA and release above.
  if(smem.keep[threadIdx.x/32][tile]) {
    float maximum[2],inv[2];
  #pragma unroll
    for(int rr=0;rr<2;++rr) {
      int group=(threadIdx.x%32)/4+rr*8;
      maximum[rr]=smem.maximum[threadIdx.x/32][group];
      inv[rr]=smem.inverse[threadIdx.x/32][group];
    }
    bool full=start>=0 && start+kColumns<=kCount && (start+kColumns-1)*16+31<=position;
    if(full) pruned_scores::write_result<T,T,true,false,true>(p,smem,acc,start,position,maximum,inv);
    else pruned_scores::write_result<T,T,true,true,true>(p,smem,acc,start,position,maximum,inv);
  }
}

template <typename T, typename O, bool Pool, bool Selection, typename Map>
__global__ __launch_bounds__(kThreads+128,1) void fused_scores_kernel(
    __grid_constant__ const Params p,__grid_constant__ const Map map) {
  using Tr=Traits<T>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& smem=*reinterpret_cast<typename Tr::template Shared<kStages,true>*>(storage);
  int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  if(tid==0) {
    for(int stage=0;stage<kStages;++stage) {smem.ready[stage].init(1);smem.released[stage].init(kThreads);}
    smem.q_ready.init(1);
    smem.cohort_done[0].init(kThreads/2);smem.cohort_done[1].init(kThreads/2);
    smem.seed_ready.init(1);smem.tail_ready.init(1);
    for(int cohort=0;cohort<2;++cohort) {smem.cohort_tail_ready[cohort].init(1);smem.cohort_tail_released[cohort].init(kThreads/2);}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();pruned_scores::load_query<T>(p,smem,map);__syncthreads();
  int normalizer_tiles=(kCount+127)/128,tiles=(kBlocks+31)/32;
  int seeds=min(kSeedTiles,tiles);
  if(tid>=kThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if(tid==kThreads) {
      int iteration=0;
      for(int tile=0;tile<normalizer_tiles;++tile,++iteration) {
        int buffer=iteration%kStages;
        if(iteration>=kStages) smem.released[buffer].wait(((iteration/kStages)-1)&1);
        pruned_scores::load_k<T>(smem,buffer,tile*128,map);
      }
      smem.seed_ready.wait(0);
      for(int j=0;j<seeds;++j,++iteration) {
        int buffer=iteration%kStages;
        if(iteration>=kStages) smem.released[buffer].wait(((iteration/kStages)-1)&1);
        pruned_scores::load_k<T>(smem,buffer,smem.order[j]*128,map);
      }
      smem.tail_ready.wait(0);
      // All seed reads and both tail lists are complete before tail_ready.
      // Reuse K buffers 0/1 under independent fresh barrier epochs.
      for(int j=0;j<max(smem.cohort_tail_count[0],smem.cohort_tail_count[1]);++j) {
#pragma unroll
        for(int cohort=0;cohort<2;++cohort) if(j<smem.cohort_tail_count[cohort]) {
          if(j>0) smem.cohort_tail_released[cohort].wait((j-1)&1);
          pruned_scores::load_tail_k<T>(smem,cohort,smem.cohort_tail_order[cohort][j]*128,map);
        }
      }
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<112>();
  smem.q_ready.wait(0);
  int64_t position=smem.positions[warp];
  {
  float maximum[2]={-INFINITY,-INFINITY},denominator[2]={0,0};
  auto acc=partition_fragment_C(typename Tr::MMA{},Shape<_64,Int<kColumns>>{});
  for(int tile=0;tile<normalizer_tiles;++tile) {
    if(tid>=kThreads/2) smem.cohort_done[0].wait(tile&1);
    else if(tile>0) smem.cohort_done[1].wait((tile-1)&1);
    int buffer=tile%kStages;smem.ready[buffer].wait((tile/kStages)&1);
    pruned_scores::issue<T>(smem,buffer,acc);warpgroup_wait<0>();smem.released[buffer].arrive();
    smem.cohort_done[tid/(kThreads/2)].arrive();
    bool full=(tile+1)*128<=kCount && ((tile+1)*128-1)*16+31<=position;
    if(full) normalize_tile<false,true>(acc,tile,kCount,position,maximum,denominator,&smem.tile_summary[warp][0][0]);
    else normalize_tile<true,true>(acc,tile,kCount,position,maximum,denominator,&smem.tile_summary[warp][0][0]);
  }
  float inv[2];
  unsigned certifiable=1;
#pragma unroll
  for(int r=0;r<2;++r) {
    float sum=denominator[r];sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);
    certifiable &= isfinite(sum) && (isfinite(maximum[r]) || maximum[r]==-INFINITY);
    inv[r]=1.0f/(sum>0 ? sum : 1);if(maximum[r]==-INFINITY) maximum[r]=0;
    if(lane%4==0) {smem.maximum[warp][lane/4+r*8]=maximum[r];smem.inverse[warp][lane/4+r*8]=inv[r];}
  }
  certifiable=__all_sync(0xffffffff,certifiable);
  if(lane==0) smem.certifiable[warp]=certifiable;
  }
  __syncwarp();build_bounds<T>(p,smem,tiles,position);
  // Different union row strides overlap neighboring warps, so all consumers
  // finish tile-max reads before any warp writes pooled QA.
  cutlass::arch::NamedBarrier::sync(kThreads,0);
  for(int b=lane;b<kBlocks;b+=32) smem.pooled[warp][b]=T(mask_pool(-INFINITY,b,position/64));
  cutlass::arch::NamedBarrier::sync(kThreads,0);
  if(warp==0) {
    unsigned sorted[2];
#pragma unroll
    for(int i=0;i<2;++i) {
      int tile=lane+i*32;unsigned bound=0;
      if(tile<tiles) for(int q=0;q<kQueryRows;++q) bound=max(bound,smem.bounds[q][tile]);
      sorted[i]=tile<tiles ? (bound<<6)|(63-tile) : 0;
    }
    sort64<2,1>(sorted);
#pragma unroll
    for(int i=0;i<2;++i) if(lane+i*32<tiles) smem.order[lane+i*32]=63-(sorted[i]&63);
  }
  cutlass::arch::NamedBarrier::sync(kThreads,0);
  if(tid==0) smem.seed_ready.arrive();
  for(int j=0;j<seeds;++j)
    score_tile<T>(p,smem,smem.order[j],normalizer_tiles+j,position);
  __syncwarp();
  unsigned cut=nosa_selection_cutoff::top33_seed_cutoff_bf16(reinterpret_cast<uint16_t const*>(smem.pooled[warp]),kBlocks,kCount,position/64,smem.order,seeds,lane);
  if(lane==0) smem.seed_cut[warp]=cut;
  for(int tile=lane;tile<tiles;tile+=32) {
    smem.keep[warp][tile]=smem.bounds[warp][tile]>=cut;

  }

  cutlass::arch::NamedBarrier::sync(kThreads,0);
  if(warp==0) {
    unsigned long long selected=0;
    for(int j=0;j<seeds;++j) selected|=1ull<<smem.order[j];
    __syncwarp();int total0=0,total1=0;
#pragma unroll
    for(int i=0;i<2;++i) {
      int tile=lane+i*32;bool keep0=false,keep1=false;
      bool extra=tile<tiles && !(selected&(1ull<<tile));
      if(extra) {
#pragma unroll
        for(int q=0;q<kQueryRows/2;++q) {
          keep0|=bool(smem.keep[q][tile]);
          keep1|=bool(smem.keep[q+kQueryRows/2][tile]);
        }
      }

      unsigned mask0=__ballot_sync(0xffffffff,keep0),mask1=__ballot_sync(0xffffffff,keep1);
      if(keep0) smem.cohort_tail_order[0][total0+__popc(mask0&((1u<<lane)-1))]=tile;
      if(keep1) smem.cohort_tail_order[1][total1+__popc(mask1&((1u<<lane)-1))]=tile;
      total0+=__popc(mask0);total1+=__popc(mask1);
    }
    if(lane==0) {
      smem.cohort_tail_count[0]=total0;smem.cohort_tail_count[1]=total1;
    }
  }
  cutlass::arch::NamedBarrier::sync(kThreads,0);
  if(tid==0) smem.tail_ready.arrive();
  int cohort=tid/(kThreads/2);
  for(int j=0;j<smem.cohort_tail_count[cohort];++j)
    score_tail_tile<T>(p,smem,smem.cohort_tail_order[cohort][j],j,position);
  __syncwarp();int row=blockIdx.x*kQueryRows+warp;
  if(row<kRows) {
    int64_t offset=(int64_t(row)*kHeads+blockIdx.y)*64;
    nosa_selection_fused::select_prefix_row(reinterpret_cast<uint16_t const*>(smem.pooled[warp]),kBlocks,position/64,
        p.ranking+blockIdx.y*64,p.ids+offset,p.valid+offset,smem.promotions[warp],smem.promoted_ids[warp],smem.seed_cut[warp]);
  }
}

template <typename T, typename O, bool Pool, bool Selection = false>
bool launch(const Params& p, cudaStream_t stream) {
  auto shape = make_shape(kCount, 128, kHeads);
  auto tma = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<const T*>(p.k)),
                  make_layout(shape, make_stride(p.kc, _1{}, p.kh))),
      typename Traits<T>::KLayout{});
  auto qshape = make_shape(make_shape(_16{}, kRows), _128{}, kHeads);
  auto qtma = make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<const T*>(p.q)),
                  make_layout(qshape, make_stride(make_stride(p.qg, p.qr), _1{}, p.qh))),
      typename Traits<T>::QLayout{}, Tile<Int<kQueryRows * 16>, _128>{}, _1{});
  TensorMaps<decltype(shape), decltype(tma), decltype(qshape), decltype(qtma)> map{shape, tma, qshape, qtma};
  auto fused = fused_scores_kernel<T, O, Pool, Selection, decltype(map)>;
  int shared = sizeof(typename Traits<T>::template Shared<kStages, Selection>);
  if (!configure(fused, shared, p.device, true)) return false;
  fused<<<dim3((kRows + kQueryRows - 1) / kQueryRows, kHeads), kThreads + 128, shared, stream>>>(p, map);
  auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
  return true;
}
}  // namespace pruned_scores
