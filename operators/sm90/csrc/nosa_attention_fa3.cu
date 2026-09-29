#include "nosa_attention.cu"
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>

#ifndef FA3_TMA
#define FA3_TMA 0
#endif

#ifndef FA3_GROUP
#define FA3_GROUP 8
#endif
#ifndef FA3_STAGES
#define FA3_STAGES 2
#endif
#ifndef FA3_KV
#define FA3_KV 64
#endif

namespace nosa_fa3 {
using namespace cute;
using tvm::ffi::TensorView;
constexpr int Group = FA3_GROUP;
constexpr int Rows = Group * 16;
constexpr int Capacity = Group * 64;
using T = cutlass::bfloat16_t;
using NativeParams = nosa_attention::Params;

struct Additional {
  NativeParams native;
  int const *pages, *members;
};

template<int N> struct SafeSoftmax : flashinfer::OnlineSoftmax<N, false> {
  using Base = flashinfer::OnlineSoftmax<N, false>;
  using Base::row_max; using Base::row_sum; using Base::scores_scale;
  CUTLASS_DEVICE SafeSoftmax() : Base(1.f) {}
  template<bool Init, class Acc> CUTLASS_DEVICE void update(Acc &acc) {
    auto scores = make_tensor(acc.data(), flashinfer::convert_layout_acc_rowcol(acc.layout()));
    auto previous = make_fragment_like(row_max);
    if constexpr (!Init) cute::copy(row_max, previous);
    flashinfer::MaxOp<float> max_op;
    CUTE_UNROLL
    for(int r=0;r<N;++r) {
      float p0=scores(r,0),p1=scores(r,1),p2=scores(r,2),p3=scores(r,3);
      CUTE_UNROLL
      for(int c=4;c<size<1>(scores);c+=4) {
        p0=max_op(p0,scores(r,c));p1=max_op(p1,scores(r,c+1));
        p2=max_op(p2,scores(r,c+2));p3=max_op(p3,scores(r,c+3));
      }
      float value=max_op(max_op(p0,p1),max_op(p2,p3));
      value=max_op(value,__shfl_xor_sync(0xffffffff,value,2));
      value=max_op(value,__shfl_xor_sync(0xffffffff,value,1));
      if constexpr (!Init)value=max_op(value,row_max(r));
      row_max(r)=value;
    }
    CUTE_UNROLL
    for (int r = 0; r < N; ++r) {
      float safe = row_max(r) == -INFINITY ? 0.f : row_max(r);
      if constexpr (!Init) {
        scores_scale(r) = exp2f(__fmul_rn(__fsub_rn(previous(r),safe),1.4426950408889634f));
        row_sum(r) *= scores_scale(r);
      }
      CUTE_UNROLL
      for (int c = 0; c < size<1>(scores); ++c) scores(r,c) = exp2f(__fmul_rn(__fsub_rn(scores(r,c),safe),1.4426950408889634f));
    }
    flashinfer::reduce_sum<Init, false>(scores, row_sum);
  }
  template<class Acc> CUTLASS_DEVICE void rescale_o(Acc &acc) {
    auto values=make_tensor(acc.data(),flashinfer::convert_layout_acc_rowcol(acc.layout()));
    CUTE_UNROLL
    for(int r=0;r<N;++r)if(scores_scale(r)!=1.f) {
      CUTE_UNROLL
      for(int c=0;c<size<1>(values);++c)values(r,c)*=scores_scale(r);
    }
  }
  template<class Acc> CUTLASS_DEVICE void finalize(Acc &acc, float pv_scale=1.f) {
    flashinfer::SumOp<float> sum;
    flashinfer::quad_allreduce_(row_sum, row_sum, sum);
    CUTE_UNROLL
    for (int r = 0; r < N; ++r) {
      float total = row_sum(r);
      scores_scale(r) = total == 0.f ? 1.f : pv_scale / total;
      row_sum(r) = row_max(r)*1.4426950408889634f + log2f(total);
    }
  }
};

struct Variant;
using BaseKT = flashinfer::AttentionKernelTraits<bool(FA3_TMA),128,128,Rows,FA3_KV,FA3_STAGES,T,T,T,int32_t,Variant>;
struct KT : BaseKT {
  struct SharedStorage : BaseKT::SharedStorage {
    int blocks[128], membership[128];
    float cis[128 * 64];
  };
};

template<int N> struct SparseSoftmax : SafeSoftmax<N> {
  int query_start,key_tokens,tile;
  CUTLASS_DEVICE SparseSoftmax(int query_start_,int key_tokens_) : SafeSoftmax<N>(),
      query_start(query_start_),key_tokens(key_tokens_),tile((key_tokens_+FA3_KV-1)/FA3_KV) {}
  template<bool Init,class Acc> CUTLASS_DEVICE void update(Acc &acc) {
    --tile;
    extern __shared__ char storage[];
    auto &s=*reinterpret_cast<KT::SharedStorage *>(storage);
    typename KT::TiledMmaQK mma;
    auto thread=mma.get_thread_slice(threadIdx.x-128);
    auto identity=cute::make_identity_tensor(select<0,1>(typename KT::TileShape_QKD{}));
    auto coords=thread.partition_C(identity);
    auto rc=make_tensor(coords.data(),flashinfer::convert_layout_acc_rowcol(coords.layout()));
    auto scores=make_tensor(acc.data(),flashinfer::convert_layout_acc_rowcol(acc.layout()));
    int blocks[FA3_KV/64],members[FA3_KV/64];
    CUTE_UNROLL
    for(int i=0;i<FA3_KV/64;++i) {
      blocks[i]=s.blocks[tile*(FA3_KV/64)+i];
      members[i]=s.membership[tile*(FA3_KV/64)+i];
    }
    // All 8 queries must select both pages, and even the earliest query
    // must see their final tokens. Prepared physical block IDs are below 2048.
    bool all_common=true;
    CUTE_UNROLL
    for(int page=0;page<FA3_KV/64;++page)
      all_common&=(members[page]==((1<<Group)-1)) &&
                  (blocks[page]*64+63<=query_start);
    if(all_common) {
      CUTE_UNROLL
      for(int c=0;c<size<1>(scores);++c) {
        int col=get<1>(rc(0,c));
        float bias=s.cis[tile*FA3_KV+col];
        CUTE_UNROLL
        for(int r=0;r<size<0>(scores);++r)
          scores(r,c)=__fadd_rn(__fmul_rn(scores(r,c),0.08838834764831845f),bias);
      }
    } else {
      int limits[FA3_KV/64][N];
      CUTE_UNROLL
      for(int page=0;page<FA3_KV/64;++page) {
        CUTE_UNROLL
        for(int r=0;r<N;++r) {
          int query=get<0>(rc(r,0))/16;
          int last=query_start+query-blocks[page]*64;
          limits[page][r]=(members[page]&(1<<query))?min(last,63):-1;
        }
      }
      CUTE_UNROLL
      for(int c=0;c<size<1>(scores);++c) {
        int col=get<1>(rc(0,c));
        int key=tile*FA3_KV+col;
        float bias=s.cis[key];
        CUTE_UNROLL
        for(int r=0;r<size<0>(scores);++r) {
          bool active=col%64<=limits[col/64][r];
          scores(r,c)=active?__fadd_rn(__fmul_rn(scores(r,c),0.08838834764831845f),bias):-INFINITY;
        }
      }
    }
    SafeSoftmax<N>::template update<Init>(acc);
  }
};

struct Variant {
  int query_start,key_tokens;
  CUTLASS_DEVICE static auto &shared() {
    extern __shared__ char storage[];
    return *reinterpret_cast<KT::SharedStorage *>(storage);
  }
  template<class P, class Coord> CUTLASS_DEVICE Variant(P const &p, Coord const &coord) {
    int batch=get<7>(coord),count=((get<6>(coord)+FA3_KV-1)/FA3_KV)*(FA3_KV/64);
    key_tokens=get<6>(coord);
    auto const &n=p.additional_params.native;
    int head=batch % n.kv_heads;
    query_start=n.query_start+(batch/n.kv_heads)*Group;
    auto &s=shared();
    int tid=threadIdx.x-128;
    for(int i=tid;i<count;i+=KT::NUM_MMA_THREADS) {
      s.blocks[i]=p.additional_params.pages[batch*Capacity+i];
      s.membership[i]=p.additional_params.members[batch*Capacity+i];
    }
    cutlass::arch::NamedBarrier::sync(KT::NUM_MMA_THREADS,7);
    for(int i=tid;i<count*64;i+=KT::NUM_MMA_THREADS) {
      int token=s.blocks[i/64]*64+i%64;
      float bias=0.f;
      if(n.bias) {
        int64_t offset=int64_t(token)*n.br+head*n.bh;
        if(n.bias_type==0)bias=static_cast<float const *>(n.bias)[offset];
        else if(n.bias_type==1)bias=float(static_cast<cutlass::bfloat16_t const *>(n.bias)[offset]);
        else bias=float(static_cast<cutlass::half_t const *>(n.bias)[offset]);
      }
      s.cis[i]=bias;
    }
    cutlass::arch::NamedBarrier::sync(KT::NUM_MMA_THREADS,7);
  }
  template<int N> CUTLASS_DEVICE auto GetAttentionUpdater() { return SparseSoftmax<N>(query_start,key_tokens); }
  template<class P> CUTLASS_DEVICE float LogitsTransform(P const &, float logits,
      uint32_t, uint32_t qi, uint32_t ki, uint32_t, uint32_t) {
    return logits;
  }
};

struct Scheduler {
  struct Params { int const *counts,*order; int queries, heads; };
  struct WorkTileInfo {
    int batch; bool valid;
    CUTLASS_DEVICE bool is_valid(Params const &) const { return valid; }
    CUTLASS_DEVICE auto get_block_coord(Params const &p) const {
      return cute::tuple{0, 0, batch % p.heads, batch * Rows, batch * Capacity,
                         int(Rows), p.counts[batch] * 64, batch};
    }
  };
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const &p) const { return {p.order?p.order[blockIdx.x]:int(blockIdx.x), true}; }
  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const &, WorkTileInfo &) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo &) const {}
  template<bool Producer> CUTLASS_DEVICE WorkTileInfo get_next_work(Params const &, WorkTileInfo const &) const { return {0, false}; }
};

// Sort this 256-work-item shape by descending paired-page work. Every CTA
// still executes the original FlashInfer entry; this helper is fully timed.
__global__ void sort_work_by_union_size(int const *counts,int *order) {
  __shared__ unsigned exchange[256];
  int tid=threadIdx.x;
  unsigned value=(((counts[tid]+1)/2)<<8)|(255-tid);
  CUTE_UNROLL
  for(int k=2;k<=256;k*=2) {
    CUTE_UNROLL
    for(int j=k/2;j>0;j/=2) {
      unsigned other;
      if(j<32)other=__shfl_xor_sync(0xffffffff,value,j);
      else {
        exchange[tid]=value;
        __syncthreads();
        other=exchange[tid^j];
        __syncthreads();
      }
      bool take_max=((tid&j)==0)==((tid&k)==0);
      value=take_max?max(value,other):min(value,other);
    }
  }
  order[tid]=255-(value&255);
}

__global__ void prepare(NativeParams p, int *pages, int *members, int *counts) {
  __shared__ unsigned membership[2048];
  __shared__ int warp_count[64], emitted;
  int query_base=blockIdx.x*Group,batch=blockIdx.x*p.kv_heads+blockIdx.y;
  int blocks=(p.tokens+63)/64;
  if(blocks>2048) {
    if(threadIdx.x==0) {
      counts[batch]=0;
      CUTE_UNROLL
      for(int i=0;i<Group/4;++i)if(query_base+i*4<p.queries)
        p.group_fallback[((query_base+i*4)/4)*p.kv_heads+blockIdx.y]=1;
    }
    return;
  }
  for(int i=threadIdx.x;i<blocks;i+=blockDim.x)membership[i]=0;
  if(threadIdx.x==0)emitted=0;
  __syncthreads();
  int warp=threadIdx.x/32,lane=threadIdx.x%32;
  if(warp<Group) {
    CUTE_UNROLL
    for(int offset=0;offset<64;offset+=32) {
      int block=nosa_attention::grouped_block(p,query_base+warp,lane+offset);
      if(block!=INT_MAX)atomicOr(membership+block,1u<<warp);
    }
  }
  __syncthreads();
  unsigned masks[8],ballots[8];
  CUTE_UNROLL
  for(int chunk=0;chunk<8;++chunk) {
    int block=chunk*256+threadIdx.x;
    masks[chunk]=block<blocks?membership[block]:0;
    ballots[chunk]=__ballot_sync(0xffffffff,masks[chunk]!=0);
    if(lane==0)warp_count[chunk*8+warp]=__popc(ballots[chunk]);
  }
  __syncthreads();
  if(warp==0) {
    int first=warp_count[lane],second=warp_count[lane+32];
    int scan_first=first,scan_second=second;
    CUTE_UNROLL
    for(int delta=1;delta<32;delta*=2) {
      int other_first=__shfl_up_sync(0xffffffff,scan_first,delta);
      int other_second=__shfl_up_sync(0xffffffff,scan_second,delta);
      if(lane>=delta){scan_first+=other_first;scan_second+=other_second;}
    }
    int first_total=__shfl_sync(0xffffffff,scan_first,31);
    warp_count[lane]=scan_first-first;
    warp_count[lane+32]=first_total+scan_second-second;
    if(lane==31)emitted=first_total+scan_second;
  }
  __syncthreads();
  bool failed=false;
  CUTE_UNROLL
  for(int chunk=0;chunk<8;++chunk)if(masks[chunk]) {
    int block=chunk*256+threadIdx.x;
    int rank=warp_count[chunk*8+warp]+__popc(ballots[chunk]&((1u<<lane)-1));
    pages[batch*Capacity+rank]=block;
    members[batch*Capacity+rank]=masks[chunk];
    failed|=(block+1)*64>p.tokens;
  }
  int fallback=__syncthreads_or(failed||emitted>128);
  if(threadIdx.x==0) {
    counts[batch]=fallback?0:emitted;
    if(FA3_KV==128 && emitted%2) {
      pages[batch*Capacity+emitted]=0;
      members[batch*Capacity+emitted]=0;
    }
    CUTE_UNROLL
    for(int i=0;i<Group/4;++i)if(query_base+i*4<p.queries)
      p.group_fallback[((query_base+i*4)/4)*p.kv_heads+blockIdx.y]=fallback;
  }
}

#if FA3_TMA
// Preserve FlashInfer's actual prefill kernel and mma_f16 scheduling. Only the
// sparse producer changes: each physical 64-token page uses two 64x64 TMA transfers.
struct ML : flashinfer::CollectiveMainloop<Additional,KT,false> {
  using Base=flashinfer::CollectiveMainloop<Additional,KT,false>;
  using LayoutT=typename Base::LayoutT;
  using QTile=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_16,_64>{}));
  using TMA_Q=decltype(make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<T const *>(nullptr)),LayoutT{}),QTile{}));
  using KVLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{},Shape<_64,_64>{}));
  using TMA_KV=decltype(make_tma_copy(SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<T const *>(nullptr)),LayoutT{}),KVLayout{}));
  struct Arguments {
    T const *Q_ptr; LayoutT layout_Q;
    T const *K_ptr; LayoutT layout_K;
    T const *V_ptr; LayoutT layout_V;
    int const *kv_indices; int window_left;
    int64_t k_page_stride,v_page_stride; uint32_t page_size;
    Additional additional_params;
  };
  struct Params {
    LayoutT layout_Q,layout_K,layout_V;
    TMA_Q tma_load_Q;
    TMA_KV tma_load_K,tma_load_V;
    int window_left;
    Additional additional_params;
  };
  static Params to_underlying_arguments(Arguments const &a) {
    auto k_layout=flashinfer::get_gmem_layout(a.additional_params.native.tokens,
        a.additional_params.native.kv_heads,128,stride<0>(a.layout_K),stride<2>(a.layout_K));
    auto v_layout=flashinfer::get_gmem_layout(a.additional_params.native.tokens,
        a.additional_params.native.kv_heads,128,stride<0>(a.layout_V),stride<2>(a.layout_V));
    auto q=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(a.Q_ptr),a.layout_Q),QTile{});
    auto k=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(a.K_ptr),k_layout),KVLayout{});
    auto v=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(a.V_ptr),v_layout),KVLayout{});
    return {a.layout_Q,k_layout,v_layout,q,k,v,a.window_left,a.additional_params};
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const &p) {
    cute::prefetch_tma_descriptor(p.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_load_V.get_tma_descriptor());
  }
  CUTLASS_DEVICE int get_num_kv_tiles(Params const &,int,int,int length) {return (length+FA3_KV-1)/FA3_KV;}
  template<bool Left,class Coord,class Sched,class Shared>
  CUTLASS_DEVICE void load(Params const &p,MainloopPipeline pipeline_k,MainloopPipeline pipeline_v,
      PipelineState &write_k,PipelineState &write_v,Shared &s,Sched &scheduler,
      typename Sched::Params const &scheduler_params,typename Sched::WorkTileInfo &work,
      Coord const &coord,int work_idx) {
    auto [qi,qh,kh,qbegin,kbegin,qlen,klen,batch]=coord;
    auto mQ=p.tma_load_Q.get_tma_tensor(p.layout_Q.shape());
    auto gK=p.tma_load_K.get_tma_tensor(p.layout_K.shape())(_,_,kh);
    auto gV=p.tma_load_V.get_tma_tensor(p.layout_V.shape())(_,_,kh);
    auto kg=flat_divide(gK,Tile<_64,_64>{});
    auto vg=flat_divide(gV,Tile<_64,_64>{});
    bool leader=cute::elect_one_sync();
    auto issue=[&](auto const &desc,auto const &g,T *buffer,auto &pipe,auto &state,int tile) {
      pipe.producer_acquire(state);
      CUTE_UNROLL
      for(int half=0;half<FA3_KV/64;++half) {
        int slot=tile*(FA3_KV/64)+half;
        int block=slot<klen/64?p.additional_params.pages[batch*Capacity+slot]
                             :(p.additional_params.native.tokens+63)/64;
        CUTE_UNROLL
        for(int dhalf=0;dhalf<2;++dhalf) {
          auto dst=make_tensor(make_smem_ptr(buffer+state.index()*FA3_KV*128+dhalf*FA3_KV*64+half*64*64),KVLayout{});
          auto slice=desc.get_slice(_0{});
          copy(desc.with(*pipe.producer_get_barrier(state)),
               slice.partition_S(g(_,_,block,dhalf)),slice.partition_D(dst));
        }
      }
      ++state;
    };
    int tile=(klen+FA3_KV-1)/FA3_KV-1;
    if(leader)issue(p.tma_load_K,kg,s.smem_k.data(),pipeline_k,write_k,tile);
    cutlass::arch::NamedBarrier::sync(KT::NUM_MMA_THREADS+KT::NUM_PRODUCER_THREADS,
        static_cast<int>(flashinfer::NamedBarriers::kQueryEmpty));
    if(leader) {
      s.barrier_Q.arrive_and_expect_tx(Base::TmaTransactionBytesQ);
      int head=batch%p.additional_params.native.kv_heads;
      int query_base=(batch/p.additional_params.native.kv_heads)*Group;
      CUTE_UNROLL
      for(int query=0;query<Group;++query) {
        auto g=flat_divide(mQ(_,_,query_base+query),Tile<_16,_64>{});
        CUTE_UNROLL
        for(int dhalf=0;dhalf<2;++dhalf) {
          auto dst=make_tensor(make_smem_ptr(s.smem_q.data()+dhalf*Rows*64+query*16*64),QTile{});
          auto slice=p.tma_load_Q.get_slice(_0{});
          copy(p.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType &>(s.barrier_Q)),
               slice.partition_S(g(_,_,head,dhalf)),slice.partition_D(dst));
        }
      }
    }
    s.barrier_O.wait((work_idx+1)%2);
    if(leader) {
      for(;tile>0;--tile) {
        issue(p.tma_load_K,kg,s.smem_k.data(),pipeline_k,write_k,tile-1);
        issue(p.tma_load_V,vg,s.smem_v.data(),pipeline_v,write_v,tile);
      }
    }
    scheduler.prefetch_next_work(scheduler_params,work);
    if(leader)issue(p.tma_load_V,vg,s.smem_v.data(),pipeline_v,write_v,0);
    scheduler.broadcast_next_work(work);
  }
};
#else
using ML = flashinfer::SparseCollectiveMainloop<Additional,KT,false>;
#endif
struct EP {
  struct Params { T *out; int queries,heads; int *fallback; };
  using Arguments=Params;
  static Params to_underlying_arguments(Arguments const &a){return a;}
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const &) {}
  CUTLASS_DEVICE void store_tail() {}
  template<class Coord,class Shared,class Acc,class Lse,class Mma>
  CUTLASS_DEVICE void store(Params const &p,Acc const &acc,Lse const &,Shared &,
      Mma mma,int tid,Coord const &coord) {
    int batch=get<7>(coord),head=batch%p.heads,query_base=(batch/p.heads)*Group;
    auto identity=make_identity_tensor(select<0,1>(typename KT::TileShape_PDV{}));
    auto rc0=mma.get_thread_slice(tid).partition_C(identity);
    auto rc=make_tensor(rc0.data(),flashinfer::convert_layout_acc_rowcol(rc0.layout()));
    auto values=make_tensor(acc.data(),flashinfer::convert_layout_acc_rowcol(acc.layout()));
    bool nonfinite=false;
    CUTE_UNROLL
    for(int r=0;r<size<0>(values);++r) {
      int row=get<0>(rc(r,0)),query=query_base+row/16;
      if(query<p.queries) {
        CUTE_UNROLL
        for(int c=0;c<size<1>(values);c+=2) {
          int col=get<1>(rc(r,c));
          int64_t offset=(int64_t(query)*p.heads*16+head*16+row%16)*128+col;
          auto pair=__floats2bfloat162_rn(values(r,c),values(r,c+1));
          __nv_bfloat162_raw bits=pair;
          nonfinite|=(bits.x&0x7f80u)==0x7f80u||(bits.y&0x7f80u)==0x7f80u;
          *reinterpret_cast<__nv_bfloat162 *>(p.out+offset)=pair;
        }
      }
    }
    if(__any_sync(0xffffffff,nonfinite)&&(tid&31)==0) {
      CUTE_UNROLL
      for(int i=0;i<Group/4;++i)if(query_base+i*4<p.queries)
        atomicExch(p.fallback+((query_base+i*4)/4)*p.heads+head,1);
    }
  }
  template<class Coord,class Shared>
  CUTLASS_DEVICE void store_zero(Params const &p,Shared &,int tid,Coord const &coord) {
    int batch=get<7>(coord),head=batch%p.heads,query_base=(batch/p.heads)*Group;
    for(int offset=tid*8;offset<Rows*128;offset+=KT::NUM_MMA_THREADS*8) {
      int row=offset/128,d=offset%128,query=query_base+row/16;
      if(query<p.queries) {
        int64_t target=(int64_t(query)*p.heads*16+head*16+row%16)*128+d;
        *reinterpret_cast<uint4 *>(p.out+target)=make_uint4(0,0,0,0);
      }
    }
  }
};

void run(NativeParams p, TensorView q, TensorView k, TensorView v, TensorView pages,
         TensorView members, TensorView counts, cudaStream_t stream) {
  prepare<<<dim3((p.queries+Group-1)/Group,p.kv_heads),256,0,stream>>>(p,static_cast<int *>(pages.data_ptr()),static_cast<int *>(members.data_ptr()),static_cast<int *>(counts.data_ptr()));
  int batches = (p.queries+Group-1)/Group*p.kv_heads;
  int *order=batches==256?static_cast<int *>(counts.data_ptr())+batches:nullptr;
  if(order)sort_work_by_union_size<<<1,256,0,stream>>>(static_cast<int const *>(counts.data_ptr()),order);
  auto qlayout = flashinfer::get_gmem_layout(p.heads,p.queries,128,q.stride(1),q.stride(0));
  auto klayout = flashinfer::get_gmem_layout(64,p.kv_heads,128,k.stride(0),k.stride(1));
  auto vlayout = flashinfer::get_gmem_layout(64,p.kv_heads,128,v.stride(0),v.stride(1));
  ML::Arguments ma{static_cast<T const *>(q.data_ptr()),qlayout,static_cast<T const *>(k.data_ptr()),klayout,static_cast<T const *>(v.data_ptr()),vlayout,static_cast<int const *>(pages.data_ptr()),-1,64*k.stride(0),64*v.stride(0),64,{p,static_cast<int const *>(pages.data_ptr()),static_cast<int const *>(members.data_ptr())}};
  EP::Arguments ea{static_cast<T *>(p.out),p.queries,p.kv_heads,p.group_fallback};
  auto kernel=flashinfer::PrefillWithKVCacheKernel<ML,EP,KT,false,false,Scheduler,false>;
  nosa_attention::configure_shared_memory(reinterpret_cast<void const *>(kernel),sizeof(KT::SharedStorage),q.device().device_id);
  kernel<<<batches,KT::NUM_THREADS,sizeof(KT::SharedStorage),stream>>>(ML::to_underlying_arguments(ma),EP::to_underlying_arguments(ea),Scheduler::Params{static_cast<int const *>(counts.data_ptr()),order,p.queries,p.kv_heads});
  using Tr=nosa_attention::Traits<T>;
  auto qs=make_shape(p.heads,128,p.queries),ks=make_shape(p.tokens,128,p.kv_heads);
  auto qt=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<T const *>(q.data_ptr())),make_layout(qs,make_stride(q.stride(1),_1{},q.stride(0)))),typename Tr::QLayout{});
  auto kt=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<T const *>(k.data_ptr())),make_layout(ks,make_stride(k.stride(0),_1{},k.stride(1)))),typename Tr::KVLayout{});
  auto vt=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<T const *>(v.data_ptr())),make_layout(ks,make_stride(v.stride(0),_1{},v.stride(1)))),typename Tr::KVLayout{});
  nosa_attention::Maps<decltype(qs),decltype(qt),decltype(ks),decltype(kt),decltype(ks),decltype(vt)> maps{qs,qt,ks,kt,ks,vt};
  auto repair=nosa_attention::attention_kernel<T,decltype(maps)>;
  nosa_attention::configure_shared_memory(reinterpret_cast<void const *>(repair),sizeof(typename Tr::Shared),q.device().device_id);
  repair<<<dim3(p.queries,p.kv_heads),256,sizeof(typename Tr::Shared),stream>>>(p,maps);
  auto error=cudaGetLastError(); TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}

void forward(TensorView q, TensorView k, TensorView v, TensorView ids, TensorView mask,
             TensorView bias, TensorView out, TensorView fallback, TensorView pages, TensorView members,
             TensorView counts, int64_t query_start) {
  TVM_FFI_ICHECK(q.dtype().code==kDLBfloat && q.size(2)==128 && q.size(1)==k.size(1)*16);
  NativeParams p{};
  p.q=q.data_ptr(); p.qr=q.stride(0); p.qh=q.stride(1);
  p.v=v.data_ptr(); p.vr=v.stride(0); p.vh=v.stride(1);
  p.ids=ids.data_ptr(); p.ids64=ids.dtype().bits==64;
  p.valid=mask.numel()?static_cast<bool const *>(mask.data_ptr()):nullptr;
  p.bias=bias.numel()?bias.data_ptr():nullptr;
  p.bias_type=bias.dtype().bits==32?0:bias.dtype().code==kDLBfloat?1:2;
  p.group_fallback=static_cast<int *>(fallback.data_ptr());
  p.out=out.data_ptr(); p.tokens=k.size(0); p.queries=q.size(0); p.heads=q.size(1); p.kv_heads=k.size(1); p.count=ids.size(2); p.query_start=query_start;
  p.ir=ids.size(0)==1?0:ids.stride(0); p.ih=ids.size(1)==1?0:ids.stride(1); p.ib=ids.stride(2);
  if(p.valid){p.mr=mask.size(0)==1?0:mask.stride(0);p.mh=mask.size(1)==1?0:mask.stride(1);p.mb=mask.stride(2);}
  if(p.bias){p.br=bias.stride(0);p.bh=bias.stride(1);}
  if(!p.queries)return;
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,q.device().device_id));
  run(p,q,k,v,pages,members,counts,stream);
}
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fa3_forward,nosa_fa3::forward);
