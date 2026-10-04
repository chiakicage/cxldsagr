// One cooperative kernel: idle FA3 producer warps fetch pinned-host sparse KV.
// Numerical mainloop, sparse masking, epilogue and repair reuse the resident code.
#include <tvm/ffi/function.h>
#pragma push_macro("TVM_FFI_DLL_EXPORT_TYPED_FUNC")
#undef TVM_FFI_DLL_EXPORT_TYPED_FUNC
#define TVM_FFI_DLL_EXPORT_TYPED_FUNC(Name, Function)
#define nosa_fa3 nosa_offload_fa3_base
#include "nosa_attention_fa3.cu"
#undef nosa_fa3
#pragma pop_macro("TVM_FFI_DLL_EXPORT_TYPED_FUNC")

namespace nosa_offload_fused {
using namespace cute;
using namespace flashinfer;
using tvm::ffi::TensorView;
namespace original = nosa_offload_fa3_base;
struct TraceVariant;
struct KT : original::KT {
  using AttentionVariant=TraceVariant;
  struct SharedStorage : original::KT::SharedStorage {int fetch_slot;};
};
using EP=original::EP;
using T=original::T;
using NativeParams=original::NativeParams;
constexpr int Group=original::Group, Rows=original::Rows, Capacity=original::Capacity;
constexpr int ProducerRegisters=24, ConsumerRegisters=240;
constexpr int FetchStripes=NOSA_FETCH_STRIPES, StripeTokens=64/FetchStripes;
static_assert(FetchStripes>0 && 64%FetchStripes==0);

__device__ __forceinline__ int64_t globaltimer() {
  int64_t ticks;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
  return ticks;
}

template<int N> struct TraceSoftmax : original::SparseSoftmax<N> {
  int64_t* trace;
  int batch,fetch_slots;
  CUTLASS_DEVICE TraceSoftmax(int start,int tokens,int64_t* trace_,int batch_,int slots_)
      :original::SparseSoftmax<N>(start,tokens),trace(trace_),batch(batch_),fetch_slots(slots_) {}
  template<bool Init,class Acc> CUTLASS_DEVICE void update(Acc& acc) {
    int tile=this->tile-1;
    int64_t* row=trace?trace+(int64_t(fetch_slots)+batch*64+tile)*4:nullptr;
    if(row && threadIdx.x==128)row[0]=globaltimer();
    original::SparseSoftmax<N>::template update<Init>(acc);
    if(row && threadIdx.x==128) {
      row[1]=globaltimer();row[2]=0;row[3]=2;
    }
  }
};

struct TraceVariant : original::Variant {
  int64_t* trace;
  int batch,fetch_slots;
  template<class P,class Coord> CUTLASS_DEVICE TraceVariant(P const& p,Coord const& coord)
      :original::Variant(p,coord),trace(p.trace),batch(get<7>(coord)),
       fetch_slots(((p.prefix+63)/64)*p.additional_params.native.kv_heads) {}
  template<int N> CUTLASS_DEVICE auto GetAttentionUpdater() {
    return TraceSoftmax<N>(query_start,key_tokens,trace,batch,fetch_slots);
  }
};

struct FetchParams {
  uint4 const *host_k,*host_v;
  uint4 *keys,*values;
  int const *first_use;
  int *ready,*queue;
  int64_t *tile_bytes,*total_bytes,*trace;
  int64_t prefix;
  int heads,fetch_ctas;
  int64_t stripe_trace_offset;
};

__device__ __forceinline__ uint4 uncached_host_load(uint4 const* address) {
  uint4 value;
  // .cv invalidates cached system-memory lines: each logical vector is
  // fetched once from host backing, including on repeated benchmark calls.
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w)
      : "l"(address) : "memory");
  return value;
}

template<int Threads,int Pages>
__device__ __forceinline__ void fetch_body(FetchParams const& p,int worker,int workers) {
  constexpr int Chunks=(64*16+Threads-1)/Threads;
  int blocks=(p.prefix+63)/64;
  int cursor=worker;
  while(cursor<blocks*p.heads) {
    int tasks[Pages];
    // Preserve unique ownership while filling the register pipeline with
    // selected pages rather than spending its slots on holes in the union.
#pragma unroll
    for(int page=0;page<Pages;++page) {
      tasks[page]=-1;
      while(cursor<blocks*p.heads) {
        int task=cursor;
        cursor+=workers;
        int ordinal=task/p.heads,head=task%p.heads;
        int block=ordinal==0?0:blocks-ordinal;
        int slot=block*p.heads+head;
        if(p.first_use[slot]!=INT_MAX && p.ready[slot]!=8){tasks[page]=slot;break;}
      }
    }
    // Instrument the complete batch's host-copy window. All rows in this
    // batch share its start; each ends after that page's stores/fence. A
    // barrier between page loads would destroy the intended memory overlap.
    if(p.trace) {
      // Exclude outstanding scans by other warps from the copy interval.
      __syncthreads();
      if(threadIdx.x==0) {
        int64_t begin=globaltimer();
#pragma unroll
        for(int page=0;page<Pages;++page)
          if(tasks[page]>=0)p.trace[int64_t(tasks[page])*4]=begin;
      }
      __syncthreads();
    }
    uint4 keys[Pages][Chunks],values[Pages][Chunks];
#pragma unroll
    for(int page=0;page<Pages;++page) {
      if(tasks[page]>=0) {
        int block=tasks[page]/p.heads,head=tasks[page]%p.heads;
        int tokens=min(int64_t(64),p.prefix-int64_t(block)*64);
#pragma unroll
        for(int chunk=0;chunk<Chunks;++chunk) {
          int item=threadIdx.x+chunk*Threads;
          if(item<tokens*16) {
            int token=item/16,column=item%16;
            int64_t offset=((int64_t(block)*64+token)*p.heads+head)*16+column;
            keys[page][chunk]=uncached_host_load(p.host_k+offset);
            values[page][chunk]=uncached_host_load(p.host_v+offset);
          }
        }
      }
    }
#pragma unroll
    for(int page=0;page<Pages;++page) {
      if(tasks[page]>=0) {
        int block=tasks[page]/p.heads,head=tasks[page]%p.heads;
        int tokens=min(int64_t(64),p.prefix-int64_t(block)*64);
#pragma unroll
        for(int chunk=0;chunk<Chunks;++chunk) {
          int item=threadIdx.x+chunk*Threads;
          if(item<tokens*16) {
            int token=item/16,column=item%16;
            int64_t offset=((int64_t(block)*64+token)*p.heads+head)*16+column;
            p.keys[offset]=keys[page][chunk];
            p.values[offset]=values[page][chunk];
          }
        }
        __threadfence();
        __syncthreads();
        if(threadIdx.x==0) {
          auto bytes=static_cast<unsigned long long>(tokens)*128*2*sizeof(uint16_t);
          if(p.trace) {
            int64_t* row=p.trace+int64_t(tasks[page])*4;
            row[1]=globaltimer();row[2]=bytes;row[3]=1;
          }
          int tile=p.first_use[tasks[page]];
          atomicAdd(reinterpret_cast<unsigned long long*>(p.tile_bytes+tile),bytes);
          atomicAdd(reinterpret_cast<unsigned long long*>(p.total_bytes),bytes);
          int* flag=p.ready+tasks[page];
          asm volatile("st.release.gpu.global.u32 [%0], 1;" :: "l"(flag) : "memory");
        }
        __syncthreads();
      }
    }
  }
}

__global__ void serialized_fetch(FetchParams p) {
  fetch_body<128,1>(p,blockIdx.x,gridDim.x);
}

__global__ void compact_fetch_queue(FetchParams p,bool head_phased) {
  // The two-head/256-batch specialization serves head 1 completely before
  // head 0, matching the persistent compute scheduler. Within each head the
  // queue still starts with page 0, followed by descending logical blocks.
  // Other geometries retain the original block-major, interleaved-head order.
  // The queue contains each selected logical (block, head) once, without holes.
  __shared__ int warp_counts[8],running;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int blocks=(p.prefix+63)/64,tasks=blocks*p.heads;
  if(threadIdx.x==0)running=0;
  __syncthreads();
  for(int base=0;base<tasks;base+=256) {
    int task=base+threadIdx.x,slot=-1;
    bool selected=false;
    if(task<tasks) {
      int ordinal=head_phased?task%blocks:task/p.heads;
      int head=head_phased?1-task/blocks:task%p.heads;
      int block=ordinal==0?0:blocks-ordinal;
      slot=block*p.heads+head;
      selected=p.first_use[slot]!=INT_MAX && p.ready[slot]!=8;
    }
    unsigned votes=__ballot_sync(0xffffffff,selected);
    int rank=__popc(votes&((1u<<lane)-1u));
    if(lane==0)warp_counts[warp]=__popc(votes);
    __syncthreads();
#pragma unroll
    for(int prior=0;prior<8;++prior)if(prior<warp)rank+=warp_counts[prior];
    if(selected)p.queue[2+running+rank]=slot;
    __syncthreads();
    if(threadIdx.x==0) {
#pragma unroll
      for(int prior=0;prior<8;++prior)running+=warp_counts[prior];
    }
    // Complete all reads and the running-count update before reusing storage.
    __syncthreads();
  }
  if(threadIdx.x==0){p.queue[0]=running;p.queue[1]=0;}
}


// Complete the 128 independent head-1 batches in the first wave, so their
// successors can start consuming head 0 while that head is still fetching.
// Preserve the resident cost ordering and logical-batch tie break within a
// head. This is only dispatched for two KV heads and exactly 256 batches.
__global__ void sort_work_by_head_and_union_size(int const *counts,int *order) {
  __shared__ unsigned exchange[256];
  int tid=threadIdx.x;
  unsigned value=((tid&1u)<<16)|(((counts[tid]+1)/2)<<8)|(255-tid);
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

__device__ __forceinline__ void fetch_spare_barrier() {
  // CUTLASS user barrier 6 maps to hardware barrier 14. The resident sparse
  // variant uses user 7 (hardware 15), and this TMA path never uses producer-WG 6.
  static_assert(int(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier)==8);
  // The memory clobber also protects the shared task broadcast from compiler
  // reordering across this subgroup barrier.
  asm volatile("bar.sync 14, 96;" ::: "memory");
}

__device__ __forceinline__ void fetch_spare_body(FetchParams const& p,volatile int& shared_slot) {
  // Only warp 1/2/3 of the producer warpgroup enter this body. Warp 0 retains
  // its original TMA role while both consumer warpgroups perform attention.
  int lane=threadIdx.x-32;
  while(true) {
    if(lane==0) {
      int task=atomicAdd(p.queue+1,1);
      shared_slot=task<p.queue[0]*FetchStripes
          ?p.queue[2+task/FetchStripes]*FetchStripes+task%FetchStripes:-1;
    }
    fetch_spare_barrier();
    int encoded=shared_slot;
    if(encoded<0)break;
    int slot=encoded/FetchStripes,stripe=encoded%FetchStripes;
    int block=slot/p.heads,head=slot%p.heads;
    int tokens=min(int64_t(64),p.prefix-int64_t(block)*64);
    int stripe_tokens=max(0,min(StripeTokens,tokens-stripe*StripeTokens));
    if(p.trace && stripe_tokens) {
      // Exclude task claiming and decoding from the host-copy interval.
      fetch_spare_barrier();
      if(lane==0) {
        auto* start=reinterpret_cast<unsigned long long*>(p.trace+int64_t(slot)*4);
        auto now=static_cast<unsigned long long>(globaltimer());
        p.trace[(int64_t(p.stripe_trace_offset)+encoded)*4]=now;
        atomicCAS(start,0ull,now);
        atomicMin(start,now);
      }
      fetch_spare_barrier();
    }
    // Keep only one K/V vector pair live within the producer's 24-register
    // budget. Other resident CTAs provide the host-memory request concurrency.
#pragma unroll 1
    for(int item=stripe*StripeTokens*16+lane;
        item<min(tokens,(stripe+1)*StripeTokens)*16;item+=96) {
      int token=item/16,column=item%16;
      int64_t offset=((int64_t(block)*64+token)*p.heads+head)*16+column;
      uint4 key=uncached_host_load(p.host_k+offset);
      uint4 value=uncached_host_load(p.host_v+offset);
      p.keys[offset]=key;
      p.values[offset]=value;
    }
    // Each writer completes its stores before the elected thread publishes the
    // ready flag. The TMA warp uses acquire + async-proxy fence before reading.
    __threadfence();
    fetch_spare_barrier();
    if(lane==0) {
      if(p.trace && stripe_tokens) {
        auto now=static_cast<unsigned long long>(globaltimer());
        auto* stripe_row=p.trace+(int64_t(p.stripe_trace_offset)+encoded)*4;
        stripe_row[1]=now;stripe_row[2]=int64_t(stripe_tokens)*128*2*sizeof(uint16_t);
        stripe_row[3]=3;
        auto* end=reinterpret_cast<unsigned long long*>(p.trace+int64_t(slot)*4+1);
        atomicMax(end,now);
      }
      unsigned done;
      int* flag=p.ready+slot;
      // Each CTA's writer fences precede this release. Acquiring the prior
      // RMW forms a chain across all stripes; a TMA acquire of FetchStripes
      // therefore observes every writer, including CTAs that finished early.
      asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
          : "=r"(done) : "l"(flag) : "memory");
      if(done==FetchStripes-1) {
        auto bytes=static_cast<unsigned long long>(tokens)*128*2*sizeof(uint16_t);
        if(p.trace) {
          int64_t* row=p.trace+int64_t(slot)*4;
          row[2]=bytes;row[3]=1;
        }
        int tile=p.first_use[slot];
        atomicAdd(reinterpret_cast<unsigned long long*>(p.tile_bytes+tile),bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(p.total_bytes),bytes);
      }
    }
    fetch_spare_barrier();
  }
}

struct Scheduler {
  struct Params { int const *counts,*order; int queries,heads,compute_ctas,batches; };
  struct WorkTileInfo {
    int ordinal;
    CUTLASS_DEVICE bool is_valid(Params const& p) const {return ordinal<p.batches;}
    CUTLASS_DEVICE auto get_block_coord(Params const& p) const {
      int batch=p.order?p.order[ordinal]:ordinal;
      return cute::tuple{0,0,batch%p.heads,batch*Rows,batch*Capacity,
                         int(Rows),p.counts[batch]*64,batch};
    }
  };
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& p) const {
    return {int(blockIdx.x)};
  }
  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&,WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}
  template<bool Producer> CUTLASS_DEVICE WorkTileInfo get_next_work(
      Params const& p,WorkTileInfo const& work) const {return {work.ordinal+p.compute_ctas};}
};

struct ML : original::ML {
  using Base=original::ML;
  struct Params : Base::Params {int const* ready;int64_t prefix;int64_t* trace;};
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
        // Only historical blocks have a pending host fetch. Padded TMA
        // pages are OOB, while all suffix tokens were staged before this launch.
        if(int64_t(block)*64<p.prefix) {
          int const* flag=p.ready+int64_t(block)*p.additional_params.native.kv_heads+kh;
          unsigned value;
          do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
            if(value!=FetchStripes)__nanosleep(64);
          } while(value!=FetchStripes);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
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

/*
 * Device-callable body derived from FlashInfer 0.6.18 prefill_sm90.cuh.
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao. Licensed under the BSD 3-Clause.
 * Modified by the FlashInfer team. Local changes make the body device-callable
 * with parameter references and assign otherwise idle producer warps to fetch.
 * The numerical math and TMA/consumer work loops remain unchanged.
 */
template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits,
          bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename TileScheduler,
          bool MULTIITEMSCORING = false>
__device__ __forceinline__ void fused_compute_body(
    typename CollectiveMainloop::Params const& mainloop_params,
    typename CollectiveEpilogue::Params const& epilogue_params,
    typename TileScheduler::Params const& scheduler_params,
    FetchParams const& fetch_params) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  using AttentionVariant = typename Ktraits::AttentionVariant;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;
  static_assert(use_tma_load_kv && Ktraits::NUM_THREADS==384 &&
                Ktraits::NUM_PRODUCER_THREADS==32 && NUM_MMA_THREADS==256,
                "Spare fetch warps require the pinned 12-warp TMA FA3 geometry");

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  }

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(/*num_threads=*/1);
    shared_storage.barrier_O.init(/*num_threads=*/1);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);
    }
  }();

  MainloopPipeline pipeline_v = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params);
    }
  }();

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer
  // blocks in the Cluster
  __syncthreads();

  uint32_t* maybe_prefix_len_ptr = nullptr;
  if constexpr (has_maybe_prefix_len_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_prefix_len_ptr = mainloop_params.additional_params.maybe_prefix_len_ptr;
  }
  uint16_t* maybe_token_pos_in_items_ptr = nullptr;
  if constexpr (has_maybe_token_pos_in_items_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_token_pos_in_items_ptr = mainloop_params.additional_params.maybe_token_pos_in_items_ptr;
  }
  uint32_t token_pos_in_items_len = 0;
  if constexpr (has_token_pos_in_items_len_v<decltype(mainloop_params.additional_params)>) {
    token_pos_in_items_len = mainloop_params.additional_params.token_pos_in_items_len;
  }
  uint16_t* maybe_max_item_len_ptr = nullptr;
  if constexpr (has_maybe_max_item_len_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_max_item_len_ptr = mainloop_params.additional_params.maybe_max_item_len_ptr;
  }

  if (warp_group_idx == 0) {  // Producer
    if constexpr (use_tma_load_kv) {
      // Dynamic registers are drawn from this CTA's launch allocation, not
      // the entire SM. 168 * 384 = 128 * 24 + 256 * 240 = 64512 registers.
      // Increasing producers to 32 would deadlock a consumer's TRY_ALLOC.
      cutlass::arch::warpgroup_reg_dealloc<ProducerRegisters>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<72>();
    }

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (!use_tma_load_kv || warp_idx_in_warpgroup == 0) {  // Load Q, K, V
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
           work_tile_info.is_valid(scheduler_params);
           work_tile_info = scheduler.template get_next_work</*is_producer=*/true>(
               scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len,
              batch_idx] = block_coord;

        if (q_tile_idx * CTA_Q >= qo_len) {
          continue;
        }
        int num_kv_tiles =
            collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
        if (num_kv_tiles <= 0) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          scheduler.broadcast_next_work(work_tile_info);
          continue;
        }
        int num_kv_tiles_outside_items_window = 0;
        int num_kv_tiles_prefix = 0;
        if constexpr (MULTIITEMSCORING) {
          auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
          auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
          auto valid_items_window_len =
              std::max(0, q_tile_idx * CTA_Q + kv_len - qo_len - max_item_len);
          num_kv_tiles_outside_items_window = valid_items_window_len / CTA_KV;
          num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
        }
        if constexpr (MULTIITEMSCORING) {
          collective_mainloop.load<LEFT_SLIDING_WINDOW>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
              num_kv_tiles_outside_items_window, num_kv_tiles_prefix);
        } else {
          collective_mainloop.template load<LEFT_SLIDING_WINDOW>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx);
        }
        ++work_idx;
      }
      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    } else if(int(blockIdx.x)<fetch_params.fetch_ctas) {
      fetch_spare_body(fetch_params,shared_storage.fetch_slot);
    }
  } else {  // Consumer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();
    } else {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 216 : 144>();
    }

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMmaPV tiled_mma_pv;

    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
    // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

    CollectiveMainloop::WarpScheduler::mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*is_producer=*/false>(scheduler_params,
                                                                                  work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
          block_coord;

      AttentionVariant variant(mainloop_params, block_coord);
      auto attention_updater =
          variant.template GetAttentionUpdater<2 * (2 * CTA_Q / NUM_MMA_THREADS)>();

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
      if (num_kv_tiles <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NUM_COPY_THREADS, block_coord);
        continue;
      }

      int swa_begin_kv_tile_idx = 0;
      int swa_end_kv_tile_idx = -1;
      if constexpr (LEFT_SLIDING_WINDOW) {
        swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(
            mainloop_params.window_left, q_tile_idx, qo_len, kv_len);
        swa_end_kv_tile_idx = get_swa_end_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                     q_tile_idx, qo_len, kv_len);
      }

      uint32_t prefix_len = 0;
      uint16_t* token_pos_in_items = nullptr;
      if constexpr (MULTIITEMSCORING) {
        prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        token_pos_in_items = maybe_token_pos_in_items_ptr + batch_idx * token_pos_in_items_len;
      }
      int num_kv_tiles_outside_items_window = 0;
      int num_kv_tiles_prefix = 0;
      if constexpr (MULTIITEMSCORING) {
        auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
        auto valid_items_window_len =
            std::max(0, q_tile_idx * CTA_Q + kv_len - qo_len - max_item_len);
        num_kv_tiles_outside_items_window = valid_items_window_len / CTA_KV;
        num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
      }
      mma_f16<Ktraits, /*LEFT_SLIDING_WINDOW=*/LEFT_SLIDING_WINDOW, CAUSAL, MULTIITEMSCORING,
              CollectiveMainloop::WarpScheduler>(
          mainloop_params, variant, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
          tOrO, attention_updater, num_kv_tiles, swa_begin_kv_tile_idx, swa_end_kv_tile_idx,
          threadIdx.x - NUM_COPY_THREADS, work_idx, q_tile_idx, shared_storage, qo_len, kv_len,
          qo_head_idx, kv_head_idx, prefix_len, token_pos_in_items,
          num_kv_tiles_outside_items_window, num_kv_tiles_prefix);
      collective_epilogue.store(epilogue_params, tOrO, attention_updater.get_lse(), shared_storage,
                                tiled_mma_pv, threadIdx.x - NUM_COPY_THREADS, block_coord);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

__global__ void __launch_bounds__(KT::NUM_THREADS,1) fused_main(
    CUTE_GRID_CONSTANT ML::Params const mainloop,
    CUTE_GRID_CONSTANT EP::Params const epilogue,
    CUTE_GRID_CONSTANT Scheduler::Params const scheduler,
    CUTE_GRID_CONSTANT FetchParams const fetch) {
  fused_compute_body<ML,EP,KT,false,false,Scheduler,false>(mainloop,epilogue,scheduler,fetch);
}

void check_cuda(cudaError_t error) {
  TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}

struct LaunchInfo {
  unsigned long long context_id=0;
  int device=-1,sms=0,active_blocks=0;
};

LaunchInfo launch_info(int device) {
  // Function attributes and occupancy belong to a CUDA context. Its unique ID
  // prevents a destroyed/recreated context from reusing stale configuration.
  static auto get_context_id=[] {
    void* entry=nullptr;
#if CUDART_VERSION >= 12050
    check_cuda(cudaGetDriverEntryPointByVersion("cuCtxGetId",&entry,12000,
                                                cudaEnableDefault,nullptr));
#else
    check_cuda(cudaGetDriverEntryPoint("cuCtxGetId",&entry,cudaEnableDefault,nullptr));
#endif
    TVM_FFI_ICHECK(entry);
    return reinterpret_cast<decltype(&cuCtxGetId)>(entry);
  }();
  unsigned long long context_id=0;
  TVM_FFI_ICHECK(get_context_id(nullptr,&context_id)==CUDA_SUCCESS);
  static thread_local LaunchInfo cached[16];
  static thread_local unsigned next=0;
  for(auto const& info:cached)
    if(info.context_id==context_id && info.device==device)return info;

  LaunchInfo info{context_id,device};
  int cooperative;
  check_cuda(cudaDeviceGetAttribute(&info.sms,cudaDevAttrMultiProcessorCount,device));
  check_cuda(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));
  TVM_FFI_ICHECK(cooperative)<<"NOSA fused offload requires cooperative kernel launch";
  auto kernel=reinterpret_cast<void const*>(fused_main);
  nosa_attention::configure_shared_memory(kernel,sizeof(KT::SharedStorage),device);
  cudaFuncAttributes attributes{};
  check_cuda(cudaFuncGetAttributes(&attributes,kernel));
  // Hopper allocates registers in 256-register units per warp. setmaxnreg
  // draws from this fixed CTA allocation, even if the SM has registers left.
  // Match cuda_occupancy.h's regsAllocatedPerWarp/regsAllocatedPerCTA formula.
  int registers_per_warp=((attributes.numRegs*32+255)/256)*256;
  int cta_pool=registers_per_warp*(KT::NUM_THREADS/32);
  constexpr int required=cutlass::NumThreadsPerWarpGroup*ProducerRegisters+
                         KT::NUM_MMA_THREADS*ConsumerRegisters;
  TVM_FFI_ICHECK(attributes.numRegs>0 && cta_pool>=required)
      <<"Fused FA3 dynamic register demand "<<required
      <<" exceeds the compiled CTA register pool "<<cta_pool
      <<" (numRegs="<<attributes.numRegs<<"); refusing a potentially deadlocking launch";
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &info.active_blocks,fused_main,KT::NUM_THREADS,sizeof(KT::SharedStorage)));
  TVM_FFI_ICHECK(info.active_blocks>0)
      <<"Fused cooperative kernel cannot fit a CTA on this device";
  cached[next++%16]=info;
  return info;
}

void launch_repair(NativeParams p,TensorView q,TensorView k,TensorView v,cudaStream_t stream) {
  using Tr=nosa_attention::Traits<T>;
  auto qs=make_shape(p.heads,128,p.queries),ks=make_shape(p.tokens,128,p.kv_heads);
  auto qt=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<T const*>(q.data_ptr())),make_layout(qs,make_stride(q.stride(1),_1{},q.stride(0)))),typename Tr::QLayout{});
  auto kt=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<T const*>(k.data_ptr())),make_layout(ks,make_stride(k.stride(0),_1{},k.stride(1)))),typename Tr::KVLayout{});
  auto vt=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(static_cast<T const*>(v.data_ptr())),make_layout(ks,make_stride(v.stride(0),_1{},v.stride(1)))),typename Tr::KVLayout{});
  nosa_attention::Maps<decltype(qs),decltype(qt),decltype(ks),decltype(kt),decltype(ks),decltype(vt)> maps{qs,qt,ks,kt,ks,vt};
  auto repair=nosa_attention::attention_kernel<T,decltype(maps)>;
  nosa_attention::configure_shared_memory(reinterpret_cast<void const*>(repair),sizeof(typename Tr::Shared),q.device().device_id);
  repair<<<dim3(p.queries,p.kv_heads),256,sizeof(typename Tr::Shared),stream>>>(p,maps);
  check_cuda(cudaGetLastError());
}

void forward(TensorView q,TensorView k,TensorView v,TensorView ids,TensorView mask,
             TensorView bias,TensorView out,TensorView fallback,TensorView pages,
             TensorView members,TensorView counts,TensorView host_k,TensorView host_v,
             TensorView first_use,TensorView ready,TensorView tile_bytes,TensorView total_bytes,
             TensorView trace,TensorView fetch_queue,int64_t query_start,int64_t query_tile_size,
             int64_t fetch_ctas,bool overlap) {
  TVM_FFI_ICHECK(q.dtype().code==kDLBfloat && q.dtype().bits==16 && q.size(2)==128 && q.size(1)==k.size(1)*16);
  TVM_FFI_ICHECK(k.IsContiguous() && v.IsContiguous() && k.size(0)==v.size(0));
  TVM_FFI_ICHECK(host_k.device().device_type==kDLCPU && host_v.device().device_type==kDLCPU && host_k.IsContiguous() && host_v.IsContiguous());
  TVM_FFI_ICHECK(query_start>=0 && query_start+q.size(0)<=k.size(0) && query_tile_size>0);
  TVM_FFI_ICHECK(fetch_ctas>0 && fetch_ctas<=INT_MAX);
  int64_t fetch_slots=((query_start+63)/64)*k.size(1);
  TVM_FFI_ICHECK(fetch_queue.device().device_type==kDLCUDA &&
      fetch_queue.device().device_id==q.device().device_id &&
      fetch_queue.dtype().code==kDLInt && fetch_queue.dtype().bits==32 &&
      fetch_queue.IsContiguous() && fetch_queue.ndim()==1 &&
      fetch_queue.numel()>=fetch_slots+2);
  TVM_FFI_ICHECK(ready.dtype().bits==32 && ready.IsContiguous() && ready.numel()>=((query_start+63)/64)*k.size(1));
  int trace_batches=((q.size(0)+Group-1)/Group)*k.size(1);
  TVM_FFI_ICHECK(trace.numel()==0 || (trace.dtype().bits==64 && trace.IsContiguous() &&
      trace.ndim()==2 && trace.size(1)==4 &&
      trace.size(0)>=fetch_slots*(1+FetchStripes)+trace_batches*64));
  NativeParams p{};
  p.q=q.data_ptr();p.qr=q.stride(0);p.qh=q.stride(1);
  p.v=v.data_ptr();p.vr=v.stride(0);p.vh=v.stride(1);
  p.ids=ids.data_ptr();p.ids64=ids.dtype().bits==64;
  p.valid=mask.numel()?static_cast<bool const*>(mask.data_ptr()):nullptr;
  p.bias=bias.numel()?bias.data_ptr():nullptr;
  p.bias_type=bias.dtype().bits==32?0:bias.dtype().code==kDLBfloat?1:2;
  p.group_fallback=static_cast<int*>(fallback.data_ptr());
  p.out=out.data_ptr();p.tokens=k.size(0);p.queries=q.size(0);p.heads=q.size(1);p.kv_heads=k.size(1);p.count=ids.size(2);p.query_start=query_start;
  p.ir=ids.size(0)==1?0:ids.stride(0);p.ih=ids.size(1)==1?0:ids.stride(1);p.ib=ids.stride(2);
  if(p.valid){p.mr=mask.size(0)==1?0:mask.stride(0);p.mh=mask.size(1)==1?0:mask.stride(1);p.mb=mask.stride(2);}
  if(p.bias){p.br=bias.stride(0);p.bh=bias.stride(1);}
  if(!p.queries)return;
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,q.device().device_id));
  void *mapped_k=nullptr,*mapped_v=nullptr;
  if(query_start) {
    check_cuda(cudaHostGetDevicePointer(&mapped_k,host_k.data_ptr(),0));
    check_cuda(cudaHostGetDevicePointer(&mapped_v,host_v.data_ptr(),0));
  }
  FetchParams fetch{static_cast<uint4 const*>(mapped_k),static_cast<uint4 const*>(mapped_v),
      static_cast<uint4*>(k.data_ptr()),static_cast<uint4*>(v.data_ptr()),
      static_cast<int const*>(first_use.data_ptr()),static_cast<int*>(ready.data_ptr()),
      static_cast<int*>(fetch_queue.data_ptr()),
      static_cast<int64_t*>(tile_bytes.data_ptr()),static_cast<int64_t*>(total_bytes.data_ptr()),
      trace.numel()?static_cast<int64_t*>(trace.data_ptr()):nullptr,
      query_start,p.kv_heads,int(fetch_ctas),fetch_slots+int64_t(trace_batches)*64};
  if(!overlap || !query_start) {
    if(query_start) {
      int tasks=((query_start+63)/64)*p.kv_heads;
      serialized_fetch<<<std::min(tasks,1024),128,0,stream>>>(fetch);
      check_cuda(cudaGetLastError());
    }
    original::run(p,q,k,v,pages,members,counts,stream);
    return;
  }
  auto launch=launch_info(q.device().device_id);
  int batches=(p.queries+Group-1)/Group*p.kv_heads;
  bool head_phased=p.kv_heads==2 && batches==256;
  int grid=std::min(batches,launch.sms);
  TVM_FFI_ICHECK(grid<=launch.active_blocks*launch.sms)
      <<"Fused cooperative grid must fit concurrently to avoid waiting-CTA deadlock";
  fetch.fetch_ctas=std::min(int(fetch_ctas),grid);
  // Bound both the compactor's padded last chunk and terminal atomic claims.
  TVM_FFI_ICHECK(fetch_slots<=(INT_MAX-std::max(grid,256))/FetchStripes)
      <<"Sparse fetch queue exceeds int32 task indexing capacity";
  compact_fetch_queue<<<1,256,0,stream>>>(fetch,head_phased);
  check_cuda(cudaGetLastError());
  original::prepare<<<dim3((p.queries+Group-1)/Group,p.kv_heads),256,0,stream>>>(p,static_cast<int*>(pages.data_ptr()),static_cast<int*>(members.data_ptr()),static_cast<int*>(counts.data_ptr()));
  int *order=batches==256?static_cast<int*>(counts.data_ptr())+batches:nullptr;
  if(order) {
    if(head_phased)sort_work_by_head_and_union_size<<<1,256,0,stream>>>(static_cast<int const*>(counts.data_ptr()),order);
    else original::sort_work_by_union_size<<<1,256,0,stream>>>(static_cast<int const*>(counts.data_ptr()),order);
  }
  auto qlayout=flashinfer::get_gmem_layout(p.heads,p.queries,128,q.stride(1),q.stride(0));
  auto klayout=flashinfer::get_gmem_layout(64,p.kv_heads,128,k.stride(0),k.stride(1));
  auto vlayout=flashinfer::get_gmem_layout(64,p.kv_heads,128,v.stride(0),v.stride(1));
  original::ML::Arguments args{static_cast<T const*>(q.data_ptr()),qlayout,
      static_cast<T const*>(k.data_ptr()),klayout,static_cast<T const*>(v.data_ptr()),vlayout,
      static_cast<int const*>(pages.data_ptr()),-1,64*k.stride(0),64*v.stride(0),64,
      {p,static_cast<int const*>(pages.data_ptr()),static_cast<int const*>(members.data_ptr())}};
  ML::Params mainloop{original::ML::to_underlying_arguments(args),static_cast<int const*>(ready.data_ptr()),query_start,fetch.trace};
  EP::Params epilogue{static_cast<T*>(p.out),p.queries,p.kv_heads,p.group_fallback};
  Scheduler::Params scheduler{static_cast<int const*>(counts.data_ptr()),order,p.queries,p.kv_heads,grid,batches};
  void* kernel_args[]={&mainloop,&epilogue,&scheduler,&fetch};
  check_cuda(cudaLaunchCooperativeKernel(reinterpret_cast<void const*>(fused_main),dim3(grid),dim3(KT::NUM_THREADS),kernel_args,sizeof(KT::SharedStorage),stream));
  // Whole cooperative main completion includes every host fetch, so the
  // unchanged numerical/tail/union-overflow repair sees a complete sparse union.
  launch_repair(p,q,k,v,stream);
}
}  // namespace nosa_offload_fused
TVM_FFI_DLL_EXPORT_TYPED_FUNC(forward,nosa_offload_fused::forward);
