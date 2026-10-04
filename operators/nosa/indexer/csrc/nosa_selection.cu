// Exact stable two-stage selection over the model's rounded pooled scores.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cstdint>

namespace nosa_selection {
using tvm::ffi::TensorView;
struct Params {
  uint16_t const* qa;
  void const *cis, *pool, *positions;
  int64_t* ids;
  bool* valid;
  int rows, heads, blocks, count, cached, cis_type, pool_type;
  int64_t query_start, cs, ch, ps, ph;
  bool bf16, contiguous, pos64;
  bool const* finite = nullptr;
};
__device__ __forceinline__ bool reject_nonfinite(Params const& p) {
  if (!p.finite || *p.finite) return false;
  for (int i=threadIdx.x; i<64; i+=blockDim.x) {
    int64_t offset=int64_t(blockIdx.x)*64+i;
    p.ids[offset]=-1;
    if(p.valid) p.valid[offset]=false;
  }
  return true;
}
__device__ __forceinline__ unsigned key16(unsigned bits) {
  if ((bits & 0x7fff) == 0) bits = 0;
  return bits & 0x8000 ? bits ^ 0xffff : bits ^ 0x8000;
}
__device__ __forceinline__ float read_value(void const* ptr, int64_t idx, int kind) {
  if (kind == 0) return __bfloat162float(static_cast<__nv_bfloat16 const*>(ptr)[idx]);
  if (kind == 1) return __half2float(static_cast<__half const*>(ptr)[idx]);
  return static_cast<float const*>(ptr)[idx];
}
__device__ __forceinline__ unsigned value_key(float value, bool bf16) {
  if (bf16) return key16(__bfloat16_as_ushort(__float2bfloat16_rn(value)));
  return key16(__half_as_ushort(__float2half_rn(value)));
}
// The broadcast word is separate from the per-warp input words. A subsequent
// reduction cannot overwrite it before every warp reaches the first barrier.
template<int Op, int Threads> __device__ __forceinline__ unsigned reduce(unsigned value, unsigned* scratch) {
  constexpr int kThreads = Threads, kWarps = Threads / 32;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if constexpr (Op == 0) value = __reduce_add_sync(0xffffffff, value);
  if constexpr (Op == 1) value = __reduce_min_sync(0xffffffff, value);
  if constexpr (Op == 2) value = __reduce_max_sync(0xffffffff, value);
  if constexpr (kWarps == 1) return value;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? scratch[lane] : (Op == 1 ? 0xffffffff : 0);
    if constexpr (Op == 0) value = __reduce_add_sync(0xffffffff, value);
    if constexpr (Op == 1) value = __reduce_min_sync(0xffffffff, value);
    if constexpr (Op == 2) value = __reduce_max_sync(0xffffffff, value);
    if (lane == 0) scratch[kWarps] = value;
  }
  __syncthreads();
  return scratch[kWarps];
}
template<int Threads, int Items> __device__ unsigned threshold(unsigned const (&keys)[Items], int blocks,
                                                   unsigned rank, unsigned inf, unsigned* scratch) {
  constexpr int kThreads = Threads;
  unsigned ninf = 0, nfinite = 0, low = 65535, high = 0, nan_count = 0;
#pragma unroll
  for (int i=0; i<Items; ++i) {
    bool valid = int(threadIdx.x) + i*kThreads < blocks;
    bool finite = valid && keys[i] > 65535-inf && keys[i] < inf;
    nan_count += valid && (keys[i] > inf || keys[i] < 65535-inf);
    ninf += valid && keys[i] == inf;
    nfinite += finite;
    if (finite) { low = min(low, keys[i]); high = max(high, keys[i]); }
  }
  nan_count = reduce<0, Threads>(nan_count, scratch);
  if (nan_count) {
    // Preserve unsigned radix order even for arbitrary NaN payloads. The model
    // rejects nonfinite inputs, but the low-level selector must remain bounded.
    low=0; high=65535;
    while(low<high) {
      unsigned middle=(low+high+1)/2, count=0;
#pragma unroll
      for(int i=0;i<Items;++i) count += int(threadIdx.x)+i*kThreads<blocks && keys[i]>=middle;
      count=reduce<0, Threads>(count,scratch);
      if(count>=rank) low=middle; else high=middle-1;
    }
    return low;
  }
  ninf = reduce<0, Threads>(ninf, scratch);
  nfinite = reduce<0, Threads>(nfinite, scratch);
  if (rank <= ninf) return inf;
  if (rank > ninf + nfinite) return 65535-inf;
  rank -= ninf;
  low = reduce<1, Threads>(low, scratch);
  high = reduce<2, Threads>(high, scratch);
  while (low < high) {
    unsigned middle = (low+high+1)/2, count=0;
#pragma unroll
    for (int i=0; i<Items; ++i) {
      bool valid = int(threadIdx.x) + i*kThreads < blocks;
      count += valid && keys[i] < inf && keys[i] >= middle;
    }
    count = reduce<0, Threads>(count, scratch);
    if (count >= rank) low = middle;
    else high = middle-1;
  }
  return low;
}
template<int Threads, int Items> __device__ void ranks(bool const (&flags)[Items], unsigned (&result)[Items], unsigned* scratch) {
  constexpr int kThreads = Threads, kWarps = Threads / 32;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if constexpr (kWarps == 1) {
    unsigned offset=0;
#pragma unroll
    for(int i=0;i<Items;++i) {
      unsigned mask=__ballot_sync(0xffffffff,flags[i]);
      result[i]=offset+__popc(mask&((1u<<lane)-1));
      offset+=__popc(mask);
    }
    return;
  }
  __syncthreads();
#pragma unroll
  for (int i=0; i<Items; ++i) {
    unsigned mask = __ballot_sync(0xffffffff, flags[i]);
    if (lane == 0) scratch[i*kWarps+warp] = __popc(mask);
    result[i] = __popc(mask & ((1u << lane)-1));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned offset=0;
    for (int i=0; i<Items*kWarps; ++i) {
      unsigned count=scratch[i]; scratch[i]=offset; offset+=count;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i=0; i<Items; ++i) result[i] += scratch[i*kWarps+warp];
  __syncthreads();
}
template<int Threads, int Items> __device__ void choose(unsigned const (&keys)[Items], bool (&chosen)[Items],
                                         int blocks, unsigned k, unsigned inf, unsigned* scratch) {
  constexpr int kThreads = Threads;
  unsigned cut=threshold<Threads>(keys, blocks, k, inf, scratch), greater=0;
  if constexpr (Threads == 32) {
#pragma unroll
    for (int i=0; i<Items; ++i) {
      bool valid=int(threadIdx.x)+i*kThreads < blocks;
      greater += valid && keys[i]>cut;
    }
    greater=reduce<0, Threads>(greater,scratch);
    unsigned offset=0, lane=threadIdx.x;
#pragma unroll
    for (int i=0; i<Items; ++i) {
      bool valid=int(threadIdx.x)+i*kThreads < blocks;
      bool equal=valid && keys[i]==cut;
      unsigned mask=__ballot_sync(0xffffffff,equal);
      unsigned rank=offset+__popc(mask&((1u<<lane)-1));
      offset+=__popc(mask);
      chosen[i]=valid && (keys[i]>cut || (equal && rank<k-greater));
    }
  } else {
    bool equal[Items]; unsigned tie_rank[Items];
#pragma unroll
    for (int i=0; i<Items; ++i) {
      bool valid=int(threadIdx.x)+i*kThreads < blocks;
      chosen[i]=valid && keys[i]>cut;
      equal[i]=valid && keys[i]==cut;
      greater += chosen[i];
    }
    greater=reduce<0, Threads>(greater,scratch);
    ranks<Threads>(equal,tie_rank,scratch);
#pragma unroll
    for (int i=0; i<Items; ++i) chosen[i] |= equal[i] && tie_rank[i] < k-greater;
  }
}
template<int Threads, int Items, bool CachedBf16> __global__ __launch_bounds__(Threads) void selection_kernel(Params p) {
  if(reject_nonfinite(p)) return;
  constexpr int kThreads = Threads, kWarps = Threads / 32;
  __shared__ unsigned scratch[Items*kWarps+kWarps+1];
  int row=blockIdx.x, head=row%p.heads;
  int64_t position=p.contiguous ? p.query_start+row/p.heads :
      (p.pos64 ? static_cast<int64_t const*>(p.positions)[row/p.heads] : static_cast<int const*>(p.positions)[row/p.heads]);
  int qblock=position/64;
  unsigned inf=p.bf16 ? 0xff80 : 0xfc00;
  unsigned keys[Items]; bool chosen[Items];
#pragma unroll
  for (int i=0; i<Items; ++i) {
    int b=threadIdx.x+i*kThreads;
    keys[i]=b<p.blocks ? key16(p.qa[int64_t(row)*p.blocks+b]) : 65535-inf;
  }
  choose<Threads>(keys,chosen,p.blocks,33,inf,scratch);
#pragma unroll
  for (int i=0; i<Items; ++i) {
    int b=threadIdx.x+i*kThreads;
    if constexpr(CachedBf16) {
      unsigned key=65535-inf;
      if (b<p.blocks && b<=qblock) {
        if(chosen[i] || b==0 || qblock<=b+16) key=inf;
        else if(b<p.cached) key=key16(static_cast<uint16_t const*>(p.pool)[int64_t(b)*p.ps+head*p.ph]);
      }
      keys[i]=key;
    } else {
    float score=-CUDART_INF_F;
    if (b < p.blocks && b <= qblock) {
      if (chosen[i] || b==0 || qblock<=b+16) score=CUDART_INF_F;
      else if (p.pool && b < p.cached) score=read_value(p.pool,int64_t(b)*p.ps+head*p.ph,p.pool_type);
      else {
#pragma unroll
        for (int w=0; w<5; ++w) {
          int c=b*4-1+w;
          if (c>=0 && c<p.count) {
            float value=read_value(p.cis,int64_t(c)*p.cs+head*p.ch,p.cis_type);
            score=(isnan(score) || isnan(value)) ? CUDART_NAN_F : fmaxf(score,value);
          }
        }
      }
    }
    keys[i]=value_key(score,p.bf16);
    }
  }
  choose<Threads>(keys,chosen,p.blocks,64,inf,scratch);
  if constexpr (Threads == 32) {
    unsigned offset=0, lane=threadIdx.x;
#pragma unroll
    for (int i=0; i<Items; ++i) {
      unsigned mask=__ballot_sync(0xffffffff,chosen[i]);
      unsigned rank=offset+__popc(mask&((1u<<lane)-1));
      offset+=__popc(mask);
      if(chosen[i]) {
        int b=threadIdx.x+i*kThreads;
        bool valid=b<=qblock;
        int64_t out=int64_t(row)*64+rank;
        p.ids[out]=valid ? b : -1;
        if(p.valid) p.valid[out]=valid;
      }
    }
  } else {
    unsigned output_rank[Items]; ranks<Threads>(chosen,output_rank,scratch);
#pragma unroll
    for (int i=0; i<Items; ++i) if (chosen[i]) {
      int b=threadIdx.x+i*kThreads;
      bool valid=b<=qblock;
      int64_t offset=int64_t(row)*64+output_rank[i];
      p.ids[offset]=valid ? b : -1;
      if(p.valid) p.valid[offset]=valid;
    }
  }
}
// For a contiguous query chunk spanning at most 17 logical blocks, every
// block at/after its first query block is local or future for every row.
// Global raw-CIS top64 from the earlier prefix therefore contains every
// possible non-promoted winner. Raising QA/local scores cannot introduce a
// raw-CIS item ranked below 64, including exact +/-infinity and smaller-ID ties.
__global__ void prepare_prefix_ranking(Params p, int prefix, unsigned* ranking) {
  constexpr int Threads=128, Items=16;
  __shared__ unsigned scratch[Items*4+5];
  unsigned keys[Items]; bool chosen[Items];
#pragma unroll
  for(int i=0;i<Items;++i) {
    int b=threadIdx.x+i*Threads;
    keys[i]=b<prefix ? key16(static_cast<uint16_t const*>(p.pool)[int64_t(b)*p.ps+blockIdx.x*p.ph]) : 0;
  }
  choose<Threads>(keys,chosen,prefix,64,0xff80,scratch);
  unsigned ranks_out[Items];
  ranks<Threads>(chosen,ranks_out,scratch);
#pragma unroll
  for(int i=0;i<Items;++i) if(chosen[i]) {
    int b=threadIdx.x+i*Threads;
    ranking[blockIdx.x*64+ranks_out[i]]=(keys[i]<<12)|(4095-b);
  }
}

template<int K, int J, bool Descending>
__device__ __forceinline__ void sort128_stage(unsigned (&values)[4]) {
  unsigned next[4];
#pragma unroll
  for(int i=0;i<4;++i) {
    unsigned other;
    if constexpr(J<32) other=__shfl_xor_sync(0xffffffff,values[i],J);
    else other=values[i^(J/32)];
    unsigned index=threadIdx.x+32*i;
    bool lower=((index&K)==0)==((index&J)==0);
    if constexpr(Descending) lower=!lower;
    next[i]=lower ? min(values[i],other) : max(values[i],other);
  }
#pragma unroll
  for(int i=0;i<4;++i) values[i]=next[i];
  if constexpr(J>1) sort128_stage<K,J/2,Descending>(values);
  else if constexpr(K<128) sort128_stage<K*2,K,Descending>(values);
}

__device__ unsigned prepare_promotions_packed(Params p, unsigned row, int qblock,
                                            unsigned* promotions, unsigned* promoted_ids) {
  unsigned packed[17];
  unsigned ninf=0,nfinite=0,nan_count=0,low=65535,high=0;
#pragma unroll
  for(int i=0;i<17;++i) {
    int a=threadIdx.x+i*64, b=a+32;
    unsigned lo=a<p.blocks ? key16(p.qa[int64_t(row)*p.blocks+a]) : 0;
    unsigned hi=b<p.blocks ? key16(p.qa[int64_t(row)*p.blocks+b]) : 0;
    packed[i]=lo|(hi<<16);
    ninf+=(a<p.blocks && lo==0xff80)+(b<p.blocks && hi==0xff80);
    bool flo=a<p.blocks && lo>127 && lo<0xff80;
    bool fhi=b<p.blocks && hi>127 && hi<0xff80;
    nfinite+=flo+fhi;
    nan_count+=(a<p.blocks && (lo<127 || lo>0xff80))+(b<p.blocks && (hi<127 || hi>0xff80));
    if(flo) {low=min(low,lo);high=max(high,lo);}
    if(fhi) {low=min(low,hi);high=max(high,hi);}
  }
  ninf=__reduce_add_sync(0xffffffff,ninf);
  nfinite=__reduce_add_sync(0xffffffff,nfinite);
  nan_count=__reduce_add_sync(0xffffffff,nan_count);
  low=__reduce_min_sync(0xffffffff,low);
  high=__reduce_max_sync(0xffffffff,high);
  unsigned cut;
  if(!nan_count && ninf>=33) cut=0xff80;
  else if(!nan_count && ninf+nfinite<33) cut=127;
  else {
    if(nan_count) {low=0;high=65535;}
    while(low<high) {
      unsigned middle=(low+high+1)/2, count=0, cmp=middle|(middle<<16);
#pragma unroll
      for(int i=0;i<17;++i) {
        unsigned flags=__vsetgeu2(packed[i],cmp);
        count+=(flags&65535)+(flags>>16);
      }
      count=__reduce_add_sync(0xffffffff,count);
      if(count>=33) low=middle;
      else high=middle-1;
    }
    cut=low;
  }
  unsigned greater=0, cmp=cut|(cut<<16);
#pragma unroll
  for(int i=0;i<17;++i) {
    unsigned flags=__vsetgtu2(packed[i],cmp);
    greater+=(flags&65535)+(flags>>16);
  }
  greater=__reduce_add_sync(0xffffffff,greater);
  unsigned tie_offset=0,promoted_count=0,lane=threadIdx.x;
#pragma unroll
  for(int i=0;i<33;++i) {
    int b=lane+i*32;
    unsigned key=(i%2 ? packed[i/2]>>16 : packed[i/2]&65535);
    bool equal=b<p.blocks && key==cut;
    unsigned mask=__ballot_sync(0xffffffff,equal);
    unsigned rank=tie_offset+__popc(mask&((1u<<lane)-1));
    tie_offset+=__popc(mask);
    bool qa=b<p.blocks && (key>cut || (equal && rank<33-greater));
    bool promoted=b<p.blocks && b<=qblock && (qa || b==0 || qblock<=b+16);
    mask=__ballot_sync(0xffffffff,promoted);
    if(lane==0) promotions[i]=mask;
    rank=promoted_count+__popc(mask&((1u<<lane)-1));
    if(promoted) promoted_ids[rank]=b;
    promoted_count+=__popc(mask);
  }
  return promoted_count;
}

__global__ __launch_bounds__(32) void selection_prefix_kernel(Params p, unsigned const* ranking) {
  if(reject_nonfinite(p)) return;
  constexpr int Items=33;
  __shared__ unsigned promotions[Items];
  __shared__ unsigned promoted_ids[64];
  unsigned row=blockIdx.x, lane=threadIdx.x, head=row%p.heads;
  int qblock=(p.query_start+row/p.heads)/64;
  unsigned promoted_count=prepare_promotions_packed(p,row,qblock,promotions,promoted_ids);
  __syncwarp();
  unsigned candidates[4];
#pragma unroll
  for(int i=0;i<2;++i) {
    unsigned packed=ranking[head*64+lane+i*32];
    unsigned b=4095-(packed&4095);
    bool duplicate=(promotions[b/32]>>(b%32))&1u;
    candidates[i]=duplicate ? 0 : packed;
    unsigned slot=lane+i*32;
    candidates[i+2]=slot<promoted_count ? (0xff80u<<12)|(4095-promoted_ids[slot]) : 0;
  }
  sort128_stage<2,1,true>(candidates);
#pragma unroll
  for(int i=0;i<4;++i) candidates[i]=i<2 ? 4095-(candidates[i]&4095) : 0xffffffff;
  sort128_stage<2,1,false>(candidates);
#pragma unroll
  for(int i=0;i<2;++i) {
    int64_t offset=int64_t(row)*64+lane+i*32;
    p.ids[offset]=candidates[i];
    if(p.valid) p.valid[offset]=true;
  }
}

// A score-fused selector consumes the same guarded prefix ranking, prepared
// before QK. It never reads QA and does not allocate or publish cache state.
void prepare_ranking(TensorView pool, TensorView ranking, int64_t prefix) {
  TVM_FFI_ICHECK(pool.device().device_type==kDLCUDA && pool.ndim()==2 && pool.size(1)>0);
  TVM_FFI_ICHECK(pool.dtype().code==kDLBfloat && pool.dtype().bits==16 && pool.dtype().lanes==1);
  TVM_FFI_ICHECK(prefix>=64 && prefix<=1056 && prefix<=pool.size(0));
  TVM_FFI_ICHECK(pool.stride(0)>=0 && pool.stride(1)>=0);
  TVM_FFI_ICHECK(ranking.device().device_type==kDLCUDA && ranking.device().device_id==pool.device().device_id);
  TVM_FFI_ICHECK(ranking.ndim()==2 && ranking.size(0)==pool.size(1) && ranking.size(1)==64 && ranking.IsContiguous());
  TVM_FFI_ICHECK(ranking.dtype().code==kDLInt && ranking.dtype().bits==32 && ranking.dtype().lanes==1);
  Params p{};
  p.pool=static_cast<char*>(pool.data_ptr())+pool.byte_offset();
  p.ps=pool.stride(0);p.ph=pool.stride(1);
  auto out=reinterpret_cast<unsigned*>(static_cast<char*>(ranking.data_ptr())+ranking.byte_offset());
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,pool.device().device_id));
  prepare_prefix_ranking<<<pool.size(1),128,0,stream>>>(p,prefix,out);
  auto error=cudaGetLastError();TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}

int dtype(TensorView t) { return t.dtype().code==kDLBfloat ? 0 : (t.dtype().bits==16 ? 1 : 2); }
void select_impl(TensorView qa, TensorView cis, TensorView pool, TensorView positions, TensorView ids,
            TensorView valid, TensorView ranking, int64_t query_start, bool has_pool, bool contiguous, bool write_mask, bool ranking_ready, bool const* finite=nullptr) {
  auto data=[](TensorView t) { return static_cast<char*>(t.data_ptr())+t.byte_offset(); };
  auto same_device=[&](TensorView t) {
    return t.device().device_type==kDLCUDA && t.device().device_id==qa.device().device_id;
  };
  auto floating=[](TensorView t) {
    return t.dtype().lanes==1 &&
      ((t.dtype().code==kDLBfloat && t.dtype().bits==16) ||
       (t.dtype().code==kDLFloat && (t.dtype().bits==16 || t.dtype().bits==32)));
  };
  TVM_FFI_ICHECK(qa.device().device_type==kDLCUDA);
  TVM_FFI_ICHECK(qa.ndim()==2 && floating(qa) && qa.dtype().bits==16 && qa.IsContiguous());
  TVM_FFI_ICHECK(same_device(ids) && ids.ndim()==3 && ids.dtype().code==kDLInt && ids.dtype().bits==64 && ids.IsContiguous());
  int rows=ids.size(0), heads=ids.size(1), blocks=qa.size(1);
  TVM_FFI_ICHECK(rows>=0 && rows<=262144 && heads>0);
  TVM_FFI_ICHECK(qa.size(0)==int64_t(rows)*heads && ids.size(2)==64 && blocks>64 && blocks<=4096);
  TVM_FFI_ICHECK(same_device(cis) && cis.ndim()==2 && cis.size(1)==heads && floating(cis));
  TVM_FFI_ICHECK(cis.size(0)>0 && cis.size(0)<=16383 && cis.stride(0)>=0 && cis.stride(1)>=0);
  if(has_pool) {
    TVM_FFI_ICHECK(same_device(pool) && pool.ndim()==2 && pool.size(1)==heads && floating(pool));
    TVM_FFI_ICHECK(pool.size(0)<=blocks && pool.stride(0)>=0 && pool.stride(1)>=0);
  }
  if(!contiguous) {
    TVM_FFI_ICHECK(same_device(positions) && positions.ndim()==1 && positions.size(0)==rows && positions.IsContiguous());
    TVM_FFI_ICHECK(positions.dtype().code==kDLInt && (positions.dtype().bits==32 || positions.dtype().bits==64));
  }
  if(write_mask) {
    TVM_FFI_ICHECK(same_device(valid) && valid.ndim()==3 && valid.IsContiguous());
    TVM_FFI_ICHECK(valid.size(0)==rows && valid.size(1)==heads && valid.size(2)==64 && valid.dtype().bits==8);
  }
  Params p{reinterpret_cast<uint16_t const*>(data(qa)),data(cis),has_pool?data(pool):nullptr,contiguous?nullptr:data(positions),
    reinterpret_cast<int64_t*>(data(ids)),write_mask?reinterpret_cast<bool*>(data(valid)):nullptr,
    rows,heads,blocks,int(cis.size(0)),has_pool?int(pool.size(0)):0,dtype(cis),has_pool?dtype(pool):0,
    query_start,cis.stride(0),cis.stride(1),has_pool?pool.stride(0):0,has_pool?pool.stride(1):0,
    qa.dtype().code==kDLBfloat,contiguous,positions.dtype().bits==64,finite};
  if(!rows) return;
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,qa.device().device_id));
#define LAUNCH(T,N,C) selection_kernel<T,N,C><<<rows*heads,T,0,stream>>>(p)
  // The at most two unstable pools are either future or inclusive local blocks,
  // so their raw CIS value cannot affect valid NOSA selection.
  bool cached_bf16=has_pool && pool.dtype().code==kDLBfloat &&
      qa.dtype().code==kDLBfloat && pool.size(0)>=blocks-2;
  int first=query_start/64, last=(query_start+rows-1)/64;
  bool shared=cached_bf16 && blocks<=1056 && contiguous && rows>=128 &&
     first>=64 && last-first<=16 && pool.size(0)>=first;
  TVM_FFI_ICHECK(!ranking_ready || shared);
  if(shared) {
    TVM_FFI_ICHECK(same_device(ranking) && ranking.IsContiguous() && ranking.ndim()==2 &&
        ranking.size(0)==heads && ranking.size(1)==64 && ranking.dtype().code==kDLInt && ranking.dtype().bits==32);
    auto table=reinterpret_cast<unsigned*>(data(ranking));
    if(!ranking_ready) prepare_prefix_ranking<<<heads,128,0,stream>>>(p,first,table);
    selection_prefix_kernel<<<rows*heads,32,0,stream>>>(p,table);
  } else if(cached_bf16 && blocks<=1056) {
    if(blocks<=128) { LAUNCH(32,4,true); }
    else if(blocks<=256) { LAUNCH(32,8,true); }
    else if(blocks<=512) { LAUNCH(32,16,true); }
    else { LAUNCH(32,33,true); }
  } else {
    if(blocks<=512) { LAUNCH(128,4,false); }
    else if(blocks<=1024) { LAUNCH(128,8,false); }
    else if(blocks<=2048) { LAUNCH(128,16,false); }
    else { LAUNCH(128,32,false); }
  }
#undef LAUNCH
  auto error=cudaGetLastError(); TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}
void select(TensorView qa, TensorView cis, TensorView pool, TensorView positions, TensorView ids,
            TensorView valid, TensorView ranking, int64_t query_start, bool has_pool, bool contiguous, bool write_mask) {
  select_impl(qa,cis,pool,positions,ids,valid,ranking,query_start,has_pool,contiguous,write_mask,false);
}
void select_ranked(TensorView qa, TensorView cis, TensorView pool, TensorView positions, TensorView ids,
                   TensorView valid, TensorView ranking, int64_t query_start, bool write_mask) {
  select_impl(qa,cis,pool,positions,ids,valid,ranking,query_start,true,true,write_mask,true);
}
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(select, nosa_selection::select);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(select_ranked, nosa_selection::select_ranked);

// A false flag must never enter the unguarded prepare_prefix_ranking helper.
// This endpoint always consumes a ranking prepared under the same flag.
#include "nosa_guarded_buffers.cuh"
namespace nosa_selection {
void select_ranked_guarded(TensorView qa, TensorView cis, TensorView pool,
                          TensorView positions, TensorView ids, TensorView valid,
                          TensorView ranking, TensorView finite, int64_t query_start) {
  auto flag = nosa_guarded_buffers::finite_pointer(finite, qa);
  TVM_FFI_ICHECK(valid.dtype().code == kDLBool && valid.dtype().bits == 8 &&
                valid.dtype().lanes == 1);
  nosa_guarded_buffers::disjoint({qa, cis, pool, ranking}, {ids, valid, finite});
  TVM_FFI_ICHECK(!nosa_guarded_buffers::overlaps(qa, ranking))
      << "prepared ranking must not overlap the score workspace";
  select_impl(qa, cis, pool, positions, ids, valid, ranking,
              query_start, true, true, true, true, flag);
}
}  // namespace nosa_selection
TVM_FFI_DLL_EXPORT_TYPED_FUNC(select_ranked_guarded, nosa_selection::select_ranked_guarded);
