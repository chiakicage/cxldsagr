// Overlap immutable-prefix CIS ranking with transactional compression.
#include "nosa_prepare.cu"

namespace nosa_prepare_ranked {
using namespace nosa_prepare;
struct RankedParams { Params prep; uint32_t* ranking; int prefix; };

__device__ __forceinline__ unsigned key16(unsigned bits) {
  if((bits&0x7fff)==0) bits=0;
  return bits&0x8000 ? bits^0xffff : bits^0x8000;
}
template<int Op> __device__ __forceinline__ unsigned reduce(unsigned value,unsigned* scratch) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  if constexpr(Op==0) value=__reduce_add_sync(0xffffffff,value);
  if constexpr(Op==1) value=__reduce_min_sync(0xffffffff,value);
  if constexpr(Op==2) value=__reduce_max_sync(0xffffffff,value);
  if(lane==0) scratch[warp]=value;
  __syncthreads();
  if(warp==0) {
    value=lane<4 ? scratch[lane] : (Op==1 ? 0xffffffff : 0);
    if constexpr(Op==0) value=__reduce_add_sync(0xffffffff,value);
    if constexpr(Op==1) value=__reduce_min_sync(0xffffffff,value);
    if constexpr(Op==2) value=__reduce_max_sync(0xffffffff,value);
    if(lane==0) scratch[4]=value;
  }
  __syncthreads();
  return scratch[4];
}

__device__ void rank_prefix(RankedParams const& r,int head,unsigned* scratch,float* pending) {
  auto const& p=r.prep;
  constexpr int Items=9;
  // At most one prefix pool is newly stable. Reconstruct it locally from
  // immutable compressed CIS plus raw suffix windows; never read another
  // compression CTA's uncommitted writes.
  if(p.old_s<r.prefix && threadIdx.x<32) {
    int block=r.prefix-1;
    float pooled=-CUDART_INF_F;
    for(int window=max(0,4*block-1);window<4*block+4;++window) {
      float value;
      if(window<p.old_c)
        value=to_float(static_cast<__nv_bfloat16 const*>(p.cc)[int64_t(window)*p.heads+head]);
      else value=cis_mean<__nv_bfloat16>(p,window,head);
      pooled=maximum(pooled,value);
    }
    if(threadIdx.x==0) *pending=pooled;
  }
  __syncthreads();
  unsigned keys[Items],low=65535,high=0;
#pragma unroll
  for(int i=0;i<Items;++i) {
    int b=threadIdx.x+i*Threads;
    unsigned key=0;
    if(b<r.prefix) {
      unsigned bits=b<p.old_s ? static_cast<uint16_t const*>(p.pool)[int64_t(b)*p.heads+head]
          : __bfloat16_as_ushort(__float2bfloat16_rn(*pending));
      key=key16(bits);low=min(low,key);high=max(high,key);
    }
    keys[i]=key;
  }
  low=reduce<1>(low,scratch);high=reduce<2>(high,scratch);
  // Search the full ordered-bit domain, including derived +/-inf and NaN
  // payloads. This is exactly the selector's stable unsigned key ordering.
  while(low<high) {
    unsigned middle=(low+high+1)/2,count=0;
#pragma unroll
    for(int i=0;i<Items;++i) count+=int(threadIdx.x)+i*Threads<r.prefix && keys[i]>=middle;
    count=reduce<0>(count,scratch);
    if(count>=64) low=middle;else high=middle-1;
  }
  unsigned cut=low,lane=threadIdx.x%32,warp=threadIdx.x/32;
  __syncthreads();
#pragma unroll
  for(int i=0;i<Items;++i) {
    bool valid=int(threadIdx.x)+i*Threads<r.prefix;
    unsigned greater=__ballot_sync(0xffffffff,valid&&keys[i]>cut);
    unsigned equal=__ballot_sync(0xffffffff,valid&&keys[i]==cut);
    if(lane==0) scratch[4*i+warp]=__popc(greater)|(__popc(equal)<<16);
  }
  __syncthreads();
  if(threadIdx.x==0) {
    unsigned greater=0;
#pragma unroll
    for(int i=0;i<Items*4;++i) greater+=scratch[i]&65535;
    scratch[Items*4]=greater;
    unsigned eq_before=0,selected_before=0;
#pragma unroll
    for(int i=0;i<Items*4;++i) {
      unsigned packed=scratch[i],eq=packed>>16;
      scratch[i]=eq_before|(selected_before<<16);
      unsigned take=min(eq,64-greater-min(64-greater,eq_before));
      eq_before+=eq;selected_before+=(packed&65535)+take;
    }
  }
  __syncthreads();
  unsigned greater=scratch[Items*4];
#pragma unroll
  for(int i=0;i<Items;++i) {
    int b=threadIdx.x+i*Threads;
    unsigned packed=scratch[4*i+warp];
    bool eq=b<r.prefix&&keys[i]==cut;
    unsigned equals=__ballot_sync(0xffffffff,eq);
    unsigned rank=(packed&65535)+__popc(equals&((1u<<lane)-1));
    bool chosen=b<r.prefix&&(keys[i]>cut || (eq&&rank<64-greater));
    unsigned chosen_mask=__ballot_sync(0xffffffff,chosen);
    rank=(packed>>16)+__popc(chosen_mask&((1u<<lane)-1));
    if(chosen) r.ranking[head*64+rank]=(keys[i]<<12)|(4095-b);
  }
}

__global__ void guarded_compression_and_ranking(RankedParams r) {
  auto const& p=r.prep;
  __shared__ int flags[5];
  union Shared {float sums[4*128];struct {unsigned scratch[41];float pending;} ranking;};
  __shared__ Shared shared;
  bool passed=true;
  for(int i=threadIdx.x;i<p.partials;i+=Threads) passed &= p.partial[i]!=0;
  passed=block_all(passed,flags);
  if(blockIdx.x==0 && threadIdx.x==0) *p.valid=passed;
  if(!passed) return;
  int compression_ctas=(p.stop-p.first)*p.heads;
  if(blockIdx.x>=compression_ctas) {
    rank_prefix(r,blockIdx.x-compression_ctas,shared.ranking.scratch,&shared.ranking.pending);
    return;
  }
  int block=p.first+blockIdx.x/p.heads,head=blockIdx.x%p.heads;
  bool make_pool=block>=p.old_s && block<p.new_s;
  float pooled=-CUDART_INF_F;
#pragma unroll
  for(int within=0;within<4;++within) {
    int window=4*block+within;
    if(window>=p.old_c && window<p.new_c) {
      key_mean<__nv_bfloat16,128>(p,window,head,shared.sums);
      if(threadIdx.x<32) {
        float value=cis_mean<__nv_bfloat16>(p,window,head);
        if(threadIdx.x==0) {
          static_cast<__nv_bfloat16*>(p.cc)[int64_t(window)*p.heads+head]=rounded<__nv_bfloat16>(value);
          pooled=maximum(pooled,value);
        }
      }
    } else if(make_pool && window<p.new_c && threadIdx.x==0) {
      pooled=maximum(pooled,to_float(static_cast<__nv_bfloat16 const*>(p.cc)[int64_t(window)*p.heads+head]));
    }
  }
  if(make_pool) {
    int left=4*block-1;
    if(left>=p.old_c && threadIdx.x<32) {
      float value=cis_mean<__nv_bfloat16>(p,left,head);
      if(threadIdx.x==0) pooled=maximum(pooled,value);
    } else if(left>=0 && threadIdx.x==0) {
      pooled=maximum(pooled,to_float(static_cast<__nv_bfloat16 const*>(p.cc)[int64_t(left)*p.heads+head]));
    }
    if(threadIdx.x==0) static_cast<__nv_bfloat16*>(p.pool)[int64_t(block)*p.heads+head]=rounded<__nv_bfloat16>(pooled);
  }
}

void prepare_ranked_out(TensorView q,TensorView k,TensorView cis,TensorView ck,TensorView cc,TensorView pool,
                        TensorView ranking,TensorView partial,TensorView valid,int64_t validated_start,
                        int64_t compressed_start,int64_t pooled_start,int64_t query_start) {
  Params p=setup(q,k,cis,ck,cc,pool,partial,valid,validated_start,compressed_start,pooled_start);
  int first=query_start/64,last=(query_start+q.size(0)-1)/64,blocks=(k.size(0)+63)/64;
  TVM_FFI_ICHECK(q.dtype().code==kDLBfloat && p.dim==128 && q.size(0)>=128 &&
      query_start==validated_start && query_start+q.size(0)<=k.size(0) &&
      first>=64 && last-first<=16 && blocks<=1056 && p.new_s>=std::max(first,blocks-2) &&
      first-p.old_s<=1);
  TVM_FFI_ICHECK(same_device(ranking,q) && ranking.ndim()==2 && ranking.size(0)==p.heads &&
      ranking.size(1)==64 && ranking.IsContiguous() && ranking.dtype().code==kDLInt && ranking.dtype().bits==32 &&
      reinterpret_cast<uintptr_t>(data(ranking))%4==0);
  RankedParams r{p,static_cast<uint32_t*>(data(ranking)),first};
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,q.device().device_id));
  finite_partials<__nv_bfloat16><<<p.partials,Threads,0,stream>>>(p);
  guarded_compression_and_ranking<<<(p.stop-p.first)*p.heads+p.heads,Threads,0,stream>>>(r);
  auto error=cudaGetLastError();TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}
}  // namespace nosa_prepare_ranked
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_ranked_out,nosa_prepare_ranked::prepare_ranked_out);
