#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Extracted from accepted selector v4. One full converged consumer warp owns
// qa_row[blocks], promotions[33], and promoted_ids[64]. All scratch regions are
// disjoint across warps. The caller must __syncwarp() after storing qa_row.
// qa_row already contains BF16-rounded pooling plus mandatory/causal masks.
// ranking_head[64] is the packed CIS prefix ranking from before this score call.
// Only the existing v4 BF16/contiguous-query/prefix-coverage guard permits use.
// Valid rows return exactly 64 ascending logical IDs and true validity bits.
namespace nosa_selection_fused {
__device__ __forceinline__ unsigned key16(unsigned bits) {
  if ((bits & 0x7fff) == 0) bits = 0;
  return bits & 0x8000 ? bits ^ 0xffff : bits ^ 0x8000;
}
template<int K, int J, bool Descending>
__device__ __forceinline__ void sort128_stage(unsigned (&values)[4]) {
  unsigned next[4];
#pragma unroll
  for(int i=0;i<4;++i) {
    unsigned other;
    if constexpr(J<32) other=__shfl_xor_sync(0xffffffff,values[i],J);
    else other=values[i^(J/32)];
    unsigned index=(threadIdx.x%32)+32*i;
    bool lower=((index&K)==0)==((index&J)==0);
    if constexpr(Descending) lower=!lower;
    next[i]=lower ? min(values[i],other) : max(values[i],other);
  }
#pragma unroll
  for(int i=0;i<4;++i) values[i]=next[i];
  if constexpr(J>1) sort128_stage<K,J/2,Descending>(values);
  else if constexpr(K<128) sort128_stage<K*2,K,Descending>(values);
}

template<int K, int J, bool Descending>
__device__ __forceinline__ void sort64_stage(unsigned (&values)[2]) {
  unsigned next[2];
#pragma unroll
  for(int i=0;i<2;++i) {
    unsigned other;
    if constexpr(J<32) other=__shfl_xor_sync(0xffffffff,values[i],J);
    else other=values[i^(J/32)];
    unsigned index=(threadIdx.x%32)+32*i;
    bool lower=((index&K)==0)==((index&J)==0);
    if constexpr(Descending) lower=!lower;
    next[i]=lower ? min(values[i],other) : max(values[i],other);
  }
#pragma unroll
  for(int i=0;i<2;++i) values[i]=next[i];
  if constexpr(J>1) sort64_stage<K,J/2,Descending>(values);
  else if constexpr(K<64) sort64_stage<K*2,K,Descending>(values);
}

__device__ __forceinline__ unsigned prepare_promotions_packed(uint16_t const* qa_row, int blocks, int qblock,
                                            unsigned* promotions, unsigned* promoted_ids, unsigned minimum_key=0) {
  unsigned packed[17];
  unsigned ninf=0,nfinite=0,nan_count=0,low=65535,high=0;
#pragma unroll
  for(int i=0;i<17;++i) {
    int a=threadIdx.x%32+i*64, b=a+32;
    unsigned lo=a<blocks ? key16(qa_row[a]) : 0;
    unsigned hi=b<blocks ? key16(qa_row[b]) : 0;
    packed[i]=lo|(hi<<16);
    ninf+=(a<blocks && lo==0xff80)+(b<blocks && hi==0xff80);
    bool flo=a<blocks && lo>127 && lo<0xff80;
    bool fhi=b<blocks && hi>127 && hi<0xff80;
    nfinite+=flo+fhi;
    nan_count+=(a<blocks && (lo<127 || lo>0xff80))+(b<blocks && (hi<127 || hi>0xff80));
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
    else low=max(low,minimum_key);
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
  unsigned tie_offset=0,promoted_count=0,lane=threadIdx.x%32;
#pragma unroll
  for(int i=0;i<33;++i) {
    int b=lane+i*32;
    unsigned key=(i%2 ? packed[i/2]>>16 : packed[i/2]&65535);
    bool equal=b<blocks && key==cut;
    unsigned mask=__ballot_sync(0xffffffff,equal);
    unsigned rank=tie_offset+__popc(mask&((1u<<lane)-1));
    tie_offset+=__popc(mask);
    bool qa=b<blocks && (key>cut || (equal && rank<33-greater));
    bool promoted=b<blocks && b<=qblock && (qa || b==0 || qblock<=b+16);
    mask=__ballot_sync(0xffffffff,promoted);
    if(lane==0) promotions[i]=mask;
    rank=promoted_count+__popc(mask&((1u<<lane)-1));
    if(promoted) promoted_ids[rank]=b;
    promoted_count+=__popc(mask);
  }
  return promoted_count;
}

__device__ __forceinline__ void select_prefix_row(
    uint16_t const* qa_row, int blocks, int qblock, unsigned const* ranking_head,
    int64_t* ids_row, bool* valid_row, unsigned* promotions, unsigned* promoted_ids, unsigned minimum_key=0) {
  unsigned lane=threadIdx.x%32;
  unsigned promoted_count=prepare_promotions_packed(qa_row,blocks,qblock,promotions,promoted_ids,minimum_key);
  __syncwarp();
  unsigned candidates[4];
#pragma unroll
  for(int i=0;i<2;++i) {
    unsigned packed=ranking_head[lane+i*32];
    unsigned b=4095-(packed&4095);
    bool duplicate=(promotions[b/32]>>(b%32))&1u;
    candidates[i]=duplicate ? 0 : packed;
    unsigned slot=lane+i*32;
    candidates[i+2]=slot<promoted_count ? (0xff80u<<12)|(4095-promoted_ids[slot]) : 0;
  }
  sort128_stage<2,1,true>(candidates);
  unsigned selected[2];
#pragma unroll
  for(int i=0;i<2;++i) selected[i]=4095-(candidates[i]&4095);
  sort64_stage<2,1,false>(selected);
#pragma unroll
  for(int i=0;i<2;++i) {
    int offset=lane+i*32;
    ids_row[offset]=selected[i];
    if(valid_row) valid_row[offset]=true;
  }
}

} // namespace nosa_selection_fused
