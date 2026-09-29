#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace nosa_selection_cutoff {
__device__ __forceinline__ unsigned key16(unsigned bits) {
  if ((bits & 0x7fff) == 0) bits = 0;
  return bits & 0x8000 ? bits ^ 0xffff : bits ^ 0x8000;
}
// Exact v4 ordered-BF16 cutoff for rank33. Each warp owns one QA row.
// Preconditions: all 32 lanes participate, lane==threadIdx.x%32,
// 33<=blocks<=1056, qa_row contains the visible BF16 bits (shared or global).
// Uncomputed values are -inf; mandatory +inf values remain in the row.
// This returns only the threshold key: equal keys still require stable-ID
// handling downstream, so tile pruning may discard only strictly lower keys.
__device__ __forceinline__ unsigned top33_cutoff_bf16(
    uint16_t const* qa_row, int blocks, unsigned lane) {
  unsigned packed[17];
  unsigned ninf=0,nfinite=0,nan_count=0,low=65535,high=0;
#pragma unroll
  for(int i=0;i<17;++i) {
    int a=lane+i*64, b=a+32;
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
  return cut;
}

// The row is at the four-seed boundary: only seed-owned 32-block regions,
// their next-block halo, and mandatory blocks can differ from initialized
// -infinity. Enumerate every such logical block once; padding contributes
// enough -infinities to preserve rank33 even for sparse/exceptional rows.
__device__ __forceinline__ unsigned top33_seed_cutoff_bf16(
    uint16_t const* qa_row, int blocks, int count, int qblock,
    int const* seed_tiles, int seeds, unsigned lane) {
  // At most 150 distinct supported blocks leaves >=42 actual -infinities
  // for blocks>=192. Their multiplicity beyond rank33 is irrelevant, even
  // when negative NaN payload keys sort below -infinity.
  if(seeds!=4 || blocks<192 || blocks>1056 || count<=0)
    return top33_cutoff_bf16(qa_row,blocks,lane);
  unsigned long long seed_mask=0;
  bool valid=true;
#pragma unroll
  for(int i=0;i<4;++i) {
    int tile=seed_tiles[i];
    if(tile<0 || tile>=(blocks+31)/32) valid=false;
    else {
      unsigned long long bit=1ull<<tile;
      valid &= !(seed_mask&bit);
      seed_mask|=bit;
    }
  }
  if(!valid) return top33_cutoff_bf16(qa_row,blocks,lane);
  unsigned packed[3];
#pragma unroll
  for(int i=0;i<2;++i) {
    int a=seed_tiles[2*i]*32+lane,b=seed_tiles[2*i+1]*32+lane;
    unsigned lo=a<blocks ? key16(qa_row[a]) : 127;
    unsigned hi=b<blocks ? key16(qa_row[b]) : 127;
    packed[i]=lo|(hi<<16);
  }
  unsigned extra=127;
  if(lane<4) {
    int tile=seed_tiles[lane],b=(tile+1)*32;
    if(b<blocks && tile*128+127<count && !(seed_mask&(1ull<<(tile+1))))
      extra=key16(qa_row[b]);
  } else if(lane==4) {
    if(!(seed_mask&1ull)) extra=key16(qa_row[0]);
  } else if(lane<22) {
    int b=qblock-16+int(lane)-5;
    if(b>0 && b<blocks) {
      int tile=b/32;
      bool body=seed_mask&(1ull<<tile);
      bool halo=b%32==0 && tile>0 && (seed_mask&(1ull<<(tile-1))) &&
          (tile-1)*128+127<count;
      if(!body && !halo) extra=key16(qa_row[b]);
    }
  }
  packed[2]=extra|(127u<<16);
  unsigned ninf=0,nfinite=0,nan_count=0,low=65535,high=0;
#pragma unroll
  for(int i=0;i<3;++i) {
    unsigned lo=packed[i]&65535,hi=packed[i]>>16;
    ninf+=(lo==0xff80)+(hi==0xff80);
    bool flo=lo>127 && lo<0xff80,fhi=hi>127 && hi<0xff80;
    nfinite+=flo+fhi;
    nan_count+=(lo<127 || lo>0xff80)+(hi<127 || hi>0xff80);
    if(flo) {low=min(low,lo);high=max(high,lo);}
    if(fhi) {low=min(low,hi);high=max(high,hi);}
  }
  ninf=__reduce_add_sync(0xffffffff,ninf);
  nfinite=__reduce_add_sync(0xffffffff,nfinite);
  nan_count=__reduce_add_sync(0xffffffff,nan_count);
  low=__reduce_min_sync(0xffffffff,low);
  high=__reduce_max_sync(0xffffffff,high);
  if(!nan_count && ninf>=33) return 0xff80;
  if(!nan_count && ninf+nfinite<33) return 127;
  if(nan_count) {low=0;high=65535;}
  while(low<high) {
    unsigned middle=(low+high+1)/2,cmp=middle|(middle<<16),count_ge=0;
#pragma unroll
    for(int i=0;i<3;++i) {
      unsigned flags=__vsetgeu2(packed[i],cmp);
      count_ge+=(flags&65535)+(flags>>16);
    }
    count_ge=__reduce_add_sync(0xffffffff,count_ge);
    if(count_ge>=33) low=middle;
    else high=middle-1;
  }
  return low;
}
}  // namespace nosa_selection_cutoff
