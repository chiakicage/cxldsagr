// Logical finite checks and transactional append-only NOSA compression.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nosa_prepare {
using tvm::ffi::TensorView;
constexpr int Threads=128, Elements=4096;

struct Scan {
  void const* data;
  int64_t elements, row_width, groups, dim, row_stride, head_stride, group_stride, feature_stride;
  int shift;
  bool inner_linear, dense;
};
struct Params {
  Scan q,k,cis;
  void const* keys;
  void const* raw_cis;
  void *ck,*cc,*pool;
  uint8_t* partial;
  bool* valid;
  int q_blocks,k_blocks,c_blocks,partials;
  int heads,dim,first,stop,old_c,new_c,old_s,new_s;
  int64_t ks0,ks1,cs0,cs1;
};

template<class T> __device__ __forceinline__ float to_float(T value) {
  if constexpr(std::is_same_v<T,__nv_bfloat16>) return __bfloat162float(value);
  else if constexpr(std::is_same_v<T,__half>) return __half2float(value);
  else return value;
}
template<class T> __device__ __forceinline__ T rounded(float value) {
  if constexpr(std::is_same_v<T,__nv_bfloat16>) return __float2bfloat16_rn(value);
  else if constexpr(std::is_same_v<T,__half>) return __float2half_rn(value);
  else return value;
}
template<class T> __device__ __forceinline__ bool finite(T value) {
  if constexpr(std::is_same_v<T,__nv_bfloat16>) return (__bfloat16_as_ushort(value)&0x7f80)!=0x7f80;
  else if constexpr(std::is_same_v<T,__half>) return (__half_as_ushort(value)&0x7c00)!=0x7c00;
  else return (__float_as_uint(value)&0x7f800000)!=0x7f800000;
}
template<class T> __device__ __forceinline__ bool finite_word(uint32_t value) {
  if constexpr(std::is_same_v<T,float>) return (value&0x7f800000)!=0x7f800000;
  else {
    constexpr uint32_t mask=std::is_same_v<T,__nv_bfloat16>?0x7f807f80:0x7c007c00;
    return __vcmpeq2(value&mask,mask)==0;
  }
}
__device__ __forceinline__ int64_t offset(Scan const& p,int64_t index) {
  if(p.dense) return index;
  int64_t row=p.shift>=0 ? index>>p.shift : index/p.row_width;
  int64_t within=index-row*p.row_width;
  if(p.inner_linear) return row*p.row_stride+within;
  int64_t feature=within%p.dim, group=within/p.dim%p.groups, head=within/(p.dim*p.groups);
  return row*p.row_stride+head*p.head_stride+group*p.group_stride+feature*p.feature_stride;
}
__device__ __forceinline__ bool block_all(bool value,int* scratch) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int all=__all_sync(0xffffffff,value);
  if(lane==0) scratch[warp]=all;
  __syncthreads();
  if(warp==0) {
    all=__reduce_and_sync(0xffffffff,lane<4?scratch[lane]:1);
    if(lane==0) scratch[4]=all;
  }
  __syncthreads();
  return scratch[4]!=0;
}

template<class T> __global__ void finite_partials(Params p) {
  __shared__ int scratch[5];
  int block=blockIdx.x,program,programs;
  Scan scan;
  if(block<p.q_blocks) {scan=p.q;program=block;programs=p.q_blocks;}
  else if(block<p.q_blocks+p.k_blocks) {scan=p.k;program=block-p.q_blocks;programs=p.k_blocks;}
  else {scan=p.cis;program=block-p.q_blocks-p.k_blocks;programs=p.c_blocks;}
  bool passed=true;
  constexpr int Vec=16/sizeof(T);
  bool vectorized=reinterpret_cast<uintptr_t>(scan.data)%16==0 &&
      (scan.dense || (scan.inner_linear && scan.row_width%Vec==0 && scan.row_stride%Vec==0));
  if(vectorized) {
    for(int64_t base=int64_t(program)*Elements;base<scan.elements;base+=int64_t(programs)*Elements) {
#pragma unroll
      for(int i=0;i<Elements/(Threads*Vec);++i) {
        int64_t index=base+threadIdx.x*Vec+i*Threads*Vec;
        if(index+Vec<=scan.elements) {
          auto address=static_cast<T const*>(scan.data)+offset(scan,index);
          uint4 value=*reinterpret_cast<uint4 const*>(address);
          passed &= finite_word<T>(value.x)&finite_word<T>(value.y)&finite_word<T>(value.z)&finite_word<T>(value.w);
        } else {
#pragma unroll
          for(int j=0;j<Vec;++j)
            if(index+j<scan.elements) passed &= finite(static_cast<T const*>(scan.data)[offset(scan,index+j)]);
        }
      }
    }
  } else {
    for(int64_t base=int64_t(program)*Elements;base<scan.elements;base+=int64_t(programs)*Elements) {
#pragma unroll
      for(int i=0;i<Elements/Threads;++i) {
        int64_t index=base+threadIdx.x+i*Threads;
        if(index<scan.elements) passed &= finite(static_cast<T const*>(scan.data)[offset(scan,index)]);
      }
    }
  }
  passed=block_all(passed,scratch);
  if(threadIdx.x==0) p.partial[block]=passed;
}

template<class T> __device__ __forceinline__ float cis_mean(Params const& p,int window,int head) {
  int lane=threadIdx.x%32;
  float value=to_float(static_cast<T const*>(p.raw_cis)[int64_t(window*16+lane)*p.cs0+head*p.cs1]);
#pragma unroll
  for(int delta=16;delta;delta/=2) value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,delta));
  return to_float(rounded<T>(value*(1.0f/32)));
}
__device__ __forceinline__ float maximum(float a,float b) {
  // The existing Triton append emits max.f32 (not max.NaN.f32), so a
  // NaN derived by finite-input FP32 overflow is ignored beside a number.
  return fmaxf(a,b);
}

template<class T,int Dim> __device__ __forceinline__ void key_mean(
    Params const& p,int window,int head,float* shared) {
  // Match Triton's 16-byte vectorized blocked layout, including its FP32
  // sequential local sums, within-warp butterfly, then cross-warp butterfly.
  constexpr int Vec=16/sizeof(T),Rows=Threads*Vec/Dim;
  constexpr int FeatureLanes=(Dim/Vec)<32?Dim/Vec:32;
  constexpr int WarpRows=32/FeatureLanes,WarpCols=Dim/(FeatureLanes*Vec);
  constexpr int RowWarps=4/WarpCols;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int row_warp=warp/WarpCols,feature=(warp%WarpCols*FeatureLanes+lane%FeatureLanes)*Vec;
  int row=row_warp*WarpRows+lane/FeatureLanes;
  union Pack {uint4 raw;T values[Vec];};
  Pack pack;
  pack.raw=*reinterpret_cast<uint4 const*>(static_cast<T const*>(p.keys)+
      int64_t(window*16+row)*p.ks0+head*p.ks1+feature);
  float values[Vec];
#pragma unroll
  for(int f=0;f<Vec;++f) values[f]=to_float(pack.values[f]);
#pragma unroll
  for(int more=Rows;more<32;more+=Rows) {
    pack.raw=*reinterpret_cast<uint4 const*>(static_cast<T const*>(p.keys)+
        int64_t(window*16+row+more)*p.ks0+head*p.ks1+feature);
#pragma unroll
    for(int f=0;f<Vec;++f) values[f]=__fadd_rn(values[f],to_float(pack.values[f]));
  }
#pragma unroll
  for(int delta=WarpRows/2;delta;delta/=2) {
#pragma unroll
    for(int f=0;f<Vec;++f)
      values[f]=__fadd_rn(values[f],__shfl_xor_sync(0xffffffff,values[f],delta*FeatureLanes));
  }
  if(lane<FeatureLanes) {
#pragma unroll
    for(int f=0;f<Vec;++f) shared[row_warp*Dim+feature+f]=values[f];
  }
  __syncthreads();
  if(row_warp==0 && lane<FeatureLanes) {
#pragma unroll
    for(int f=0;f<Vec;++f) {
      float sum;
      if constexpr(RowWarps==4)
        sum=__fadd_rn(__fadd_rn(shared[feature+f],shared[2*Dim+feature+f]),
                     __fadd_rn(shared[Dim+feature+f],shared[3*Dim+feature+f]));
      else sum=__fadd_rn(shared[feature+f],shared[Dim+feature+f]);
      pack.values[f]=rounded<T>(sum*(1.0f/32));
    }
    *reinterpret_cast<uint4*>(static_cast<T*>(p.ck)+(int64_t(window)*p.heads+head)*Dim+feature)=pack.raw;
  }
  __syncthreads();
}

template<class T,int Dim> __global__ void guarded_compression(Params p) {
  __shared__ int scratch[5];
  __shared__ float sums[4*Dim];
  bool passed=true;
  for(int i=threadIdx.x;i<p.partials;i+=Threads) passed &= p.partial[i]!=0;
  passed=block_all(passed,scratch);
  if(blockIdx.x==0 && threadIdx.x==0) *p.valid=passed;
  // The first kernel has completed before this kernel starts on the same
  // stream. Every CTA sees the complete validation result before any write.
  if(!passed || blockIdx.x>=(p.stop-p.first)*p.heads) return;
  int block=p.first+blockIdx.x/p.heads,head=blockIdx.x%p.heads;
  bool make_pool=block>=p.old_s && block<p.new_s;
  float pooled=-CUDART_INF_F;
#pragma unroll
  for(int within=0;within<4;++within) {
    int window=4*block+within;
    if(window>=p.old_c && window<p.new_c) {
      key_mean<T,Dim>(p,window,head,sums);
      if(threadIdx.x<32) {
        float value=cis_mean<T>(p,window,head);
        if(threadIdx.x==0) {
          static_cast<T*>(p.cc)[int64_t(window)*p.heads+head]=rounded<T>(value);
          pooled=maximum(pooled,value);
        }
      }
    } else if(make_pool && window<p.new_c && threadIdx.x==0) {
      pooled=maximum(pooled,to_float(static_cast<T const*>(p.cc)[int64_t(window)*p.heads+head]));
    }
  }
  if(make_pool) {
    int left=4*block-1;
    if(left>=p.old_c && threadIdx.x<32) {
      float value=cis_mean<T>(p,left,head);
      if(threadIdx.x==0) pooled=maximum(pooled,value);
    } else if(left>=0 && threadIdx.x==0) {
      pooled=maximum(pooled,to_float(static_cast<T const*>(p.cc)[int64_t(left)*p.heads+head]));
    }
    if(threadIdx.x==0) static_cast<T*>(p.pool)[int64_t(block)*p.heads+head]=rounded<T>(pooled);
  }
}

void* data(TensorView t) {return static_cast<char*>(t.data_ptr())+t.byte_offset();}
bool same_device(TensorView a,TensorView b) {
  return a.device().device_type==b.device().device_type && a.device().device_id==b.device().device_id;
}
Scan make_scan(TensorView t,int64_t skip=0) {
  int rank=t.ndim();
  int64_t dim=rank==2?1:t.size(rank-1),groups=rank==2?t.size(1):t.size(rank-2);
  int64_t heads=rank==4?t.size(1):1;
  int64_t ds=rank==2?0:t.stride(rank-1),gs=rank==2?t.stride(1):t.stride(rank-2),hs=rank==4?t.stride(1):0;
  int64_t width=heads*groups*dim;
  bool inner=(dim==1||ds==1)&&(groups==1||gs==dim)&&(heads==1||hs==groups*dim);
  int shift=(width>0 && (width&(width-1))==0)?__builtin_ctzll(width):-1;
  return {static_cast<char*>(data(t))+skip*t.stride(0)*(t.dtype().bits/8),(t.size(0)-skip)*width,width,groups,dim,t.stride(0),hs,gs,ds,shift,inner,inner&&t.stride(0)==width};
}

Params setup(TensorView q,TensorView k,TensorView cis,TensorView ck,TensorView cc,TensorView pool,
             TensorView partial,TensorView valid,int64_t validated_start,int64_t compressed_start,int64_t pooled_start) {
  TVM_FFI_ICHECK((q.ndim()==3||q.ndim()==4) && k.ndim()==3 && cis.ndim()==2);
  TVM_FFI_ICHECK(q.device().device_type==kDLCUDA && q.size(0)>0 && q.size(0)<=262144);
  int64_t length=k.size(0),heads=k.size(1),dim=k.size(2);
  TVM_FFI_ICHECK(length<=262144 && heads>0 && (dim==64||dim==128||dim==256) && k.stride(2)==1);
  TVM_FFI_ICHECK(cis.size(0)==length && cis.size(1)==heads);
  TVM_FFI_ICHECK(q.dtype()==k.dtype() && q.dtype().lanes==1 &&
      ((q.dtype().code==kDLBfloat && q.dtype().bits==16)||
       (q.dtype().code==kDLFloat && (q.dtype().bits==16||q.dtype().bits==32))));
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(data(k))%16==0 &&
      k.stride(0)%(128/k.dtype().bits)==0 && k.stride(1)%(128/k.dtype().bits)==0);
  for(auto tensor:{q,k,cis,ck,cc,pool}) {
    TVM_FFI_ICHECK(same_device(tensor,q) && tensor.dtype()==q.dtype());
    for(int d=0;d<tensor.ndim();++d) TVM_FFI_ICHECK(tensor.stride(d)>=0);
  }
  for(int d=1;d<q.ndim();++d) TVM_FFI_ICHECK(q.size(d)>0);
  int count=std::max<int64_t>(0,length/16-1),stable=std::max<int64_t>(0,(length-16)/64);
  TVM_FFI_ICHECK(validated_start>=0 && validated_start<=length && compressed_start>=0 && compressed_start<=count && pooled_start>=0 && pooled_start<=stable);
  TVM_FFI_ICHECK(ck.ndim()==3 && ck.size(0)>=count && ck.size(1)==heads && ck.size(2)==dim && ck.IsContiguous());
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(data(ck))%16==0);
  TVM_FFI_ICHECK(cc.ndim()==2 && cc.size(0)>=count && cc.size(1)==heads && cc.IsContiguous());
  TVM_FFI_ICHECK(pool.ndim()==2 && pool.size(0)>=stable && pool.size(1)==heads && pool.IsContiguous());
  Params p{};p.q=make_scan(q);p.k=make_scan(k,validated_start);p.cis=make_scan(cis,validated_start);
  p.q_blocks=std::min<int64_t>(1024,(p.q.elements+Elements-1)/Elements);
  p.k_blocks=std::min<int64_t>(1024,(p.k.elements+Elements-1)/Elements);
  p.c_blocks=std::min<int64_t>(1024,(p.cis.elements+Elements-1)/Elements);
  p.partials=p.q_blocks+p.k_blocks+p.c_blocks;
  TVM_FFI_ICHECK(same_device(partial,q) && partial.IsContiguous() && partial.ndim()==1 && partial.size(0)>=p.partials && partial.dtype().code==kDLUInt && partial.dtype().bits==8);
  TVM_FFI_ICHECK(same_device(valid,q) && valid.IsContiguous() && valid.ndim()==0 && valid.dtype().bits==8);
  p.keys=data(k);p.raw_cis=data(cis);p.ck=data(ck);p.cc=data(cc);p.pool=data(pool);
  p.partial=static_cast<uint8_t*>(data(partial));p.valid=static_cast<bool*>(data(valid));
  p.heads=heads;p.dim=dim;p.old_c=compressed_start;p.new_c=count;p.old_s=pooled_start;p.new_s=stable;
  p.first=std::min(int(compressed_start/4),int(pooled_start));
  p.stop=(compressed_start==count&&pooled_start==stable)?p.first:std::max((count+3)/4,stable);
  p.ks0=k.stride(0);p.ks1=k.stride(1);p.cs0=cis.stride(0);p.cs1=cis.stride(1);
  return p;
}

void prepare_out(TensorView q,TensorView k,TensorView cis,TensorView ck,TensorView cc,TensorView pool,
                 TensorView partial,TensorView valid,int64_t validated_start,int64_t compressed_start,int64_t pooled_start) {
  Params p=setup(q,k,cis,ck,cc,pool,partial,valid,validated_start,compressed_start,pooled_start);
  int dim=p.dim;
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,q.device().device_id));
#define COMPRESS(T,D) guarded_compression<T,D><<<std::max(1,(p.stop-p.first)*p.heads),Threads,0,stream>>>(p)
#define LAUNCH(T) finite_partials<T><<<p.partials,Threads,0,stream>>>(p); if(dim==64) {COMPRESS(T,64);} else if(dim==128) {COMPRESS(T,128);} else {COMPRESS(T,256);}
  if(q.dtype().code==kDLBfloat) {LAUNCH(__nv_bfloat16);}
  else if(q.dtype().bits==16) {LAUNCH(__half);}
  else {LAUNCH(float);}
#undef LAUNCH
#undef COMPRESS
  auto error=cudaGetLastError();TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}
}  // namespace nosa_prepare
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_out,nosa_prepare::prepare_out);
