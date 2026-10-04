// Checked, request-owned ranked preparation and selection on one CUDA stream.
#include "nosa_indexer.cu"
#include "nosa_prepare_ranked.cu"

namespace nosa_indexer_checked {
using tvm::ffi::TensorView;
using nosa_prepare::data;
using nosa_prepare::same_device;

void shape(TensorView value, TensorView q, std::initializer_list<int64_t> dimensions,
           int code, int bits) {
  TVM_FFI_ICHECK(same_device(value,q) && value.IsContiguous() &&
      value.ndim()==dimensions.size() && value.dtype().code==code && value.dtype().bits==bits);
  int axis=0;
  for(auto size:dimensions) TVM_FFI_ICHECK(value.size(axis++)==size);
}

bool overlaps(TensorView left, TensorView right) {
  if(!same_device(left,right)) return false;
  auto begin=[](TensorView value){return reinterpret_cast<uintptr_t>(data(value));};
  auto bytes=[](TensorView value){
    int64_t span=1;
    for(int d=0;d<value.ndim();++d) {
      if(value.size(d)==0) return int64_t(0);
      span+=(value.size(d)-1)*value.stride(d);
    }
    return span*value.dtype().bits/8;
  };
  auto l=begin(left),r=begin(right);
  auto lb=bytes(left),rb=bytes(right);
  return lb>0 && rb>0 && l<r+rb && r<l+lb;
}

template<bool Checked>
bool run(TensorView q, TensorView k, TensorView cis,
         TensorView ck, TensorView cc, TensorView pool,
         TensorView workspace, TensorView normalizers, TensorView ids, TensorView valid,
         TensorView ranking, TensorView partial, TensorView finite, TensorView host_finite,
         int64_t validated_start, int64_t compressed_start, int64_t pooled_start,
         int64_t query_start) {
  // All geometry checks precede the first launch. The regular async APIs keep
  // their existing fallback/capture behavior; this endpoint is explicitly checked.
  auto p=nosa_prepare::setup(q,k,cis,ck,cc,pool,partial,finite,
                            validated_start,compressed_start,pooled_start);
  int64_t rows=q.size(0),heads=k.size(1),blocks=(k.size(0)+63)/64;
  TVM_FFI_ICHECK(q.ndim()==4 && q.size(1)==heads && q.size(2)==16 && q.size(3)==128 &&
      q.dtype().code==kDLBfloat && q.dtype().bits==16 && q.stride(3)==1 && rows>=128 &&
      p.new_c>=2047 && blocks<=1056);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(data(q))%16==0);
  for(int d=0;d<3;++d) TVM_FFI_ICHECK(q.stride(d)>0 && q.stride(d)%8==0);
  TVM_FFI_ICHECK(ck.size(0)==p.new_c && cc.size(0)==p.new_c && pool.size(0)==p.new_s);
  int64_t first=query_start/64,last=(query_start+rows-1)/64;
  TVM_FFI_ICHECK(query_start==validated_start && query_start+rows==k.size(0) &&
      first>=64 && last-first<=16 && p.new_s>=std::max(first,blocks-2) && first-p.old_s<=1);
  shape(workspace,q,{rows*heads,blocks},kDLBfloat,16);
  shape(normalizers,q,{1,rows,heads,16,2},kDLFloat,32);
  shape(ids,q,{rows,heads,64},kDLInt,64);
  TVM_FFI_ICHECK(same_device(valid,q) && valid.IsContiguous() && valid.ndim()==3 &&
      valid.size(0)==rows && valid.size(1)==heads && valid.size(2)==64 && valid.dtype().bits==8);
  shape(ranking,q,{heads,64},kDLInt,32);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(data(ranking))%4==0);
  TVM_FFI_ICHECK(host_finite.device().device_type==kDLCPU && host_finite.ndim()==0 &&
      host_finite.dtype().bits==8 && host_finite.IsContiguous());
  // Preparation writes derived records and ranking before selection consumes
  // them. Every writable buffer must be disjoint from inputs and other outputs,
  // including workspace: a short-row score launch otherwise corrupts ranking.
  const std::initializer_list<TensorView> outputs{
      ck,cc,pool,workspace,normalizers,ids,valid,ranking,partial,finite};
  for(auto left=outputs.begin();left!=outputs.end();++left) {
    for(auto input:{q,k,cis}) {
      TVM_FFI_ICHECK(!overlaps(*left,input))
          <<"writable indexer buffers must not overlap inputs or other outputs";
    }
    for(auto right=left+1;right!=outputs.end();++right) {
      TVM_FFI_ICHECK(!overlaps(*left,*right))
          <<"writable indexer buffers must not overlap inputs or other outputs";
    }
  }
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA,q.device().device_id));
  cudaStreamCaptureStatus capture;
  auto error=cudaStreamIsCapturing(stream,&capture);
  TVM_FFI_ICHECK(error==cudaSuccess && capture==cudaStreamCaptureStatusNone)
      <<"checked ranked indexer cannot be CUDA Graph captured";
  nosa_prepare_ranked::prepare_ranked_out(q,k,cis,ck,cc,pool,ranking,partial,finite,
      validated_start,compressed_start,pooled_start,query_start);
  if constexpr(Checked) {
    error=cudaMemcpyAsync(data(host_finite),data(finite),1,cudaMemcpyDeviceToHost,stream);
    TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
    error=cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
    if(!*static_cast<bool*>(data(host_finite))) return false;
  }
  // The asynchronous path gates every score/selection kernel on the device
  // flag. Failed preparation emits only safe empty logical selections. The
  // model checks all layer flags before it commits or returns an output.
  nosa_indexer::select_impl(q,ck,cc,pool,workspace,normalizers,ids,valid,ranking,query_start,true,
      Checked ? nullptr : static_cast<bool const*>(data(finite)));
  return true;
}
} // namespace nosa_indexer_checked
TVM_FFI_DLL_EXPORT_TYPED_FUNC(checked_ranked_indexer_out,nosa_indexer_checked::run<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(async_ranked_indexer_out,nosa_indexer_checked::run<false>);
