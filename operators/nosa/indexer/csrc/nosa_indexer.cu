// Submit score and stable selection from one native entry point.
#include "nosa_scores.cu"
#include "nosa_selection.cu"
namespace nosa_indexer {
void select_impl(tvm::ffi::TensorView q, tvm::ffi::TensorView k,
            tvm::ffi::TensorView cis, tvm::ffi::TensorView pool,
            tvm::ffi::TensorView workspace, tvm::ffi::TensorView normalizers,
            tvm::ffi::TensorView ids, tvm::ffi::TensorView valid, tvm::ffi::TensorView ranking, int64_t query_start, bool ranking_ready, bool const* finite=nullptr) {
  int64_t blocks = workspace.size(1);
  int64_t rows=q.size(0),first=query_start/64,last=(query_start+rows-1)/64;
  bool fused=q.dtype().code==kDLBfloat && pool.dtype().code==kDLBfloat &&
      rows>=1024 && rows<=1088 && ((rows+15)/16)*q.size(1)>=128 &&
      blocks<=1056 && first>=64 && last-first<=16 && pool.size(0)>=blocks-2 && pool.size(0)>=first;
  if(fused) {
    if(!ranking_ready) nosa_selection::prepare_ranking(pool,ranking,first);
    if(fused_select_impl(q,k,ranking,ids,valid,query_start,blocks,finite)) return;
  }
  nosa_scores::scores_impl(q, k, q, workspace, normalizers, query_start, blocks, true, true, finite);
  nosa_selection::select_impl(workspace, cis, pool, q, ids, valid, ranking, query_start, true, true, true, ranking_ready, finite);
}
void select(tvm::ffi::TensorView q, tvm::ffi::TensorView k,
            tvm::ffi::TensorView cis, tvm::ffi::TensorView pool,
            tvm::ffi::TensorView workspace, tvm::ffi::TensorView normalizers,
            tvm::ffi::TensorView ids, tvm::ffi::TensorView valid, tvm::ffi::TensorView ranking, int64_t query_start) {
  select_impl(q,k,cis,pool,workspace,normalizers,ids,valid,ranking,query_start,false);
}
void select_ranked(tvm::ffi::TensorView q, tvm::ffi::TensorView k,
                   tvm::ffi::TensorView cis, tvm::ffi::TensorView pool,
                   tvm::ffi::TensorView workspace, tvm::ffi::TensorView normalizers,
                   tvm::ffi::TensorView ids, tvm::ffi::TensorView valid, tvm::ffi::TensorView ranking, int64_t query_start) {
  select_impl(q,k,cis,pool,workspace,normalizers,ids,valid,ranking,query_start,true);
}
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(indexer_out, nosa_indexer::select);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(indexer_ranked_out, nosa_indexer::select_ranked);
