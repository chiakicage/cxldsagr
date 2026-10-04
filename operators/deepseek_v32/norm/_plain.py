"""Typed-I/O specialization of FlashInfer's plain RMSNorm kernel.

Derived from flashinfer/norm/kernels/rmsnorm.py, copyright (c) 2025 FlashInfer
team, licensed under Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0).
Only the kernel body is specialized. Constructor and launch are inherited.
"""

import cutlass
from cutlass import Float32, Int64, cute
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import predicate_k, row_reduce_sum_multirow

from operators.deepseek_v32.norm._layout import check_layout


class LocalPlainRMSNorm(RMSNormKernel):
    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        idX = cute.make_identity_tensor(mX.shape)
        idY = cute.make_identity_tensor(mY.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        cY = cute.local_tile(idY, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # Each atom transfers the same four logical values as the Float32
        # baseline. Physical width belongs to the operand, not the constructor.
        copy_atom_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=self.vec_size * mX.element_type.width,
        )
        copy_atom_w = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=self.vec_size * mW.element_type.width,
        )
        copy_atom_y = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=self.vec_size * mY.element_type.width,
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=self.vec_size * mX.element_type.width,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
            canonical_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=self.copy_bits
            )
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_x, tv_layout, tiler_mn)
            canonical_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=self.copy_bits
            )

        tiled_copy_W = cute.make_tiled_copy(copy_atom_w, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_y, tv_layout, tiler_mn)
        canonical_copy = cute.make_tiled_copy(canonical_atom, tv_layout, tiler_mn)
        # Equality of the full static TV maps establishes logical coordinates
        # for every thread/value; no retile or reduction reordering is hidden.
        check_layout(
            "input_source_tv",
            tiled_copy_load.layout_src_tv_tiled,
            canonical_copy.layout_src_tv_tiled,
        )
        check_layout(
            "weight_source_tv", tiled_copy_W.layout_src_tv_tiled, canonical_copy.layout_src_tv_tiled
        )
        check_layout(
            "output_destination_tv",
            tiled_copy_store.layout_dst_tv_tiled,
            canonical_copy.layout_dst_tv_tiled,
        )

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)
        thr_canonical = canonical_copy.get_slice(tidx)
        canonical_coords = thr_canonical.partition_S(cX)
        canonical_fragment = cute.make_fragment_like(canonical_coords, Float32)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)
        check_layout("input_register", tXrX.layout, canonical_fragment.layout)
        check_layout("weight_register", tXrW.layout, canonical_fragment.layout)
        check_layout("output_register", tXrO.layout, canonical_fragment.layout)
        check_layout("input_coordinates", tXcX.layout, canonical_coords.layout)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        tYpY = predicate_k(thr_copy_O.partition_D(cY), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.copy(copy_atom_w, tWgW, tWrW, pred=tWpW)
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_x, tXgX, tXrX, pred=tXpX)
            cute.copy(copy_atom_w, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)
        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mY.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_y, tXrO, tXgO, pred=tYpY)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()
