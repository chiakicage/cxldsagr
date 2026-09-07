# Nsight Systems decode profile summary

Command:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o docs/model_decode_v32_nsys_b64_l8192 .venv/bin/python model_run/deepseek_v32_decode.py --batch-sizes 64 --history-lens 8192 --warmups 1 --iters 3 --post-proj deepgemm
```

Report: docs/model_decode_v32_nsys_b64_l8192.nsys-rep
SQLite export: docs/model_decode_v32_nsys_b64_l8192.sqlite

High-level totals:

- CUDA kernels: 68020 launches, 132.987 ms total GPU kernel time.
- CUDA memcpy: 9954 ops, 6.937 ms GPU memcpy time.
- CUDA memset: 24 ops, 0.032 ms GPU memset time.
- Runtime launch APIs: 68020 calls, 237.977 ms CPU API time, 3.499 us avg.

Top CUDA runtime APIs:

| API | Calls | Total ms | Avg us |
| --- | ---: | ---: | ---: |
| cudaLaunchKernel_v7000 | 67597 | 235.203 | 3.479 |
| cuLibraryLoadData | 26 | 136.892 | 5265.078 |
| cudaMemcpyAsync_v3020 | 9954 | 27.310 | 2.744 |
| cudaStreamSynchronize_v3020 | 263 | 11.929 | 45.358 |
| cudaFree_v3020 | 15 | 7.937 | 529.117 |
| cuKernelGetName | 67597 | 4.926 | 0.073 |
| cudaLaunchKernelExC_v11060 | 33 | 1.906 | 57.763 |
| cudaMalloc_v3020 | 26 | 1.837 | 70.644 |
| cuLibraryLoadFromFile | 22 | 1.218 | 55.356 |
| cuKernelGetFunction | 22 | 1.054 | 47.892 |
| cuLaunchKernelEx | 390 | 0.868 | 2.226 |
| cudaDeviceSynchronize_v3020 | 15 | 0.368 | 24.502 |
| cudaStreamIsCapturing_v10000 | 297 | 0.112 | 0.378 |
| cudaMemsetAsync_v3020 | 24 | 0.096 | 3.998 |
| cuTensorMapEncodeTiled | 578 | 0.092 | 0.159 |

Top CUDA kernels:

| Kernel | Instances | Total ms | Avg us |
| --- | ---: | ---: | ---: |
| void sparse_mla_decode_v2_kernel<(ModelType)0, (ComputeMode)0, (int)128, (bool)0>(const __nv_bfloat16 *, const unsign... | 11 | 18.210 | 1655.460 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<floa... | 3303 | 10.927 | 3.308 |
| void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::na... | 3351 | 8.062 | 2.406 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda()... | 3339 | 8.036 | 2.407 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorB... | 3710 | 7.271 | 1.960 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<float>, std::array<char *, (unsigned lo... | 3601 | 6.860 | 1.905 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<int, int, int, at::native::BitwiseAn... | 6580 | 4.657 | 0.708 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::ConvertToFloat8E4M3fnOp<float>, std::array<char *,... | 3088 | 4.019 | 1.302 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda()... | 3302 | 3.600 | 1.090 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda()... | 3290 | 3.581 | 1.089 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy... | 134 | 3.571 | 26.650 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AbsFunctor<c10::BFloat16>, std::array<char *, (uns... | 3303 | 3.362 | 1.018 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda()... | 526 | 3.207 | 6.096 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::B... | 13 | 3.106 | 238.950 |
| void at::native::<unnamed>::searchsorted_cuda_kernel<float, long>(T2 *, const T1 *, const T1 *, const long *, long, l... | 263 | 3.106 | 11.808 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::reciprocal_kernel_cuda(at::TensorIteratorBase &)::... | 3303 | 2.687 | 0.814 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::launch_clamp_scalar(at::TensorIteratorB... | 3290 | 2.533 | 0.770 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<int>, std::array<char *, (unsigned... | 3290 | 2.513 | 0.764 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunct... | 48 | 2.499 | 52.071 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, float, at::native::bin... | 3351 | 2.458 | 0.734 |

Interpretation:

- The trace is launch/API heavy: cudaLaunchKernel alone has 67,597 calls and about 235 ms of CPU API time in this captured process.
- Total GPU kernel time is much smaller than the observed wall/API activity. The dominant GPU kernel by total time is sparse MLA decode, but it is only about 18 ms across 11 instances in the whole capture.
- Most kernel instances are tiny PyTorch elementwise/reduce/copy kernels from activation quantization and packing loops, matching the torch profiler conclusion that the current Python-level decode path is launch/scheduling dominated rather than compute-kernel dominated.
