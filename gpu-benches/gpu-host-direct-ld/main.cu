#include "../gpu-error.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <string>
#include <sys/mman.h>
#include <vector>

using namespace std;

struct Result {
  double bestMs;
  double medianMs;
  double bestGBs;
  double medianGBs;
};

static Result summarize(vector<float> timesMs, size_t bytes) {
  sort(timesMs.begin(), timesMs.end());
  double bestMs = timesMs.front();
  double medianMs = timesMs[timesMs.size() / 2];
  return {bestMs, medianMs, (double)bytes / (bestMs * 1.0e-3) / 1.0e9,
          (double)bytes / (medianMs * 1.0e-3) / 1.0e9};
}

static string humanBytes(size_t bytes) {
  if (bytes % (1024ull * 1024 * 1024) == 0)
    return to_string(bytes / (1024ull * 1024 * 1024)) + " GiB";
  if (bytes % (1024ull * 1024) == 0)
    return to_string(bytes / (1024ull * 1024)) + " MiB";
  return to_string(bytes) + " B";
}

static void printPcieInfo(int cudaDevice) {
  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDeviceProperties(&prop, cudaDevice));
  cout << "Device: " << prop.name << ", SM " << prop.major << prop.minor
       << ", SMs=" << prop.multiProcessorCount << "\n";
  cout << "PCI bus id: " << prop.pciDomainID << ":" << prop.pciBusID << ":"
       << prop.pciDeviceID << "\n";

  nvmlReturn_t st = nvmlInit_v2();
  if (st != NVML_SUCCESS) {
    cout << "NVML unavailable: " << nvmlErrorString(st) << "\n";
    return;
  }
  nvmlDevice_t dev;
  st = nvmlDeviceGetHandleByIndex_v2(cudaDevice, &dev);
  if (st == NVML_SUCCESS) {
    unsigned int currGen = 0, maxGen = 0, currWidth = 0, maxWidth = 0;
    nvmlDeviceGetCurrPcieLinkGeneration(dev, &currGen);
    nvmlDeviceGetMaxPcieLinkGeneration(dev, &maxGen);
    nvmlDeviceGetCurrPcieLinkWidth(dev, &currWidth);
    nvmlDeviceGetMaxPcieLinkWidth(dev, &maxWidth);
    cout << "PCIe link: current Gen" << currGen << " x" << currWidth
         << ", max Gen" << maxGen << " x" << maxWidth << "\n";
  }
  nvmlShutdown();
}

static void printSmapsInfo(const string &name, const void *ptr) {
  ifstream smaps("/proc/self/smaps");
  if (!smaps)
    return;

  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  string line;
  bool inMapping = false;
  cout << "  " << name << " mapping info:\n";
  while (getline(smaps, line)) {
    uintptr_t lo = 0, hi = 0;
    if (sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2) {
      inMapping = (addr >= lo && addr < hi);
      if (inMapping)
        cout << "    " << line << "\n";
      continue;
    }
    if (inMapping &&
        (line.rfind("KernelPageSize:", 0) == 0 ||
         line.rfind("MMUPageSize:", 0) == 0 ||
         line.rfind("AnonHugePages:", 0) == 0 ||
         line.rfind("VmFlags:", 0) == 0)) {
      cout << "    " << line << "\n";
    }
  }
}

static void fillHost(uint4 *data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    data[i] = make_uint4((uint32_t)i, (uint32_t)(i >> 32), 0x12345678u,
                         0x9abcdef0u);
  }
}

static uint4 *allocHugeHost(size_t bytes) {
  void *ptr = nullptr;
  int rc = posix_memalign(&ptr, 2ull * 1024 * 1024, bytes);
  if (rc != 0)
    return nullptr;
  madvise(ptr, bytes, MADV_HUGEPAGE);
  return static_cast<uint4 *>(ptr);
}

__device__ __forceinline__ uint4 ld_global_v4_u32(const uint4 *ptr) {
  uint4 v;
  uint64_t addr = __cvta_generic_to_global(ptr);
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_global_nc_v4_u32(const uint4 *ptr) {
  uint4 v;
  uint64_t addr = __cvta_generic_to_global(ptr);
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_global_cg_v4_u32(const uint4 *ptr) {
  uint4 v;
  uint64_t addr = __cvta_generic_to_global(ptr);
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_global_ca_v4_u32(const uint4 *ptr) {
  uint4 v;
  uint64_t addr = __cvta_generic_to_global(ptr);
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_global_cv_v4_u32(const uint4 *ptr) {
  uint4 v;
  uint64_t addr = __cvta_generic_to_global(ptr);
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

__device__ __forceinline__ void cp_async_bulk_prefetch_l2(const void *src,
                                                          uint32_t bytes) {
#if __CUDA_ARCH__ >= 900
  uint64_t gmem = __cvta_generic_to_global(src);
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(gmem),
               "r"(bytes)
               : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
#if __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}

template <int N> __device__ __forceinline__ void cp_async_bulk_wait_group_read() {
#if __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, uint32_t count) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t *mbar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar,
                                                          uint32_t txBytes) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("{\n .reg .b64 state;\n"
               " mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
               "}\n" ::"r"(addr),
               "r"(txBytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar,
                                                     uint32_t phase) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n"
                 " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(addr), "r"(phase));
  }
}

__device__ __forceinline__ void cp_async_bulk_g2s(void *smemDst,
                                                  const void *gmemSrc,
                                                  uint32_t bytes,
                                                  uint64_t *mbar) {
  uint32_t dstAddr = static_cast<uint32_t>(__cvta_generic_to_shared(smemDst));
  uint32_t mbarAddr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
               " [%0], [%1], %2, [%3];\n" ::"r"(dstAddr),
               "l"(gmemSrc), "r"(bytes), "r"(mbarAddr));
}

__device__ __forceinline__ void cp_async_4B(void *smemDst,
                                            const void *gmemSrc) {
  uint32_t dstAddr = static_cast<uint32_t>(__cvta_generic_to_shared(smemDst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dstAddr),
               "l"(gmemSrc));
}

__device__ __forceinline__ void cp_async_8B(void *smemDst,
                                            const void *gmemSrc) {
  uint32_t dstAddr = static_cast<uint32_t>(__cvta_generic_to_shared(smemDst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dstAddr),
               "l"(gmemSrc));
}

__device__ __forceinline__ void cp_async_16B(void *smemDst,
                                             const void *gmemSrc) {
  uint32_t dstAddr = static_cast<uint32_t>(__cvta_generic_to_shared(smemDst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dstAddr),
               "l"(gmemSrc));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <int Mode>
__device__ __forceinline__ uint4 loadMode(const uint4 *ptr) {
  if constexpr (Mode == 0) {
    return *ptr;
  } else if constexpr (Mode == 1) {
    return ld_global_v4_u32(ptr);
  } else if constexpr (Mode == 2) {
    return ld_global_nc_v4_u32(ptr);
  } else if constexpr (Mode == 3) {
    return ld_global_cg_v4_u32(ptr);
  } else if constexpr (Mode == 4) {
    return ld_global_ca_v4_u32(ptr);
  } else {
    return ld_global_cv_v4_u32(ptr);
  }
}

template <int Mode, int Unroll>
__global__ void directLdKernel(const uint4 *__restrict__ src,
                               unsigned long long *__restrict__ partial,
                               size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t warpContiguousStride = (size_t)blockDim.x * gridDim.x;
  size_t loopStride = warpContiguousStride * Unroll;
  unsigned long long acc = 0;

  for (size_t base = tid; base < n; base += loopStride) {
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      size_t i = base + (size_t)u * warpContiguousStride;
      if (i < n) {
        uint4 v = loadMode<Mode>(src + i);
        acc += (unsigned long long)v.x + v.y + v.z + v.w;
      }
    }
  }

  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename T>
__device__ __forceinline__ unsigned long long sumValue(T v) {
  return (unsigned long long)v;
}

template <>
__device__ __forceinline__ unsigned long long sumValue<uint2>(uint2 v) {
  return (unsigned long long)v.x + v.y;
}

template <>
__device__ __forceinline__ unsigned long long sumValue<uint4>(uint4 v) {
  return (unsigned long long)v.x + v.y + v.z + v.w;
}

template <>
__device__ __forceinline__ unsigned long long sumValue<ulonglong4>(
    ulonglong4 v) {
  return v.x + v.y + v.z + v.w;
}

template <typename T>
__global__ void directLdWidthKernel(const T *__restrict__ src,
                                    unsigned long long *__restrict__ partial,
                                    size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  unsigned long long acc = 0;
  for (size_t i = tid; i < n; i += stride) {
    acc += sumValue<T>(src[i]);
  }
  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename T, int ILP>
__global__ void directLdWidthIlpKernel(const T *__restrict__ src,
                                       unsigned long long *__restrict__ partial,
                                       size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t warpContiguousStride = (size_t)blockDim.x * gridDim.x;
  size_t loopStride = warpContiguousStride * ILP;
  unsigned long long acc[ILP];
#pragma unroll
  for (int u = 0; u < ILP; ++u) {
    acc[u] = 0;
  }

  for (size_t base = tid; base < n; base += loopStride) {
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      size_t i = base + (size_t)u * warpContiguousStride;
      if (i < n) {
        acc[u] += sumValue<T>(src[i]);
      }
    }
  }

  unsigned long long total = 0;
#pragma unroll
  for (int u = 0; u < ILP; ++u) {
    total += acc[u];
  }
  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = total;
}

template <int Mode, int PrefetchMode>
__global__ void directLdPrefetchKernel(const uint4 *__restrict__ src,
                                       unsigned long long *__restrict__ partial,
                                       size_t n, int aheadDistance) {
  size_t ctaBase = (size_t)blockIdx.x * blockDim.x;
  size_t ctaStride = (size_t)blockDim.x * gridDim.x;
  unsigned long long acc = 0;

  for (size_t base = ctaBase; base < n; base += ctaStride) {
    if constexpr (PrefetchMode == 1) {
      if (threadIdx.x == 0) {
        size_t remaining = n - base;
        uint32_t vecs = (uint32_t)(remaining < (size_t)blockDim.x
                                       ? remaining
                                       : (size_t)blockDim.x);
        cp_async_bulk_prefetch_l2(src + base, vecs * sizeof(uint4));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
      }
      __syncthreads();
    } else if constexpr (PrefetchMode == 2) {
      size_t pfBase = base + (size_t)aheadDistance * ctaStride;
      if (threadIdx.x == 0 && pfBase < n) {
        size_t remaining = n - pfBase;
        uint32_t vecs = (uint32_t)(remaining < (size_t)blockDim.x
                                       ? remaining
                                       : (size_t)blockDim.x);
        cp_async_bulk_prefetch_l2(src + pfBase, vecs * sizeof(uint4));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<4>();
      }
    }

    size_t i = base + threadIdx.x;
    if (i < n) {
      uint4 v = loadMode<Mode>(src + i);
      acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
  }

  if constexpr (PrefetchMode == 2) {
    if (threadIdx.x == 0) {
      cp_async_bulk_wait_group_read<0>();
    }
  }
  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int Mode>
__global__ void directCopyKernel(const uint4 *__restrict__ src,
                                 uint4 *__restrict__ dst, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    dst[i] = loadMode<Mode>(src + i);
  }
}

template <int TileVecs>
__global__ void cpAsyncBulkReadKernel(const uint4 *__restrict__ src,
                                      unsigned long long *__restrict__ partial,
                                      size_t n) {
  extern __shared__ __align__(16) unsigned char smemRaw[];
  uint4 *tile = reinterpret_cast<uint4 *>(smemRaw);
  uint64_t *mbar =
      reinterpret_cast<uint64_t *>(smemRaw + TileVecs * sizeof(uint4));

  if (threadIdx.x == 0) {
    mbarrier_init(mbar, 1);
  }
  __syncthreads();

  unsigned long long acc = 0;
  uint32_t phase = 0;
  size_t tiles = (n + TileVecs - 1) / TileVecs;
  for (size_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    size_t start = t * (size_t)TileVecs;
    size_t remaining = n - start;
    uint32_t vecs =
        (uint32_t)(remaining < (size_t)TileVecs ? remaining : (size_t)TileVecs);
    uint32_t bytes = vecs * sizeof(uint4);

    if (threadIdx.x == 0) {
      mbarrier_arrive_expect_tx(mbar, bytes);
      cp_async_bulk_g2s(tile, src + start, bytes, mbar);
      mbarrier_wait_parity(mbar, phase);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < vecs; i += blockDim.x) {
      uint4 v = tile[i];
      acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    __syncthreads();
    phase ^= 1;
  }

  if (threadIdx.x == 0) {
    mbarrier_inval(mbar);
  }
  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename T, int TileBytes>
__global__ void cpAsyncReadKernel(const uint4 *__restrict__ src,
                                  unsigned long long *__restrict__ partial,
                                  size_t bytes) {
  extern __shared__ __align__(16) unsigned char smemRaw[];
  const char *srcBytes = reinterpret_cast<const char *>(src);
  T *tile = reinterpret_cast<T *>(smemRaw);
  unsigned long long acc = 0;

  constexpr int TileElems = TileBytes / sizeof(T);
  size_t n = bytes / sizeof(T);
  size_t tiles = (n + TileElems - 1) / TileElems;
  for (size_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    size_t start = t * (size_t)TileElems;
    size_t remaining = n - start;
    uint32_t elems = (uint32_t)(remaining < (size_t)TileElems
                                    ? remaining
                                    : (size_t)TileElems);

    for (uint32_t i = threadIdx.x; i < elems; i += blockDim.x) {
      const void *gmem = srcBytes + (start + i) * sizeof(T);
      if constexpr (sizeof(T) == 4) {
        cp_async_4B(tile + i, gmem);
      } else if constexpr (sizeof(T) == 8) {
        cp_async_8B(tile + i, gmem);
      } else {
        cp_async_16B(tile + i, gmem);
      }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < elems; i += blockDim.x) {
      acc += sumValue<T>(tile[i]);
    }
    __syncthreads();
  }

  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int Mode, int Unroll>
static Result benchDirectLd(const uint4 *mapped, size_t bytes, int reps,
                            int grid, int block) {
  size_t n = bytes / sizeof(uint4);
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, (size_t)grid * block *
                                     sizeof(unsigned long long)));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    directLdKernel<Mode, Unroll><<<grid, block>>>(mapped, partial, n);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));
  return summarize(times, bytes);
}

template <int TileVecs>
static Result benchCpAsyncBulkRead(const uint4 *mapped, size_t bytes, int reps,
                                   int grid, int block) {
  size_t n = bytes / sizeof(uint4);
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, (size_t)grid * block *
                                     sizeof(unsigned long long)));

  size_t smemBytes = TileVecs * sizeof(uint4) + 16;
  GPU_ERROR(cudaFuncSetAttribute(
      cpAsyncBulkReadKernel<TileVecs>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smemBytes));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    cpAsyncBulkReadKernel<TileVecs>
        <<<grid, block, smemBytes>>>(mapped, partial, n);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));
  return summarize(times, bytes);
}

template <typename T, int TileBytes>
static Result benchCpAsyncRead(const uint4 *mapped, size_t bytes, int reps,
                               int grid, int block) {
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, (size_t)grid * block *
                                     sizeof(unsigned long long)));

  size_t smemBytes = TileBytes;
  GPU_ERROR(cudaFuncSetAttribute(cpAsyncReadKernel<T, TileBytes>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smemBytes));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    cpAsyncReadKernel<T, TileBytes><<<grid, block, smemBytes>>>(
        mapped, partial, bytes);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));
  return summarize(times, bytes);
}

template <typename T>
static Result benchDirectLdWidth(const void *mapped, size_t bytes, int reps,
                                 int grid, int block) {
  size_t n = bytes / sizeof(T);
  const T *typed = static_cast<const T *>(mapped);
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, (size_t)grid * block *
                                     sizeof(unsigned long long)));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    directLdWidthKernel<T><<<grid, block>>>(typed, partial, n);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));
  return summarize(times, n * sizeof(T));
}

template <typename T, int ILP>
static Result benchDirectLdWidthIlp(const void *mapped, size_t bytes, int reps,
                                    int grid, int block) {
  size_t n = bytes / sizeof(T);
  const T *typed = static_cast<const T *>(mapped);
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, (size_t)grid * block *
                                     sizeof(unsigned long long)));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    directLdWidthIlpKernel<T, ILP><<<grid, block>>>(typed, partial, n);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));
  return summarize(times, n * sizeof(T));
}

template <int Mode, int PrefetchMode>
static Result benchDirectLdPrefetch(const uint4 *mapped, size_t bytes, int reps,
                                    int grid, int block, int aheadDistance) {
  size_t n = bytes / sizeof(uint4);
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, (size_t)grid * block *
                                     sizeof(unsigned long long)));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    directLdPrefetchKernel<Mode, PrefetchMode>
        <<<grid, block>>>(mapped, partial, n, aheadDistance);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));
  return summarize(times, bytes);
}

template <int Mode>
static Result benchDirectCopy(const uint4 *mapped, uint4 *deviceDst,
                              size_t bytes, int reps, int grid, int block) {
  size_t n = bytes / sizeof(uint4);

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    directCopyKernel<Mode><<<grid, block>>>(mapped, deviceDst, n);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  return summarize(times, bytes);
}

static void printResult(const string &name, const Result &r) {
  cout << left << setw(18) << name << right << setw(11) << fixed
       << setprecision(3) << r.bestMs << setw(11) << r.medianMs << setw(13)
       << setprecision(2) << r.bestGBs << setw(13) << r.medianGBs << "\n";
}

static void runLoads(const string &label, const uint4 *mapped, uint4 *deviceDst,
                     size_t bytes, int reps, int grid, int block) {
  cout << "\n[" << label << "]\n";
  cout << left << setw(18) << "kernel" << right << setw(11) << "best ms"
       << setw(11) << "med ms" << setw(13) << "best GB/s" << setw(13)
       << "med GB/s" << "\n";
  cout << string(66, '-') << "\n";
  printResult("c++ uint4 u1",
              benchDirectLd<0, 1>(mapped, bytes, reps, grid, block));
  printResult("width u32",
              benchDirectLdWidth<uint32_t>(mapped, bytes, reps, grid, block));
  printResult("width uint2",
              benchDirectLdWidth<uint2>(mapped, bytes, reps, grid, block));
  printResult("uint2 ilp2",
              benchDirectLdWidthIlp<uint2, 2>(mapped, bytes, reps, grid,
                                              block));
  printResult("uint2 ilp4",
              benchDirectLdWidthIlp<uint2, 4>(mapped, bytes, reps, grid,
                                              block));
  printResult("uint2 ilp8",
              benchDirectLdWidthIlp<uint2, 8>(mapped, bytes, reps, grid,
                                              block));
  printResult("width uint4",
              benchDirectLdWidth<uint4>(mapped, bytes, reps, grid, block));
  printResult("width u64x4",
              benchDirectLdWidth<ulonglong4>(mapped, bytes, reps, grid, block));
  printResult("c++ uint4 u4",
              benchDirectLd<0, 4>(mapped, bytes, reps, grid, block));
  printResult("ptx ld.v4 u1",
              benchDirectLd<1, 1>(mapped, bytes, reps, grid, block));
  printResult("ptx ld.nc u1",
              benchDirectLd<2, 1>(mapped, bytes, reps, grid, block));
  printResult("ptx ld.cg u1",
              benchDirectLd<3, 1>(mapped, bytes, reps, grid, block));
  printResult("ptx ld.ca u1",
              benchDirectLd<4, 1>(mapped, bytes, reps, grid, block));
  printResult("ptx ld.cv u1",
              benchDirectLd<5, 1>(mapped, bytes, reps, grid, block));
  printResult("ptx ld.v4 u4",
              benchDirectLd<1, 4>(mapped, bytes, reps, grid, block));
  printResult("prefetch wait",
              benchDirectLdPrefetch<1, 1>(mapped, bytes, reps, grid, block,
                                           0));
  printResult("prefetch ahead",
              benchDirectLdPrefetch<1, 2>(mapped, bytes, reps, grid, block,
                                           4));
  printResult("bulk 4KiB",
              benchCpAsyncBulkRead<256>(mapped, bytes, reps, grid, block));
  printResult("bulk 16KiB",
              benchCpAsyncBulkRead<1024>(mapped, bytes, reps, grid, block));
  printResult("bulk 64KiB",
              benchCpAsyncBulkRead<4096>(mapped, bytes, reps, grid, block));
  printResult("cp.async16 4K",
              benchCpAsyncRead<uint4, 4 * 1024>(mapped, bytes, reps, grid,
                                                block));
  printResult("cp.async4 16K",
              benchCpAsyncRead<uint32_t, 16 * 1024>(mapped, bytes, reps, grid,
                                                    block));
  printResult("cp.async8 16K",
              benchCpAsyncRead<uint64_t, 16 * 1024>(mapped, bytes, reps, grid,
                                                    block));
  printResult("cp.async16 16K",
              benchCpAsyncRead<uint4, 16 * 1024>(mapped, bytes, reps, grid,
                                                 block));
  printResult("cp.async4 64K",
              benchCpAsyncRead<uint32_t, 64 * 1024>(mapped, bytes, reps, grid,
                                                    block));
  printResult("cp.async8 64K",
              benchCpAsyncRead<uint64_t, 64 * 1024>(mapped, bytes, reps, grid,
                                                    block));
  printResult("cp.async16 64K",
              benchCpAsyncRead<uint4, 64 * 1024>(mapped, bytes, reps, grid,
                                                 block));
}

int main(int argc, char **argv) {
  size_t bytes = 1024ull * 1024 * 1024;
  int reps = 9;
  int grid = 8192;
  int block = 256;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--bytes" && i + 1 < argc) {
      bytes = stoull(argv[++i]);
    } else if (arg == "--reps" && i + 1 < argc) {
      reps = stoi(argv[++i]);
    } else if (arg == "--grid" && i + 1 < argc) {
      grid = stoi(argv[++i]);
    } else if (arg == "--block" && i + 1 < argc) {
      block = stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      cout << "Usage: " << argv[0]
           << " [--bytes N] [--reps N] [--grid N] [--block N]\n";
      return 0;
    } else {
      cerr << "Unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (bytes < 1024 || bytes % sizeof(uint4) != 0) {
    cerr << "--bytes must be >= 1024 and divisible by 16\n";
    return 2;
  }

  GPU_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  int dev = 0;
  GPU_ERROR(cudaGetDevice(&dev));
  printPcieInfo(dev);
  cout << "Direct GPU ld.global from CPU pinned mapped host memory\n";
  cout << "buffer=" << humanBytes(bytes) << ", reps=" << reps
       << ", grid=" << grid << ", block=" << block << "\n";

  uint4 *deviceDst = nullptr;
  GPU_ERROR(cudaMalloc(&deviceDst, bytes));

  uint4 *hostAlloc = nullptr;
  uint4 *mappedHostAlloc = nullptr;
  GPU_ERROR(cudaHostAlloc(&hostAlloc, bytes, cudaHostAllocMapped));
  fillHost(hostAlloc, bytes / sizeof(uint4));
  GPU_ERROR(cudaHostGetDevicePointer(&mappedHostAlloc, hostAlloc, 0));
  runLoads("cudaHostAllocMapped, default 4KiB pages", mappedHostAlloc,
           deviceDst, bytes, reps, grid, block);

  uint4 *hugeHost = allocHugeHost(bytes);
  if (!hugeHost) {
    cerr << "huge host allocation failed\n";
    GPU_ERROR(cudaFreeHost(hostAlloc));
    return 1;
  }
  fillHost(hugeHost, bytes / sizeof(uint4));
  printSmapsInfo("hugeHost before cudaHostRegister", hugeHost);
  GPU_ERROR(cudaHostRegister(hugeHost, bytes, cudaHostRegisterMapped));
  printSmapsInfo("hugeHost after cudaHostRegister", hugeHost);
  uint4 *mappedHuge = nullptr;
  GPU_ERROR(cudaHostGetDevicePointer(&mappedHuge, hugeHost, 0));
  runLoads("THP-backed cudaHostRegisterMapped", mappedHuge, deviceDst, bytes,
           reps, grid, block);

  GPU_ERROR(cudaHostUnregister(hugeHost));
  free(hugeHost);
  GPU_ERROR(cudaFreeHost(hostAlloc));
  GPU_ERROR(cudaFree(deviceDst));
  return 0;
}
