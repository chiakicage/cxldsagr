#include "../MeasurementSeries.hpp"
#include "../gpu-error.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static constexpr int kBlockThreads = 256;
static constexpr int kPrefetchWaitN = 4;

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

__global__ void initKernel(uint4 *data, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    data[i] = make_uint4((uint32_t)i, (uint32_t)(i >> 32), 0x12345678u,
                         0x9abcdef0u);
  }
}

template <int Mode>
__global__ void streamKernel(const uint4 *__restrict__ data,
                             unsigned long long *__restrict__ partial,
                             size_t vecs, size_t tileVecs,
                             size_t prefetchDistanceTiles) {
  const size_t tiles = (vecs + tileVecs - 1) / tileVecs;
  unsigned long long acc = 0;

  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const size_t start = tile * tileVecs;
    const size_t remaining = vecs - start;
    const size_t curVecs = remaining < tileVecs ? remaining : tileVecs;

    if constexpr (Mode == 1) {
      const size_t pfTile = tile + prefetchDistanceTiles * (size_t)gridDim.x;
      if (threadIdx.x == 0 && pfTile < tiles) {
        const size_t pfStart = pfTile * tileVecs;
        const size_t pfRemaining = vecs - pfStart;
        const size_t pfVecs = pfRemaining < tileVecs ? pfRemaining : tileVecs;
        cp_async_bulk_prefetch_l2(data + pfStart,
                                  (uint32_t)(pfVecs * sizeof(uint4)));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<kPrefetchWaitN>();
      }
    } else if constexpr (Mode == 2) {
      if (threadIdx.x == 0) {
        cp_async_bulk_prefetch_l2(data + start,
                                  (uint32_t)(curVecs * sizeof(uint4)));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
      }
      __syncthreads();
    }

    for (size_t i = threadIdx.x; i < curVecs; i += blockDim.x) {
      uint4 v = data[start + i];
      acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
  }

  if constexpr (Mode == 1) {
    if (threadIdx.x == 0) {
      cp_async_bulk_wait_group_read<0>();
    }
  }

  partial[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int Mode>
__global__ void timedTileReadKernel(const uint4 *__restrict__ data,
                                    unsigned long long *__restrict__ out,
                                    size_t vecs, size_t tileVecs,
                                    int trials) {
  extern __shared__ unsigned long long smem[];
  unsigned long long cycles = 0;
  unsigned long long acc = 0;
  const size_t tiles = (vecs + tileVecs - 1) / tileVecs;

  for (int t = 0; t < trials; ++t) {
    const size_t tile = (size_t)t % tiles;
    const size_t start = tile * tileVecs;
    const size_t remaining = vecs - start;
    const size_t curVecs = remaining < tileVecs ? remaining : tileVecs;

    if constexpr (Mode == 1) {
      if (threadIdx.x == 0) {
        cp_async_bulk_prefetch_l2(data + start,
                                  (uint32_t)(curVecs * sizeof(uint4)));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
      }
    }
    __syncthreads();

    unsigned long long begin = clock64();
    for (size_t i = threadIdx.x; i < curVecs; i += blockDim.x) {
      uint4 v = data[start + i];
      acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    unsigned long long end = clock64();
    cycles += end - begin;
    __syncthreads();
  }

  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[0] = smem[0];
    out[1] = cycles / (unsigned long long)trials;
  }
}

struct StreamResult {
  double ms;
  double gbps;
};

static string mbString(size_t bytes) {
  return to_string(bytes / 1024 / 1024) + " MiB";
}

template <int Mode>
static StreamResult measureStream(const uint4 *ptr, size_t bytes,
                                  size_t tileBytes, size_t prefetchDistance,
                                  int grid, int reps) {
  const size_t vecs = bytes / sizeof(uint4);
  const size_t tileVecs = tileBytes / sizeof(uint4);

  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial,
                       (size_t)grid * kBlockThreads *
                           sizeof(unsigned long long)));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int r = 0; r < reps + 5; ++r) {
    GPU_ERROR(cudaEventRecord(start));
    streamKernel<Mode><<<grid, kBlockThreads>>>(ptr, partial, vecs, tileVecs,
                                                prefetchDistance);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());

    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (r >= 5) {
      times.push_back(ms);
    }
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(partial));

  sort(times.begin(), times.end());
  const double ms = times[times.size() / 2];
  return {ms, (double)bytes / (ms * 1.0e-3) / 1.0e9};
}

template <int Mode>
static double measureTimedTileCycles(const uint4 *ptr, size_t bytes,
                                     size_t tileBytes, int trials) {
  const size_t vecs = bytes / sizeof(uint4);
  const size_t tileVecs = tileBytes / sizeof(uint4);

  unsigned long long *out = nullptr;
  GPU_ERROR(cudaMalloc(&out, 2 * sizeof(unsigned long long)));
  timedTileReadKernel<Mode><<<1, kBlockThreads,
                              kBlockThreads * sizeof(unsigned long long)>>>(
      ptr, out, vecs, tileVecs, trials);
  GPU_ERROR(cudaGetLastError());

  unsigned long long host[2] = {};
  GPU_ERROR(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  GPU_ERROR(cudaFree(out));
  return (double)host[1];
}

static void fillHost(uint4 *data, size_t vecs) {
  for (size_t i = 0; i < vecs; ++i) {
    data[i] = make_uint4((uint32_t)i, (uint32_t)(i >> 32), 0x12345678u,
                         0x9abcdef0u);
  }
}

int main(int argc, char **argv) {
  size_t bytes = 512ull * 1024 * 1024;
  size_t tileBytes = 16ull * 1024;
  size_t prefetchDistance = 2;
  int reps = 15;
  int trials = 1024;
  int grid = 0;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--bytes" && i + 1 < argc) {
      bytes = stoull(argv[++i]);
    } else if (arg == "--tile-bytes" && i + 1 < argc) {
      tileBytes = stoull(argv[++i]);
    } else if (arg == "--prefetch-distance" && i + 1 < argc) {
      prefetchDistance = stoull(argv[++i]);
    } else if (arg == "--grid" && i + 1 < argc) {
      grid = stoi(argv[++i]);
    } else if (arg == "--reps" && i + 1 < argc) {
      reps = stoi(argv[++i]);
    } else if (arg == "--trials" && i + 1 < argc) {
      trials = stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      cout << "Usage: " << argv[0]
           << " [--bytes N] [--tile-bytes N] [--prefetch-distance N]"
           << " [--grid N] [--reps N] [--trials N]\n";
      return 0;
    } else {
      cerr << "Unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (bytes < tileBytes || bytes % sizeof(uint4) != 0) {
    cerr << "--bytes must be >= --tile-bytes and divisible by 16\n";
    return 2;
  }
  if (tileBytes < 16 || tileBytes % 16 != 0 ||
      tileBytes % sizeof(uint4) != 0) {
    cerr << "--tile-bytes must be a multiple of 16\n";
    return 2;
  }

  GPU_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDeviceProperties(&prop, 0));
  if (grid == 0) {
    grid = prop.multiProcessorCount * 4;
  }

  cout << "Device: " << prop.name << ", SM " << prop.major << prop.minor
       << ", SMs=" << prop.multiProcessorCount << "\n";
  cout << "Benchmark: cp.async.bulk.prefetch.L2.global before normal loads\n";
  cout << "buffer=" << mbString(bytes) << " tile=" << tileBytes
       << " B grid=" << grid << " block=" << kBlockThreads
       << " prefetch-distance=" << prefetchDistance << " reps=" << reps
       << " timed-trials=" << trials << "\n\n";

  uint4 *hostData = nullptr;
  uint4 *mappedData = nullptr;
  GPU_ERROR(cudaHostAlloc(&hostData, bytes, cudaHostAllocMapped));
  fillHost(hostData, bytes / sizeof(uint4));
  GPU_ERROR(cudaHostGetDevicePointer(&mappedData, hostData, 0));

  uint4 *deviceData = nullptr;
  GPU_ERROR(cudaMalloc(&deviceData, bytes));
  initKernel<<<grid, kBlockThreads>>>(deviceData, bytes / sizeof(uint4));
  GPU_ERROR(cudaDeviceSynchronize());

  auto runOne = [&](const char *label, const uint4 *ptr) {
    cout << label << "\n";
    auto base = measureStream<0>(ptr, bytes, tileBytes, prefetchDistance, grid,
                                 reps);
    auto ahead = measureStream<1>(ptr, bytes, tileBytes, prefetchDistance, grid,
                                  reps);
    auto wait = measureStream<2>(ptr, bytes, tileBytes, prefetchDistance, grid,
                                 reps);

    cout << fixed << setprecision(3);
    cout << "  stream no-prefetch       " << setw(8) << base.gbps
         << " GB/s  " << base.ms << " ms\n";
    cout << "  stream prefetch-ahead    " << setw(8) << ahead.gbps
         << " GB/s  " << ahead.ms << " ms\n";
    cout << "  stream prefetch+wait     " << setw(8) << wait.gbps
         << " GB/s  " << wait.ms << " ms\n";

    double coldCycles = measureTimedTileCycles<0>(ptr, bytes, tileBytes, trials);
    double prefetchedCycles =
        measureTimedTileCycles<1>(ptr, bytes, tileBytes, trials);
    cout << "  timed tile read cycles   no-prefetch=" << coldCycles
         << " after-prefetch+wait=" << prefetchedCycles << "\n\n";
  };

  runOne("mapped pinned host memory", mappedData);
  runOne("device memory control", deviceData);

  GPU_ERROR(cudaFree(deviceData));
  GPU_ERROR(cudaFreeHost(hostData));
  return 0;
}
