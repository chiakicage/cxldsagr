#include "../MeasurementSeries.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

__global__ void initKernel(uint4 *data, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
    data[i] = make_uint4((uint32_t)i, (uint32_t)(i >> 32), 0x12345678u,
                         0x9abcdef0u);
  }
}

__global__ void streamReadKernel(const uint4 *__restrict__ data,
                                 unsigned long long *__restrict__ sink,
                                 size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  unsigned long long acc = 0;
  for (size_t i = tid; i < n; i += stride) {
    uint4 v = data[i];
    acc += (unsigned long long)v.x + v.y + v.z + v.w;
  }
  atomicAdd(sink, acc);
}

__global__ void pointerChaseKernel(const uint64_t *__restrict__ ptrs,
                                   unsigned long long *__restrict__ sink,
                                   size_t iters) {
  const uint64_t *p = ptrs;
  const int unroll = 32;
#pragma unroll 1
  for (size_t i = 0; i < iters; i += unroll) {
#pragma unroll
    for (int u = 0; u < unroll; ++u) {
      p = reinterpret_cast<const uint64_t *>(*p);
    }
  }
  if (threadIdx.x == 0) {
    sink[0] = reinterpret_cast<unsigned long long>(p);
  }
}

struct StreamResult {
  double ms;
  double gbps;
};

static StreamResult measureStream(const uint4 *dptr, size_t bytes, int reps) {
  const size_t n = bytes / sizeof(uint4);
  unsigned long long *sink = nullptr;
  GPU_ERROR(cudaMalloc(&sink, sizeof(unsigned long long)));

  int block = 256;
  int grid = 4096;
  vector<float> times;
  times.reserve(reps);

  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int r = 0; r < reps + 5; ++r) {
    GPU_ERROR(cudaMemset(sink, 0, sizeof(unsigned long long)));
    GPU_ERROR(cudaEventRecord(start));
    streamReadKernel<<<grid, block>>>(dptr, sink, n);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());

    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (r >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(sink));

  sort(times.begin(), times.end());
  double ms = times[times.size() / 2];
  return {ms, (double)bytes / (ms * 1.0e-3) / 1.0e9};
}

static double measureLatencyCycles(const uint64_t *dptr, size_t bytes, int reps,
                                   unsigned int clockMHz, size_t minIters,
                                   bool doWarmup, size_t stride) {
  const size_t count = bytes / (sizeof(uint64_t) * stride);
  const size_t iters = max(count, minIters);
  unsigned long long *sink = nullptr;
  GPU_ERROR(cudaMalloc(&sink, sizeof(unsigned long long)));

  MeasurementSeries times;
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  if (doWarmup) {
    pointerChaseKernel<<<1, 1>>>(dptr, sink, iters);
    GPU_ERROR(cudaDeviceSynchronize());
  }

  for (int r = 0; r < reps; ++r) {
    GPU_ERROR(cudaEventRecord(start));
    pointerChaseKernel<<<1, 1>>>(dptr, sink, iters);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    GPU_ERROR(cudaGetLastError());

    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    times.add(ms * 1.0e-3);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaFree(sink));

  return times.median() / (double)iters * clockMHz * 1000.0 * 1000.0;
}

static void fillPointerCycle(uint64_t *host, uint64_t *deviceBase, size_t bytes,
                             size_t stride) {
  const size_t count = bytes / (sizeof(uint64_t) * stride);
  vector<size_t> order(count);
  for (size_t i = 0; i < count; ++i)
    order[i] = i;

  mt19937_64 rng(12345);
  shuffle(order.begin() + 1, order.end(), rng);

  for (size_t i = 0; i < count; ++i) {
    size_t cur = order[i];
    size_t next = order[(i + 1) % count];
    host[cur * stride] = reinterpret_cast<uint64_t>(deviceBase + next * stride);
  }
}

static string mbString(size_t bytes) {
  return to_string(bytes / 1024 / 1024) + "MB";
}

int main(int argc, char **argv) {
  size_t bytes = 256ull * 1024 * 1024;
  size_t latBytes = 16ull * 1024 * 1024;
  size_t deviceL2LatBytes = 16ull * 1024 * 1024;
  size_t deviceLatBytes = 256ull * 1024 * 1024;
  int reps = 15;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--bytes" && i + 1 < argc) {
      bytes = stoull(argv[++i]);
    } else if (arg == "--lat-bytes" && i + 1 < argc) {
      latBytes = stoull(argv[++i]);
    } else if (arg == "--device-l2-lat-bytes" && i + 1 < argc) {
      deviceL2LatBytes = stoull(argv[++i]);
    } else if (arg == "--device-lat-bytes" && i + 1 < argc) {
      deviceLatBytes = stoull(argv[++i]);
    } else if (arg == "--reps" && i + 1 < argc) {
      reps = stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      cout << "Usage: " << argv[0]
           << " [--bytes N] [--lat-bytes N] [--device-lat-bytes N]"
           << " [--device-l2-lat-bytes N] [--reps N]\n";
      return 0;
    } else {
      cerr << "Unknown argument: " << arg << "\n";
      return 2;
    }
  }

  if (bytes < 1024 || bytes % sizeof(uint4) != 0) {
    cerr << "--bytes must be at least 1024 and divisible by 16\n";
    return 2;
  }
  if (latBytes < 1024 || latBytes % sizeof(uint64_t) != 0) {
    cerr << "--lat-bytes must be at least 1024 and divisible by 8\n";
    return 2;
  }
  if (deviceLatBytes < 1024 || deviceLatBytes % sizeof(uint64_t) != 0) {
    cerr << "--device-lat-bytes must be at least 1024 and divisible by 8\n";
    return 2;
  }
  if (deviceL2LatBytes < 1024 ||
      deviceL2LatBytes % sizeof(uint64_t) != 0) {
    cerr << "--device-l2-lat-bytes must be at least 1024 and divisible by 8\n";
    return 2;
  }

  GPU_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDeviceProperties(&prop, 0));
  unsigned int clockMHz = getGPUClock();

  cout << "Device: " << prop.name << ", SM " << prop.major << prop.minor
       << ", SMs=" << prop.multiProcessorCount << ", clock=" << clockMHz
       << " MHz\n";
  cout << "Mapped pinned host memory read from GPU; device memory is a control\n";
  cout << "stream-buffer=" << mbString(bytes)
       << " mapped-latency-buffer=" << mbString(latBytes)
       << " device-l2-latency-buffer=" << mbString(deviceL2LatBytes)
       << " device-latency-buffer=" << mbString(deviceLatBytes)
       << " reps=" << reps << "\n\n";

  uint4 *hostData = nullptr;
  uint4 *mappedData = nullptr;
  GPU_ERROR(cudaHostAlloc(&hostData, bytes, cudaHostAllocMapped));
  for (size_t i = 0; i < bytes / sizeof(uint4); ++i) {
    hostData[i] = make_uint4((uint32_t)i, (uint32_t)(i >> 32), 0x12345678u,
                             0x9abcdef0u);
  }
  GPU_ERROR(cudaHostGetDevicePointer(&mappedData, hostData, 0));

  uint4 *deviceData = nullptr;
  GPU_ERROR(cudaMalloc(&deviceData, bytes));
  initKernel<<<4096, 256>>>(deviceData, bytes / sizeof(uint4));
  GPU_ERROR(cudaDeviceSynchronize());

  auto hostStream = measureStream(mappedData, bytes, reps);
  auto devStream = measureStream(deviceData, bytes, reps);

  uint64_t *hostPtrs = nullptr;
  uint64_t *mappedPtrs = nullptr;
  uint64_t *devicePtrs = nullptr;
  vector<uint64_t> tmp(deviceL2LatBytes / sizeof(uint64_t));
  GPU_ERROR(cudaHostAlloc(&hostPtrs, latBytes, cudaHostAllocMapped));
  GPU_ERROR(cudaHostGetDevicePointer(&mappedPtrs, hostPtrs, 0));
  GPU_ERROR(cudaMalloc(&devicePtrs, deviceL2LatBytes));

  const size_t lineStride = 128 / sizeof(uint64_t);
  fillPointerCycle(hostPtrs, mappedPtrs, latBytes, lineStride);
  fillPointerCycle(tmp.data(), devicePtrs, deviceL2LatBytes, lineStride);
  GPU_ERROR(cudaMemcpy(devicePtrs, tmp.data(), deviceL2LatBytes,
                       cudaMemcpyHostToDevice));

  double hostCycles =
      measureLatencyCycles(mappedPtrs, latBytes, reps, clockMHz, 0, false,
                           lineStride);
  double devCycles =
      measureLatencyCycles(devicePtrs, deviceL2LatBytes, reps, clockMHz,
                           1000000, true, lineStride);

  double devGlobalCycles = devCycles;
  uint64_t *deviceGlobalPtrs = nullptr;
  if (deviceLatBytes != deviceL2LatBytes) {
    vector<uint64_t> globalTmp(deviceLatBytes / sizeof(uint64_t));
    GPU_ERROR(cudaMalloc(&deviceGlobalPtrs, deviceLatBytes));
    fillPointerCycle(globalTmp.data(), deviceGlobalPtrs, deviceLatBytes,
                     lineStride);
    GPU_ERROR(cudaMemcpy(deviceGlobalPtrs, globalTmp.data(), deviceLatBytes,
                         cudaMemcpyHostToDevice));
    devGlobalCycles =
        measureLatencyCycles(deviceGlobalPtrs, deviceLatBytes, reps, clockMHz,
                             1000000, true, lineStride);
  }

  cout << setw(10) << "location" << setw(12) << "pattern" << setw(12) << "ms"
       << setw(12) << "GB/s" << setw(14) << "lat cycles" << "\n";
  cout << string(60, '-') << "\n";
  cout << fixed << setprecision(3);
  cout << setw(10) << "mapped" << setw(12) << "seq-read" << setw(12)
       << hostStream.ms << setw(12) << setprecision(2) << hostStream.gbps
       << setw(14) << "-" << "\n";
  cout << setw(10) << "device" << setw(12) << "seq-read" << setw(12)
       << setprecision(3) << devStream.ms << setw(12) << setprecision(2)
       << devStream.gbps << setw(14) << "-" << "\n";
  cout << setw(10) << "mapped" << setw(12) << "pchase" << setw(12) << "-"
       << setw(12) << "-" << setw(14) << setprecision(1) << hostCycles << "\n";
  cout << setw(10) << "device-l2" << setw(12) << "pchase" << setw(12) << "-"
       << setw(12) << "-" << setw(14) << setprecision(1) << devCycles << "\n";
  cout << setw(10) << "device-gbl" << setw(12) << "pchase" << setw(12) << "-"
       << setw(12) << "-" << setw(14) << setprecision(1) << devGlobalCycles
       << "\n";

  GPU_ERROR(cudaFreeHost(hostPtrs));
  GPU_ERROR(cudaFreeHost(hostData));
  GPU_ERROR(cudaFree(devicePtrs));
  if (deviceGlobalPtrs != nullptr)
    GPU_ERROR(cudaFree(deviceGlobalPtrs));
  GPU_ERROR(cudaFree(deviceData));
  return 0;
}
