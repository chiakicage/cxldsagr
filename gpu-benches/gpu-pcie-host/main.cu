#include "../gpu-error.h"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <sys/mman.h>
#include <string>
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
  cout << "Async copy engines: " << prop.asyncEngineCount
       << ", canMapHostMemory=" << prop.canMapHostMemory
       << ", concurrentKernels=" << prop.concurrentKernels << "\n";
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
  } else {
    cout << "NVML device handle unavailable: " << nvmlErrorString(st) << "\n";
  }
  nvmlShutdown();
}

__global__ void initKernel(uint4 *data, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    data[i] = make_uint4((unsigned)i, (unsigned)(i >> 32), 0x12345678u,
                         0x9abcdef0u);
  }
}

__global__ void mappedReadKernel(const uint4 *__restrict__ src,
                                 unsigned long long *__restrict__ partial,
                                 size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  unsigned long long acc = 0;
  for (size_t i = tid; i < n; i += stride) {
    uint4 v = src[i];
    acc += (unsigned long long)v.x + v.y + v.z + v.w;
  }
  partial[tid] = acc;
}

__global__ void mappedWriteKernel(uint4 *__restrict__ dst, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    dst[i] = make_uint4((unsigned)i, (unsigned)(i >> 32), 0x456789abu,
                        0xcdef0123u);
  }
}

static Result benchMappedRead(const uint4 *mapped, size_t bytes, int reps,
                              int grid, int block) {
  size_t n = bytes / sizeof(uint4);
  unsigned long long *partial = nullptr;
  GPU_ERROR(cudaMalloc(&partial, grid * block * sizeof(unsigned long long)));

  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    mappedReadKernel<<<grid, block>>>(mapped, partial, n);
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

static Result benchMappedWrite(uint4 *mapped, size_t bytes, int reps, int grid,
                               int block) {
  size_t n = bytes / sizeof(uint4);
  vector<float> times;
  times.reserve(reps);
  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start));
    mappedWriteKernel<<<grid, block>>>(mapped, n);
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

static Result benchMemcpy(void *dst, const void *src, size_t bytes,
                          cudaMemcpyKind kind, int reps) {
  vector<float> times;
  times.reserve(reps);
  cudaStream_t stream;
  cudaEvent_t start, stop;
  GPU_ERROR(cudaStreamCreate(&stream));
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < reps + 5; ++i) {
    GPU_ERROR(cudaEventRecord(start, stream));
    GPU_ERROR(cudaMemcpyAsync(dst, src, bytes, kind, stream));
    GPU_ERROR(cudaEventRecord(stop, stream));
    GPU_ERROR(cudaEventSynchronize(stop));
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  GPU_ERROR(cudaStreamDestroy(stream));
  return summarize(times, bytes);
}

static Result benchMemcpyBidirectional(void *dDst, const void *hSrc,
                                       void *hDst, const void *dSrc,
                                       size_t bytes, int reps) {
  vector<float> times;
  times.reserve(reps);
  cudaStream_t h2dStream, d2hStream;
  GPU_ERROR(cudaStreamCreateWithFlags(&h2dStream, cudaStreamNonBlocking));
  GPU_ERROR(cudaStreamCreateWithFlags(&d2hStream, cudaStreamNonBlocking));

  for (int i = 0; i < reps + 5; ++i) {
    auto t0 = chrono::steady_clock::now();
    GPU_ERROR(cudaMemcpyAsync(dDst, hSrc, bytes, cudaMemcpyHostToDevice,
                              h2dStream));
    GPU_ERROR(cudaMemcpyAsync(hDst, dSrc, bytes, cudaMemcpyDeviceToHost,
                              d2hStream));
    GPU_ERROR(cudaStreamSynchronize(h2dStream));
    GPU_ERROR(cudaStreamSynchronize(d2hStream));
    auto t1 = chrono::steady_clock::now();
    float ms = chrono::duration<float, milli>(t1 - t0).count();
    if (i >= 5)
      times.push_back(ms);
  }

  GPU_ERROR(cudaStreamDestroy(h2dStream));
  GPU_ERROR(cudaStreamDestroy(d2hStream));
  return summarize(times, bytes * 2);
}

static Result benchMemcpyMulti(void *dst, const void *src, size_t bytes,
                               cudaMemcpyKind kind, int reps, int streams) {
  vector<float> times;
  times.reserve(reps);
  vector<cudaStream_t> s(streams);
  for (int i = 0; i < streams; ++i) {
    GPU_ERROR(cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking));
  }

  size_t chunk = bytes / streams / 4096 * 4096;
  if (chunk == 0) {
    chunk = bytes / streams / sizeof(uint4) * sizeof(uint4);
  }
  size_t copied = chunk * streams;
  const char *srcBytes = static_cast<const char *>(src);
  char *dstBytes = static_cast<char *>(dst);

  for (int r = 0; r < reps + 5; ++r) {
    auto t0 = chrono::steady_clock::now();
    for (int i = 0; i < streams; ++i) {
      GPU_ERROR(cudaMemcpyAsync(dstBytes + (size_t)i * chunk,
                                srcBytes + (size_t)i * chunk, chunk, kind,
                                s[i]));
    }
    for (int i = 0; i < streams; ++i) {
      GPU_ERROR(cudaStreamSynchronize(s[i]));
    }
    auto t1 = chrono::steady_clock::now();
    float ms = chrono::duration<float, milli>(t1 - t0).count();
    if (r >= 5)
      times.push_back(ms);
  }

  for (auto stream : s) {
    GPU_ERROR(cudaStreamDestroy(stream));
  }
  return summarize(times, copied);
}

static void *allocRegisteredCandidate(size_t bytes, bool huge) {
  void *ptr = nullptr;
  int rc = posix_memalign(&ptr, huge ? 2ull * 1024 * 1024 : 4096ull, bytes);
  if (rc != 0) {
    return nullptr;
  }
  if (huge) {
    madvise(ptr, bytes, MADV_HUGEPAGE);
  }
  return ptr;
}

static void printSmapsInfo(const string &name, const void *ptr) {
  ifstream smaps("/proc/self/smaps");
  if (!smaps) {
    return;
  }

  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  string line;
  bool inMapping = false;
  cout << "  " << name << " mapping info:\n";
  while (getline(smaps, line)) {
    uintptr_t lo = 0, hi = 0;
    if (sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2) {
      inMapping = (addr >= lo && addr < hi);
      if (inMapping) {
        cout << "    " << line << "\n";
      }
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

static void printResult(const string &name, const Result &r) {
  cout << left << setw(22) << name << right << setw(11) << fixed
       << setprecision(3) << r.bestMs << setw(11) << r.medianMs << setw(13)
       << setprecision(2) << r.bestGBs << setw(13) << r.medianGBs << "\n";
}

static void runSuite(const string &label, unsigned int hostFlags, size_t bytes,
                     int reps, int grid, int block, uint4 *deviceSrc,
                     uint4 *deviceDst) {
  uint4 *hostSrc = nullptr;
  uint4 *hostDst = nullptr;
  uint4 *mappedSrc = nullptr;
  uint4 *mappedDst = nullptr;

  GPU_ERROR(cudaHostAlloc(&hostSrc, bytes, hostFlags));
  GPU_ERROR(cudaHostAlloc(&hostDst, bytes, hostFlags));
  GPU_ERROR(cudaHostGetDevicePointer(&mappedSrc, hostSrc, 0));
  GPU_ERROR(cudaHostGetDevicePointer(&mappedDst, hostDst, 0));

  size_t n = bytes / sizeof(uint4);
  for (size_t i = 0; i < n; ++i) {
    hostSrc[i] = make_uint4((unsigned)i, (unsigned)(i >> 32), 0x11111111u,
                            0x22222222u);
  }

  Result mappedRead = benchMappedRead(mappedSrc, bytes, reps, grid, block);
  Result mappedWrite = benchMappedWrite(mappedDst, bytes, reps, grid, block);
  Result h2d =
      benchMemcpy(deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice, reps);
  Result d2h =
      benchMemcpy(hostDst, deviceSrc, bytes, cudaMemcpyDeviceToHost, reps);
  Result bidir =
      benchMemcpyBidirectional(deviceDst, hostSrc, hostDst, deviceSrc, bytes,
                               reps);
  Result h2d2 =
      benchMemcpyMulti(deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice, reps,
                       2);
  Result h2d4 =
      benchMemcpyMulti(deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice, reps,
                       4);
  Result d2h2 =
      benchMemcpyMulti(hostDst, deviceSrc, bytes, cudaMemcpyDeviceToHost, reps,
                       2);
  Result d2h4 =
      benchMemcpyMulti(hostDst, deviceSrc, bytes, cudaMemcpyDeviceToHost, reps,
                       4);

  cout << "\n[" << label << "]\n";
  cout << left << setw(22) << "path" << right << setw(11) << "best ms"
       << setw(11) << "med ms" << setw(13) << "best GB/s" << setw(13)
       << "med GB/s" << "\n";
  cout << string(70, '-') << "\n";
  printResult("mapped read", mappedRead);
  printResult("mapped write", mappedWrite);
  printResult("cudaMemcpy H2D", h2d);
  printResult("cudaMemcpy D2H", d2h);
  printResult("cudaMemcpy H2D+D2H", bidir);
  printResult("cudaMemcpy H2D x2", h2d2);
  printResult("cudaMemcpy H2D x4", h2d4);
  printResult("cudaMemcpy D2H x2", d2h2);
  printResult("cudaMemcpy D2H x4", d2h4);

  GPU_ERROR(cudaFreeHost(hostDst));
  GPU_ERROR(cudaFreeHost(hostSrc));
}

static void runRegisteredSuite(const string &label, bool huge, size_t bytes,
                               int reps, uint4 *deviceSrc, uint4 *deviceDst) {
  uint4 *hostSrc = static_cast<uint4 *>(allocRegisteredCandidate(bytes, huge));
  uint4 *hostDst = static_cast<uint4 *>(allocRegisteredCandidate(bytes, huge));
  if (!hostSrc || !hostDst) {
    cerr << label << " host allocation failed\n";
    free(hostDst);
    free(hostSrc);
    return;
  }

  size_t n = bytes / sizeof(uint4);
  for (size_t i = 0; i < n; ++i) {
    hostSrc[i] = make_uint4((unsigned)i, (unsigned)(i >> 32), 0x33333333u,
                            0x44444444u);
    hostDst[i] = make_uint4(0, 0, 0, 0);
  }

  printSmapsInfo("hostSrc before cudaHostRegister", hostSrc);

  GPU_ERROR(cudaHostRegister(hostSrc, bytes, cudaHostRegisterMapped));
  GPU_ERROR(cudaHostRegister(hostDst, bytes, cudaHostRegisterMapped));
  uint4 *mappedSrc = nullptr;
  uint4 *mappedDst = nullptr;
  GPU_ERROR(cudaHostGetDevicePointer(&mappedSrc, hostSrc, 0));
  GPU_ERROR(cudaHostGetDevicePointer(&mappedDst, hostDst, 0));

  Result h2d =
      benchMemcpy(deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice, reps);
  Result d2h =
      benchMemcpy(hostDst, deviceSrc, bytes, cudaMemcpyDeviceToHost, reps);
  Result mappedRead = benchMappedRead(mappedSrc, bytes, reps, 8192, 256);
  Result mappedWrite = benchMappedWrite(mappedDst, bytes, reps, 8192, 256);
  Result h2d4 =
      benchMemcpyMulti(deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice, reps,
                       4);
  Result d2h4 =
      benchMemcpyMulti(hostDst, deviceSrc, bytes, cudaMemcpyDeviceToHost, reps,
                       4);

  printSmapsInfo("hostSrc after cudaHostRegister", hostSrc);

  cout << "\n[" << label << "]\n";
  cout << left << setw(22) << "path" << right << setw(11) << "best ms"
       << setw(11) << "med ms" << setw(13) << "best GB/s" << setw(13)
       << "med GB/s" << "\n";
  cout << string(70, '-') << "\n";
  printResult("mapped read", mappedRead);
  printResult("mapped write", mappedWrite);
  printResult("cudaMemcpy H2D", h2d);
  printResult("cudaMemcpy D2H", d2h);
  printResult("cudaMemcpy H2D x4", h2d4);
  printResult("cudaMemcpy D2H x4", d2h4);

  GPU_ERROR(cudaHostUnregister(hostDst));
  GPU_ERROR(cudaHostUnregister(hostSrc));
  free(hostDst);
  free(hostSrc);
}

int main(int argc, char **argv) {
  size_t bytes = 512ull * 1024 * 1024;
  int reps = 15;
  int block = 256;
  int grid = 8192;

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

  cout << "buffer=" << humanBytes(bytes) << ", reps=" << reps
       << ", grid=" << grid << ", block=" << block << "\n\n";

  size_t n = bytes / sizeof(uint4);
  uint4 *deviceSrc = nullptr;
  uint4 *deviceDst = nullptr;
  GPU_ERROR(cudaMalloc(&deviceSrc, bytes));
  GPU_ERROR(cudaMalloc(&deviceDst, bytes));
  initKernel<<<grid, block>>>(deviceSrc, n);
  initKernel<<<grid, block>>>(deviceDst, n);
  GPU_ERROR(cudaDeviceSynchronize());

  runSuite("default pinned mapped", cudaHostAllocMapped, bytes, reps, grid,
           block, deviceSrc, deviceDst);
  runSuite("write-combined pinned mapped",
           cudaHostAllocMapped | cudaHostAllocWriteCombined, bytes, reps, grid,
           block, deviceSrc, deviceDst);
  runRegisteredSuite("4KiB-aligned malloc + cudaHostRegister", false, bytes,
                     reps, deviceSrc, deviceDst);
  runRegisteredSuite("2MiB-aligned malloc + MADV_HUGEPAGE + cudaHostRegister",
                     true, bytes, reps, deviceSrc, deviceDst);

  GPU_ERROR(cudaFree(deviceDst));
  GPU_ERROR(cudaFree(deviceSrc));
  return 0;
}
