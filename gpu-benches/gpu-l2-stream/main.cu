#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"
#include <iomanip>
#include <iostream>
using namespace std;

const int64_t max_buffer_size = 128l * 1024 * 1024 + 2;
const int64_t iteration_count = 1024l * 1024 * 1024 + 2;
double *dA, *dB, *dC, *dD;

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 const double *__restrict__ C,
                                 const double *__restrict__ D, const size_t N,
                                 bool secretlyFalse);

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.23;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T, int length>
__global__ void write_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % length;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.23;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T, int length>
__global__ void read_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % length;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  double temp = B[tidx];

  if (secretlyFalse || temp == 123.0)
    A[tidx] = temp + spoiler[tidx];
}

template <typename T, int length>
__global__ void scale_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % length;

  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = B[tidx] * 1.2;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T, int length>
__global__ void triad_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];

  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % length;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = B[tidx] * D[tidx] + C[tidx];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

void measureFunc(kernel_ptr_type func, int streamCount, int blockSize,
                 int blocksPerSM) {

#ifdef __NVCC__
  GPU_ERROR(cudaFuncSetAttribute(
      func, cudaFuncAttributePreferredSharedMemoryCarveout, 20));
#endif

  int maxActiveBlocks = 0;
  int spoilerSize = 1024;

  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, func, 32, spoilerSize));

  while (maxActiveBlocks > blocksPerSM) {
    spoilerSize += 256;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, func, 32, spoilerSize));
    // std::cout << maxActiveBlocks << " " << spoilerSize << "\n";
  }

  if (maxActiveBlocks == 0)
    std::cout << "Configure " << maxActiveBlocks << " instead of "
              << blocksPerSM << "\n";

  MeasurementSeries time;

  func<<<iteration_count / blockSize + 1, blockSize, spoilerSize>>>(
      dA, dB, dC, dD, iteration_count, false);

  GPU_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  for (int iter = 0; iter < 11; iter++) {
    GPU_ERROR(cudaEventCreate(&start));
    GPU_ERROR(cudaEventCreate(&stop));
    GPU_ERROR(cudaEventRecord(start));
    func<<<iteration_count / blockSize + 1, blockSize, spoilerSize>>>(
        dA, dB, dC, dD, iteration_count, false);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    time.add(milliseconds / 1000);
  }

  MeasurementSeries DRAMread;
  MeasurementSeries DRAMwrite;
  MeasurementSeries L2read;
  MeasurementSeries L2write;

  cout << fixed << setprecision(0);
  for (int i = 0; i < 0; i++) {
    measureDRAMBytesStart();
    func<<<iteration_count / blockSize + 1, blockSize, spoilerSize>>>(
        dA, dB, dC, dD, iteration_count, false);
    auto metrics = measureDRAMBytesStop();
    DRAMread.add(metrics[0]);
    DRAMwrite.add(metrics[1]);

    measureL2BytesStart();
    func<<<iteration_count / blockSize + 1, blockSize, spoilerSize>>>(
        dA, dB, dC, dD, iteration_count, false);
    metrics = measureL2BytesStop();
    L2read.add(metrics[0]);
    L2write.add(metrics[1]);
  }

  /*cout << setw(5) << DRAMread.median() / time.median() / 1.0e9 << " " <<
  setw(5)
       << DRAMwrite.median() / time.median() / 1.0e9 << " ";

  cout << setw(5) << L2read.median() / time.median() / 1.0e9 << " " << setw(5)
       << L2write.median() / time.median() / 1.0e9 << " ";
*/
  cout << fixed << setprecision(0) << " " << setw(5)
       << streamCount * iteration_count * sizeof(double) / time.minValue() *
              1e-9
       << " " << fixed << setprecision(0) << " " << setw(5)
       << streamCount * iteration_count * sizeof(double) / time.median() * 1e-9
       << " " << fixed << setprecision(0) << " " << setw(5)
       << streamCount * iteration_count * sizeof(double) / time.maxValue() *
              1e-9
       << " ";
  cout.flush();
}

void measureKernels(vector<pair<kernel_ptr_type, int>> kernels, int blockSize,
                    int blocksPerSM, int length) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  if (deviceName.starts_with("AMD Radeon RX 6")) {
    prop.maxThreadsPerMultiProcessor = 1024;
    prop.multiProcessorCount *= 2;
  }

  // std::cout << prop.maxThreadsPerMultiProcessor << " "
  //           << prop.maxThreadsPerBlock << "\n";

  if (blockSize * blocksPerSM > prop.maxThreadsPerMultiProcessor ||
      blockSize > prop.maxThreadsPerBlock)
    return;

  int smCount = prop.multiProcessorCount;
  cout << setprecision(1)                                      //
       << setw(4) << blockSize << "   "                        //
       << setw(2) << blocksPerSM << "  "                       //
       << setw(7) << smCount * blockSize * blocksPerSM << "  " //
       << setw(7) << length * sizeof(double) * 2 / 1024 << "  " << setw(5)
       << (float)(blockSize * blocksPerSM) / prop.maxThreadsPerMultiProcessor *
              100.0
       << "%     |  GB/s: ";

  for (auto kernel : kernels) {
    measureFunc(kernel.first, kernel.second, blockSize, blocksPerSM);
  }

  cout << "\n";
}

template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
  (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

template <auto n, typename F> constexpr void for_sequence(F f) {
  for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}

size_t constexpr expSeries(int start, float exponent, int i) {
  float val = start;
  for (int n = 0; n < i; n++) {
    val = val * exponent + 1;
  }
  return (int)val;
}

int main(int argc, char **argv) {
  // initMeasureMetric();
  unsigned int clock = getGPUClock();
  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));

  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dA, dA, dA, dA,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dB, dB, dB, dB,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dC, dC, dC, dC,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dD, dD, dD, dD,
                                                    max_buffer_size, false);
  GPU_ERROR(cudaDeviceSynchronize());

  // cout << "block smBlocks   threads    occ%   |                init"
  //      << "       read       scale     triad       3pt        5pt\n";

  for_sequence<112>([](auto i) {
    if constexpr (i == 5 || i == 19 || i == 44 || i == 60 || i == 76 ||
                  i == 86 || i == 93 || i == 103 || i == 111) {
    const int length = expSeries(2, 1.04, i) * (8 * 1024) + 8161;

    vector<pair<kernel_ptr_type, int>> kernels = {
        {read_kernel<double, length>, 1},
        {scale_kernel<double, length>, 2},
        {triad_kernel<double, length>, 4},
        {write_kernel<double, length>, 1}};

    // measureKernels(kernels, 16, 1, length);
    // measureKernels(kernels, 32, 1, length);
    //
    // measureKernels(kernels, 48, 1, length);
    // measureKernels(kernels, 64, 1, length);
    // measureKernels(kernels, 80, 1, length);
    measureKernels(kernels, 96, 1, length);
    measureKernels(kernels, 112, 1, length);

    for (int warpCount = 2; warpCount <= 80; warpCount++) {

      int threadCount = warpCount * 64;

      if (threadCount / 64 % 4 != 0 && threadCount > 512)
        continue;

      if (threadCount / 64 % 2 == 0)
        measureKernels(kernels, threadCount / 2, 2, length);
      else if (warpCount < 4)
        measureKernels(kernels, threadCount, 1, length);
    }
    std::cout << "\n";
    }
  });

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
}
