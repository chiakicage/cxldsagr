#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <vector>

using namespace sycl;
using dtype = double;

int main(int argc, char **argv) {
  const int N = 64;
  std::cout << std::setw(13) << "data set"   //
       << std::setw(12) << "exec time"  //
       << std::setw(11) << "spread"     //
       << std::setw(15) << "Eff. bw\n"; //

  sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::enable_profiling{}};
  std::cout << "Running on GPU:" << q.get_device().get_info<sycl::info::device::name>()<< std::endl;
     

  for (int blockRun = 3; blockRun < 10000; blockRun += max(1.0, blockRun * 0.1)) {
    const int blockSize = 1024;
    const int blockCount = 200000;

    std::vector<double> time;

    for (int i = 0; i < 11; i++) {
      const size_t bufferCount = blockRun * blockSize * N + i * 128;
      dtype *dA = malloc_device<dtype>(bufferCount, q);
      dtype *dB = malloc_device<dtype>(bufferCount, q);

      q.parallel_for(range<1>(bufferCount), [=](id<1> idx) {
        dA[idx] = dtype(1.1);
        dB[idx] = dtype(1.1);
      }).wait();

      auto start = std::chrono::high_resolution_clock::now();
      q.parallel_for(nd_range<1>(range<1>(blockCount * blockSize), range<1>(blockSize)), [=](nd_item<1> item) {
        int threadIdx = item.get_local_id(0);
        int blockIdx = item.get_group(0);

        dtype localSum = dtype(0);
        for (int i = 0; i < N / 2; i++) {
          int idx = (blockSize * blockRun * i + (blockIdx % blockRun) * blockSize) * 2 + threadIdx;
          localSum += dB[idx] * dB[idx + blockSize];
        }
        localSum *= (dtype)1.3;
        if (threadIdx > 1233 || localSum == (dtype)23.12)
          dA[threadIdx] += localSum;
      }).wait();
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsedtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      time.push_back(std::chrono::duration<double>(elapsedtime).count());

      free(dA, q);
      free(dB, q);
    }

    std::sort(time.begin(), time.end());
    double blockDV = N * blockSize * sizeof(dtype);
    double bw = blockDV * blockCount / time[0] / 1.0e9; // time min value

    std::cout << std::fixed << std::setprecision(0) << std::setw(10) << blockDV / 1024 << " kB"
              << std::fixed << std::setprecision(0) << std::setw(10) << blockDV * blockRun / 1024 << " kB"
              << std::fixed << std::setprecision(0) << std::setw(10) << (time[0] * 1000.0) << "ms"
              << std::setprecision(1) << std::setw(10)
              << abs(*(begin(time)) - *(end(time) - 1)) /
                     std::accumulate(begin(time) + 1, end(time) - 1, 0.0) / (time.size() - 2) * 100
              << "%" << std::setw(10) << bw << " GB/s   " << std::endl;
  }
}