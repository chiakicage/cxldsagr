// Standalone CUDA benchmark for mapped pinned-memory KV-record reads.
//
// Purpose:
//   Measure GPU direct reads from CPU pinned mapped memory when each sparse
//   index reads a whole KV-sized contiguous record, not a single 4B word.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120a benchmarks/mapped_kv_record_kernel.cu \
//     -o /tmp/mapped_kv_record_bench
//
// Run:
//   /tmp/mapped_kv_record_bench
//   /tmp/mapped_kv_record_bench --records 524288 --reps 20

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__device__ __forceinline__ size_t permute_idx(size_t i, size_t mask) {
    return (i * 11400714819323198485ull) & mask;
}

// One warp cooperatively reads one KV record. Each record is contiguous
// record_bytes bytes; record_bytes should be a multiple of 16.
__global__ void record_read_kernel(
    const uint4* __restrict__ records,
    unsigned long long* __restrict__ partial,
    size_t n_records,
    int vecs_per_record,
    bool random_access) {
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    size_t n_warps = (gridDim.x * blockDim.x) >> 5;
    int lane = threadIdx.x & 31;
    size_t mask = n_records - 1;
    unsigned long long acc = 0;

    for (size_t r = warp_id; r < n_records; r += n_warps) {
        size_t rec = random_access ? permute_idx(r, mask) : r;
        const uint4* base = records + rec * (size_t)vecs_per_record;

        for (int v = lane; v < vecs_per_record; v += 32) {
            uint4 x = base[v];
            acc += (unsigned long long)x.x + x.y + x.z + x.w;
        }
    }

    atomicAdd(partial, acc);
}

struct Result {
    double ms;
    double gbps;
};

static Result run_record_read(
    uint4* dptr,
    size_t n_records,
    int record_bytes,
    int reps,
    bool random_access) {
    int vecs_per_record = record_bytes / 16;
    unsigned long long* partial = nullptr;
    CHECK_CUDA(cudaMalloc(&partial, sizeof(unsigned long long)));

    int block = 256;
    int grid = 4096;
    std::vector<float> times_ms;
    times_ms.reserve(reps);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int r = 0; r < reps + 5; ++r) {
        CHECK_CUDA(cudaMemset(partial, 0, sizeof(unsigned long long)));
        CHECK_CUDA(cudaEventRecord(start));
        record_read_kernel<<<grid, block>>>(
            dptr, partial, n_records, vecs_per_record, random_access);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        if (r >= 5) times_ms.push_back(ms);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(partial));

    std::sort(times_ms.begin(), times_ms.end());
    double med_ms = times_ms[times_ms.size() / 2];
    double bytes = static_cast<double>(n_records) * record_bytes;
    double gbps = bytes / (med_ms * 1e-3) / 1e9;
    return {med_ms, gbps};
}

static Result bench_mapped_record_read(
    size_t n_records,
    int record_bytes,
    int reps,
    bool random_access) {
    uint4* hptr = nullptr;
    uint4* dptr = nullptr;
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    CHECK_CUDA(cudaHostAlloc(
        &hptr, n_records * static_cast<size_t>(record_bytes), cudaHostAllocMapped));

    size_t words = n_records * static_cast<size_t>(record_bytes) / sizeof(uint32_t);
    uint32_t* hp32 = reinterpret_cast<uint32_t*>(hptr);
    for (size_t i = 0; i < words; ++i) {
        hp32[i] = static_cast<uint32_t>(i);
    }

    CHECK_CUDA(cudaHostGetDevicePointer(&dptr, hptr, 0));
    Result out = run_record_read(dptr, n_records, record_bytes, reps, random_access);
    CHECK_CUDA(cudaFreeHost(hptr));
    return out;
}

static Result bench_device_record_read(
    size_t n_records,
    int record_bytes,
    int reps,
    bool random_access) {
    uint4* dptr = nullptr;
    CHECK_CUDA(cudaMalloc(&dptr, n_records * static_cast<size_t>(record_bytes)));
    CHECK_CUDA(cudaMemset(dptr, 1, n_records * static_cast<size_t>(record_bytes)));
    Result out = run_record_read(dptr, n_records, record_bytes, reps, random_access);
    CHECK_CUDA(cudaFree(dptr));
    return out;
}

static size_t parse_size(const char* s) {
    return static_cast<size_t>(std::strtoull(s, nullptr, 10));
}

int main(int argc, char** argv) {
    size_t n_records = 1ull << 19;
    int reps = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--records" && i + 1 < argc) {
            n_records = parse_size(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [--records N] [--reps N]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if ((n_records & (n_records - 1)) != 0) {
        fprintf(stderr, "--records must be a power of two for the permutation.\n");
        return 2;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, SM %d%d, SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("GPU direct read of CPU mapped pinned KV-sized records\n");
    printf("one warp reads one contiguous record; random means random record order\n");
    printf("records=%zu reps=%d\n\n", n_records, reps);

    printf("%8s %5s %9s %10s %10s %10s\n",
           "records", "recB", "location", "pattern", "ms", "GB/s");
    printf("--------------------------------------------------------------\n");

    int record_sizes[] = {528, 656};
    for (int recb : record_sizes) {
        for (int loc = 0; loc < 2; ++loc) {
            for (int pat = 0; pat < 2; ++pat) {
                bool mapped = (loc == 0);
                bool random = (pat == 1);
                Result r = mapped
                    ? bench_mapped_record_read(n_records, recb, reps, random)
                    : bench_device_record_read(n_records, recb, reps, random);
                printf("%8zu %5d %9s %10s %10.3f %10.2f\n",
                       n_records, recb,
                       mapped ? "mapped" : "device",
                       random ? "random" : "seq",
                       r.ms, r.gbps);
            }
        }
    }

    return 0;
}
