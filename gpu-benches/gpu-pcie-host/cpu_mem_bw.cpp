#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

static string humanBytes(size_t bytes) {
  if (bytes % (1024ull * 1024 * 1024) == 0)
    return to_string(bytes / (1024ull * 1024 * 1024)) + " GiB";
  if (bytes % (1024ull * 1024) == 0)
    return to_string(bytes / (1024ull * 1024)) + " MiB";
  return to_string(bytes) + " B";
}

template <typename F>
static double timeMs(F &&f) {
  auto t0 = chrono::steady_clock::now();
  f();
  auto t1 = chrono::steady_clock::now();
  return chrono::duration<double, milli>(t1 - t0).count();
}

static void parallelFor(size_t n, int threads, const function<void(size_t, size_t)> &f) {
  vector<thread> workers;
  workers.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    size_t begin = n * t / threads;
    size_t end = n * (t + 1) / threads;
    workers.emplace_back([=, &f]() { f(begin, end); });
  }
  for (auto &w : workers)
    w.join();
}

static void printResult(const string &name, size_t bytes, vector<double> ms) {
  sort(ms.begin(), ms.end());
  double best = ms.front();
  double med = ms[ms.size() / 2];
  cout << left << setw(12) << name << right << setw(11) << fixed
       << setprecision(3) << best << setw(11) << med << setw(13)
       << setprecision(2) << (double)bytes / (best * 1.0e-3) / 1.0e9
       << setw(13) << (double)bytes / (med * 1.0e-3) / 1.0e9 << "\n";
}

int main(int argc, char **argv) {
  size_t bytes = 1024ull * 1024 * 1024;
  int reps = 7;
  int threads = thread::hardware_concurrency();

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--bytes" && i + 1 < argc) {
      bytes = stoull(argv[++i]);
    } else if (arg == "--reps" && i + 1 < argc) {
      reps = stoi(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      threads = stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      cout << "Usage: " << argv[0]
           << " [--bytes N] [--reps N] [--threads N]\n";
      return 0;
    } else {
      cerr << "Unknown argument: " << arg << "\n";
      return 2;
    }
  }

  bytes = bytes / sizeof(uint64_t) * sizeof(uint64_t);
  size_t n = bytes / sizeof(uint64_t);
  uint64_t *a = static_cast<uint64_t *>(aligned_alloc(4096, bytes));
  uint64_t *b = static_cast<uint64_t *>(aligned_alloc(4096, bytes));
  if (!a || !b) {
    cerr << "allocation failed\n";
    return 1;
  }

  parallelFor(n, threads, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      a[i] = i;
      b[i] = ~i;
    }
  });

  atomic<uint64_t> sink{0};
  vector<double> readMs, writeMs, copyMs;
  readMs.reserve(reps);
  writeMs.reserve(reps);
  copyMs.reserve(reps);

  for (int r = 0; r < reps; ++r) {
    readMs.push_back(timeMs([&] {
      parallelFor(n, threads, [&](size_t begin, size_t end) {
        uint64_t local = 0;
        for (size_t i = begin; i < end; ++i)
          local += a[i];
        sink.fetch_add(local, memory_order_relaxed);
      });
    }));

    writeMs.push_back(timeMs([&] {
      parallelFor(n, threads, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
          b[i] = i * 3 + r;
      });
    }));

    copyMs.push_back(timeMs([&] {
      parallelFor(bytes, threads, [&](size_t begin, size_t end) {
        begin = begin / 64 * 64;
        end = end / 64 * 64;
        memcpy(reinterpret_cast<char *>(b) + begin,
               reinterpret_cast<char *>(a) + begin, end - begin);
      });
    }));
  }

  cout << "CPU memory benchmark: buffer=" << humanBytes(bytes)
       << ", threads=" << threads << ", reps=" << reps
       << ", sink=" << sink.load(memory_order_relaxed) << "\n";
  cout << left << setw(12) << "path" << right << setw(11) << "best ms"
       << setw(11) << "med ms" << setw(13) << "best GB/s" << setw(13)
       << "med GB/s" << "\n";
  cout << string(60, '-') << "\n";
  printResult("read", bytes, readMs);
  printResult("write", bytes, writeMs);
  printResult("copy", bytes, copyMs);

  free(b);
  free(a);
  return 0;
}
