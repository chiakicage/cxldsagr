#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

enum class Op {
  Read,
  Read2,
  Write,
  Copy,
  NtWrite,
  NtCopy,
  Stop,
};

struct Shared {
  pthread_barrier_t startBarrier;
  pthread_barrier_t endBarrier;
  atomic<int> op{0};
  atomic<unsigned long long> sink{0};
};

struct Worker {
  int tid;
  int cpu;
  size_t begin;
  size_t end;
  uint64_t *a;
  uint64_t *b;
  Shared *shared;
};

static vector<int> parseCpuList(const string &s) {
  vector<int> cpus;
  stringstream ss(s);
  string part;
  while (getline(ss, part, ',')) {
    size_t dash = part.find('-');
    if (dash == string::npos) {
      cpus.push_back(stoi(part));
    } else {
      int lo = stoi(part.substr(0, dash));
      int hi = stoi(part.substr(dash + 1));
      for (int c = lo; c <= hi; ++c) {
        cpus.push_back(c);
      }
    }
  }
  return cpus;
}

static string humanBytes(size_t bytes) {
  if (bytes % (1024ull * 1024 * 1024) == 0) {
    return to_string(bytes / (1024ull * 1024 * 1024)) + " GiB";
  }
  if (bytes % (1024ull * 1024) == 0) {
    return to_string(bytes / (1024ull * 1024)) + " MiB";
  }
  return to_string(bytes) + " B";
}

static void pinThisThread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  int rc = sched_setaffinity(0, sizeof(set), &set);
  if (rc != 0) {
    cerr << "warning: sched_setaffinity failed for CPU " << cpu << "\n";
  }
}

static void ntFill64(uint64_t *ptr, size_t begin, size_t end, uint64_t value) {
  size_t i = begin;
  const __m256i v = _mm256_set1_epi64x((long long)value);
  for (; i + 4 <= end; i += 4) {
    _mm256_stream_si256(reinterpret_cast<__m256i *>(ptr + i), v);
  }
  for (; i < end; ++i) {
    ptr[i] = value;
  }
  _mm_sfence();
}

static void ntCopy64(uint64_t *dst, const uint64_t *src, size_t begin,
                     size_t end) {
  size_t i = begin;
  for (; i + 4 <= end; i += 4) {
    __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(src + i));
    _mm256_stream_si256(reinterpret_cast<__m256i *>(dst + i), v);
  }
  for (; i < end; ++i) {
    dst[i] = src[i];
  }
  _mm_sfence();
}

static void workerMain(Worker w) {
  pinThisThread(w.cpu);

  for (size_t i = w.begin; i < w.end; ++i) {
    w.a[i] = (uint64_t)i;
    w.b[i] = ~((uint64_t)i);
  }

  while (true) {
    pthread_barrier_wait(&w.shared->startBarrier);
    Op op = static_cast<Op>(w.shared->op.load(memory_order_relaxed));
    if (op == Op::Stop) {
      pthread_barrier_wait(&w.shared->endBarrier);
      break;
    }

    if (op == Op::Read) {
      uint64_t local = 0;
      for (size_t i = w.begin; i < w.end; ++i) {
        local += w.a[i];
      }
      w.shared->sink.fetch_add(local, memory_order_relaxed);
    } else if (op == Op::Read2) {
      uint64_t local = 0;
      for (size_t i = w.begin; i < w.end; ++i) {
        local += w.a[i] + w.b[i];
      }
      w.shared->sink.fetch_add(local, memory_order_relaxed);
    } else if (op == Op::Write) {
      for (size_t i = w.begin; i < w.end; ++i) {
        w.b[i] = i * 7 + (uint64_t)w.tid;
      }
    } else if (op == Op::Copy) {
      memcpy(w.b + w.begin, w.a + w.begin, (w.end - w.begin) * sizeof(uint64_t));
    } else if (op == Op::NtWrite) {
      ntFill64(w.b, w.begin, w.end, 0x123456789abcdef0ull + (uint64_t)w.tid);
    } else if (op == Op::NtCopy) {
      ntCopy64(w.b, w.a, w.begin, w.end);
    }

    pthread_barrier_wait(&w.shared->endBarrier);
  }
}

static double runOnce(Shared &shared, Op op) {
  shared.op.store(static_cast<int>(op), memory_order_relaxed);
  auto t0 = chrono::steady_clock::now();
  pthread_barrier_wait(&shared.startBarrier);
  pthread_barrier_wait(&shared.endBarrier);
  auto t1 = chrono::steady_clock::now();
  return chrono::duration<double, milli>(t1 - t0).count();
}

static void printResult(const string &name, size_t appBytes, size_t trafficBytes,
                        const vector<double> &times) {
  vector<double> t = times;
  sort(t.begin(), t.end());
  double best = t.front();
  double med = t[t.size() / 2];
  cout << left << setw(12) << name << right << fixed << setprecision(3)
       << setw(11) << best << setw(11) << med << setprecision(2) << setw(13)
       << (double)appBytes / (best * 1.0e-3) / 1.0e9 << setw(13)
       << (double)trafficBytes / (best * 1.0e-3) / 1.0e9 << setw(13)
       << (double)trafficBytes / (med * 1.0e-3) / 1.0e9 << "\n";
}

int main(int argc, char **argv) {
  size_t bytes = 1024ull * 1024 * 1024;
  int reps = 9;
  string cpuSpec = "0,2,4,6,8,10";
  string opSpec = "read,read2,write,copy,nt-write,nt-copy";

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--bytes" && i + 1 < argc) {
      bytes = stoull(argv[++i]);
    } else if (arg == "--reps" && i + 1 < argc) {
      reps = stoi(argv[++i]);
    } else if (arg == "--cpus" && i + 1 < argc) {
      cpuSpec = argv[++i];
    } else if (arg == "--ops" && i + 1 < argc) {
      opSpec = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      cout << "Usage: " << argv[0]
           << " [--bytes N] [--reps N] [--cpus 0,2,4,6,8,10]"
           << " [--ops read,read2,write,copy,nt-write,nt-copy]\n";
      return 0;
    } else {
      cerr << "Unknown argument: " << arg << "\n";
      return 2;
    }
  }

  bytes = bytes / 4096 * 4096;
  size_t n = bytes / sizeof(uint64_t);
  vector<int> cpus = parseCpuList(cpuSpec);
  if (cpus.empty()) {
    cerr << "empty CPU list\n";
    return 2;
  }

  uint64_t *a = static_cast<uint64_t *>(aligned_alloc(4096, bytes));
  uint64_t *b = static_cast<uint64_t *>(aligned_alloc(4096, bytes));
  if (!a || !b) {
    cerr << "allocation failed\n";
    return 1;
  }

  Shared shared;
  pthread_barrier_init(&shared.startBarrier, nullptr, (unsigned)cpus.size() + 1);
  pthread_barrier_init(&shared.endBarrier, nullptr, (unsigned)cpus.size() + 1);

  vector<thread> threads;
  for (size_t t = 0; t < cpus.size(); ++t) {
    size_t begin = (n * t / cpus.size()) / 4 * 4;
    size_t end = (n * (t + 1) / cpus.size()) / 4 * 4;
    Worker w{(int)t, cpus[t], begin, end, a, b, &shared};
    threads.emplace_back(workerMain, w);
  }

  vector<pair<string, Op>> allOps = {{"read", Op::Read},
                                     {"read2", Op::Read2},
                                     {"write", Op::Write},
                                     {"copy", Op::Copy},
                                     {"nt-write", Op::NtWrite},
                                     {"nt-copy", Op::NtCopy}};
  vector<pair<string, Op>> ops;
  stringstream opStream(opSpec);
  string opName;
  while (getline(opStream, opName, ',')) {
    auto it = find_if(allOps.begin(), allOps.end(),
                      [&](const auto &entry) { return entry.first == opName; });
    if (it == allOps.end()) {
      cerr << "Unknown op: " << opName << "\n";
      return 2;
    }
    ops.push_back(*it);
  }
  if (ops.empty()) {
    cerr << "empty op list\n";
    return 2;
  }
  vector<vector<double>> allTimes(ops.size());

  for (int warm = 0; warm < 3; ++warm) {
    for (auto &op : ops) {
      runOnce(shared, op.second);
    }
  }

  for (int r = 0; r < reps; ++r) {
    for (size_t i = 0; i < ops.size(); ++i) {
      allTimes[i].push_back(runOnce(shared, ops[i].second));
    }
  }

  shared.op.store(static_cast<int>(Op::Stop), memory_order_relaxed);
  pthread_barrier_wait(&shared.startBarrier);
  pthread_barrier_wait(&shared.endBarrier);
  for (auto &t : threads) {
    t.join();
  }

  cout << "CPU STREAM-like memory benchmark: buffer=" << humanBytes(bytes)
       << " per array, cpus=" << cpuSpec << ", reps=" << reps
       << ", ops=" << opSpec << ", sink="
       << shared.sink.load(memory_order_relaxed) << "\n";
  cout << left << setw(12) << "path" << right << setw(11) << "best ms"
       << setw(11) << "med ms" << setw(13) << "app GB/s" << setw(13)
       << "traffic GB/s" << setw(13) << "med traffic" << "\n";
  cout << string(73, '-') << "\n";

  for (size_t i = 0; i < ops.size(); ++i) {
    size_t traffic = bytes;
    size_t appBytes = bytes;
    if (ops[i].second == Op::Read2) {
      appBytes = 2 * bytes;
      traffic = 2 * bytes;
    }
    if (ops[i].second == Op::Copy || ops[i].second == Op::NtCopy) {
      traffic = 2 * bytes;
    } else if (ops[i].second == Op::Write) {
      traffic = 2 * bytes;
    }
    printResult(ops[i].first, appBytes, traffic, allTimes[i]);
  }

  pthread_barrier_destroy(&shared.startBarrier);
  pthread_barrier_destroy(&shared.endBarrier);
  free(b);
  free(a);
  return 0;
}
