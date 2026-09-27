/*
 * Pinned, first-touch, AVX-512 CPU memory-bandwidth measurement.
 * Build: gcc -O3 -march=native -std=c11 -pthread -Wall -Wextra dram_bench.c -lm
 * All reported rates use decimal GB/s. No allocation, initialization or
 * validation is included in timed intervals. Timings include release/completion
 * barriers, and use one wall clock for the complete group of worker threads.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <immintrin.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/perf_event.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAX_OPS 4
#define MAX_NODES 4096
#define HUGE_ALIGN ((size_t)2 * 1024 * 1024)

enum operation { OP_READ, OP_NT_WRITE, OP_NT_COPY, OP_NT_TRIAD };
static const char *op_names[] = {"read", "nt-write", "nt-copy", "nt-triad"};
static const int streams[] = {1, 1, 2, 3};

struct config {
    int cpus[CPU_SETSIZE];
    int threads;
    size_t mib_per_thread;
    size_t array_bytes;
    int warmup, reps, passes;
    enum operation ops[MAX_OPS];
    int op_count;
    bool imc;
};

struct worker {
    int id, actual_cpu;
    void *mapping;
    size_t mapping_bytes;
    double *a, *b, *c;
    int hugepage_advice_failed;
    double read_sum, source_sum;
    double checksum[MAX_OPS], expected_checksum[MAX_OPS];
    uint64_t mismatches[MAX_OPS];
};

struct placement {
    unsigned long long pages[MAX_NODES];
    unsigned long long anonymous_hugepage_kib;
    int numa_maps_readable, smaps_readable, matched_mappings;
};

static struct config cfg = {.mib_per_thread = 128, .warmup = 2, .reps = 7, .passes = 2};
static pthread_barrier_t barrier;

#define MAX_IMC_COUNTERS 256
struct counter_value { uint64_t value, enabled, running; };
struct imc_counter { int fd, socket, is_write, pmu, cpu; };
struct imc_state {
    struct imc_counter counters[MAX_IMC_COUNTERS];
    int count, sockets[CPU_SETSIZE], socket_count;
};
struct imc_result {
    double read_bytes, write_bytes, min_running_fraction, capture_seconds;
    struct counter_value deltas[MAX_IMC_COUNTERS];
};

static void die(const char *message) {
    fprintf(stderr, "dram_bench: %s\n", message);
    exit(EXIT_FAILURE);
}

static void system_die(const char *message) {
    fprintf(stderr, "dram_bench: %s: %s\n", message, strerror(errno));
    exit(EXIT_FAILURE);
}

static void synchronize(void) {
    int error = pthread_barrier_wait(&barrier);
    if (error && error != PTHREAD_BARRIER_SERIAL_THREAD)
        die("pthread_barrier_wait failed");
}

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts))
        system_die("clock_gettime");
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void read_sysfs(const char *path, char *buffer, size_t size) {
    FILE *file = fopen(path, "r");
    if (!file) system_die(path);
    if (!fgets(buffer, (int)size, file)) die("cannot read required IMC sysfs metadata");
    fclose(file);
    buffer[strcspn(buffer, "\r\n")] = '\0';
}

static int cpu_socket(int cpu) {
    char path[256], buffer[128], *end;
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    read_sysfs(path, buffer, sizeof(buffer));
    long value = strtol(buffer, &end, 10);
    if (*end || value < 0 || value > INT_MAX) die("invalid CPU socket metadata");
    return (int)value;
}

static void imc_setup(struct imc_state *state) {
    if (!cfg.imc) return;
    for (int t = 0; t < cfg.threads; ++t) {
        int socket = cpu_socket(cfg.cpus[t]), found = 0;
        for (int i = 0; i < state->socket_count; ++i)
            if (state->sockets[i] == socket) found = 1;
        if (!found) state->sockets[state->socket_count++] = socket;
    }
    if (state->socket_count * 16 > MAX_IMC_COUNTERS)
        die("too many sockets for IMC counter collection");
    /* Deliberately restricted to the eight programmable EMR IMC PMUs with
     * verified CAS encodings/units; free_running PMUs are never selected. */
    for (int pmu = 0; pmu < 8; ++pmu) {
        char base[256], path[512], buffer[256], cpumask[256], *end;
        snprintf(base, sizeof(base), "/sys/bus/event_source/devices/uncore_imc_%d", pmu);
        snprintf(path, sizeof(path), "%s/type", base);
        read_sysfs(path, buffer, sizeof(buffer));
        unsigned long type = strtoul(buffer, &end, 10);
        if (*end || type > UINT_MAX) die("invalid IMC PMU type");
        snprintf(path, sizeof(path), "%s/cpumask", base);
        read_sysfs(path, cpumask, sizeof(cpumask));
        int representatives[CPU_SETSIZE];
        for (int i = 0; i < state->socket_count; ++i) representatives[i] = -1;
        char *save;
        for (char *token = strtok_r(cpumask, ",", &save); token;
                token = strtok_r(NULL, ",", &save)) {
            long first = strtol(token, &end, 10), last = first;
            if (end == token || first < 0 || first >= CPU_SETSIZE)
                die("invalid IMC representative CPU");
            if (*end == '-') last = strtol(end + 1, &end, 10);
            if (*end || last < first || last >= CPU_SETSIZE)
                die("invalid IMC representative CPU range");
            for (long cpu = first; cpu <= last; ++cpu) {
                int socket = cpu_socket((int)cpu);
                for (int i = 0; i < state->socket_count; ++i)
                    if (state->sockets[i] == socket && representatives[i] < 0)
                        representatives[i] = (int)cpu;
            }
        }
        for (int mode = 0; mode < 2; ++mode) {
            const char *event = mode ? "cas_count_write" : "cas_count_read";
            snprintf(path, sizeof(path), "%s/events/%s", base, event);
            read_sysfs(path, buffer, sizeof(buffer));
            if (strcmp(buffer, mode ? "event=0x05,umask=0xf0" : "event=0x05,umask=0xcf"))
                die("IMC event encoding differs from supported EMR encoding");
            snprintf(path, sizeof(path), "%s/events/%s.unit", base, event);
            read_sysfs(path, buffer, sizeof(buffer));
            if (strcmp(buffer, "MiB")) die("unsupported IMC event unit");
            snprintf(path, sizeof(path), "%s/events/%s.scale", base, event);
            read_sysfs(path, buffer, sizeof(buffer));
            double scale = strtod(buffer, &end);
            if (*end || scale * 1048576.0 != 64.0)
                die("IMC event scale is not 64 bytes per event");
            for (int i = 0; i < state->socket_count; ++i) {
                if (representatives[i] < 0) die("IMC lacks a representative for requested socket");
                struct perf_event_attr attr = {0};
                attr.size = sizeof(attr);
                attr.type = (uint32_t)type;
                attr.config = mode ? 0xf005 : 0xcf05;
                attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                int fd = (int)syscall(SYS_perf_event_open, &attr, -1, representatives[i], -1, 0);
                if (fd < 0) system_die("perf_event_open IMC (explicit --imc requires counter access)");
                state->counters[state->count++] =
                    (struct imc_counter){fd, state->sockets[i], mode, pmu, representatives[i]};
            }
        }
    }
}

static void imc_snapshot(const struct imc_state *state, struct counter_value *values) {
    for (int i = 0; i < state->count; ++i) {
        ssize_t bytes;
        do { bytes = read(state->counters[i].fd, &values[i], sizeof(values[i])); }
        while (bytes < 0 && errno == EINTR);
        if (bytes != (ssize_t)sizeof(values[i])) die("cannot read complete IMC counter value");
    }
}

static struct imc_result imc_delta(const struct imc_state *state,
                                  const struct counter_value *before,
                                  const struct counter_value *after,
                                  double capture_seconds) {
    struct imc_result out = {.min_running_fraction = 1.0, .capture_seconds = capture_seconds};
    for (int i = 0; i < state->count; ++i) {
        if (after[i].enabled < before[i].enabled || after[i].running < before[i].running ||
                after[i].value < before[i].value)
            die("IMC counter decreased during measurement");
        uint64_t enabled = after[i].enabled - before[i].enabled;
        uint64_t running = after[i].running - before[i].running;
        if (!enabled || !running || running > enabled)
            die("IMC counter had no valid running time");
        double fraction = (double)running / (double)enabled;
        out.deltas[i] = (struct counter_value){after[i].value - before[i].value, enabled, running};
        double bytes = (double)(after[i].value - before[i].value) * 64.0 / fraction;
        if (state->counters[i].is_write) out.write_bytes += bytes;
        else out.read_bytes += bytes;
        if (fraction < out.min_running_fraction) out.min_running_fraction = fraction;
    }
    return out;
}

/* Eight independent accumulators avoid a serial floating-point add chain. */
__attribute__((noinline))
static double read_kernel(const double *restrict b, size_t n, int passes) {
    __m512d s0 = _mm512_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    __m512d s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; i += 64) {
            s0 = _mm512_add_pd(s0, _mm512_load_pd(b + i));
            s1 = _mm512_add_pd(s1, _mm512_load_pd(b + i + 8));
            s2 = _mm512_add_pd(s2, _mm512_load_pd(b + i + 16));
            s3 = _mm512_add_pd(s3, _mm512_load_pd(b + i + 24));
            s4 = _mm512_add_pd(s4, _mm512_load_pd(b + i + 32));
            s5 = _mm512_add_pd(s5, _mm512_load_pd(b + i + 40));
            s6 = _mm512_add_pd(s6, _mm512_load_pd(b + i + 48));
            s7 = _mm512_add_pd(s7, _mm512_load_pd(b + i + 56));
        }
        /* Prevent inter-pass load hoisting even under aggressive optimization. */
        __asm__ __volatile__("" ::: "memory");
    }
    s0 = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
    s4 = _mm512_add_pd(_mm512_add_pd(s4, s5), _mm512_add_pd(s6, s7));
    return _mm512_reduce_add_pd(_mm512_add_pd(s0, s4));
}

__attribute__((noinline))
static void write_kernel(double *restrict a, size_t n, int passes) {
    __m512d value = _mm512_set1_pd(4.0);
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; i += 64) {
            _mm512_stream_pd(a + i, value);
            _mm512_stream_pd(a + i + 8, value);
            _mm512_stream_pd(a + i + 16, value);
            _mm512_stream_pd(a + i + 24, value);
            _mm512_stream_pd(a + i + 32, value);
            _mm512_stream_pd(a + i + 40, value);
            _mm512_stream_pd(a + i + 48, value);
            _mm512_stream_pd(a + i + 56, value);
        }
        _mm_sfence();
    }
}

__attribute__((noinline))
static void copy_kernel(double *restrict a, const double *restrict b, size_t n, int passes) {
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; i += 64) {
            _mm512_stream_pd(a + i, _mm512_load_pd(b + i));
            _mm512_stream_pd(a + i + 8, _mm512_load_pd(b + i + 8));
            _mm512_stream_pd(a + i + 16, _mm512_load_pd(b + i + 16));
            _mm512_stream_pd(a + i + 24, _mm512_load_pd(b + i + 24));
            _mm512_stream_pd(a + i + 32, _mm512_load_pd(b + i + 32));
            _mm512_stream_pd(a + i + 40, _mm512_load_pd(b + i + 40));
            _mm512_stream_pd(a + i + 48, _mm512_load_pd(b + i + 48));
            _mm512_stream_pd(a + i + 56, _mm512_load_pd(b + i + 56));
        }
        _mm_sfence();
    }
}

__attribute__((noinline))
static void triad_kernel(double *restrict a, const double *restrict b,
                         const double *restrict c, size_t n, int passes) {
    __m512d scalar = _mm512_set1_pd(3.0);
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; i += 64) {
            for (size_t j = 0; j < 64; j += 8) {
                __m512d value = _mm512_fmadd_pd(scalar, _mm512_load_pd(c + i + j),
                                               _mm512_load_pd(b + i + j));
                _mm512_stream_pd(a + i + j, value);
            }
        }
        _mm_sfence();
    }
}

static double source_b(size_t i, int id) {
    return 1.0 + (double)(i & 1023) / 1024.0 + (double)id / 256.0;
}

static double source_c(size_t i, int id) {
    return 2.0 + (double)(i & 511) / 512.0 + (double)id / 128.0;
}

static void initialize(struct worker *w) {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cfg.cpus[w->id], &affinity);
    int error = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
    if (error) {
        fprintf(stderr, "dram_bench: cannot pin worker %d to CPU %d: %s\n",
                w->id, cfg.cpus[w->id], strerror(error));
        exit(EXIT_FAILURE);
    }
    w->actual_cpu = sched_getcpu();
    if (w->actual_cpu != cfg.cpus[w->id])
        die("worker CPU differs from requested affinity");

    /* PROT_NONE padding keeps each worker's VMA distinct for numa_maps auditing.
     * The usable region is 2 MiB aligned; padding reserves address space only. */
    size_t data_bytes = 3 * cfg.array_bytes;
    w->mapping_bytes = data_bytes + 2 * HUGE_ALIGN;
    w->mapping = mmap(NULL, w->mapping_bytes, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w->mapping == MAP_FAILED)
        system_die("mmap arrays");
    uintptr_t aligned = ((uintptr_t)w->mapping + HUGE_ALIGN) & ~(HUGE_ALIGN - 1);
    w->a = (double *)aligned;
    w->b = (double *)(aligned + cfg.array_bytes);
    w->c = (double *)(aligned + 2 * cfg.array_bytes);
    if (mprotect(w->a, data_bytes, PROT_READ | PROT_WRITE))
        system_die("mprotect arrays");
    if (madvise(w->a, data_bytes, MADV_HUGEPAGE))
        w->hugepage_advice_failed = 1;

    size_t n = cfg.array_bytes / sizeof(double);
    for (size_t i = 0; i < n; ++i) {
        w->a[i] = 0.0;
        w->b[i] = source_b(i, w->id);
        w->c[i] = source_c(i, w->id);
    }
    /* n is a multiple of 1024. Compute the expected checksum independently of
     * the measured read kernel, from the mean of its repeating source pattern. */
    w->source_sum = (double)n * (1.0 + (double)w->id / 256.0 + 1023.0 / 2048.0);
}

static void validate(struct worker *w, int oi) {
    enum operation op = cfg.ops[oi];
    size_t n = cfg.array_bytes / sizeof(double);
    if (op == OP_READ) {
        w->checksum[oi] = w->read_sum;
        w->expected_checksum[oi] = w->source_sum * cfg.passes;
        if (!isfinite(w->read_sum) || fabs(w->read_sum - w->expected_checksum[oi]) >
                1e-12 * fabs(w->expected_checksum[oi]))
            w->mismatches[oi] = 1;
        return;
    }
    double checksum = 0.0, expected_sum = 0.0;
    uint64_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        double expected = op == OP_NT_WRITE ? 4.0 : source_b(i, w->id);
        if (op == OP_NT_TRIAD)
            expected += 3.0 * source_c(i, w->id);
        if (w->a[i] != expected)
            ++mismatches;
        checksum += w->a[i];
        expected_sum += expected;
    }
    w->checksum[oi] = checksum;
    w->expected_checksum[oi] = expected_sum;
    w->mismatches[oi] = mismatches;
}

static void *worker_main(void *argument) {
    struct worker *w = argument;
    initialize(w);
    synchronize();
    size_t n = cfg.array_bytes / sizeof(double);
    for (int oi = 0; oi < cfg.op_count; ++oi) {
        for (int rep = 0; rep < cfg.warmup + cfg.reps; ++rep) {
            synchronize(); /* Ready: initialization/validation is complete. */
            synchronize(); /* Start: the master has started its clock. */
            switch (cfg.ops[oi]) {
            case OP_READ: w->read_sum = read_kernel(w->b, n, cfg.passes); break;
            case OP_NT_WRITE: write_kernel(w->a, n, cfg.passes); break;
            case OP_NT_COPY: copy_kernel(w->a, w->b, n, cfg.passes); break;
            case OP_NT_TRIAD: triad_kernel(w->a, w->b, w->c, n, cfg.passes); break;
            }
            synchronize(); /* Done: includes completion of all NT stores. */
            synchronize(); /* The master's clock has stopped before any validation. */
        }
        validate(w, oi);
        synchronize();
    }
    /* Keep mappings alive until the master has read placement and results. */
    synchronize();
    if (munmap(w->mapping, w->mapping_bytes))
        system_die("munmap arrays");
    return NULL;
}

static int mapping_worker(uintptr_t address, const struct worker *workers) {
    for (int t = 0; t < cfg.threads; ++t)
        if (address == (uintptr_t)workers[t].a)
            return t;
    return -1;
}

static struct placement inspect_placement(const struct worker *workers) {
    struct placement out = {0};
    char *line = NULL;
    size_t capacity = 0;
    FILE *file = fopen("/proc/self/numa_maps", "r");
    if (file) {
        out.numa_maps_readable = 1;
        while (getline(&line, &capacity, file) >= 0) {
            char *end;
            uintptr_t address = (uintptr_t)strtoull(line, &end, 16);
            if (end == line || mapping_worker(address, workers) < 0)
                continue;
            ++out.matched_mappings;
            char *save, *word = strtok_r(end, " \n", &save);
            while (word) {
                int node;
                unsigned long long pages;
                if (sscanf(word, "N%d=%llu", &node, &pages) == 2 &&
                        node >= 0 && node < MAX_NODES)
                    out.pages[node] += pages;
                word = strtok_r(NULL, " \n", &save);
            }
        }
        fclose(file);
    }
    file = fopen("/proc/self/smaps", "r");
    if (file) {
        out.smaps_readable = 1;
        int selected = 0;
        while (getline(&line, &capacity, file) >= 0) {
            unsigned long long start, finish, kib;
            if (sscanf(line, "%llx-%llx", &start, &finish) == 2)
                selected = mapping_worker((uintptr_t)start, workers) >= 0;
            else if (selected && sscanf(line, "AnonHugePages: %llu kB", &kib) == 1)
                out.anonymous_hugepage_kib += kib;
        }
        fclose(file);
    }
    free(line);
    return out;
}

static unsigned long long parse_number(const char *text, bool allow_zero) {
    char *end;
    errno = 0;
    if (!*text || *text == '-')
        die("numeric option must be an unsigned integer");
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || (!allow_zero && !value))
        die("invalid numeric option");
    return value;
}

static void parse_cpus(const char *text) {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
        system_die("sched_getaffinity");
    bool seen[CPU_SETSIZE] = {false};
    const char *p = text;
    cfg.threads = 0;
    if (!*p)
        die("CPU list cannot be empty");
    while (*p) {
        char *end;
        errno = 0;
        long first = strtol(p, &end, 10), last;
        if (errno || end == p || first < 0 || first >= CPU_SETSIZE)
            die("invalid CPU list");
        p = end;
        last = first;
        if (*p == '-') {
            ++p;
            last = strtol(p, &end, 10);
            if (errno || end == p || last < first || last >= CPU_SETSIZE)
                die("invalid CPU range");
            p = end;
        }
        for (long cpu = first; cpu <= last; ++cpu) {
            if (seen[cpu])
                die("CPU list contains a duplicate CPU");
            if (!CPU_ISSET(cpu, &allowed))
                die("requested CPU is outside process allowed affinity");
            seen[cpu] = true;
            cfg.cpus[cfg.threads++] = (int)cpu;
        }
        if (!*p)
            break;
        if (*p != ',' || !p[1])
            die("invalid CPU-list separator");
        ++p;
    }
}

static void parse_ops(const char *text) {
    char *copy = strdup(text), *save;
    if (!copy)
        system_die("strdup ops");
    bool seen[MAX_OPS] = {false};
    cfg.op_count = 0;
    for (char *word = strtok_r(copy, ",", &save); word;
            word = strtok_r(NULL, ",", &save)) {
        int op;
        for (op = 0; op < MAX_OPS; ++op)
            if (!strcmp(word, op_names[op]))
                break;
        if (op == MAX_OPS || seen[op])
            die("unknown or duplicated operation");
        seen[op] = true;
        cfg.ops[cfg.op_count++] = (enum operation)op;
    }
    free(copy);
    if (!cfg.op_count)
        die("operation list cannot be empty");
}

static void usage(const char *program) {
    printf("Usage: %s --cpus CPU_LIST [options]\n"
           "  --cpus 0-47,96-143       Explicit worker CPUs, including range syntax\n"
           "  --mib-per-thread 128    Size of EACH of a/b/c, MiB per worker\n"
           "  --warmup 2              Untimed warmup repetitions per operation\n"
           "  --reps 7                Recorded repetitions per operation\n"
           "  --passes 2              Full-array passes in each repetition\n"
           "  --ops read,nt-write,nt-copy,nt-triad\n"
           "  --imc                    Also read socket-wide EMR IMC CAS counters\n"
           "  --help\n"
           "Output: one JSON object. Copy payload counts read + write; its\n"
           "one_direction_GBps counts copied bytes only. Triad counts 2 reads\n"
           "+ 1 write. All stores are non-temporal, followed by sfence.\n",
           program);
}

int main(int argc, char **argv) {
    static const struct option options[] = {
        {"cpus", required_argument, NULL, 'c'},
        {"mib-per-thread", required_argument, NULL, 'm'},
        {"warmup", required_argument, NULL, 'w'},
        {"reps", required_argument, NULL, 'r'},
        {"passes", required_argument, NULL, 'p'},
        {"ops", required_argument, NULL, 'o'},
        {"imc", no_argument, NULL, 'i'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    parse_ops("read,nt-write,nt-copy,nt-triad");
    int option;
    while ((option = getopt_long(argc, argv, "", options, NULL)) != -1) {
        unsigned long long value;
        switch (option) {
        case 'c': parse_cpus(optarg); break;
        case 'm':
            value = parse_number(optarg, false);
            if (value > (SIZE_MAX - 2 * HUGE_ALIGN) / (3 * 1024 * 1024))
                die("array size overflows address space");
            cfg.mib_per_thread = (size_t)value;
            break;
        case 'w': case 'r': case 'p':
            value = parse_number(optarg, option == 'w');
            if (value > INT_MAX / 2)
                die("iteration count is too large");
            if (option == 'w') cfg.warmup = (int)value;
            if (option == 'r') cfg.reps = (int)value;
            if (option == 'p') cfg.passes = (int)value;
            break;
        case 'o': parse_ops(optarg); break;
        case 'i': cfg.imc = true; break;
        case 'h': usage(argv[0]); return EXIT_SUCCESS;
        default: return EXIT_FAILURE;
        }
    }
    if (optind != argc || !cfg.threads)
        die("--cpus is required; positional arguments are not accepted");
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("fma"))
        die("this benchmark requires AVX-512F and FMA");
    cfg.array_bytes = cfg.mib_per_thread * 1024 * 1024;
    if (cfg.array_bytes > SIZE_MAX / (3 * (size_t)cfg.threads))
        die("total array size overflows address space");
    struct worker *workers = calloc((size_t)cfg.threads, sizeof(*workers));
    pthread_t *threads = calloc((size_t)cfg.threads, sizeof(*threads));
    double *seconds = calloc((size_t)cfg.op_count * cfg.reps, sizeof(*seconds));
    struct imc_result *imc_results = calloc((size_t)cfg.op_count * cfg.reps, sizeof(*imc_results));
    if (!workers || !threads || !seconds || !imc_results)
        system_die("allocate metadata");
    if (pthread_barrier_init(&barrier, NULL, (unsigned)cfg.threads + 1))
        die("pthread_barrier_init failed");
    for (int t = 0; t < cfg.threads; ++t) {
        workers[t].id = t;
        int error = pthread_create(&threads[t], NULL, worker_main, &workers[t]);
        if (error) {
            fprintf(stderr, "dram_bench: pthread_create: %s\n", strerror(error));
            return EXIT_FAILURE;
        }
    }
    synchronize();
    struct placement placement = inspect_placement(workers);
    struct imc_state imc = {0};
    struct counter_value counter_before[MAX_IMC_COUNTERS], counter_after[MAX_IMC_COUNTERS];
    imc_setup(&imc);
    for (int oi = 0; oi < cfg.op_count; ++oi) {
        for (int rep = 0; rep < cfg.warmup + cfg.reps; ++rep) {
            synchronize();
            double capture_start = cfg.imc ? monotonic_seconds() : 0.0;
            if (cfg.imc) imc_snapshot(&imc, counter_before);
            double start = monotonic_seconds();
            synchronize();
            synchronize();
            double duration = monotonic_seconds() - start;
            if (cfg.imc) imc_snapshot(&imc, counter_after);
            double capture_duration = cfg.imc ? monotonic_seconds() - capture_start : 0.0;
            synchronize();
            if (rep >= cfg.warmup) {
                seconds[(size_t)oi * cfg.reps + rep - cfg.warmup] = duration;
                if (cfg.imc)
                    imc_results[(size_t)oi * cfg.reps + rep - cfg.warmup] =
                        imc_delta(&imc, counter_before, counter_after, capture_duration);
            }
        }
        synchronize();
    }

    int advice_failures = 0;
    for (int t = 0; t < cfg.threads; ++t)
        advice_failures += workers[t].hugepage_advice_failed;
    printf("{\n  \"schema_version\": 1,\n  \"threads\": %d,\n  \"cpus\": [", cfg.threads);
    for (int t = 0; t < cfg.threads; ++t)
        printf("%s%d", t ? ", " : "", cfg.cpus[t]);
    printf("],\n  \"mib_per_thread\": %zu,\n  \"array_bytes_per_thread\": %zu,\n"
           "  \"allocated_bytes\": %zu,\n  \"warmup\": %d,\n  \"reps\": %d,\n"
           "  \"passes\": %d,\n  \"clock\": \"CLOCK_MONOTONIC_RAW\",\n"
           "  \"dtype\": \"float64\",\n  \"isa\": \"AVX-512F/FMA\",\n"
           "  \"timing\": \"master wall time, includes start/done barriers\",\n"
           "  \"numa_first_touch\": {\n"
           "    \"method\": \"pinned worker initializes all arrays; inherited Linux memory policy\",\n"
           "    \"madvise_hugepage_failures\": %d,\n"
           "    \"numa_maps_readable\": %s,\n    \"smaps_readable\": %s,\n"
           "    \"matched_array_mappings\": %d,\n    \"page_size_bytes\": %ld,\n"
           "    \"anonymous_hugepage_bytes\": %llu,\n"
           "    \"array_pages_by_node\": {",
           cfg.mib_per_thread, cfg.array_bytes, 3 * cfg.array_bytes * cfg.threads,
           cfg.warmup, cfg.reps, cfg.passes, advice_failures,
           placement.numa_maps_readable ? "true" : "false",
           placement.smaps_readable ? "true" : "false", placement.matched_mappings,
           sysconf(_SC_PAGESIZE), placement.anonymous_hugepage_kib * 1024);
    bool comma = false;
    for (int node = 0; node < MAX_NODES; ++node) {
        if (!placement.pages[node])
            continue;
        printf("%s\"N%d\": %llu", comma ? ", " : "", node, placement.pages[node]);
        comma = true;
    }
    printf("}\n  },\n  \"imc\": {\"enabled\": %s", cfg.imc ? "true" : "false");
    if (cfg.imc) {
        printf(", \"counter_count\": %d, \"sockets\": [", imc.count);
        for (int i = 0; i < imc.socket_count; ++i)
            printf("%s%d", i ? ", " : "", imc.sockets[i]);
        printf("], \"bytes_per_event\": 64, \"pmus\": \"uncore_imc_0..7\", "
               "\"read_config\": \"0xcf05\", \"write_config\": \"0xf005\", "
               "\"scaling\": \"per-counter delta_value * delta_time_enabled / delta_time_running\", "
               "\"rate_denominator\": \"imc_capture_seconds; outer interval around sequential snapshots\", "
               "\"scope\": \"all traffic on selected sockets, including background processes; "
               "counter capture is slightly wider than payload timing; not exact benchmark traffic\"");
        printf(", \"counter_delta_columns\": [\"events\", \"time_enabled_ns\", \"time_running_ns\"], "
               "\"counters\": [");
        for (int i = 0; i < imc.count; ++i) {
            struct imc_counter c = imc.counters[i];
            printf("%s{\"pmu\": %d, \"socket\": %d, \"cpu\": %d, \"direction\": \"%s\"}",
                   i ? ", " : "", c.pmu, c.socket, c.cpu, c.is_write ? "write" : "read");
        }
        printf("]");
    }
    printf("},\n  \"results\": [\n");
    bool valid = true;
    for (int oi = 0; oi < cfg.op_count; ++oi) {
        enum operation op = cfg.ops[oi];
        size_t bytes_per_pass = cfg.array_bytes * cfg.threads * streams[op];
        double checksum = 0.0, expected = 0.0;
        uint64_t mismatches = 0;
        for (int t = 0; t < cfg.threads; ++t) {
            checksum += workers[t].checksum[oi];
            expected += workers[t].expected_checksum[oi];
            mismatches += workers[t].mismatches[oi];
        }
        if (mismatches)
            valid = false;
        printf("    {\"op\": \"%s\", \"bytes_per_pass\": %zu, "
               "\"active_working_set_bytes\": %zu, \"seconds\": [",
               op_names[op], bytes_per_pass, bytes_per_pass);
        for (int rep = 0; rep < cfg.reps; ++rep)
            printf("%s%.9f", rep ? ", " : "", seconds[(size_t)oi * cfg.reps + rep]);
        printf("], \"payload_GBps\": [");
        for (int rep = 0; rep < cfg.reps; ++rep)
            printf("%s%.9f", rep ? ", " : "", (double)bytes_per_pass * cfg.passes /
                   seconds[(size_t)oi * cfg.reps + rep] / 1e9);
        if (op == OP_NT_COPY) {
            printf("], \"one_direction_GBps\": [");
            for (int rep = 0; rep < cfg.reps; ++rep)
                printf("%s%.9f", rep ? ", " : "", (double)cfg.array_bytes * cfg.threads *
                       cfg.passes / seconds[(size_t)oi * cfg.reps + rep] / 1e9);
        }
        printf("]");
        if (cfg.imc) {
            const char *fields[] = {"imc_read_GBps", "imc_write_GBps", "imc_total_GBps",
                                    "imc_min_running_fraction", "imc_capture_seconds"};
            for (int field = 0; field < 5; ++field) {
                printf(", \"%s\": [", fields[field]);
                for (int rep = 0; rep < cfg.reps; ++rep) {
                    struct imc_result r = imc_results[(size_t)oi * cfg.reps + rep];
                    double value = field == 0 ? r.read_bytes / r.capture_seconds / 1e9 :
                                   field == 1 ? r.write_bytes / r.capture_seconds / 1e9 :
                                   field == 2 ? (r.read_bytes + r.write_bytes) / r.capture_seconds / 1e9 :
                                   field == 3 ? r.min_running_fraction : r.capture_seconds;
                    printf("%s%.9f", rep ? ", " : "", value);
                }
                printf("]");
            }
            printf(", \"imc_counter_deltas\": [");
            for (int rep = 0; rep < cfg.reps; ++rep) {
                printf("%s[", rep ? ", " : "");
                for (int i = 0; i < imc.count; ++i) {
                    struct counter_value delta = imc_results[(size_t)oi * cfg.reps + rep].deltas[i];
                    printf("%s[%" PRIu64 ", %" PRIu64 ", %" PRIu64 "]", i ? ", " : "",
                           delta.value, delta.enabled, delta.running);
                }
                printf("]");
            }
            printf("]");
        }
        printf(", \"validation\": {\"passed\": %s, \"mismatches\": %" PRIu64
               ", \"checksum\": %.17g, \"expected_checksum\": %.17g}}%s\n",
               mismatches ? "false" : "true", mismatches, checksum, expected,
               oi + 1 < cfg.op_count ? "," : "");
    }
    printf("  ],\n  \"validation_passed\": %s\n}\n", valid ? "true" : "false");
    synchronize();
    for (int t = 0; t < cfg.threads; ++t)
        if (pthread_join(threads[t], NULL))
            die("pthread_join failed");
    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < imc.count; ++i) close(imc.counters[i].fd);
    free(imc_results);
    free(seconds);
    free(threads);
    free(workers);
    if (!valid)
        fprintf(stderr, "dram_bench: result validation FAILED\n");
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
