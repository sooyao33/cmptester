// vramcheck - VRAM integrity + deterministic compute verifier for the unlocked CMP 170HX.
//
// The unlock re-maps memory geometry, exposing far more DRAM than the stock label, and this
// SKU reports no ECC counters. So corruption, if it exists, is silent. This tool makes it loud.
//
// Phase A (memory): moving-inversions over the whole allocation. Each pattern is written, then
//   verified by RECOMPUTING the expected value from the global index in registers - so a wrong
//   readback can only come from storage, not from a stale copy of the expectation.
// Phase B (compute): an iterated splitmix64 chain per element. Integer-only, so it is bit-exact
//   and deterministic - no floating-point rounding ambiguity. Verified three ways:
//     1. a CPU-computed ground-truth sample (independent of the GPU entirely),
//     2. a whole-buffer checksum that must be bit-identical across every round,
//     3. per-element mismatch counting with the first offenders reported by address.
//
// Build: nvcc -O3 -arch=sm_80 -o vramcheck vramcheck.cu
// Exit code 0 = clean, 1 = corruption detected, 2 = setup error.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(2); } } while (0)

struct ErrRec { unsigned long long idx, expected, got; };
static const int MAX_ERR_REC = 32;

__host__ __device__ __forceinline__ uint64_t sm64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// mode: 0=zeros 1=ones 2=0xAA55.. 3=0x55AA.. 4=index-derived pseudorandom
__host__ __device__ __forceinline__ uint64_t pattern(uint64_t gidx, uint64_t seed, int mode, int invert) {
    uint64_t v;
    switch (mode) {
        case 0: v = 0ULL; break;
        case 1: v = ~0ULL; break;
        case 2: v = 0xAA55AA55AA55AA55ULL; break;
        case 3: v = 0x55AA55AA55AA55AAULL; break;
        default: v = sm64(seed ^ gidx); break;
    }
    return invert ? ~v : v;
}

__global__ void kFill(uint64_t* buf, size_t n, uint64_t base, uint64_t seed, int mode, int invert) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        buf[i] = pattern(base + i, seed, mode, invert);
}

__global__ void kVerify(const uint64_t* buf, size_t n, uint64_t base, uint64_t seed, int mode,
                        int invert, unsigned long long* errCount, ErrRec* errs) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint64_t want = pattern(base + i, seed, mode, invert);
        uint64_t got  = buf[i];
        if (got != want) {
            unsigned long long slot = atomicAdd(errCount, 1ULL);
            if (slot < MAX_ERR_REC) {
                errs[slot].idx = base + i;
                errs[slot].expected = want;
                errs[slot].got = got;
            }
        }
    }
}

// Seed the buffer for the compute phase.
__global__ void kSeedWork(uint64_t* buf, size_t n, uint64_t base, uint64_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        buf[i] = sm64(seed ^ (base + i));
}

// Deterministic ALU work: iterate the hash chain in registers, then store.
__global__ void kWork(uint64_t* buf, size_t n, int iters) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint64_t x = buf[i];
        #pragma unroll 4
        for (int k = 0; k < iters; ++k) x = sm64(x);
        buf[i] = x;
    }
}

__global__ void kChecksum(const uint64_t* buf, size_t n, unsigned long long* outXor,
                          unsigned long long* outSum) {
    __shared__ unsigned long long sx[256], ss[256];
    unsigned long long x = 0, s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint64_t v = buf[i];
        x ^= v; s += v;
    }
    sx[threadIdx.x] = x; ss[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) { sx[threadIdx.x] ^= sx[threadIdx.x + off]; ss[threadIdx.x] += ss[threadIdx.x + off]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicXor(outXor, sx[0]); atomicAdd(outSum, ss[0]); }
}

struct Chunk { uint64_t* ptr; size_t n; uint64_t base; };

int main(int argc, char** argv) {
    double gbWanted = 0.0;      // 0 = auto (free - headroom)
    int rounds = 3, iters = 96, sampleN = 4096, inject = 0;
    uint64_t seed = 0xC0FFEEULL;
    double headroomGB = 1.5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](double d) { return (i + 1 < argc) ? atof(argv[++i]) : d; };
        if      (a == "--gb")     gbWanted = next(0);
        else if (a == "--rounds") rounds   = (int)next(3);
        else if (a == "--iters")  iters    = (int)next(96);
        else if (a == "--sample") sampleN  = (int)next(4096);
        else if (a == "--seed")   seed     = (uint64_t)next(0xC0FFEE);
        else if (a == "--inject") inject   = (int)next(0);
        else if (a == "--help") {
            printf("usage: vramcheck [--gb N] [--rounds N] [--iters N] [--sample N] [--seed N] [--inject N]\n");
            printf("  --inject N   deliberately corrupt N elements to prove the checker detects faults\n");
            return 0;
        }
    }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t freeB = 0, totB = 0;
    CHECK(cudaMemGetInfo(&freeB, &totB));

    printf("device      : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("vram        : %.1f GiB total, %.1f GiB free\n", totB / 1073741824.0, freeB / 1073741824.0);

    size_t target = gbWanted > 0 ? (size_t)(gbWanted * 1073741824.0)
                                 : (freeB > (size_t)(headroomGB * 1073741824.0)
                                    ? freeB - (size_t)(headroomGB * 1073741824.0) : 0);
    if (target < (1ULL << 28)) { fprintf(stderr, "not enough free VRAM\n"); return 2; }

    // Chunked allocation: many 1 GiB blocks rather than one giant block.
    const size_t CH = 1ULL << 30;
    std::vector<Chunk> chunks;
    size_t got = 0; uint64_t base = 0;
    while (got < target) {
        size_t want = (target - got < CH) ? (target - got) : CH;
        want &= ~(size_t)7;
        uint64_t* p = nullptr;
        if (cudaMalloc(&p, want) != cudaSuccess) { cudaGetLastError(); break; }
        chunks.push_back({p, want / 8, base});
        base += want / 8; got += want;
    }
    if (chunks.empty()) { fprintf(stderr, "allocation failed\n"); return 2; }
    size_t totalElems = base;
    printf("allocated   : %.2f GiB in %zu chunks (%zu M elements)\n",
           got / 1073741824.0, chunks.size(), (size_t)(totalElems / 1048576));
    printf("params      : rounds=%d iters=%d sample=%d seed=0x%llX\n\n",
           rounds, iters, sampleN, (unsigned long long)seed);

    unsigned long long *dErrCount, *dXor, *dSum;
    ErrRec* dErrs;
    CHECK(cudaMalloc(&dErrCount, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dXor, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dSum, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dErrs, sizeof(ErrRec) * MAX_ERR_REC));

    int block = 256, grid = prop.multiProcessorCount * 16;
    unsigned long long totalErrors = 0;

    auto reportErrs = [&](const char* label, unsigned long long cnt) {
        totalErrors += cnt;
        if (!cnt) { printf("    %-28s OK\n", label); return; }
        printf("    %-28s *** %llu MISMATCHES ***\n", label, cnt);
        std::vector<ErrRec> h(MAX_ERR_REC);
        CHECK(cudaMemcpy(h.data(), dErrs, sizeof(ErrRec) * MAX_ERR_REC, cudaMemcpyDeviceToHost));
        unsigned long long show = cnt < MAX_ERR_REC ? cnt : MAX_ERR_REC;
        for (unsigned long long i = 0; i < show; ++i)
            printf("        elem %llu (byte 0x%llx): expected %016llx got %016llx (xor %016llx)\n",
                   h[i].idx, h[i].idx * 8, h[i].expected, h[i].got, h[i].expected ^ h[i].got);
    };

    // Negative control: flip bits in N spread-out elements so we can prove detection works.
    auto injectFaults = [&](int count) {
        for (int s = 0; s < count; ++s) {
            uint64_t gidx = (uint64_t)(((double)s + 0.5) / count * totalElems);
            for (auto& ch : chunks) {
                if (gidx >= ch.base && gidx < ch.base + ch.n) {
                    uint64_t v = 0;
                    CHECK(cudaMemcpy(&v, ch.ptr + (gidx - ch.base), 8, cudaMemcpyDeviceToHost));
                    v ^= (1ULL << (s % 64));
                    CHECK(cudaMemcpy(ch.ptr + (gidx - ch.base), &v, 8, cudaMemcpyHostToDevice));
                    break;
                }
            }
        }
    };
    if (inject > 0)
        printf(">>> FAULT INJECTION ACTIVE: corrupting %d elements per check (negative control)\n\n", inject);

    // ---------------- Phase A: memory patterns, moving inversions ----------------
    const char* pnames[] = {"zeros", "ones", "0xAA55...", "0x55AA...", "pseudorandom"};
    printf("PHASE A - memory patterns (write, verify, invert, verify)\n");
    for (int mode = 0; mode <= 4; ++mode) {
        for (int inv = 0; inv <= 1; ++inv) {
            for (auto& c : chunks) kFill<<<grid, block>>>(c.ptr, c.n, c.base, seed, mode, inv);
            CHECK(cudaDeviceSynchronize());
            if (inject > 0) injectFaults(inject);
            CHECK(cudaMemset(dErrCount, 0, sizeof(unsigned long long)));
            for (auto& c : chunks)
                kVerify<<<grid, block>>>(c.ptr, c.n, c.base, seed, mode, inv, dErrCount, dErrs);
            CHECK(cudaDeviceSynchronize());
            unsigned long long cnt = 0;
            CHECK(cudaMemcpy(&cnt, dErrCount, sizeof(cnt), cudaMemcpyDeviceToHost));
            char lbl[64];
            snprintf(lbl, sizeof lbl, "%s%s", pnames[mode], inv ? " (inverted)" : "");
            reportErrs(lbl, cnt);
        }
    }

    // ---------------- Phase B: deterministic compute ----------------
    printf("\nPHASE B - deterministic compute (%d hash iterations/element)\n", iters);
    unsigned long long refXor = 0, refSum = 0;
    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    for (int r = 0; r < rounds; ++r) {
        for (auto& c : chunks) kSeedWork<<<grid, block>>>(c.ptr, c.n, c.base, seed);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(t0));
        for (auto& c : chunks) kWork<<<grid, block>>>(c.ptr, c.n, iters);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaDeviceSynchronize());
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (inject > 0) injectFaults(inject);

        CHECK(cudaMemset(dXor, 0, sizeof(unsigned long long)));
        CHECK(cudaMemset(dSum, 0, sizeof(unsigned long long)));
        for (auto& c : chunks) kChecksum<<<grid, block>>>(c.ptr, c.n, dXor, dSum);
        CHECK(cudaDeviceSynchronize());
        unsigned long long hx = 0, hs = 0;
        CHECK(cudaMemcpy(&hx, dXor, sizeof(hx), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&hs, dSum, sizeof(hs), cudaMemcpyDeviceToHost));

        double gops = (double)totalElems * iters / (ms / 1000.0) / 1e9;
        printf("    round %d/%d  xor=%016llx sum=%016llx  %.2f s  %.1f Ghash/s",
               r + 1, rounds, hx, hs, ms / 1000.0, gops);

        if (r == 0) { refXor = hx; refSum = hs; printf("  (reference)\n"); }
        else if (hx == refXor && hs == refSum) printf("  MATCH\n");
        else { printf("  *** CHECKSUM DIVERGED ***\n"); totalErrors++; }

        // Independent CPU ground truth on a spread sample - the GPU is not consulted.
        unsigned long long bad = 0;
        for (int s = 0; s < sampleN; ++s) {
            uint64_t gidx = (uint64_t)((double)s / sampleN * totalElems);
            const Chunk* c = nullptr;
            for (auto& ch : chunks)
                if (gidx >= ch.base && gidx < ch.base + ch.n) { c = &ch; break; }
            if (!c) continue;
            uint64_t want = sm64(seed ^ gidx);
            for (int k = 0; k < iters; ++k) want = sm64(want);
            uint64_t hostGot = 0;
            CHECK(cudaMemcpy(&hostGot, c->ptr + (gidx - c->base), 8, cudaMemcpyDeviceToHost));
            if (hostGot != want) {
                if (bad < 8)
                    printf("        CPU-truth mismatch elem %llu: expected %016llx got %016llx\n",
                           (unsigned long long)gidx, (unsigned long long)want,
                           (unsigned long long)hostGot);
                bad++;
            }
        }
        totalErrors += bad;
        printf("    %-28s %s (%d elements)\n", "cpu ground-truth sample",
               bad ? "*** FAILED ***" : "OK", sampleN);
    }

    for (auto& c : chunks) cudaFree(c.ptr);
    printf("\n==================================================\n");
    if (totalErrors == 0) printf("RESULT: CLEAN - no corruption across %.2f GiB\n", got / 1073741824.0);
    else                  printf("RESULT: FAILED - %llu error(s) detected\n", totalErrors);
    printf("==================================================\n");
    return totalErrors ? 1 : 0;
}
