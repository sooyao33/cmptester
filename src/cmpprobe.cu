// cmpprobe - prove a CMP 170HX unlock with measurements, not with what nvidia-smi claims.
//
//   1. device properties (SM count: 70 old unlock / 74 new unlock / stock is lower)
//   2. cudaMemGetInfo
//   3. allocate every 1 GiB chunk the driver gives, free 2 for headroom (same policy as
//      gpuqual, which avoids the fault-prone top of the unlocked region), write an
//      address-derived pattern into ALL of it, verify it, report errors above/below 8 GiB
//   4. HBM bandwidth (copy kernel, best of N)              ~1.2 TB/s expected
//   5. FP32 FMA throughput (register-only)                 scales with SM count x clock
//   6. tensor cores: FP16 WMMA matmul on exact small integers, every element compared bit-exactly with an
//      independent CUDA-core FP32 matmul + CPU spot checks; reports tensor TFLOPS
//   7. pinned host<->device bandwidth                      reveals the PCIe link (Gen1/2, x1/x4/x16)
//
// Build: nvcc -O3 -gencode arch=compute_80,code=sm_80 -o cmpprobe cmpprobe.cu
// Usage: cmpprobe [--device N] [--headroom-gib 2] [--no-alloc] [--inject]
//        --inject must report exactly 2 bad words (1 below, 1 above 8 GiB) or the checker is broken
// Output: one JSON object per line. Exit 0 = ran clean, 1 = memory errors, 2 = setup error.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("{\"ev\":\"fatal\",\"err\":\"%s\",\"at\":\"%s:%d\"}\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(2); } } while (0)

static const size_t CHUNK = 1ULL << 30;
static const unsigned long long WORDS = CHUNK / 8;

__device__ __forceinline__ unsigned long long mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__global__ void k_fill(unsigned long long *p, unsigned long long base, unsigned long long seed) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < WORDS;
         i += (unsigned long long)gridDim.x * blockDim.x)
        p[i] = mix((base + i) ^ seed);
}

__global__ void k_check(const unsigned long long *p, unsigned long long base, unsigned long long seed,
                        unsigned long long *bad, unsigned long long *first) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < WORDS;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long v = __ldcv(p + i);   // force a DRAM read, not L2
        if (v != mix((base + i) ^ seed)) {
            if (atomicAdd(bad, 1ULL) == 0) *first = base + i;
        }
    }
}

__global__ void k_flip(unsigned long long *p, unsigned long long word, int bit) { p[word] ^= (1ULL << bit); }

__global__ void k_copy(const float4 *__restrict__ a, float4 *__restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b[i] = a[i];
}

__global__ void k_fma(float *out, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f, c = 0.9999f, d = blockIdx.x * 1e-4f;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 16; j++) { a = fmaf(a, b, c); d = fmaf(d, c, b); }
    }
    if (a + d == 12345.678f) out[0] = a + d;   // keep the compiler honest
}


// ---- tensor-core check: integers in [-2,2], K=4096 -> every partial sum exact in fp16 inputs / fp32 accum
static const int TM = 4096;
__host__ __device__ __forceinline__ float av(int i, int k) { return (float)(((i * 7 + k * 3) % 5) - 2); }
__host__ __device__ __forceinline__ float bv(int k, int j) { return (float)(((k * 5 + j * 11) % 5) - 2); }
__global__ void k_tc_fill(__half *A, __half *B) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (size_t)TM * TM; t += (size_t)gridDim.x * blockDim.x) {
        int r = t / TM, c = t % TM; A[t] = __float2half(av(r, c)); B[t] = __float2half(bv(r, c));
    }
}
__global__ void k_wmma(const __half *A, const __half *B, float *C) {   // one warp per 16x16 tile, row-major
    int tn = blockIdx.x, tm = blockIdx.y * 4 + threadIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    for (int k = 0; k < TM; k += 16) {
        wmma::load_matrix_sync(fa, A + (size_t)tm * 16 * TM + k, TM);
        wmma::load_matrix_sync(fb, B + (size_t)k * TM + tn * 16, TM);
        wmma::mma_sync(fc, fa, fb, fc);
    }
    wmma::store_matrix_sync(C + (size_t)tm * 16 * TM + tn * 16, fc, TM, wmma::mem_row_major);
}
__global__ void k_ref(float *C) {   // CUDA cores, recomputes inputs from the formula (no shared state with the TC path)
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    float acc = 0.f;
    for (int k = 0; k < TM; k++) acc = fmaf(av(i, k), bv(k, j), acc);
    C[(size_t)i * TM + j] = acc;
}
__global__ void k_cmp(const float *X, const float *Y, unsigned long long *bad) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (size_t)TM * TM; t += (size_t)gridDim.x * blockDim.x)
        if (__float_as_uint(X[t]) != __float_as_uint(Y[t])) atomicAdd(bad, 1ULL);
}

int main(int argc, char **argv) {
    int dev = 0, headroom = 2, do_alloc = 1, inject = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--headroom-gib") && i + 1 < argc) headroom = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-alloc")) do_alloc = 0;
        else if (!strcmp(argv[i], "--inject")) inject = 1;   // self-test: flip 2 known bits, must be caught
    }
    CK(cudaSetDevice(dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
    int clk = 0, mclk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
    printf("{\"ev\":\"props\",\"device\":%d,\"bus\":\"%s\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,"
           "\"l2_mib\":%d,\"bus_width_bits\":%d,\"sm_clock_mhz\":%d,\"mem_clock_mhz\":%d,\"total_global_mib\":%zu}\n",
           dev, bus, p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize >> 20,
           p.memoryBusWidth, clk / 1000, mclk / 1000, p.totalGlobalMem >> 20);

    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    printf("{\"ev\":\"meminfo\",\"total_mib\":%zu,\"free_mib\":%zu}\n", tot >> 20, fr >> 20);

    int rc = 0;
    if (do_alloc) {
        std::vector<unsigned long long *> ch;
        for (;;) { void *q = nullptr; if (cudaMalloc(&q, CHUNK) != cudaSuccess) { cudaGetLastError(); break; }
                   ch.push_back((unsigned long long *)q); }
        size_t got = ch.size();
        for (int i = 0; i < headroom && !ch.empty(); i++) { cudaFree(ch.back()); ch.pop_back(); }
        unsigned long long *d_bad, *d_first; CK(cudaMalloc(&d_bad, 8)); CK(cudaMalloc(&d_first, 8));
        unsigned long long total_bad = 0, below = 0, above = 0;
        for (int pass = 0; pass < 2; pass++) {
            unsigned long long seed = pass ? ~0x5EEDULL : 0x5EEDULL;   // pass 2 = different data in every cell
            for (size_t c = 0; c < ch.size(); c++) k_fill<<<p.multiProcessorCount * 8, 256>>>(ch[c], c * WORDS, seed);
            CK(cudaDeviceSynchronize());
            if (inject && pass == 0 && !ch.empty()) {      // one flip below 8 GiB, one in the last chunk
                k_flip<<<1, 1>>>(ch[0], 12345, 7);
                k_flip<<<1, 1>>>(ch[ch.size() - 1], WORDS - 777, 63);
                CK(cudaDeviceSynchronize());
                printf("{\"ev\":\"inject\",\"flips\":[[0,12345,7],[%zu,%llu,63]]}\n", ch.size() - 1, WORDS - 777);
            }
            for (size_t c = 0; c < ch.size(); c++) {
                CK(cudaMemset(d_bad, 0, 8));
                k_check<<<p.multiProcessorCount * 8, 256>>>(ch[c], c * WORDS, seed, d_bad, d_first);
                unsigned long long b = 0, f = 0;
                CK(cudaMemcpy(&b, d_bad, 8, cudaMemcpyDeviceToHost));
                if (b) { CK(cudaMemcpy(&f, d_first, 8, cudaMemcpyDeviceToHost));
                         printf("{\"ev\":\"mem_error\",\"pass\":%d,\"chunk\":%zu,\"bad_words\":%llu,\"first_word\":%llu}\n", pass, c, b, f);
                         total_bad += b; if (c < 8) below += b; else above += b; }
            }
        }
        printf("{\"ev\":\"alloc_verify\",\"chunks_obtained\":%zu,\"gib_verified\":%zu,\"passes\":2,"
               "\"bad_words\":%llu,\"bad_below_8gib\":%llu,\"bad_above_8gib\":%llu,\"injected\":%d,\"result\":\"%s\"}\n",
               got, ch.size(), total_bad, below, above, inject, total_bad ? "FAIL" : "CLEAN");
        if (total_bad) rc = 1;
        for (auto q : ch) cudaFree(q);
        cudaFree(d_bad); cudaFree(d_first);
    }

    {   // HBM bandwidth: 2 GiB read + 2 GiB write per pass
        size_t bytes = 2ULL << 30, n = bytes / 16;
        float4 *a, *b; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
        CK(cudaMemset(a, 1, bytes));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int r = 0; r < 8; r++) {
            cudaEventRecord(e0); k_copy<<<p.multiProcessorCount * 16, 256>>>(a, b, n); cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); if (r && ms < best) best = ms;
        }
        printf("{\"ev\":\"hbm_bw\",\"gb_s\":%.1f}\n", 2.0 * bytes / (best * 1e6));
        cudaFree(a); cudaFree(b);
    }
    {   // FP32 FMA: 32 FMA (=64 flop) per inner step
        float *o; CK(cudaMalloc(&o, 4));
        int blocks = p.multiProcessorCount * 8, threads = 256, iters = 4096;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int r = 0; r < 6; r++) {
            cudaEventRecord(e0); k_fma<<<blocks, threads>>>(o, iters); cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); if (r && ms < best) best = ms;
        }
        double flop = (double)blocks * threads * iters * 64.0;
        printf("{\"ev\":\"fp32\",\"tflops\":%.2f}\n", flop / (best * 1e9));
        cudaFree(o);
    }
    {   // tensor cores
        __half *A, *B; float *C, *R; unsigned long long *bad;
        size_t n = (size_t)TM * TM;
        CK(cudaMalloc(&A, n * 2)); CK(cudaMalloc(&B, n * 2)); CK(cudaMalloc(&C, n * 4)); CK(cudaMalloc(&R, n * 4)); CK(cudaMalloc(&bad, 8));
        k_tc_fill<<<p.multiProcessorCount * 8, 256>>>(A, B);
        k_ref<<<dim3(TM / 256, TM), 256>>>(R);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f; unsigned long long totbad = 0;
        for (int r = 0; r < 5; r++) {
            CK(cudaMemset(C, 0xff, n * 4));
            cudaEventRecord(e0); k_wmma<<<dim3(TM / 16, TM / 64), dim3(32, 4)>>>(A, B, C); cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); if (r && ms < best) best = ms;
            if (inject && r == 0) { float v = 99999.f; CK(cudaMemcpy(C + 777777, &v, 4, cudaMemcpyHostToDevice)); }
            CK(cudaMemset(bad, 0, 8)); k_cmp<<<p.multiProcessorCount * 8, 256>>>(C, R, bad);
            unsigned long long b = 0; CK(cudaMemcpy(&b, bad, 8, cudaMemcpyDeviceToHost)); totbad += b;
        }
        std::vector<float> row(TM); int cpubad = 0;
        for (int s = 0; s < 16; s++) {   // CPU ground truth for 16 rows x 16 columns
            int i = (s * 997) % TM; CK(cudaMemcpy(row.data(), C + (size_t)i * TM, TM * 4, cudaMemcpyDeviceToHost));
            for (int q = 0; q < 16; q++) { int j = (q * 263 + s) % TM; double acc = 0; for (int k = 0; k < TM; k++) acc += av(i, k) * bv(k, j);
                                           if ((float)acc != row[j]) cpubad++; }
        }
        printf("{\"ev\":\"tensor\",\"tflops_fp16\":%.1f,\"runs\":5,\"elements_compared\":%zu,\"bad_vs_cudacore\":%llu,\"bad_vs_cpu\":%d,\"injected\":%d,\"result\":\"%s\"}\n",
               2.0 * TM * TM * (double)TM / (best * 1e9), n * 5, totbad, cpubad, inject, (totbad || cpubad) ? "FAIL" : "CLEAN");
        if (totbad || cpubad) rc = 1;
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(R); cudaFree(bad);
    }
    {   // PCIe pinned transfers
        size_t bytes = 256ULL << 20; void *h, *d;
        CK(cudaMallocHost(&h, bytes)); CK(cudaMalloc(&d, bytes));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float h2d = 1e30f, d2h = 1e30f, ms;
        for (int r = 0; r < 4; r++) {
            cudaEventRecord(e0); CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); cudaEventRecord(e1);
            cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (ms < h2d) h2d = ms;
            cudaEventRecord(e0); CK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)); cudaEventRecord(e1);
            cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (ms < d2h) d2h = ms;
        }
        printf("{\"ev\":\"pcie_bw\",\"h2d_gb_s\":%.2f,\"d2h_gb_s\":%.2f}\n", bytes / (h2d * 1e6), bytes / (d2h * 1e6));
        cudaFreeHost(h); cudaFree(d);
    }
    printf("{\"ev\":\"done\",\"rc\":%d}\n", rc);
    return rc;
}
