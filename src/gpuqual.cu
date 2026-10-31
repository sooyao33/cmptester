// gpuqual.cu -- GPU memory / compute error-detection worker for NVIDIA CMP 170HX (GA100, 64GiB, no ECC)
//
// Emits newline-delimited JSON events on stdout. The Python supervisor owns all
// persistent state; this worker is a single "session" that runs until its budget
// expires, it is told to stop, or its CUDA context dies.
//
// HARD RULE: this program NEVER writes any power management setting. It reads
// power state via NVML for telemetry and alarms only.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdarg>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <csignal>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvml.h>

// ---------------------------------------------------------------- constants

#define CHUNK_BYTES   (1ULL << 30)      // 1 GiB per allocation chunk
#define U4_PER_CHUNK  (CHUNK_BYTES / 16ULL)
#define MAX_CHUNKS    96
#define ERR_CAP       4096              // device-side error record capacity
#define MAX_TRIAGE_PER_PASS 64          // bound triage work if the card is a disaster
#define STOCK_FB_BYTES (0x200000000ULL) // 8 GiB -- stock CMP 170HX framebuffer

// Invertible 32-bit transform constants (TINV computed+asserted on host)
#define TMUL 0x2545F491u
#define TROT 13

enum Pattern { PAT_PRNG=0, PAT_ZERO, PAT_ONES, PAT_A5, PAT_5A, PAT_ADDR, PAT_WALK1, PAT_NUM };
static const char* PAT_NAME[] = {"prng","zero","ones","a5","5a","addr","walk1"};

enum Phase { PH_FILL=0, PH_MOVINV, PH_RMW, PH_SCATTER, PH_ALU, PH_RETENTION, PH_KNOWNBAD };
static const char* PH_NAME[] = {"P0_fill","P1_movinv","P2_rmw","P3_scatter","P4_alu_soak","P5_retention","PX_knownbad"};

// ---------------------------------------------------------------- device side

struct ErrRec {
    unsigned long long gidx;   // global uint4 index across the whole array
    unsigned int chunk;
    unsigned int phase;
    uint4 exp;
    uint4 obs;
};

struct Triage {
    uint4 rd_cached;      // normal load (may hit L2)
    uint4 rd_uncached;    // ld.global.cv -- bypasses caches, forces DRAM read
    uint4 after_rewrite;  // uncached read after writing the correct value back
};

__host__ __device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// The golden reference: every 16-byte word is a pure function of its global
// index. No storage anywhere -- this is what makes 64 GiB verifiable.
__device__ __forceinline__ uint4 pattern_val(unsigned long long gidx,
                                             unsigned long long seed,
                                             int pat, int inv) {
    uint4 r;
    switch (pat) {
    case PAT_PRNG: {
        unsigned long long a = splitmix64((gidx << 1) ^ seed);
        unsigned long long b = splitmix64(((gidx << 1) | 1ULL) ^ seed);
        r = make_uint4((unsigned)a, (unsigned)(a >> 32), (unsigned)b, (unsigned)(b >> 32));
        break; }
    case PAT_ZERO:  r = make_uint4(0u, 0u, 0u, 0u); break;
    case PAT_ONES:  r = make_uint4(~0u, ~0u, ~0u, ~0u); break;
    case PAT_A5:    r = make_uint4(0xAAAAAAAAu, 0x55555555u, 0xAAAAAAAAu, 0x55555555u); break;
    case PAT_5A:    r = make_uint4(0x55555555u, 0xAAAAAAAAu, 0x55555555u, 0xAAAAAAAAu); break;
    case PAT_ADDR: {
        unsigned lo = (unsigned)gidx, hi = (unsigned)(gidx >> 32);
        r = make_uint4(lo, hi, ~lo, ~hi); break; }
    case PAT_WALK1: {
        unsigned w = 1u << (unsigned)(gidx & 31ULL);
        r = make_uint4(w, ~w, w, ~w); break; }
    default: r = make_uint4(0u, 0u, 0u, 0u);
    }
    if (inv) { r.x = ~r.x; r.y = ~r.y; r.z = ~r.z; r.w = ~r.w; }
    return r;
}

__device__ __forceinline__ bool u4ne(const uint4 &a, const uint4 &b) {
    return (a.x ^ b.x) | (a.y ^ b.y) | (a.z ^ b.z) | (a.w ^ b.w);
}

__device__ __forceinline__ void record_err(unsigned long long *errcount,
                                           ErrRec *recs, unsigned int *nrec,
                                           unsigned long long gidx, unsigned chunk,
                                           unsigned phase, uint4 e, uint4 o) {
    atomicAdd(errcount, 1ULL);
    unsigned s = atomicAdd(nrec, 1u);
    if (s < ERR_CAP) {
        ErrRec r; r.gidx = gidx; r.chunk = chunk; r.phase = phase; r.exp = e; r.obs = o;
        recs[s] = r;
    }
}

__device__ __forceinline__ uint4 ld_uncached(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

// --- P0: fill -------------------------------------------------------------
__global__ void k_fill(uint4 *p, unsigned long long n, unsigned long long base,
                       unsigned long long seed, int pat, int inv) {
    unsigned long long i  = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long st = gridDim.x * (unsigned long long)blockDim.x;
    for (; i < n; i += st) p[i] = pattern_val(base + i, seed, pat, inv);
}

// --- P0/P5: pure verify ---------------------------------------------------
__global__ void k_verify(const uint4 * __restrict__ p, unsigned long long n,
                         unsigned long long base, unsigned long long seed,
                         int pat, int inv, unsigned chunk, unsigned phase,
                         unsigned long long *errcount, ErrRec *recs, unsigned *nrec) {
    unsigned long long i  = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long st = gridDim.x * (unsigned long long)blockDim.x;
    for (; i < n; i += st) {
        uint4 e = pattern_val(base + i, seed, pat, inv);
        uint4 v = p[i];
        if (u4ne(v, e)) record_err(errcount, recs, nrec, base + i, chunk, phase, e, v);
    }
}

// --- P1: moving inversions (verify pattern, write its complement) ---------
// dir=0 ascending, dir=1 descending -- catches address-order-dependent coupling.
__global__ void k_movinv(uint4 *p, unsigned long long n, unsigned long long base,
                         unsigned long long seed, int pat, int inv, int dir,
                         unsigned chunk, unsigned phase,
                         unsigned long long *errcount, ErrRec *recs, unsigned *nrec) {
    unsigned long long i  = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long st = gridDim.x * (unsigned long long)blockDim.x;
    for (; i < n; i += st) {
        unsigned long long idx = dir ? (n - 1ULL - i) : i;
        uint4 e = pattern_val(base + idx, seed, pat, inv);
        uint4 v = p[idx];
        if (u4ne(v, e)) record_err(errcount, recs, nrec, base + idx, chunk, phase, e, v);
        p[idx] = pattern_val(base + idx, seed, pat, !inv);
    }
}

// --- P2: read-modify-write transform hammer -------------------------------
// State at round r is tmix(golden, K_r). The value written is derived from the
// value actually READ, so there is a genuine data dependency through DRAM.
__device__ __forceinline__ unsigned rotl32(unsigned x, int k) { return (x << k) | (x >> (32 - k)); }
__device__ __forceinline__ unsigned rotr32(unsigned x, int k) { return (x >> k) | (x << (32 - k)); }
__device__ __forceinline__ unsigned tmix(unsigned x, unsigned k)   { return rotl32((x ^ k) * TMUL, TROT); }
__device__ __forceinline__ unsigned tunmix(unsigned y, unsigned k, unsigned tinv) { return (rotr32(y, TROT) * tinv) ^ k; }

__device__ __forceinline__ uint4 tmix4(uint4 g, unsigned k) {
    return make_uint4(tmix(g.x, k), tmix(g.y, k), tmix(g.z, k), tmix(g.w, k));
}

// Prime the array directly into round-r transform space.
__global__ void k_fill_rmw(uint4 *p, unsigned long long n, unsigned long long base,
                           unsigned long long seed, int pat, unsigned kr) {
    unsigned long long i  = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long st = gridDim.x * (unsigned long long)blockDim.x;
    for (; i < n; i += st) p[i] = tmix4(pattern_val(base + i, seed, pat, 0), kr);
}

__global__ void k_rmw(uint4 *p, unsigned long long n, unsigned long long base,
                      unsigned long long seed, int pat, unsigned kr, unsigned krn,
                      unsigned tinv, unsigned chunk, unsigned phase,
                      unsigned long long *errcount, ErrRec *recs, unsigned *nrec) {
    unsigned long long i  = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long st = gridDim.x * (unsigned long long)blockDim.x;
    for (; i < n; i += st) {
        uint4 g = pattern_val(base + i, seed, pat, 0);
        uint4 e = tmix4(g, kr);
        uint4 v = p[i];
        if (u4ne(v, e)) record_err(errcount, recs, nrec, base + i, chunk, phase, e, v);
        // derive the next state from what we actually read (true RMW chain)
        uint4 w = make_uint4(tmix(tunmix(v.x, kr, tinv), krn),
                             tmix(tunmix(v.y, kr, tinv), krn),
                             tmix(tunmix(v.z, kr, tinv), krn),
                             tmix(tunmix(v.w, kr, tinv), krn));
        p[i] = w;
    }
}

// --- P3: scattered / random-access verify ---------------------------------
// Streaming access hits DRAM rows very differently from random access. This
// drives a much higher row-activation rate against the same data.
__global__ void k_scatter(const uint4 * __restrict__ p, unsigned long long n,
                          unsigned long long base, unsigned long long seed,
                          int pat, unsigned long long salt, unsigned iters,
                          unsigned chunk, unsigned phase,
                          unsigned long long *errcount, ErrRec *recs, unsigned *nrec) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long s = splitmix64(tid ^ salt);
    for (unsigned k = 0; k < iters; k++) {
        s = splitmix64(s);
        unsigned long long idx = s % n;
        uint4 e = pattern_val(base + idx, seed, pat, 0);
        uint4 v = ld_uncached(p + idx);   // force DRAM, defeat the 32MB L2
        if (u4ne(v, e)) record_err(errcount, recs, nrec, base + idx, chunk, phase, e, v);
    }
}

// --- P4: ALU known-answer test (silent data corruption detector) -----------
// x_{n+1} = A*x_n + C  has closed form  x_n = A^n*x0 + C*S_n,  S_n = sum A^i.
// Pure registers, no memory traffic, 64-bit IMADs -> maximum power draw.
__device__ __forceinline__ void geo_pow(unsigned long long A, unsigned long long nn,
                                        unsigned long long &An, unsigned long long &Sn) {
    unsigned long long rA = 1ULL, rS = 0ULL, cA = A, cS = 1ULL, k = nn;
    while (k) {
        if (k & 1ULL) { rS = rS + rA * cS; rA = rA * cA; }
        cS = cS + cA * cS;   // S(2m) = S(m)*(1 + A^m)
        cA = cA * cA;
        k >>= 1;
    }
    An = rA; Sn = rS;
}

__global__ void k_alu(unsigned long long A, unsigned long long C,
                      unsigned long long iters, unsigned long long salt,
                      unsigned long long *badcount, ErrRec *recs, unsigned *nrec,
                      int inject) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long x0 = splitmix64(tid ^ salt);
    unsigned long long x = x0;
    for (unsigned long long i = 0; i < iters; i++) x = A * x + C;
    if (inject && tid == 12345ULL) x ^= 0x40ULL;    // self-test: fake an SDC
    unsigned long long An, Sn; geo_pow(A, iters, An, Sn);
    unsigned long long e = An * x0 + C * Sn;
    if (x != e) {
        atomicAdd(badcount, 1ULL);
        unsigned s = atomicAdd(nrec, 1u);
        if (s < ERR_CAP) {
            ErrRec r; r.gidx = tid; r.chunk = 0xFFFFFFFFu; r.phase = PH_ALU;
            r.exp = make_uint4((unsigned)e, (unsigned)(e >> 32), 0u, 0u);
            r.obs = make_uint4((unsigned)x, (unsigned)(x >> 32), 0u, 0u);
            recs[s] = r;
        }
    }
}

// --- triage ladder --------------------------------------------------------
// Runs on a single thread against one address. Ordering matters: read before
// any write, uncached read to force a real DRAM access, then rewrite + reread.
__global__ void k_triage(uint4 *addr, uint4 expected, Triage *out, int sabotage) {
    out->rd_cached   = *addr;
    out->rd_uncached = ld_uncached(addr);
    *addr = expected;
    __threadfence();
    if (sabotage) {                 // self-test only: emulate a stuck cell
        uint4 s = expected; s.x ^= 1u;
        *addr = s;
        __threadfence();
    }
    out->after_rewrite = ld_uncached(addr);
}

// --- fault injection (self-test) ------------------------------------------
__global__ void k_inject(uint4 *p, unsigned long long idx, unsigned lane, unsigned bit) {
    uint4 v = p[idx];
    if      (lane == 0) v.x ^= (1u << bit);
    else if (lane == 1) v.y ^= (1u << bit);
    else if (lane == 2) v.z ^= (1u << bit);
    else                v.w ^= (1u << bit);
    p[idx] = v;
}

// ---------------------------------------------------------------- host side

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double wall_s(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void emit(const char *fmt, ...) {
    char body[4096];
    va_list ap; va_start(ap, fmt); vsnprintf(body, sizeof(body), fmt, ap); va_end(ap);
    printf("{\"t\":%.3f,%s}\n", wall_s(), body);
    fflush(stdout);
}

#define CUCK(call, what) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    emit("\"ev\":\"cuda_error\",\"where\":\"%s\",\"code\":%d,\"msg\":\"%s\"", \
         what, (int)_e, cudaGetErrorString(_e)); return -1; } } while (0)

// ---- NVML telemetry ------------------------------------------------------
struct Tele {
    unsigned gpu_t, mem_t, pwr_mw, plimit_mw, sm_clk, mem_clk, util;
    unsigned long long throttle;
    int ok;
    int temp_ok;      // GPU temperature specifically was read successfully
    int memtemp_ok;   // HBM temperature was read successfully
};
static nvmlDevice_t g_nvml;
static int g_nvml_ok = 0;

static Tele sample_tele(void) {
    Tele t; memset(&t, 0, sizeof(t));
    if (!g_nvml_ok) return t;
    unsigned v;
    nvmlTemperature_t tv; memset(&tv, 0, sizeof(tv));
    tv.version = nvmlTemperature_v1; tv.sensorType = NVML_TEMPERATURE_GPU;
    if (nvmlDeviceGetTemperatureV(g_nvml, &tv) == NVML_SUCCESS) {
        t.gpu_t = (unsigned)tv.temperature; t.temp_ok = 1;
    }
    if (nvmlDeviceGetPowerUsage(g_nvml, &v) == NVML_SUCCESS) t.pwr_mw = v;
    if (nvmlDeviceGetPowerManagementLimit(g_nvml, &v) == NVML_SUCCESS) t.plimit_mw = v;
    if (nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &v) == NVML_SUCCESS) t.sm_clk = v;
    if (nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_MEM, &v) == NVML_SUCCESS) t.mem_clk = v;
    nvmlUtilization_t u;
    if (nvmlDeviceGetUtilizationRates(g_nvml, &u) == NVML_SUCCESS) t.util = u.gpu;
    unsigned long long thr;
    if (nvmlDeviceGetCurrentClocksEventReasons(g_nvml, &thr) == NVML_SUCCESS) t.throttle = thr;
    nvmlFieldValue_t fv; memset(&fv, 0, sizeof(fv));
    fv.fieldId = NVML_FI_DEV_MEMORY_TEMP;
    if (nvmlDeviceGetFieldValues(g_nvml, 1, &fv) == NVML_SUCCESS && fv.nvmlReturn == NVML_SUCCESS) {
        t.mem_t = (unsigned)fv.value.siVal; t.memtemp_ok = 1;
    }
    t.ok = 1;
    return t;
}
static void emit_tele(const char *tag, const Tele &t) {
    emit("\"ev\":\"telemetry\",\"tag\":\"%s\",\"gpu_c\":%u,\"mem_c\":%u,\"pwr_w\":%.1f,"
         "\"plimit_w\":%.1f,\"sm_mhz\":%u,\"mem_mhz\":%u,\"util\":%u,\"throttle\":\"0x%llx\"",
         tag, t.gpu_t, t.mem_t, t.pwr_mw / 1000.0, t.plimit_mw / 1000.0,
         t.sm_clk, t.mem_clk, t.util, t.throttle);
}

// ---- config --------------------------------------------------------------
struct Cfg {
    double budget_s      = 3600.0;   // how long this session may run
    double scale         = 1.0;      // phase-duration scale (smoke mode)
    int    supercycle0   = 0;
    unsigned long long seed0 = 0x1234ABCDULL;
    int    size_gib      = 0;        // 0 = greedy auto
    // thermal policy (user-supplied empirical numbers for this card)
    double t_target      = 69.0;     // known-good steady state at 140W
    double t_drift       = 76.0;     // sustained 5min above this => THERMAL_DRIFT finding
    double t_backoff     = 78.0;     // duty-cycle backoff engages
    double t_critical    = 82.0;     // alarm level
    double t_pause       = 80.0;     // stop GPU work and wait for cooldown
    double t_halt        = 84.0;     // stop the run entirely; do NOT auto-resume
    double t_resume      = 70.0;     // resume only after cooling back below this
    double t_mem_pause   = 84.0;     // HBM runs ~5C hotter than the core
    double t_mem_halt    = 88.0;
    double plimit_w      = 140.0;    // expected; any deviation is CRITICAL
    int    self_test     = 0;        // inject faults to qualify the harness
    int    device        = 0;        // which physical GPU to test
    std::vector<unsigned long long> known_bad;
};

// ---- global device resources --------------------------------------------
static std::vector<void*>       g_chunks;
static unsigned long long      *d_errcount = nullptr;
static unsigned long long      *d_alubad   = nullptr;
static ErrRec                  *d_recs     = nullptr;
static unsigned int            *d_nrec     = nullptr;
static Triage                  *d_triage   = nullptr;
static int g_grid = 560, g_block = 256;
static unsigned g_tinv = 0;

// running tallies for this session
static unsigned long long s_mismatch = 0, s_hard = 0, s_soft = 0, s_transient = 0, s_sdc = 0;
static unsigned long long s_bitreads = 0, s_aluops = 0;
static double s_backoff_s = 0;
static std::vector<unsigned long long> s_newbad;

static unsigned popcnt4(uint4 a, uint4 b) {
    return __builtin_popcount(a.x ^ b.x) + __builtin_popcount(a.y ^ b.y)
         + __builtin_popcount(a.z ^ b.z) + __builtin_popcount(a.w ^ b.w);
}

// Map a global uint4 index to a byte offset within the tested array. This is a
// virtual-address-space offset, not a physical one, but it is the only handle
// we have -- and crucially it lets us bucket errors above/below the 8 GiB
// stock framebuffer boundary that the driver unlock extends past.
static unsigned long long gidx_to_off(unsigned long long gidx) { return gidx * 16ULL; }

static int run_triage(const Cfg &cfg, const ErrRec &r, const char *phase_name) {
    unsigned chunk = r.chunk;
    if (chunk >= g_chunks.size()) return 0;
    unsigned long long local = r.gidx - (unsigned long long)chunk * U4_PER_CHUNK;
    uint4 *addr = ((uint4*)g_chunks[chunk]) + local;

    int sabotage = 0;
    if (cfg.self_test) {   // emulate a stuck cell for exactly one seeded address
        for (size_t i = 0; i < cfg.known_bad.size(); i++)
            if (cfg.known_bad[i] == r.gidx) sabotage = 1;
    }

    Triage h; memset(&h, 0, sizeof(h));
    k_triage<<<1,1>>>(addr, r.exp, d_triage, sabotage);
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    if (cudaMemcpy(&h, d_triage, sizeof(Triage), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;

    // Independent read back over PCIe -- a second, physically different path.
    uint4 hostread; memset(&hostread, 0, sizeof(hostread));
    cudaMemcpy(&hostread, addr, sizeof(uint4), cudaMemcpyDeviceToHost);

    int ok_cached   = !memcmp(&h.rd_cached,     &r.exp, sizeof(uint4));
    int ok_uncached = !memcmp(&h.rd_uncached,   &r.exp, sizeof(uint4));
    int ok_rewrite  = !memcmp(&h.after_rewrite, &r.exp, sizeof(uint4));
    int ok_host     = !memcmp(&hostread,        &r.exp, sizeof(uint4));

    const char *cls;
    if (!ok_rewrite)                     { cls = "HARD_FAULT";     s_hard++; s_newbad.push_back(r.gidx); }
    else if (ok_cached && ok_uncached)   { cls = "TRANSIENT_READ"; s_transient++; }
    else                                 { cls = "SOFT_FLIP";      s_soft++; }

    Tele t = sample_tele();
    unsigned long long off = gidx_to_off(r.gidx);
    emit("\"ev\":\"triage\",\"class\":\"%s\",\"phase\":\"%s\",\"gidx\":%llu,\"chunk\":%u,"
         "\"byte_off\":%llu,\"above_stock_fb\":%d,"
         "\"exp\":\"%08x%08x%08x%08x\",\"obs\":\"%08x%08x%08x%08x\","
         "\"xor\":\"%08x%08x%08x%08x\",\"bits\":%u,"
         "\"reread_cached_ok\":%d,\"reread_uncached_ok\":%d,\"host_read_ok\":%d,\"rewrite_ok\":%d,"
         "\"gpu_c\":%u,\"mem_c\":%u,\"pwr_w\":%.1f,\"sm_mhz\":%u",
         cls, phase_name, r.gidx, r.chunk, off, off >= STOCK_FB_BYTES ? 1 : 0,
         r.exp.w, r.exp.z, r.exp.y, r.exp.x, r.obs.w, r.obs.z, r.obs.y, r.obs.x,
         r.exp.w ^ r.obs.w, r.exp.z ^ r.obs.z, r.exp.y ^ r.obs.y, r.exp.x ^ r.obs.x,
         popcnt4(r.exp, r.obs),
         ok_cached, ok_uncached, ok_host, ok_rewrite,
         t.gpu_t, t.mem_t, t.pwr_mw / 1000.0, t.sm_clk);
    return 0;
}

// Drain device error records after a kernel batch, triage each one.
static int drain_errors(const Cfg &cfg, unsigned phase) {
    unsigned h_n = 0; unsigned long long h_c = 0;
    if (cudaMemcpy(&h_n, d_nrec, sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (cudaMemcpy(&h_c, d_errcount, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (h_c == 0) return 0;

    s_mismatch += h_c;
    unsigned ncap = h_n > ERR_CAP ? ERR_CAP : h_n;
    std::vector<ErrRec> recs(ncap);
    if (ncap) cudaMemcpy(recs.data(), d_recs, ncap * sizeof(ErrRec), cudaMemcpyDeviceToHost);

    emit("\"ev\":\"mismatch_batch\",\"phase\":\"%s\",\"count\":%llu,\"recorded\":%u,\"truncated\":%d",
         PH_NAME[phase], h_c, ncap, h_n > ERR_CAP ? 1 : 0);

    unsigned lim = ncap < MAX_TRIAGE_PER_PASS ? ncap : MAX_TRIAGE_PER_PASS;
    for (unsigned i = 0; i < lim; i++) run_triage(cfg, recs[i], PH_NAME[phase]);
    if (ncap > lim)
        emit("\"ev\":\"triage_capped\",\"phase\":\"%s\",\"triaged\":%u,\"available\":%u", PH_NAME[phase], lim, ncap);

    cudaMemset(d_nrec, 0, sizeof(unsigned));
    cudaMemset(d_errcount, 0, sizeof(unsigned long long));
    return 0;
}

// ---- thermal governor ----------------------------------------------------
// Never touches the power limit. Enforces headroom purely by inserting idle
// time. Also treats drift away from the known-good 69C baseline as a finding.
struct Governor {
    double duty = 1.0;
    double drift_since = -1;
    int    drift_flagged = 0;
    unsigned peak_gpu = 0, peak_mem = 0;
    double last_plimit_w = -1;   // last observed limit, to alarm only on change
    int    tele_fail = 0;        // consecutive failed telemetry reads
    int    pauses = 0;           // thermal pauses this session
};

// Set when the run must stop for thermal safety. The card is left idle and the
// supervisor is told (via exit code) NOT to restart the worker.
static int s_thermal_halt = 0;
#define EXIT_THERMAL_HALT 20

static void govern(const Cfg &cfg, Governor &gv, double batch_s) {
    Tele t = sample_tele();

    // FAIL-SAFE: a temperature we cannot read must be treated as dangerous, never
    // as 0 C. A silently dead sensor is exactly how a card cooks unattended.
    if (!t.ok || !t.temp_ok) {
        gv.tele_fail++;
        if (gv.tele_fail == 3)
            emit("\"ev\":\"CRITICAL\",\"kind\":\"telemetry_blind\",\"fails\":%d,\"note\":"
                 "\"cannot read GPU temperature; throttling to minimum as a precaution\"",
                 gv.tele_fail);
        gv.duty = 0.05;
        usleep(500000);
        if (gv.tele_fail >= 20) {
            emit("\"ev\":\"CRITICAL\",\"kind\":\"THERMAL_HALT\",\"reason\":\"telemetry_blind\","
                 "\"fails\":%d,\"note\":\"halting: cannot verify card temperature\"", gv.tele_fail);
            s_thermal_halt = 1; g_stop = 1;
        }
        return;
    }
    gv.tele_fail = 0;

    // HALT: stop the run outright and leave the card idle. Never auto-resumes.
    if (t.gpu_t >= cfg.t_halt || (t.memtemp_ok && t.mem_t >= cfg.t_mem_halt)) {
        emit("\"ev\":\"CRITICAL\",\"kind\":\"THERMAL_HALT\",\"reason\":\"over_limit\","
             "\"gpu_c\":%u,\"mem_c\":%u,\"halt_gpu_c\":%.0f,\"halt_mem_c\":%.0f,\"pwr_w\":%.1f,"
             "\"note\":\"halting run; card left idle and will NOT auto-resume\"",
             t.gpu_t, t.mem_t, cfg.t_halt, cfg.t_mem_halt, t.pwr_mw / 1000.0);
        s_thermal_halt = 1; g_stop = 1;
        return;
    }

    // PAUSE: stop all GPU work and wait for a real cooldown before continuing.
    if (t.gpu_t >= cfg.t_pause || (t.memtemp_ok && t.mem_t >= cfg.t_mem_pause)) {
        gv.pauses++;
        emit("\"ev\":\"CRITICAL\",\"kind\":\"THERMAL_PAUSE\",\"gpu_c\":%u,\"mem_c\":%u,"
             "\"pause_c\":%.0f,\"resume_below_c\":%.0f,\"pause_num\":%d",
             t.gpu_t, t.mem_t, cfg.t_pause, cfg.t_resume, gv.pauses);
        double p0 = now_s();
        int fatal = 0;
        while (!g_stop) {
            usleep(2000000);
            Tele c = sample_tele();
            if (!c.ok || !c.temp_ok) {
                if (now_s() - p0 > 120.0) { fatal = 1; break; }
                continue;
            }
            if (c.gpu_t >= cfg.t_halt || (c.memtemp_ok && c.mem_t >= cfg.t_mem_halt)) { fatal = 1; break; }
            if (c.gpu_t <= cfg.t_resume && (!c.memtemp_ok || c.mem_t <= cfg.t_resume + 6.0)) break;
            // Idle and still not cooling means the cooling itself has failed.
            if (now_s() - p0 > 900.0) { fatal = 1; break; }
        }
        Tele c2 = sample_tele();
        if (fatal || gv.pauses > 20) {
            emit("\"ev\":\"CRITICAL\",\"kind\":\"THERMAL_HALT\",\"reason\":\"%s\","
                 "\"gpu_c\":%u,\"mem_c\":%u,\"pauses\":%d,\"waited_s\":%.0f,"
                 "\"note\":\"halting run; card left idle and will NOT auto-resume\"",
                 gv.pauses > 20 ? "too_many_pauses" : "did_not_cool",
                 c2.gpu_t, c2.mem_t, gv.pauses, now_s() - p0);
            s_thermal_halt = 1; g_stop = 1;
            return;
        }
        emit("\"ev\":\"thermal_resume\",\"gpu_c\":%u,\"mem_c\":%u,\"cooled_for_s\":%.0f",
             c2.gpu_t, c2.mem_t, now_s() - p0);
        s_backoff_s += now_s() - p0;
        gv.duty = 0.20;   // resume gently rather than straight back to full load
    }

    if (t.gpu_t > gv.peak_gpu) gv.peak_gpu = t.gpu_t;
    if (t.mem_t > gv.peak_mem) gv.peak_mem = t.mem_t;

    // The hazard this guards is a limit set ABOVE the safe ceiling. A limit at
    // or below it is an operator choice and is safe by definition -- record it
    // once as a change in test conditions, never alarm, and never re-emit while
    // it holds steady (this fires per kernel batch).
    if (cfg.plimit_w > 0 && t.plimit_mw > 0) {
        double lw = t.plimit_mw / 1000.0;
        if (fabs(lw - gv.last_plimit_w) > 0.5) {
            if (lw > cfg.plimit_w + 0.5)
                emit("\"ev\":\"CRITICAL\",\"kind\":\"power_limit_above_ceiling\","
                     "\"ceiling_w\":%.1f,\"observed_w\":%.1f", cfg.plimit_w, lw);
            else
                emit("\"ev\":\"config_change\",\"kind\":\"power_limit_set\",\"observed_w\":%.1f,"
                     "\"ceiling_w\":%.1f,\"note\":\"at or below safe ceiling; test condition recorded\"",
                     lw, cfg.plimit_w);
            gv.last_plimit_w = lw;
        }
    }
    if (t.gpu_t >= cfg.t_critical)
        emit("\"ev\":\"CRITICAL\",\"kind\":\"thermal\",\"gpu_c\":%u,\"mem_c\":%u,\"limit_c\":%.0f",
             t.gpu_t, t.mem_t, cfg.t_critical);

    // sustained drift above the known-good baseline => cooling degradation
    if (t.gpu_t >= cfg.t_drift) {
        if (gv.drift_since < 0) gv.drift_since = now_s();
        else if (!gv.drift_flagged && now_s() - gv.drift_since > 300.0) {
            gv.drift_flagged = 1;
            emit("\"ev\":\"finding\",\"kind\":\"THERMAL_DRIFT\",\"gpu_c\":%u,\"mem_c\":%u,"
                 "\"baseline_c\":%.0f,\"sustained_s\":%.0f,\"pwr_w\":%.1f",
                 t.gpu_t, t.mem_t, cfg.t_target, now_s() - gv.drift_since, t.pwr_mw / 1000.0);
        }
    } else { gv.drift_since = -1; gv.drift_flagged = 0; }

    double old = gv.duty;
    int hot = (t.gpu_t >= cfg.t_backoff) ||
              (t.memtemp_ok && t.mem_t >= cfg.t_backoff + 5.0);
    int cool = (t.gpu_t <= cfg.t_backoff - 3.0) &&
               (!t.memtemp_ok || t.mem_t <= cfg.t_backoff + 2.0);
    if (hot)       gv.duty = std::max(0.05, gv.duty * 0.70);
    else if (cool) gv.duty = std::min(1.0, gv.duty * 1.10);

    if (gv.duty < 0.999 && old >= 0.999)
        emit("\"ev\":\"backoff_start\",\"gpu_c\":%u,\"mem_c\":%u,\"duty\":%.3f,\"threshold_c\":%.0f",
             t.gpu_t, t.mem_t, gv.duty, cfg.t_backoff);
    if (gv.duty >= 0.999 && old < 0.999)
        emit("\"ev\":\"backoff_end\",\"gpu_c\":%u,\"mem_c\":%u", t.gpu_t, t.mem_t);

    if (gv.duty < 0.999) {
        double sleep_s = batch_s * (1.0 / gv.duty - 1.0);
        // Cap generously: a 5s cap turned a 5% duty target into ~37% on 3s
        // batches, defeating the backoff exactly when it mattered most.
        sleep_s = std::min(sleep_s, 60.0);
        s_backoff_s += sleep_s;
        usleep((useconds_t)(sleep_s * 1e6));
    }
}

// A long phase can run for 25-100 minutes without producing any event on a
// healthy card. Without a periodic heartbeat the supervisor's hang watchdog
// reads that silence as a dead kernel and kills a perfectly good worker.
static double s_last_hb = 0;
static void heartbeat(const char *phase, int cycle, double elapsed, double budget,
                      const Governor &gv) {
    double t = now_s();
    if (t - s_last_hb < 30.0) return;
    s_last_hb = t;
    Tele te = sample_tele();
    emit("\"ev\":\"heartbeat\",\"phase\":\"%s\",\"cycle\":%d,\"phase_s\":%.0f,"
         "\"phase_budget_s\":%.0f,\"phase_pct\":%.1f,\"mismatch\":%llu,\"hard\":%llu,"
         "\"gpu_c\":%u,\"mem_c\":%u,\"pwr_w\":%.1f,\"sm_mhz\":%u,\"duty\":%.2f",
         phase, cycle, elapsed, budget, budget > 0 ? 100.0 * elapsed / budget : 0.0,
         s_mismatch, s_hard, te.gpu_t, te.mem_t, te.pwr_mw / 1000.0, te.sm_clk, gv.duty);
}

// ---- phases --------------------------------------------------------------
static unsigned long long chunk_words(size_t c) { return U4_PER_CHUNK; }
static unsigned long long chunk_base(size_t c)  { return (unsigned long long)c * U4_PER_CHUNK; }

static int pass_fill(const Cfg &cfg, unsigned long long seed, int pat, int inv) {
    for (size_t c = 0; c < g_chunks.size(); c++)
        k_fill<<<g_grid, g_block>>>((uint4*)g_chunks[c], chunk_words(c), chunk_base(c), seed, pat, inv);
    CUCK(cudaDeviceSynchronize(), "fill");
    return 0;
}

static int pass_verify(const Cfg &cfg, unsigned long long seed, int pat, int inv, unsigned phase) {
    for (size_t c = 0; c < g_chunks.size(); c++)
        k_verify<<<g_grid, g_block>>>((const uint4*)g_chunks[c], chunk_words(c), chunk_base(c),
                                      seed, pat, inv, (unsigned)c, phase, d_errcount, d_recs, d_nrec);
    CUCK(cudaDeviceSynchronize(), "verify");
    s_bitreads += (unsigned long long)g_chunks.size() * CHUNK_BYTES * 8ULL;
    return drain_errors(cfg, phase);
}

int main(int argc, char **argv) {
    Cfg cfg;
    const char *tag = "session";
    for (int i = 1; i < argc; i++) {
        if      (!strncmp(argv[i], "--budget=",  9)) cfg.budget_s   = atof(argv[i] + 9);
        else if (!strncmp(argv[i], "--scale=",   8)) cfg.scale      = atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--seed=",    7)) cfg.seed0      = strtoull(argv[i] + 7, 0, 0);
        else if (!strncmp(argv[i], "--cycle=",   8)) cfg.supercycle0= atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--size-gib=",11)) cfg.size_gib  = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--t-backoff=",12)) cfg.t_backoff= atof(argv[i] + 12);
        else if (!strncmp(argv[i], "--t-drift=", 10)) cfg.t_drift   = atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--t-critical=",13)) cfg.t_critical = atof(argv[i] + 13);
        else if (!strncmp(argv[i], "--t-pause=",  10)) cfg.t_pause     = atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--t-halt=",    9)) cfg.t_halt      = atof(argv[i] + 9);
        else if (!strncmp(argv[i], "--t-resume=", 11)) cfg.t_resume    = atof(argv[i] + 11);
        else if (!strncmp(argv[i], "--t-mem-pause=",14)) cfg.t_mem_pause = atof(argv[i] + 14);
        else if (!strncmp(argv[i], "--t-mem-halt=",13)) cfg.t_mem_halt  = atof(argv[i] + 13);
        else if (!strncmp(argv[i], "--plimit=",  9)) cfg.plimit_w   = atof(argv[i] + 9);
        else if (!strcmp (argv[i], "--self-test"))   cfg.self_test  = 1;
        else if (!strncmp(argv[i], "--device=",  9)) cfg.device     = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--tag=",     6)) tag            = argv[i] + 6;
    }

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);
    setvbuf(stdout, NULL, _IOLBF, 0);

    // modular inverse of TMUL mod 2^32 (Newton), asserted -- never hardcoded
    unsigned inv = TMUL;
    for (int i = 0; i < 5; i++) inv *= (2u - TMUL * inv);
    g_tinv = inv;
    if (TMUL * g_tinv != 1u) { emit("\"ev\":\"fatal\",\"msg\":\"tinv check failed\""); return 2; }

    // Bind to the requested card BEFORE any allocation: every cudaMalloc and
    // kernel launch below lands on the current device.
    int ndev = 0;
    CUCK(cudaGetDeviceCount(&ndev), "getdevicecount");
    if (cfg.device < 0 || cfg.device >= ndev) {
        emit("\"ev\":\"fatal\",\"msg\":\"--device=%d out of range; %d GPU(s) visible\"",
             cfg.device, ndev);
        return 2;
    }
    CUCK(cudaSetDevice(cfg.device), "setdevice");

    cudaDeviceProp prop;
    CUCK(cudaGetDeviceProperties(&prop, cfg.device), "getprops");

    // Resolve NVML BY PCI BUS ID, never by index. NVML enumerates independently
    // of CUDA and ignores CUDA_VISIBLE_DEVICES, so an index taken from one and
    // used in the other can silently point at the OTHER card -- which would mean
    // hammering one GPU while the thermal governor watches its neighbour. The
    // bus id is the only identifier both APIs agree on.
    char busid[64] = {0};
    int have_bus = (cudaDeviceGetPCIBusId(busid, sizeof(busid), cfg.device) == cudaSuccess);
    if (nvmlInit() == NVML_SUCCESS && have_bus &&
        nvmlDeviceGetHandleByPciBusId(busid, &g_nvml) == NVML_SUCCESS)
        g_nvml_ok = 1;
    if (!g_nvml_ok)
        // Not fatal here: govern() treats unreadable telemetry as dangerous and
        // throttles, then halts. Say so loudly rather than running blind quietly.
        emit("\"ev\":\"CRITICAL\",\"kind\":\"nvml_bind_failed\",\"device\":%d,\"pci\":\"%s\","
             "\"note\":\"no telemetry handle for the card under test; governor will halt\"",
             cfg.device, busid);

    int bpm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, (void*)k_verify, g_block, 0);
    g_grid = bpm * prop.multiProcessorCount;

    emit("\"ev\":\"session_start\",\"tag\":\"%s\",\"gpu\":\"%s\",\"device\":%d,\"pci\":\"%s\","
         "\"nvml\":%d,\"sms\":%d,\"l2_kb\":%d,"
         "\"grid\":%d,\"block\":%d,\"budget_s\":%.0f,\"scale\":%.4f,\"self_test\":%d,\"pid\":%d",
         tag, prop.name, cfg.device, busid, g_nvml_ok,
         prop.multiProcessorCount, prop.l2CacheSize / 1024,
         g_grid, g_block, cfg.budget_s, cfg.scale, cfg.self_test, (int)getpid());

    // small allocations first, so the greedy loop cannot starve them
    CUCK(cudaMalloc(&d_errcount, sizeof(unsigned long long)), "malloc errcount");
    CUCK(cudaMalloc(&d_alubad,   sizeof(unsigned long long)), "malloc alubad");
    CUCK(cudaMalloc(&d_recs,     ERR_CAP * sizeof(ErrRec)),   "malloc recs");
    CUCK(cudaMalloc(&d_nrec,     sizeof(unsigned)),           "malloc nrec");
    CUCK(cudaMalloc(&d_triage,   sizeof(Triage)),             "malloc triage");
    cudaMemset(d_errcount, 0, sizeof(unsigned long long));
    cudaMemset(d_alubad,   0, sizeof(unsigned long long));
    cudaMemset(d_nrec,     0, sizeof(unsigned));

    // greedy allocation, then back off to leave driver headroom
    size_t want = cfg.size_gib > 0 ? (size_t)cfg.size_gib : MAX_CHUNKS;
    while (g_chunks.size() < want) {
        void *p = nullptr;
        if (cudaMalloc(&p, CHUNK_BYTES) != cudaSuccess) { cudaGetLastError(); break; }
        g_chunks.push_back(p);
    }
    if (cfg.size_gib == 0) {
        for (int i = 0; i < 2 && !g_chunks.empty(); i++) { cudaFree(g_chunks.back()); g_chunks.pop_back(); }
    }
    if (g_chunks.empty()) { emit("\"ev\":\"fatal\",\"msg\":\"no memory allocated\""); return 3; }

    size_t total_mib = g_chunks.size() * 1024;
    emit("\"ev\":\"alloc\",\"chunks\":%zu,\"gib\":%zu,\"bytes\":%llu,\"stock_fb_gib\":8",
         g_chunks.size(), g_chunks.size(), (unsigned long long)g_chunks.size() * CHUNK_BYTES);

    Governor gv;
    Tele t0 = sample_tele(); emit_tele("baseline", t0);

    double t_start = now_s();
    unsigned long long seed = cfg.seed0;
    int cycle = cfg.supercycle0;

    // Phase budgets (seconds), scaled for smoke runs
    const double D_MOVINV  = 1500.0 * cfg.scale;
    const double D_RMW     = 2700.0 * cfg.scale;
    const double D_SCATTER = 1500.0 * cfg.scale;
    const double D_ALU     = 6000.0 * cfg.scale;

    while (!g_stop && (now_s() - t_start) < cfg.budget_s) {
        cycle++;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        emit("\"ev\":\"cycle_start\",\"cycle\":%d,\"seed\":\"0x%llx\"", cycle, seed);

        // ---- P0: fill + immediate verify ----
        double p0 = now_s();
        emit("\"ev\":\"phase_start\",\"phase\":\"%s\",\"cycle\":%d", PH_NAME[PH_FILL], cycle);
        if (pass_fill(cfg, seed, PAT_PRNG, 0)) return 4;

        if (cfg.self_test) {   // qualify the detector: inject known faults
            unsigned long long inj_soft = (unsigned long long)(g_chunks.size() / 2) * U4_PER_CHUNK + 777777ULL;
            unsigned long long inj_hard = (unsigned long long)(g_chunks.size() / 3) * U4_PER_CHUNK + 424242ULL;
            cfg.known_bad.clear(); cfg.known_bad.push_back(inj_hard);
            size_t cs = inj_soft / U4_PER_CHUNK, ch = inj_hard / U4_PER_CHUNK;
            k_inject<<<1,1>>>((uint4*)g_chunks[cs], inj_soft % U4_PER_CHUNK, 1, 19);
            k_inject<<<1,1>>>((uint4*)g_chunks[ch], inj_hard % U4_PER_CHUNK, 3, 5);
            cudaDeviceSynchronize();
            emit("\"ev\":\"self_test_inject\",\"soft_gidx\":%llu,\"hard_gidx\":%llu,"
                 "\"expect\":\"1 SOFT_FLIP + 1 HARD_FAULT, 1 bit each\"", inj_soft, inj_hard);
        }

        if (pass_verify(cfg, seed, PAT_PRNG, 0, PH_FILL)) return 4;
        emit("\"ev\":\"phase_end\",\"phase\":\"%s\",\"cycle\":%d,\"dur_s\":%.1f,\"bitreads\":%llu",
             PH_NAME[PH_FILL], cycle, now_s() - p0, s_bitreads);
        govern(cfg, gv, now_s() - p0);

        // ---- P1: moving inversions, several patterns, both directions ----
        {
            double ps = now_s();
            emit("\"ev\":\"phase_start\",\"phase\":\"%s\",\"cycle\":%d", PH_NAME[PH_MOVINV], cycle);
            const int pats[] = { PAT_A5, PAT_ADDR, PAT_ONES, PAT_WALK1 };
            int pi = 0, inv_state = 0;
            while (!g_stop && now_s() - ps < D_MOVINV) {
                int pat = pats[pi % 4];
                if (pass_fill(cfg, seed, pat, 0)) return 4;
                inv_state = 0;
                for (int rep = 0; rep < 2 && !g_stop; rep++) {
                    for (int dir = 0; dir < 2 && !g_stop; dir++) {
                        double b = now_s();
                        for (size_t c = 0; c < g_chunks.size(); c++)
                            k_movinv<<<g_grid, g_block>>>((uint4*)g_chunks[c], chunk_words(c), chunk_base(c),
                                                          seed, pat, inv_state, dir, (unsigned)c, PH_MOVINV,
                                                          d_errcount, d_recs, d_nrec);
                        CUCK(cudaDeviceSynchronize(), "movinv");
                        s_bitreads += (unsigned long long)g_chunks.size() * CHUNK_BYTES * 8ULL;
                        if (drain_errors(cfg, PH_MOVINV)) return 4;
                        inv_state = !inv_state;
                        govern(cfg, gv, now_s() - b);
                        heartbeat(PH_NAME[PH_MOVINV], cycle, now_s() - ps, D_MOVINV, gv);
                    }
                }
                pi++;
            }
            emit("\"ev\":\"phase_end\",\"phase\":\"%s\",\"cycle\":%d,\"dur_s\":%.1f,\"bitreads\":%llu",
                 PH_NAME[PH_MOVINV], cycle, now_s() - ps, s_bitreads);
        }

        // ---- P2: read-modify-write transform hammer ----
        {
            double ps = now_s();
            emit("\"ev\":\"phase_start\",\"phase\":\"%s\",\"cycle\":%d", PH_NAME[PH_RMW], cycle);
            unsigned round = 0;
            unsigned kr = (unsigned)splitmix64(seed ^ 0xC0FFEEULL) | 1u;
            // establish round-0 state directly in transform space
            for (size_t c = 0; c < g_chunks.size(); c++)
                k_fill_rmw<<<g_grid, g_block>>>((uint4*)g_chunks[c], chunk_words(c), chunk_base(c),
                                                seed, PAT_PRNG, kr);
            CUCK(cudaDeviceSynchronize(), "rmw prime");

            unsigned k_cur = kr;
            while (!g_stop && now_s() - ps < D_RMW) {
                double b = now_s();
                unsigned k_next = k_cur * 2654435761u + 0x9E3779B9u;
                for (size_t c = 0; c < g_chunks.size(); c++)
                    k_rmw<<<g_grid, g_block>>>((uint4*)g_chunks[c], chunk_words(c), chunk_base(c), seed, PAT_PRNG,
                                               k_cur, k_next, g_tinv, (unsigned)c, PH_RMW,
                                               d_errcount, d_recs, d_nrec);
                CUCK(cudaDeviceSynchronize(), "rmw");
                s_bitreads += (unsigned long long)g_chunks.size() * CHUNK_BYTES * 8ULL;
                if (drain_errors(cfg, PH_RMW)) return 4;
                k_cur = k_next; round++;
                govern(cfg, gv, now_s() - b);
                heartbeat(PH_NAME[PH_RMW], cycle, now_s() - ps, D_RMW, gv);
            }
            emit("\"ev\":\"phase_end\",\"phase\":\"%s\",\"cycle\":%d,\"dur_s\":%.1f,\"rounds\":%u,\"bitreads\":%llu",
                 PH_NAME[PH_RMW], cycle, now_s() - ps, round, s_bitreads);
        }

        // ---- P3: scattered access ----
        {
            double ps = now_s();
            emit("\"ev\":\"phase_start\",\"phase\":\"%s\",\"cycle\":%d", PH_NAME[PH_SCATTER], cycle);
            if (pass_fill(cfg, seed, PAT_PRNG, 0)) return 4;
            unsigned long long salt = seed;
            unsigned long long iterations = 0;
            while (!g_stop && now_s() - ps < D_SCATTER) {
                double b = now_s();
                salt = splitmix64(salt);
                for (size_t c = 0; c < g_chunks.size(); c++)
                    k_scatter<<<g_grid, g_block>>>((const uint4*)g_chunks[c], chunk_words(c), chunk_base(c),
                                                   seed, PAT_PRNG, salt + c, 64u, (unsigned)c, PH_SCATTER,
                                                   d_errcount, d_recs, d_nrec);
                CUCK(cudaDeviceSynchronize(), "scatter");
                unsigned long long reads = (unsigned long long)g_grid * g_block * 64ULL * g_chunks.size();
                s_bitreads += reads * 128ULL;
                iterations += reads;
                if (drain_errors(cfg, PH_SCATTER)) return 4;
                govern(cfg, gv, now_s() - b);
                heartbeat(PH_NAME[PH_SCATTER], cycle, now_s() - ps, D_SCATTER, gv);
            }
            emit("\"ev\":\"phase_end\",\"phase\":\"%s\",\"cycle\":%d,\"dur_s\":%.1f,\"rand_reads\":%llu",
                 PH_NAME[PH_SCATTER], cycle, now_s() - ps, iterations);
        }

        // ---- P4: ALU soak at full power, 60 GiB held golden and hot ----
        // This is the most production-relevant test: memory sits untouched at
        // the card's thermal ceiling, then P5 verifies it survived.
        double retention_start = 0;
        {
            double ps = now_s();
            emit("\"ev\":\"phase_start\",\"phase\":\"%s\",\"cycle\":%d", PH_NAME[PH_ALU], cycle);
            if (pass_fill(cfg, seed, PAT_PRNG, 0)) return 4;
            if (pass_verify(cfg, seed, PAT_PRNG, 0, PH_ALU)) return 4;
            retention_start = now_s();

            const unsigned long long A = 6364136223846793005ULL, C = 1442695040888963407ULL;
            const unsigned long long ITERS = 400000ULL;
            unsigned long long salt = seed, launches = 0;
            while (!g_stop && now_s() - ps < D_ALU) {
                double b = now_s();
                salt = splitmix64(salt);
                k_alu<<<g_grid, g_block>>>(A, C, ITERS, salt, d_alubad, d_recs, d_nrec,
                                           (cfg.self_test && launches == 3) ? 1 : 0);
                CUCK(cudaDeviceSynchronize(), "alu");
                unsigned long long bad = 0;
                cudaMemcpy(&bad, d_alubad, sizeof(bad), cudaMemcpyDeviceToHost);
                s_aluops += (unsigned long long)g_grid * g_block * ITERS * 2ULL;
                if (bad) {
                    s_sdc += bad;
                    Tele t = sample_tele();
                    emit("\"ev\":\"finding\",\"kind\":\"COMPUTE_SDC\",\"threads_bad\":%llu,\"launch\":%llu,"
                         "\"iters\":%llu,\"gpu_c\":%u,\"pwr_w\":%.1f,\"sm_mhz\":%u",
                         bad, launches, ITERS, t.gpu_t, t.pwr_mw / 1000.0, t.sm_clk);
                    cudaMemset(d_alubad, 0, sizeof(unsigned long long));
                    cudaMemset(d_nrec,   0, sizeof(unsigned));
                }
                launches++;
                govern(cfg, gv, now_s() - b);
                heartbeat(PH_NAME[PH_ALU], cycle, now_s() - ps, D_ALU, gv);
            }
            Tele t = sample_tele();
            emit("\"ev\":\"phase_end\",\"phase\":\"%s\",\"cycle\":%d,\"dur_s\":%.1f,\"launches\":%llu,"
                 "\"alu_ops\":%llu,\"peak_gpu_c\":%u,\"peak_mem_c\":%u,\"pwr_w\":%.1f",
                 PH_NAME[PH_ALU], cycle, now_s() - ps, launches, s_aluops, gv.peak_gpu, gv.peak_mem,
                 t.pwr_mw / 1000.0);
        }

        // ---- P5: retention verify after the hot soak ----
        {
            double ps = now_s();
            double held = ps - retention_start;
            emit("\"ev\":\"phase_start\",\"phase\":\"%s\",\"cycle\":%d,\"held_s\":%.0f",
                 PH_NAME[PH_RETENTION], cycle, held);
            unsigned long long before = s_mismatch;
            if (pass_verify(cfg, seed, PAT_PRNG, 0, PH_RETENTION)) return 4;
            unsigned long long bits = (unsigned long long)g_chunks.size() * CHUNK_BYTES * 8ULL;
            emit("\"ev\":\"phase_end\",\"phase\":\"%s\",\"cycle\":%d,\"dur_s\":%.1f,"
                 "\"held_s\":%.0f,\"bits\":%llu,\"bit_hours\":%.6g,\"errors\":%llu,"
                 "\"peak_gpu_c\":%u,\"peak_mem_c\":%u",
                 PH_NAME[PH_RETENTION], cycle, now_s() - ps, held, bits,
                 (double)bits * held / 3600.0, s_mismatch - before, gv.peak_gpu, gv.peak_mem);
        }

        // ---- known-bad retest: confirm reproducibility of any hard fault ----
        if (!cfg.known_bad.empty() && !cfg.self_test) {
            for (size_t i = 0; i < cfg.known_bad.size(); i++) {
                ErrRec r; memset(&r, 0, sizeof(r));
                r.gidx = cfg.known_bad[i];
                r.chunk = (unsigned)(r.gidx / U4_PER_CHUNK);
                r.phase = PH_KNOWNBAD;
                if (r.chunk >= g_chunks.size()) continue;
                run_triage(cfg, r, PH_NAME[PH_KNOWNBAD]);
            }
        }

        Tele te = sample_tele();
        // A cycle interrupted by SIGTERM still falls through to P5; mark it so
        // the supervisor does not credit it as a completed cycle.
        emit("\"ev\":\"cycle_end\",\"cycle\":%d,\"complete\":%d,\"elapsed_s\":%.0f,\"mismatch\":%llu,"
             "\"hard\":%llu,\"soft\":%llu,\"transient\":%llu,\"sdc\":%llu,"
             "\"bitreads\":%llu,\"alu_ops\":%llu,\"backoff_s\":%.1f,"
             "\"peak_gpu_c\":%u,\"peak_mem_c\":%u,\"duty\":%.3f",
             cycle, g_stop ? 0 : 1, now_s() - t_start, s_mismatch, s_hard, s_soft, s_transient, s_sdc,
             s_bitreads, s_aluops, s_backoff_s, gv.peak_gpu, gv.peak_mem, gv.duty);
        emit_tele("cycle_end", te);
    }

    emit("\"ev\":\"session_end\",\"cycles\":%d,\"elapsed_s\":%.0f,\"stopped\":%d,"
         "\"mismatch\":%llu,\"hard\":%llu,\"soft\":%llu,\"transient\":%llu,\"sdc\":%llu,"
         "\"bitreads\":%llu,\"alu_ops\":%llu,\"gib\":%zu,\"backoff_s\":%.1f,"
         "\"thermal_halt\":%d,\"thermal_pauses\":%d,"
         "\"peak_gpu_c\":%u,\"peak_mem_c\":%u",
         cycle, now_s() - t_start, (int)g_stop, s_mismatch, s_hard, s_soft, s_transient, s_sdc,
         s_bitreads, s_aluops, g_chunks.size(), s_backoff_s, s_thermal_halt, gv.pauses,
         gv.peak_gpu, gv.peak_mem);

    for (size_t i = 0; i < g_chunks.size(); i++) cudaFree(g_chunks[i]);
    if (g_nvml_ok) nvmlShutdown();
    return s_thermal_halt ? EXIT_THERMAL_HALT : 0;
}
