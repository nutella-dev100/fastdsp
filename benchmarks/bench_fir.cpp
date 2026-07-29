#include "bench_util.hpp"

#include <fastdsp/filters/fir_filter.hpp>
#include <fastdsp/filters/fir_design.hpp>
#include <fastdsp/simd/ops.hpp>
#include <fastdsp/simd/ops_avx2.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <random>
#include <vector>

using namespace fastdsp;

// ---------------------------------------------------------------------------
// Backend 2: the fair scalar baseline.
//
// simd::ScalarOps takes plain pointers, so at -O2 the compiler must assume y
// and x may overlap and will not auto-vectorise. Inside FIRFilter the two are
// always distinct (output block vs scratch delay line), so __restrict is a
// promise the library could legitimately make. Quote the SIMD win against
// THIS baseline, not the unannotated one.
//
// The library keeps the permissive signature because test_fir_simd.cpp
// deliberately covers the aliased y == x case; promoting this into ScalarOps
// means narrowing that contract and deleting that test.
// ---------------------------------------------------------------------------
struct ScalarRestrictOps{
    static constexpr const char* name() noexcept { return "scalar+restrict"; }

    template<FloatingPoint T>
    static constexpr std::size_t width() noexcept { return 1; }

    template<FloatingPoint T>
    static void axpy(T* __restrict y, T a, const T* __restrict x, std::size_t n) noexcept {
        for(std::size_t i = 0; i < n; ++i) y[i] += a * x[i];
    }
};

// ---------------------------------------------------------------------------
// Backend 4: fused kernel prototype.
//
// The axpy formulation walks the output block once per tap, so it loads and
// stores y `taps` times: 32 B loaded from the delay line + 32 B loaded from y
// + 32 B stored to y per 8 outputs per tap, i.e. 6 bytes of L1 traffic per
// flop. That is what caps it, not arithmetic.
//
// This holds four accumulator vectors (32 outputs) in registers across ALL
// taps, so y is touched exactly once. Traffic drops to 2 bytes per flop, and
// the four independent accumulator chains hide FMA latency.
//
// Not in the library yet: this is the measurement that would justify putting
// it there.
// ---------------------------------------------------------------------------
static void fir_fused_avx2(const float* s, const float* h, std::size_t taps,
                           float* y, std::size_t m, std::size_t tail) noexcept
{
    std::size_t i = 0;
    const std::size_t vend = (m >= 32) ? (m - (m % 32)) : 0;
    for(; i < vend; i += 32){
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const float* base = s + tail + i;
        for(std::size_t k = 0; k < taps; ++k){
            const __m256 hk = _mm256_broadcast_ss(h + k);
            const float* p = base - k;              // sliding window: unaligned by construction
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p     ), hk, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p +  8), hk, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), hk, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), hk, a3);
        }
        _mm256_storeu_ps(y + i,      a0);
        _mm256_storeu_ps(y + i +  8, a1);
        _mm256_storeu_ps(y + i + 16, a2);
        _mm256_storeu_ps(y + i + 24, a3);
    }
    for(; i < m; ++i){
        float acc = 0.0f;
        for(std::size_t k = 0; k < taps; ++k) acc += h[k] * s[tail + i - k];
        y[i] = acc;
    }
}

// Streaming wrapper mirroring FIRFilter's history handling, so the comparison
// is like-for-like: same per-block history copy, same blocking. Duplicated
// here rather than in the library because this is still a prototype.
class FusedFIR{
    private:
        Signal<float> h_;
        std::vector<float> hist_, scratch_;
        std::size_t taps_, block_;
    public:
        FusedFIR(const Signal<float>& h, std::size_t block)
            : h_(h), taps_(h.size()), block_(block)
        {
            hist_.assign(taps_ - 1, 0.0f);
            scratch_.assign(taps_ - 1 + block_, 0.0f);
        }
        void process(const float* in, float* out, std::size_t n){
            const std::size_t tail = taps_ - 1;
            for(std::size_t pos = 0; pos < n; pos += block_){
                const std::size_t m = std::min(block_, n - pos);
                std::copy(hist_.begin(), hist_.end(), scratch_.begin());
                std::copy(in + pos, in + pos + m, scratch_.begin() + tail);
                fir_fused_avx2(scratch_.data(), h_.data(), taps_, out + pos, m, tail);
                if(tail) std::copy(scratch_.begin() + m, scratch_.begin() + m + tail, hist_.begin());
            }
        }
};

// ---------------------------------------------------------------------------

static Signal<float> noise(std::size_t n, unsigned seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    Signal<float> x(n, 48000.0);
    for(std::size_t i = 0; i < n; ++i) x[i] = d(rng);
    return x;
}

// Buffers allocated and the filter constructed outside the timed region: this
// measures steady-state streaming throughput, which is what a real-time path
// cares about, not one-off setup.
template<class Ops>
static bench::Result run(const Signal<float>& h, const Signal<float>& x,
                         Signal<float>& y, std::size_t block)
{
    FIRFilter<float, Ops> f(h, block);
    return bench::measure([&]{
        f.process(x.data(), y.data(), x.size());
        bench::keep(y[y.size() - 1]);      // stop the call being elided entirely
    });
}

static bench::Result run_fused(const Signal<float>& h, const Signal<float>& x,
                               Signal<float>& y, std::size_t block)
{
    FusedFIR f(h, block);
    return bench::measure([&]{
        f.process(x.data(), y.data(), x.size());
        bench::keep(y[y.size() - 1]);
    });
}

// A benchmark of a kernel that computes the wrong thing is worse than no
// benchmark. Check before timing, every run.
static void verify_backends(){
    const auto hd = design_lowpass<double>(65, 4000.0, 48000.0);
    Signal<float> h(hd.size(), 48000.0);
    for(std::size_t i = 0; i < hd.size(); ++i) h[i] = static_cast<float>(hd[i]);

    const auto x = noise(5000, 42);
    Signal<float> ref(x.size(), 48000.0), got(x.size(), 48000.0);

    FIRFilter<float, simd::ScalarOps> r(h, 512);
    r.process(x.data(), ref.data(), x.size());

    double worst = 0.0;
    FIRFilter<float, simd::Avx2Ops> v(h, 512);
    v.process(x.data(), got.data(), x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
        worst = std::max(worst, std::abs(static_cast<double>(got[i] - ref[i])));

    FusedFIR f(h, 512);
    f.process(x.data(), got.data(), x.size());
    for(std::size_t i = 0; i < x.size(); ++i)
        worst = std::max(worst, std::abs(static_cast<double>(got[i] - ref[i])));

    if(worst > 1e-4){
        std::fprintf(stderr, "backend mismatch, max abs error %.3e -- benchmark aborted\n", worst);
        std::exit(EXIT_FAILURE);
    }
    std::printf("backends agree with scalar to %.2e abs\n\n", worst);
}

static void sweep_taps(){
    constexpr std::size_t kN = 1u << 20;   // ~4 MB: well past L2, so this is the
                                           // kernel and not a cache-resident
                                           // best case
    const Signal<float> x = noise(kN, 1);
    Signal<float> y(kN, 48000.0);
    constexpr std::size_t kBlock = FIRFilter<float>::kDefaultBlock;

    std::printf("FIR throughput, %zu samples/call, float, block=%zu\n", kN, kBlock);
    std::printf("%6s %11s %11s %11s %11s %11s %10s\n",
                "taps", "scalar", "restrict", "avx2 axpy", "avx2 fused",
                "fused/restr", "GFLOP/s");
    std::printf("%6s %11s %11s %11s %11s %11s %10s\n",
                "", "Ms/s", "Ms/s", "Ms/s", "Ms/s", "", "(fused)");

    for(std::size_t taps : {16u, 32u, 64u, 128u, 256u, 512u}){
        const auto hd = design_lowpass<double>(taps | 1u, 4000.0, 48000.0);
        Signal<float> h(hd.size(), 48000.0);
        for(std::size_t i = 0; i < hd.size(); ++i) h[i] = static_cast<float>(hd[i]);

        const auto s = run<simd::ScalarOps>(h, x, y, kBlock);
        const auto r = run<ScalarRestrictOps>(h, x, y, kBlock);
        const auto v = run<simd::Avx2Ops>(h, x, y, kBlock);
        const auto u = run_fused(h, x, y, kBlock);

        std::printf("%6zu %11.1f %11.1f %11.1f %11.1f %10.2fx %10.1f\n",
                    h.size(),
                    bench::msamples_per_s(kN, s.median_s),
                    bench::msamples_per_s(kN, r.median_s),
                    bench::msamples_per_s(kN, v.median_s),
                    bench::msamples_per_s(kN, u.median_s),
                    r.median_s / u.median_s,
                    bench::gflops(kN * h.size(), u.median_s));
    }
    std::printf("\nNOTE: at -O3 GCC auto-vectorises both scalar columns. Build the same\n"
                "      sweep at -O2 and the scalar numbers fall by ~4x while the\n"
                "      intrinsic ones do not move -- that gap is a compiler-flag\n"
                "      artefact, not a result. Quote fused/restrict at -O3.\n\n");
}

// Justifies FIRFilter::kDefaultBlock. Small blocks pay the per-block fixed
// costs (history copy, zero-fill, loop setup) too often; large ones push the
// scratch buffer out of L1. The default should sit at the start of the flat
// region: larger buys nothing and only forces a caller to hand over more
// latency.
static void sweep_block(){
    constexpr std::size_t kN = 1u << 20;
    const Signal<float> x = noise(kN, 2);
    Signal<float> y(kN, 48000.0);

    const auto hd = design_lowpass<double>(65, 4000.0, 48000.0);
    Signal<float> h(hd.size(), 48000.0);
    for(std::size_t i = 0; i < hd.size(); ++i) h[i] = static_cast<float>(hd[i]);

    std::printf("Internal block size sweep, 65 taps, float\n");
    std::printf("%8s %13s %13s\n", "block", "axpy Ms/s", "fused Ms/s");
    for(std::size_t b : {16u, 32u, 64u, 128u, 256u, 512u, 1024u, 4096u, 16384u}){
        const auto v = run<simd::Avx2Ops>(h, x, y, b);
        const auto u = run_fused(h, x, y, b);
        std::printf("%8zu %13.1f %13.1f\n", b,
                    bench::msamples_per_s(kN, v.median_s),
                    bench::msamples_per_s(kN, u.median_s));
    }
    std::printf("\n");
}

// The gap between median and minimum says whether the machine was quiet
// enough for any of the above to mean anything.
static void report_noise_floor(){
    constexpr std::size_t kN = 1u << 20;
    const Signal<float> x = noise(kN, 3);
    Signal<float> y(kN, 48000.0);
    const auto hd = design_lowpass<double>(65, 4000.0, 48000.0);
    Signal<float> h(hd.size(), 48000.0);
    for(std::size_t i = 0; i < hd.size(); ++i) h[i] = static_cast<float>(hd[i]);

    const auto v = run<simd::Avx2Ops>(h, x, y, FIRFilter<float>::kDefaultBlock);
    std::printf("timing spread (avx2 axpy, 65 taps): median %.4f ms, min %.4f ms, +%.1f%%\n",
                v.median_s * 1e3, v.min_s * 1e3,
                100.0 * (v.median_s - v.min_s) / v.min_s);
    std::printf("  above a few percent means the machine was busy; quiesce and re-run\n"
                "  before quoting anything.\n\n");
}

int main(){
    bench::print_environment();
    verify_backends();
    sweep_taps();
    sweep_block();
    report_noise_floor();
    return 0;
}