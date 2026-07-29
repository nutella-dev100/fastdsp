#include "bench_util.hpp"

#include <fastdsp/convolution/convolve.hpp>
#include <fastdsp/convolution/convolver.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#if defined(__AVX2__) && defined(__FMA__)
    #include <fastdsp/transforms/detail/simd_policy.hpp>
#endif
#include <fastdsp/core/Signal.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace fastdsp;

// This benchmark exists to settle a TODO that is already written into the
// library. convolve.hpp says of its two dispatch thresholds:
//
//     "Both were measured (doubles, GCC 13, ScalarPolicy), not estimated, but
//      on a different machine than the target. Re-measure before trusting
//      them."
//
// So: re-measure them, on this machine, and print what the constants should
// be. A benchmark that changes a line of source is worth more than one that
// produces a number for a README.
//
// Everything is timed the way convolve() actually calls it -- in particular
// the overlap-add path constructs a fresh Convolver per call, because that is
// what the dispatcher does. Timing a pre-built Convolver against direct
// convolution would answer a different question and would move the crossover
// in the wrong direction.

using Policy = detail::ScalarPolicy;

static Signal<double> noise(std::size_t n, unsigned seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    Signal<double> x(n, 48000.0);
    for(std::size_t i = 0; i < n; ++i) x[i] = d(rng);
    return x;
}

static double max_abs_diff(const Signal<double>& a, const Signal<double>& b){
    if(a.size() != b.size()) return 1e30;
    double w = 0.0;
    for(std::size_t i = 0; i < a.size(); ++i) w = std::max(w, std::abs(a[i] - b[i]));
    return w;
}

// convolve_direct is the independent oracle -- if the FFT paths have drifted,
// every number below is meaningless, so check before timing anything.
static void verify(){
    const auto x = noise(1000, 1);
    const auto h = noise(97, 2);
    const auto ref = convolve_direct(x, h);

    const double e_fft = max_abs_diff(convolve_fft<double, Policy>(x, h), ref);
    const double e_ola = max_abs_diff(Convolver<double, Policy>(h).convolve(x), ref);
    const double e_dis = max_abs_diff(convolve<double, Policy>(x, h), ref);

    // Scale tolerance with the transform size: FFT convolution error grows
    // roughly as sqrt(N) * eps * signal magnitude.
    if(e_fft > 1e-9 || e_ola > 1e-9 || e_dis > 1e-9){
        std::fprintf(stderr, "convolution paths disagree (fft %.2e, ola %.2e, dispatch %.2e)"
                             " -- benchmark aborted\n", e_fft, e_ola, e_dis);
        std::exit(EXIT_FAILURE);
    }
    std::printf("fft / overlap-add / dispatch agree with direct to %.2e abs\n\n",
                std::max({e_fft, e_ola, e_dis}));
}

// --- threshold 1: direct vs overlap-add ------------------------------------
// Governs kDirectConvolutionMaxTaps. Direct is O(nx*nh) but with a tiny
// constant and perfect locality; overlap-add pays two transforms per block
// plus plan construction. The crossover is where the kernel gets long enough
// that the transforms win.
static void sweep_direct_vs_ola(){
    std::printf("threshold 1: direct vs overlap-add (kDirectConvolutionMaxTaps)\n");
    std::printf("%9s %7s %13s %13s %10s %9s\n",
                "nx", "taps", "direct ms", "ola ms", "ola/direct", "winner");

    for(std::size_t nx : {4096u, 65536u}){
        const auto x = noise(nx, 3);
        std::size_t crossover = 0;

        for(std::size_t nh : {8u, 16u, 32u, 64u, 96u, 128u, 256u}){
            const auto h = noise(nh, 4);

            const auto d = bench::measure([&]{
                auto y = convolve_direct(x, h);
                bench::keep(y[0]);
            });
            const auto o = bench::measure([&]{
                auto y = Convolver<double, Policy>(h).convolve(x);
                bench::keep(y[0]);
            });

            const double ratio = o.median_s / d.median_s;
            if(crossover == 0 && ratio < 1.0) crossover = nh;

            std::printf("%9zu %7zu %13.4f %13.4f %9.2fx %9s\n",
                        nx, nh, d.median_s * 1e3, o.median_s * 1e3, ratio,
                        ratio < 1.0 ? "ola" : "direct");
        }
        std::printf("  -> nx=%zu: overlap-add first wins at taps=%zu\n\n",
                    nx, crossover);
    }
    std::printf("  set kDirectConvolutionMaxTaps just below the smaller crossover.\n"
                "  current value in convolve.hpp: %zu\n\n", kDirectConvolutionMaxTaps);
}

// --- threshold 2: one-shot FFT vs overlap-add ------------------------------
// Governs the `hi < 2 * lo` rule. One transform pair over the whole signal is
// sized by nx even when h is short, so its cost grows as nx log nx with a
// transform far larger than the kernel needs; overlap-add keeps the transform
// pinned to a multiple of the kernel length. They should tie near nx == nh and
// diverge fast after that.
static void sweep_oneshot_vs_ola(){
    std::printf("threshold 2: one-shot FFT vs overlap-add (the `hi < 2*lo` rule)\n");
    std::printf("%7s %6s %9s %14s %13s %11s %9s\n",
                "taps", "nx/nh", "nx", "one-shot ms", "ola ms", "ola/oneshot", "winner");

    for(std::size_t nh : {512u, 2048u, 8192u}){
        const auto h = noise(nh, 5);
        for(std::size_t mult : {1u, 2u, 4u, 16u}){
            const std::size_t nx = nh * mult;
            const auto x = noise(nx, 6);

            const auto f = bench::measure([&]{
                auto y = convolve_fft<double, Policy>(x, h);
                bench::keep(y[0]);
            });
            const auto o = bench::measure([&]{
                auto y = Convolver<double, Policy>(h).convolve(x);
                bench::keep(y[0]);
            });

            const double ratio = o.median_s / f.median_s;
            std::printf("%7zu %6zu %9zu %14.4f %13.4f %10.2fx %9s\n",
                        nh, mult, nx, f.median_s * 1e3, o.median_s * 1e3, ratio,
                        ratio < 1.0 ? "ola" : "one-shot");
        }
        std::printf("\n");
    }
}

// --- what Convolver is actually for ----------------------------------------
// The dispatcher builds a Convolver per call, so the plan and the kernel
// spectrum are thrown away every time. The reason the class exists is the
// fixed-kernel streaming case, where that setup is paid once. This measures
// how much that amortisation is worth -- i.e. how much a caller gains by
// holding a Convolver instead of calling convolve() in a loop.
static void sweep_amortised(){
    std::printf("Convolver setup amortisation (fixed kernel, repeated inputs)\n");
    std::printf("%7s %9s %15s %15s %10s\n",
                "taps", "nx", "per-call ms", "amortised ms", "saving");

    for(std::size_t nh : {256u, 1024u, 4096u}){
        const auto h = noise(nh, 7);
        const std::size_t nx = 65536;
        const auto x = noise(nx, 8);

        const auto fresh = bench::measure([&]{
            auto y = Convolver<double, Policy>(h).convolve(x);
            bench::keep(y[0]);
        });

        const Convolver<double, Policy> reused(h);
        const auto amort = bench::measure([&]{
            auto y = reused.convolve(x);
            bench::keep(y[0]);
        });

        std::printf("%7zu %9zu %15.4f %15.4f %9.2fx\n", nh, nx,
                    fresh.median_s * 1e3, amort.median_s * 1e3,
                    fresh.median_s / amort.median_s);
    }
    std::printf("\n");
}

// --- what the SIMD butterfly is worth at the application level -------------
// The FFT microbenchmark measures the transform; this measures what a caller
// actually feels, which is always less because the transform is only part of
// the work (packing, pointwise multiply, overlap accumulation are all scalar).
#if defined(__AVX2__) && defined(__FMA__)
static void sweep_policy(){
    std::printf("ScalarPolicy vs SimdPolicy through overlap-add convolution\n");
    std::printf("%7s %9s %13s %13s %10s\n", "taps", "nx", "scalar ms", "simd ms", "speedup");

    for(std::size_t nh : {256u, 1024u, 4096u}){
        const auto h = noise(nh, 9);
        const std::size_t nx = 65536;
        const auto x = noise(nx, 10);

        const Convolver<double, detail::ScalarPolicy> cs(h);
        const Convolver<double, detail::SimdPolicy>   cv(h);

        if(max_abs_diff(cs.convolve(x), cv.convolve(x)) > 1e-9){
            std::fprintf(stderr, "policies disagree -- benchmark aborted\n");
            std::exit(EXIT_FAILURE);
        }

        const auto s = bench::measure([&]{ auto y = cs.convolve(x); bench::keep(y[0]); });
        const auto v = bench::measure([&]{ auto y = cv.convolve(x); bench::keep(y[0]); });

        std::printf("%7zu %9zu %13.4f %13.4f %9.2fx\n", nh, nx,
                    s.median_s * 1e3, v.median_s * 1e3, s.median_s / v.median_s);
    }
    std::printf("\n");
}
#endif

int main(){
    bench::print_environment();
    verify();
    sweep_direct_vs_ola();
    sweep_oneshot_vs_ola();
    sweep_amortised();
#if defined(__AVX2__) && defined(__FMA__)
    sweep_policy();
#endif
    return 0;
}