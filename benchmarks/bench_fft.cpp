#include "bench_util.hpp"

#include <fastdsp/transforms/fft.hpp>
#include <fastdsp/transforms/fft_plan.hpp>
#include <fastdsp/transforms/rfft.hpp>
#include <fastdsp/transforms/rfft_plan.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#include <fastdsp/transforms/detail/simd_policy.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <fastdsp/core/Signal.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace fastdsp;

// ---------------------------------------------------------------------------
// Why every timing here refills its input first.
//
// FFTPlan::forward is IN-PLACE, and a forward transform multiplies magnitudes
// by roughly sqrt(N). The harness calls its lambda hundreds or thousands of
// times, so transforming the same buffer repeatedly means magnitudes grow as
// N^(k/2): for float at N=1024 that overflows to inf in about 26 calls. The
// tail of the run then measures arithmetic on inf and denormals, which on x86
// is slow enough to make a perfectly good kernel look terrible.
//
// So each iteration copies a pristine input in, and a second timing measures
// the copy alone so it can be subtracted. The copy fraction is printed rather
// than hidden, because subtracting one measurement from another is exactly the
// kind of step that quietly invents a result.
// ---------------------------------------------------------------------------

struct Timing {
    double net_s;       // transform only, copy subtracted
    double copy_frac;   // how much of the raw measurement was the copy
    double spread;      // (median - min) / min, i.e. how noisy the machine was
};

template<FloatingPoint T>
static void fill_noise(ComplexSignal<T>& c, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t i = 0; i < c.size(); ++i) {
        c.real_data()[i] = static_cast<T>(d(rng));
        c.imag_data()[i] = static_cast<T>(d(rng));
    }
}

template<class Policy, FloatingPoint T>
static Timing time_forward(std::size_t N) {
    const FFTPlan<T, Policy> plan(N);         // built outside the timed region

    ComplexSignal<T> src(N, 48000.0), work(N, 48000.0);
    fill_noise(src, 7);

    auto refill = [&] {
        std::copy(src.real_data(), src.real_data() + N, work.real_data());
        std::copy(src.imag_data(), src.imag_data() + N, work.imag_data());
        bench::clobber();      // the copy must actually happen
        };

    const auto full = bench::measure([&] { refill(); plan.forward(work); bench::keep(work.real_data()[0]); });
    const auto copy = bench::measure([&] { refill();                     bench::keep(work.real_data()[0]); });

    return { full.median_s - copy.median_s,
             copy.median_s / full.median_s,
             (full.median_s - full.min_s) / full.min_s };
}

// The conventional operation count for a radix-2 complex FFT. Not a real flop
// count -- it is the figure of merit FFTW and friends report, so quoting it
// keeps this comparable in form to published numbers without pretending a
// comparison has been run.
static double fft_gflops(std::size_t N, double seconds) {
    const double ops = 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
    return ops / seconds / 1e9;
}

// ---------------------------------------------------------------------------

template<FloatingPoint T>
static void verify(const char* type_name) {
    for (std::size_t N : {16u, 64u, 1024u, 4096u}) {
        ComplexSignal<T> a(N, 48000.0), b(N, 48000.0);
        fill_noise(a, 11);
        std::copy(a.real_data(), a.real_data() + N, b.real_data());
        std::copy(a.imag_data(), a.imag_data() + N, b.imag_data());

        const FFTPlan<T, detail::ScalarPolicy> ps(N);
        const FFTPlan<T, detail::SimdPolicy>   pv(N);
        ps.forward(a);
        pv.forward(b);

        double worst = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(a.real_data()[i] - b.real_data()[i])));
            worst = std::max(worst, std::abs(static_cast<double>(a.imag_data()[i] - b.imag_data()[i])));
            scale = std::max(scale, std::abs(static_cast<double>(a.real_data()[i])));
        }
        // Relative tolerance: FFT output magnitude scales with sqrt(N), so an
        // absolute bound would be meaningless across the sweep.
        const double rel = worst / std::max(scale, 1e-30);
        const double limit = std::is_same_v<T, float> ? 1e-4 : 1e-11;
        if (rel > limit) {
            std::fprintf(stderr, "%s: policies disagree at N=%zu (rel %.2e) -- aborted\n",
                type_name, N, rel);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("%-7s policies agree across sizes\n", type_name);
}

template<FloatingPoint T>
static void sweep_fft(const char* type_name) {
    std::printf("\nComplex FFT, %s, in-place, plan reused\n", type_name);
    std::printf("%9s %12s %12s %9s %11s %11s %8s\n",
        "N", "scalar ms", "simd ms", "speedup", "scalar GF", "simd GF", "copy%");

    for (std::size_t N : {64u, 256u, 1024u, 4096u, 16384u, 65536u, 262144u, 1048576u}) {
        const auto s = time_forward<detail::ScalarPolicy, T>(N);
        const auto v = time_forward<detail::SimdPolicy, T>(N);

        std::printf("%9zu %12.5f %12.5f %8.2fx %11.2f %11.2f %7.0f%%\n",
            N, s.net_s * 1e3, v.net_s * 1e3, s.net_s / v.net_s,
            fft_gflops(N, s.net_s), fft_gflops(N, v.net_s),
            100.0 * v.copy_frac);
    }

    std::printf("  note: SimdPolicy falls back to scalar when half < width<%s>() = %zu,\n"
        "        so the first %zu butterfly stages of every transform are scalar\n"
        "        regardless of N. That is why small N shows little or no gain.\n",
        type_name, detail::SimdPolicy::width<T>(),
        static_cast<std::size_t>(std::bit_width(detail::SimdPolicy::width<T>()) - 1));
}

// ---------------------------------------------------------------------------
// Isolating the gather.
//
// simd_policy.hpp loads twiddles with _mm256_i32gather_ps whenever stride != 1,
// and takes a contiguous load only in the final stage. Gather has poor
// throughput on most x86 parts -- it is decomposed into individual loads
// internally -- so if the SIMD speedup above is disappointing, this is the
// first place to look. Same butterfly, same size, only the twiddle access
// pattern differs.
// ---------------------------------------------------------------------------
template<FloatingPoint T>
static void bench_butterfly_access(const char* type_name) {
    constexpr std::size_t half = 4096;

    std::printf("\nButterfly twiddle access, %s, half=%zu\n", type_name, half);
    std::printf("%9s %14s %14s %10s\n", "stride", "scalar ms", "simd ms", "speedup");

    for (std::size_t stride : {1u, 2u, 8u, 64u}) {
        std::vector<T> tw_re(half * stride), tw_im(half * stride);
        for (std::size_t i = 0; i < tw_re.size(); ++i) {
            const double a = -2.0 * 3.14159265358979 * double(i) / double(tw_re.size() * 2);
            tw_re[i] = static_cast<T>(std::cos(a));
            tw_im[i] = static_cast<T>(std::sin(a));
        }

        std::vector<T> src_re(2 * half), src_im(2 * half), re(2 * half), im(2 * half);
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for (std::size_t i = 0; i < src_re.size(); ++i) {
            src_re[i] = static_cast<T>(d(rng));
            src_im[i] = static_cast<T>(d(rng));
        }

        // The butterfly roughly doubles magnitudes per call, so this needs the
        // same refill treatment as the transforms above.
        auto refill = [&] {
            std::copy(src_re.begin(), src_re.end(), re.begin());
            std::copy(src_im.begin(), src_im.end(), im.begin());
            bench::clobber();
            };

        const auto sc = bench::measure([&] {
            refill();
            detail::ScalarPolicy::butterfly_block(re.data(), im.data(), tw_re.data(), tw_im.data(), 0, half, stride);
            bench::keep(re[0]);
            });
        const auto sv = bench::measure([&] {
            refill();
            detail::SimdPolicy::butterfly_block(re.data(), im.data(), tw_re.data(), tw_im.data(), 0, half, stride);
            bench::keep(re[0]);
            });
        const auto cp = bench::measure([&] { refill(); bench::keep(re[0]); });

        const double s = sc.median_s - cp.median_s;
        const double v = sv.median_s - cp.median_s;
        std::printf("%9zu %14.5f %14.5f %9.2fx%s\n", stride, s * 1e3, v * 1e3, s / v,
            stride == 1 ? "   <- contiguous load" : "   <- gather");
    }
}

// ---------------------------------------------------------------------------

template<FloatingPoint T>
static void sweep_rfft(const char* type_name) {
    std::printf("\nReal FFT, %s (returns N/2+1 bins; allocates its output per call)\n", type_name);
    std::printf("%9s %12s %12s %9s %14s\n",
        "N", "scalar ms", "simd ms", "speedup", "vs complex FFT");

    for (std::size_t N : {1024u, 16384u, 262144u, 1048576u}) {
        Signal<T> x(N, 48000.0);
        std::mt19937 rng(13);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for (std::size_t i = 0; i < N; ++i) x[i] = static_cast<T>(d(rng));

        // No refill needed: RFFTPlan::forward takes a const input and returns a
        // fresh spectrum, so the input never degrades.
        const RFFTPlan<T, detail::ScalarPolicy> rs(N);
        const RFFTPlan<T, detail::SimdPolicy>   rv(N);

        const auto s = bench::measure([&] { auto X = rs.forward(x); bench::keep(X.real_data()[0]); });
        const auto v = bench::measure([&] { auto X = rv.forward(x); bench::keep(X.real_data()[0]); });

        // What the real-input specialisation buys over just zero-filling the
        // imaginary part and running a full N-point complex transform.
        const auto c = time_forward<detail::SimdPolicy, T>(N);

        std::printf("%9zu %12.5f %12.5f %8.2fx %13.2fx\n",
            N, s.median_s * 1e3, v.median_s * 1e3,
            s.median_s / v.median_s, c.net_s / v.median_s);
    }
}

// ---------------------------------------------------------------------------
// Plan construction is O(N) transcendental calls, which is not cheap relative
// to one O(N log N) transform. This is the measurement that justifies FFTPlan
// existing at all next to the single-shot fft() free function.
// ---------------------------------------------------------------------------
template<FloatingPoint T>
static void sweep_plan_cost(const char* type_name) {
    std::printf("\nPlan construction vs one forward transform, %s\n", type_name);
    std::printf("%9s %14s %14s %12s\n", "N", "plan ms", "forward ms", "plan/forward");

    for (std::size_t N : {1024u, 16384u, 262144u, 1048576u}) {
        const auto p = bench::measure([&] {
            FFTPlan<T, detail::SimdPolicy> plan(N);
            bench::keep(plan.size());
            });
        const auto f = time_forward<detail::SimdPolicy, T>(N);
        std::printf("%9zu %14.5f %14.5f %11.2fx\n",
            N, p.median_s * 1e3, f.net_s * 1e3, p.median_s / f.net_s);
    }
    std::printf("  a ratio above 1 means the single-shot fft() free function spends more\n"
        "  time building its twiddle table than transforming; reuse a plan.\n");
}

int main() {
    bench::print_environment();

    verify<float>("float");
    verify<double>("double");

    sweep_fft<float>("float");
    sweep_fft<double>("double");

    bench_butterfly_access<float>("float");
    bench_butterfly_access<double>("double");

    sweep_rfft<float>("float");
    sweep_rfft<double>("double");

    sweep_plan_cost<float>("float");

    return 0;
}