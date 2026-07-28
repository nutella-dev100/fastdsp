#include <fastdsp/transforms/fft_plan.hpp>
#include <fastdsp/transforms/detail/simd_policy.hpp>
#include <fastdsp/transforms/dft.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

using fastdsp::ComplexSignal;
using fastdsp::FFTPlan;
using fastdsp::DFT;
using fastdsp::detail::ScalarPolicy;
using fastdsp::detail::SimdPolicy;

template<typename T>
static bool close(T a, T b, T tol) { return std::fabs(a - b) <= tol; }

template<typename T>
static ComplexSignal<T> random_signal(std::size_t N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    ComplexSignal<T> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x.real_data()[i] = dist(rng);
        x.imag_data()[i] = dist(rng);
    }
    return x;
}

// The plan's stated debugging strategy: same input, both policies, diff them.
template<typename T>
static void test_simd_matches_scalar(std::size_t N, T tol, unsigned seed) {
    ComplexSignal<T> a = random_signal<T>(N, seed);
    ComplexSignal<T> b = a;

    FFTPlan<T, ScalarPolicy>(N).forward(a);
    FFTPlan<T, SimdPolicy>(N).forward(b);

    for (std::size_t i = 0; i < N; ++i) {
        assert(close(a.real_data()[i], b.real_data()[i], tol));
        assert(close(a.imag_data()[i], b.imag_data()[i], tol));
    }
    printf("[PASS] simd == scalar (N=%zu, %s)\n", N,
           sizeof(T) == sizeof(float) ? "float" : "double");
}

// Independent ground truth, so a bug shared by both policies can't hide.
template<typename T>
static void test_simd_matches_dft(std::size_t N, T tol, unsigned seed) {
    ComplexSignal<T> x = random_signal<T>(N, seed);
    ComplexSignal<T> reference = DFT<T>().forward(x);

    FFTPlan<T, SimdPolicy>(N).forward(x);

    for (std::size_t i = 0; i < N; ++i) {
        assert(close(x.real_data()[i], reference.real_data()[i], tol));
        assert(close(x.imag_data()[i], reference.imag_data()[i], tol));
    }
    printf("[PASS] simd == DFT (N=%zu, %s)\n", N,
           sizeof(T) == sizeof(float) ? "float" : "double");
}

template<typename T>
static void test_simd_roundtrip(std::size_t N, T tol, unsigned seed) {
    ComplexSignal<T> x = random_signal<T>(N, seed);
    ComplexSignal<T> original = x;

    const FFTPlan<T, SimdPolicy> plan(N);
    plan.forward(x);
    plan.inverse(x);

    for (std::size_t i = 0; i < N; ++i) {
        assert(close(x.real_data()[i], original.real_data()[i], tol));
        assert(close(x.imag_data()[i], original.imag_data()[i], tol));
    }
    printf("[PASS] simd roundtrip (N=%zu, %s)\n", N,
           sizeof(T) == sizeof(float) ? "float" : "double");
}

int main() {
    // N=4 and N=8 stay entirely on the scalar fallback path (half < width);
    // N>=16 (float) / N>=8 (double) start exercising real vector iterations.
    // Covering both sides of that boundary is the point of this size list.
    for (std::size_t N : {4u, 8u, 16u, 32u, 64u, 256u, 1024u}) {
        test_simd_matches_scalar<float>(N, 1e-4f, 11);
        test_simd_matches_scalar<double>(N, 1e-12, 12);
    }

    test_simd_matches_dft<float>(16, 1e-4f, 21);
    test_simd_matches_dft<double>(16, 1e-11, 22);
    test_simd_matches_dft<double>(64, 1e-10, 23);

    test_simd_roundtrip<float>(1024, 1e-4f, 31);
    test_simd_roundtrip<double>(1024, 1e-11, 32);

    printf("All SIMD tests passed.\n");
    return 0;
}