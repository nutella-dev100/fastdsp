#include <fastdsp/transforms/fft.hpp>
#include <fastdsp/transforms/fft_plan.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

using fastdsp::ComplexSignal;
using fastdsp::FFTPlan;
using fastdsp::fft;
using fastdsp::ifft;

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

template<typename T>
static void test_free_function_roundtrip(std::size_t N, T tol, unsigned seed) {
    ComplexSignal<T> x = random_signal<T>(N, seed);
    ComplexSignal<T> original = x;

    fft(x);
    ifft(x);

    for (std::size_t i = 0; i < N; ++i) {
        assert(close(x.real_data()[i], original.real_data()[i], tol));
        assert(close(x.imag_data()[i], original.imag_data()[i], tol));
    //     printf("i=%zu got=%.9f expected=%.9f diff=%.9f\n",
    //    i, x.real_data()[i], original.real_data()[i],
    //    std::fabs(x.real_data()[i] - original.real_data()[i]));
    }
    printf("[PASS] fft/ifft roundtrip (N=%zu, %s)\n", N,
           sizeof(T) == sizeof(float) ? "float" : "double");
}

template<typename T>
static void test_plan_roundtrip(std::size_t N, T tol, unsigned seed) {
    FFTPlan<T> plan(N);
    ComplexSignal<T> x = random_signal<T>(N, seed);
    ComplexSignal<T> original = x;

    plan.forward(x);
    plan.inverse(x);

    for (std::size_t i = 0; i < N; ++i) {
        assert(close(x.real_data()[i], original.real_data()[i], tol));
        assert(close(x.imag_data()[i], original.imag_data()[i], tol));
    //     printf("i=%zu got=%.9f expected=%.9f diff=%.9f\n",
    //    i, x.real_data()[i], original.real_data()[i],
    //    std::fabs(x.real_data()[i] - original.real_data()[i]));
    }
    printf("[PASS] FFTPlan forward/inverse roundtrip (N=%zu, %s)\n", N,
           sizeof(T) == sizeof(float) ? "float" : "double");
}

int main() {
    test_free_function_roundtrip<float>(8, 1e-5f, 1);
    test_free_function_roundtrip<float>(64, 1e-4f, 2);
    test_free_function_roundtrip<double>(8, 1e-12, 3);
    test_free_function_roundtrip<double>(64, 1e-11, 4);

    test_plan_roundtrip<double>(16, 1e-12, 5);
    test_plan_roundtrip<float>(16, 1e-5f, 6);

    printf("All roundtrip tests passed.\n");
    return 0;
}
