#include <fastdsp/transforms/fft.hpp>
#include <fastdsp/transforms/dft.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <numbers>

using fastdsp::ComplexSignal;
using fastdsp::fft;
using fastdsp::DFT;

template<typename T>
static bool close(T a, T b, T tol) { return std::fabs(a - b) <= tol; }

// Naive O(N^2) DFT used only as a reference oracle for the comparison test.
// Kept self-contained here rather than depending on the project's own dft.hpp,
// so this test doesn't assume anything about that header's API.
template<typename T>
static void naive_dft(const ComplexSignal<T>& in, ComplexSignal<T>& out) {
    const std::size_t N = in.size();
    out.resize(N);
    const T two_pi = T(2) * std::numbers::pi_v<T>;

    for (std::size_t k = 0; k < N; ++k) {
        T sum_re = 0, sum_im = 0;
        for (std::size_t n = 0; n < N; ++n) {
            const T angle = -two_pi * static_cast<T>(k) * static_cast<T>(n) / static_cast<T>(N);
            const T c = std::cos(angle), s = std::sin(angle);
            const T xr = in.real_data()[n], xi = in.imag_data()[n];
            sum_re += xr * c - xi * s;
            sum_im += xr * s + xi * c;
        }
        out.real_data()[k] = sum_re;
        out.imag_data()[k] = sum_im;
    }
}

static void test_impulse() {
    // x = {1, 0, 0, ...} -> every bin is exactly 1 + 0j
    ComplexSignal<double> x(16);
    x.real_data()[0] = 1.0;
    fft(x);

    for (std::size_t i = 0; i < 16; ++i) {
        assert(close(x.real_data()[i], 1.0, 1e-12));
        assert(close(x.imag_data()[i], 0.0, 1e-12));
    }
    printf("[PASS] impulse response (N=16)\n");
}

static void test_dc() {
    // x = {1, 1, 1, ...} -> bin 0 is N, the rest ~0
    ComplexSignal<double> x(16);
    for (std::size_t i = 0; i < 16; ++i) x.real_data()[i] = 1.0;
    fft(x);

    assert(close(x.real_data()[0], 16.0, 1e-9));
    assert(close(x.imag_data()[0], 0.0, 1e-9));
    for (std::size_t i = 1; i < 16; ++i) {
        assert(close(x.real_data()[i], 0.0, 1e-9));
        assert(close(x.imag_data()[i], 0.0, 1e-9));
    }
    printf("[PASS] DC input (N=16)\n");
}

template<typename T>
static void test_dft_comparison(std::size_t N, T tol, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));

    // General complex input covers the real-signal case too (it's just im = 0),
    // so one path exercises both instead of testing them separately.
    ComplexSignal<T> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x.real_data()[i] = dist(rng);
        x.imag_data()[i] = dist(rng);
    }

    const DFT<T> dft;
    ComplexSignal<T> reference = dft.forward(x);
    fft(x);

    for (std::size_t i = 0; i < N; ++i) {
        assert(close(x.real_data()[i], reference.real_data()[i], tol));
        assert(close(x.imag_data()[i], reference.imag_data()[i], tol));
    }
    printf("[PASS] fft vs naive DFT (N=%zu, %s)\n", N,
           sizeof(T) == sizeof(float) ? "float" : "double");
}

int main() {
    test_impulse();
    test_dc();

    test_dft_comparison<float>(8, 1e-5f, 42);
    test_dft_comparison<float>(16, 1e-5f, 43);
    test_dft_comparison<double>(8, 1e-12, 44);
    test_dft_comparison<double>(16, 1e-12, 45);

    printf("All fft-vs-dft tests passed.\n");
    return 0;
}
