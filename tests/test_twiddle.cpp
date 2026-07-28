#include <fastdsp/transforms/detail/fft_common.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>

using fastdsp::detail::TwiddleTable;

template<typename T>
static bool close(T a, T b, T tol) { return std::fabs(a - b) <= tol; }

static void test_size() {
    TwiddleTable<double> tw(8);
    assert(tw.size() == 4); // N/2, not N
    printf("[PASS] TwiddleTable size == N/2\n");
}

static void test_n8_values() {
    // Plan's worked check: real_ should read 1, 0.7071, 0, -0.7071
    TwiddleTable<double> tw(8);
    const double* re = tw.real();
    const double* im = tw.imag();

    const double tol = 1e-4;
    assert(close(re[0],  1.0,      tol));
    assert(close(re[1],  0.70710678, tol));
    assert(close(re[2],  0.0,      tol));
    assert(close(re[3], -0.70710678, tol));

    // imag_[k] = sin(-2*pi*k/N): 0, -0.7071, -1, -0.7071
    assert(close(im[0],  0.0,        tol));
    assert(close(im[1], -0.70710678, tol));
    assert(close(im[2], -1.0,        tol));
    assert(close(im[3], -0.70710678, tol));

    printf("[PASS] TwiddleTable N=8 values match plan's worked example\n");
}

static void test_unit_magnitude() {
    // Every twiddle factor must sit on the unit circle.
    TwiddleTable<float> tw(64);
    for (std::size_t k = 0; k < tw.size(); ++k) {
        const float mag = std::sqrt(tw.real()[k] * tw.real()[k] +
                                     tw.imag()[k] * tw.imag()[k]);
        assert(close(mag, 1.0f, 1e-5f));
    }
    printf("[PASS] TwiddleTable unit-magnitude (N=64)\n");
}

int main() {
    test_size();
    test_n8_values();
    test_unit_magnitude();
    printf("All twiddle tests passed.\n");
    return 0;
}
