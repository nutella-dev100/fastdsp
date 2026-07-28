#include <fastdsp/transforms/rfft.hpp>
#include <fastdsp/transforms/rfft_plan.hpp>
#include <fastdsp/transforms/fft.hpp>
#include <fastdsp/transforms/dft.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>

using namespace fastdsp;

template<typename T>
static Signal<T> random_real(std::size_t N, std::mt19937& rng){
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    Signal<T> x(N);
    for(std::size_t i = 0; i < N; ++i) x[i] = dist(rng);
    return x;
}

template<typename T>
static void check(std::size_t N, T tol, bool use_dft, std::mt19937& rng){
    const std::size_t M = N / 2;
    const Signal<T> x = random_real<T>(N, rng);

    const ComplexSignal<T> R = rfft(x);
    assert(R.size() == M + 1);

    // 1. differential against the complex FFT with a zeroed imaginary part
    {
        ComplexSignal<T> c(N);
        for(std::size_t i = 0; i < N; ++i){ c.real()[i] = x[i]; c.imag()[i] = T(0); }
        fft(c);
        for(std::size_t k = 0; k <= M; ++k){
            assert(std::abs(R.real_data()[k] - c.real_data()[k]) < tol);
            assert(std::abs(R.imag_data()[k] - c.imag_data()[k]) < tol);
        }
    }

    // 2. independent oracle (DFT is O(N^2), small sizes only).
    //    DFT::forward returns all N bins; the first M+1 are the half spectrum.
    if(use_dft){
        const DFT<T> dft;
        const ComplexSignal<T> D = dft.forward(x);
        for(std::size_t k = 0; k <= M; ++k){
            assert(std::abs(R.real_data()[k] - D.real_data()[k]) < tol);
            assert(std::abs(R.imag_data()[k] - D.imag_data()[k]) < tol);
        }
    }

    // 3. DC and Nyquist are exactly real
    assert(R.imag_data()[0] == T(0));
    assert(R.imag_data()[M] == T(0));

    // 4. round trip
    {
        const Signal<T> y = irfft(R);
        assert(y.size() == N);
        for(std::size_t i = 0; i < N; ++i) assert(std::abs(y[i] - x[i]) < tol);
    }

    // 5. the plan agrees with the single-shot path, and is usable when const
    {
        const RFFTPlan<T> plan(N);
        assert(plan.spectrum_size() == M + 1);

        const ComplexSignal<T> P = plan.forward(x);
        for(std::size_t k = 0; k <= M; ++k){
            assert(P.real_data()[k] == R.real_data()[k]);
            assert(P.imag_data()[k] == R.imag_data()[k]);
        }

        const Signal<T> y = plan.inverse(P);
        for(std::size_t i = 0; i < N; ++i) assert(std::abs(y[i] - x[i]) < tol);
    }
}

// impulse -> every bin exactly 1+0j; catches packing and bit-reversal errors
template<typename T>
static void check_impulse(std::size_t N){
    Signal<T> x(N);
    for(std::size_t i = 0; i < N; ++i) x[i] = T(0);
    x[0] = T(1);

    const ComplexSignal<T> R = rfft(x);
    for(std::size_t k = 0; k < R.size(); ++k){
        assert(R.real_data()[k] == T(1));
        assert(R.imag_data()[k] == T(0));
    }
}

// DC -> bin 0 = N, rest ~0; catches twiddle sign errors
template<typename T>
static void check_dc(std::size_t N, T tol){
    Signal<T> x(N);
    for(std::size_t i = 0; i < N; ++i) x[i] = T(1);

    const ComplexSignal<T> R = rfft(x);
    assert(std::abs(R.real_data()[0] - static_cast<T>(N)) < tol);
    assert(std::abs(R.imag_data()[0]) < tol);
    for(std::size_t k = 1; k < R.size(); ++k){
        assert(std::abs(R.real_data()[k]) < tol);
        assert(std::abs(R.imag_data()[k]) < tol);
    }
}

template<typename T>
static void run_all(T tol, std::mt19937& rng){
    // N=2,4 exercise the degenerate M=1,2 half transforms;
    // N>=32 (float) puts the half FFT past the SIMD width boundary
    const std::size_t small[] = {2, 4, 8, 16, 32, 64};
    for(std::size_t N : small) check<T>(N, tol, true, rng);

    const std::size_t large[] = {256, 1024, 4096};
    for(std::size_t N : large) check<T>(N, tol, false, rng);

    for(std::size_t N : small){ check_impulse<T>(N); check_dc<T>(N, tol); }
    check_impulse<T>(1024);
    check_dc<T>(1024, tol);
    std::cout << "Tests Passed" << std::endl;
}

int main(){
    std::mt19937 rng(12345);
    run_all<double>(1e-10, rng);
    run_all<float>(1e-3f, rng);
    return 0;
}