#pragma once

#if !defined(__AVX2__) || !defined(__FMA__)
    #error "ops_avx2.hpp requires -mavx2 -mfma"
#endif

#include <fastdsp/core/Types.hpp>
#include <immintrin.h>
#include <cstddef>
#include <type_traits>

namespace fastdsp::simd{

struct Avx2Ops{
    static constexpr const char* name() noexcept { return "avx2"; }

    template<FloatingPoint T>
    static constexpr std::size_t width() noexcept {
        return std::is_same_v<T, float> ? 8 : 4;
    }

    template<FloatingPoint T>
    static void axpy(T* y, T a, const T* x, std::size_t n) noexcept {
        if constexpr (std::is_same_v<T, float>) axpy_f32(y, a, x, n);
        else axpy_f64(y, a, x, n);
    }

private:
    // loadu/storeu throughout, deliberately. In the FIR inner loop x is a
    // sliding window into the delay line, so the load offset shifts by one
    // element per tap: no allocator can make these aligned. Alignment would
    // only help the y side, and splitting the kernel to get it is not worth
    // the code.
    //
    // Every iteration touches disjoint memory, so there is no loop-carried
    // dependency and no need to hand-unroll for FMA latency.
    static void axpy_f32(float* y, float a, const float* x, std::size_t n) noexcept {
        const __m256 va = _mm256_set1_ps(a);
        // Compute the end of the vector region rather than testing i + 8 <= n:
        // that test can overflow for absurd n, and GCC then cannot bound the
        // tail loop (-Waggressive-loop-optimizations fires).
        const std::size_t vend = n - (n % 8);
        std::size_t i = 0;
        for(; i < vend; i += 8){
            const __m256 vx = _mm256_loadu_ps(x + i);
            const __m256 vy = _mm256_loadu_ps(y + i);
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(vx, va, vy));
        }
        // Unlike the butterfly, n here is a block length and not a power of
        // two, so the tail is real.
        for(; i < n; ++i) y[i] += a * x[i];
    }

    static void axpy_f64(double* y, double a, const double* x, std::size_t n) noexcept {
        const __m256d va = _mm256_set1_pd(a);
        const std::size_t vend = n - (n % 4);
        std::size_t i = 0;
        for(; i < vend; i += 4){
            const __m256d vx = _mm256_loadu_pd(x + i);
            const __m256d vy = _mm256_loadu_pd(y + i);
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(vx, va, vy));
        }
        for(; i < n; ++i) y[i] += a * x[i];
    }
};

}