#pragma once

#include <fastdsp/core/Types.hpp>
#include <cstddef>

// TODO : Move the butterfly implementation into this directory

namespace fastdsp::simd{

struct ScalarOps{
    static constexpr const char* name() noexcept { return "scalar"; }

    template<FloatingPoint T>
    static constexpr std::size_t width() noexcept { return 1; }

    // y[i] += a * x[i]  for i in [0, n).
    template<FloatingPoint T>
    static void axpy(T* y, T a, const T* x, std::size_t n) noexcept {
        for(std::size_t i = 0; i < n; ++i) y[i] += a * x[i];
    }
};

}

// --- default selection -----------------------------------------------------
// __AVX2__ is defined by the compiler whenever -mavx2 is on, so this needs no
// separate FASTDSP_HAS_AVX2 define. Note that DefaultOps therefore differs
// between translation units compiled with different flags: anything templated
// on it must not be shared across such a boundary. Header-only single-target
// executables (which is what every test here is) are unaffected.

#if defined(__AVX2__) && defined(__FMA__)
    #include <fastdsp/simd/ops_avx2.hpp>
    namespace fastdsp::simd { using DefaultOps = Avx2Ops; }
#else
    namespace fastdsp::simd { using DefaultOps = ScalarOps; }
#endif