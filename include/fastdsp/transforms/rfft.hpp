#pragma once

#include <fastdsp/transforms/detail/rfft_impl.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#if defined(FASTDSP_HAS_AVX2)
  #include <fastdsp/transforms/detail/simd_policy.hpp>
#endif
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <cstddef>

namespace fastdsp{

// Single-use: both twiddle tables are built on every call. Use RFFTPlan for
// repeated transforms of one size.

// N real samples -> N/2 + 1 bins (the spectrum is Hermitian; the rest is mirror).
template<FloatingPoint T>
ComplexSignal<T> rfft(const Signal<T>& x){
    const std::size_t N = x.size();
    if(N == 0) return ComplexSignal<T>();

    return detail::rfft_impl<detail::ScalarPolicy>(
        x, detail::TwiddleTable<T>(N / 2), detail::TwiddleTable<T>(N));
}

// N/2 + 1 bins -> N real samples.
template<FloatingPoint T>
Signal<T> irfft(const ComplexSignal<T>& X){
    const std::size_t bins = X.size();
    if(bins < 2) return Signal<T>();

    const std::size_t M = bins - 1;
    return detail::irfft_impl<detail::ScalarPolicy>(
        X, detail::TwiddleTable<T>(M), detail::TwiddleTable<T>(2 * M));
}

}