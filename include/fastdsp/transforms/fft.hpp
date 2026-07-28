#pragma once

#include <fastdsp/transforms/detail/fft_impl.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#if defined(FASTDSP_HAS_AVX2)
  #include <fastdsp/transforms/detail/simd_policy.hpp>
#endif
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <cstddef>

namespace fastdsp{

//these are intended as single use function, where twiddle table is calc everytime. No cached table
template<FloatingPoint T>
void fft(ComplexSignal<T> &x){
    detail::fft_impl<detail::ScalarPolicy>(x, detail::TwiddleTable<T>(x.size()));
}

template<FloatingPoint T>
void ifft(ComplexSignal<T>& x) {
    const std::size_t N = x.size();
    T* im = x.imag_data();

    //conjugate
    for (std::size_t i = 0; i < N; ++i) im[i] = -im[i];
    fft(x);

    const T inv = T(1) / static_cast<T>(N);
    T* re = x.real_data();
    //scale and conjugate
    for (std::size_t i = 0; i < N; ++i) { re[i] *= inv; im[i] *= -inv; }
}

}