#pragma once

#include <fastdsp/transforms/detail/fft_common.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <bit>
#include <cassert>
#include <cstddef>

namespace fastdsp::detail{

template<class Policy, FloatingPoint T>
void fft_impl(ComplexSignal<T>& x, const TwiddleTable<T> &tw){
    const std::size_t N = x.size();
    if(N < 1) return;
    assert(std::has_single_bit(N));
    assert(tw.size() == N / 2);

    bit_reverse(x);

    T* re = x.real_data();
    T* im = x.imag_data();

    //butterfly
    for(std::size_t len = 2; len <= N; len <<= 1){
        const std::size_t half = len >> 1;
        const std::size_t stride = N / len;
        for(std::size_t base = 0; base < N; base += len){
            Policy::butterfly_block(re, im, tw.real(), tw.imag(), base, half, stride);
        }
    }
}

}