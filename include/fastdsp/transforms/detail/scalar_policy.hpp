#pragma once

#include <fastdsp/core/Types.hpp>
#include <cstddef>

namespace fastdsp::detail{

struct ScalarPolicy{
    template<FloatingPoint T>
    static void butterfly_block(T* re, T* im,
                                const T* tw_re, const T* tw_im,
                                std::size_t base, std::size_t half, std::size_t stride)
    noexcept{
        for(std::size_t j = 0; j < half; j++){
            const std::size_t i0 = base + j;
            const std::size_t i1 = i0 + half;
            const std::size_t t = j * stride;

            const T wr = tw_re[t], wi = tw_im[t];
            const T xr = re[i1], xi = im[i1];

            const T pre = xr * wr - xi * wi;
            const T pim = xr * wi + xi * wr;
            
            re[i1] = re[i0] - pre;
            im[i1] = im[i0] - pim;

            re[i0] += pre;
            im[i0] += pim;
        }
    }
};

}