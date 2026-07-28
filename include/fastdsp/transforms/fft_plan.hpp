#pragma once

#include <fastdsp/transforms/detail/fft_impl.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#include <fastdsp/transforms/detail/fft_common.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <bit>
#include <cassert>
#include <cstddef>

namespace fastdsp{

template<FloatingPoint T, class Policy = detail::ScalarPolicy>
class FFTPlan{
    private:
        std::size_t size_;
        detail::TwiddleTable<T> twiddle_;

    public:
        explicit FFTPlan(std::size_t N) : size_(N), twiddle_(N){
            assert(std::has_single_bit(N));
        }

        std::size_t size() const noexcept {
            return size_;
        }

        void forward(ComplexSignal<T> &x) const {
            detail::fft_impl<Policy>(x, twiddle_);
        }

        void inverse(ComplexSignal<T> &x) const {
            //IFFT(X) = 1/N x (FFT(X*))*

            const std::size_t N = x.size();
            assert(N == size_);

            T* im = x.imag_data();

            //getting conjugate
            for(std::size_t i = 0; i < N; i++) im[i] = -im[i];
            forward(x);

            const T inv = T(1) / static_cast<T>(N);
            T* re = x.real_data();

            //fft(x) ka conjugate
            for(std::size_t i = 0; i < N; i++){
                re[i] *= inv;
                im[i] *= -inv;
            }
        }
};
    
}