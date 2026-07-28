#pragma once

#include <fastdsp/transforms/detail/rfft_impl.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#include <fastdsp/transforms/detail/fft_common.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <bit>
#include <cassert>
#include <cstddef>

namespace fastdsp{

template<FloatingPoint T, class Policy = detail::ScalarPolicy>
class RFFTPlan{
    private:
        std::size_t size_;                  // N, the real-domain length
        detail::TwiddleTable<T> half_;      // for the N/2-point complex FFT
        detail::TwiddleTable<T> fix_;       // exp(-2*pi*i*k/N), k < N/2

    public:
        explicit RFFTPlan(std::size_t N) : size_(N), half_(N / 2), fix_(N){
            assert(N >= 2 && std::has_single_bit(N));
        }

        std::size_t size() const noexcept { return size_; }
        std::size_t spectrum_size() const noexcept { return size_ / 2 + 1; }

        ComplexSignal<T> forward(const Signal<T>& x) const {
            assert(x.size() == size_);
            return detail::rfft_impl<Policy>(x, half_, fix_);
        }

        Signal<T> inverse(const ComplexSignal<T>& X) const {
            assert(X.size() == spectrum_size());
            return detail::irfft_impl<Policy>(X, half_, fix_);
        }
};

}