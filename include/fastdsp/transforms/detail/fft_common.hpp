#pragma once

#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/Complex.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <fastdsp/core/Types.hpp>
#include <cmath>
#include <numbers>
#include <utility>
#include <bit>

namespace fastdsp::detail{

template<FloatingPoint T>
class TwiddleTable{
    // TODO: extend twiddle table to avoid gather in simd instructions

    private:
        std::vector<T> real_;
        std::vector<T> imag_;
    
    public:
        explicit TwiddleTable(std::size_t N) : real_(N / 2), imag_(N / 2){
            // 2*pi/N
            const T theta = -T(2) * std::numbers::pi_v<T> / static_cast<T>(N);

            for(std::size_t k = 0; k < N / 2; ++k){
                const T angle = theta * static_cast<T>(k);
                real_[k] = std::cos(angle);
                imag_[k] = std::sin(angle);
            }
        }

        std::size_t size() const noexcept { return real_.size(); }

        const T* real() const noexcept { return real_.data(); }
        const T* imag() const noexcept { return imag_.data(); }

};

inline std::size_t reverse_bits(std::size_t value, unsigned bits) noexcept {
    std::size_t reversed = 0;

    for(unsigned i = 0; i < bits; ++i){
        reversed <<= 1;
        reversed |= (value & 1);
        value >>= 1;
    }

    return reversed;
}

template<FloatingPoint T>
void bit_reverse(ComplexSignal<T>& signal) noexcept
{
    const std::size_t N = signal.size();

    if(N <= 1)
        return;

    const unsigned bits = std::bit_width(N) - 1;

    T* real = signal.real_data();
    T* imag = signal.imag_data();

    for(std::size_t i = 0; i < N; ++i)
    {
        const std::size_t j = reverse_bits(i, bits);

        if(j <= i)
            continue;

        std::swap(real[i], real[j]);
        std::swap(imag[i], imag[j]);
    }
}

}