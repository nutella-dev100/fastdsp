#pragma once

#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/Complex.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <fastdsp/core/Types.hpp>
#include <cmath>
#include <numbers>

namespace fastdsp{
    template<FloatingPoint T>
    class DFT{
        public:
            DFT() noexcept = default;
            ~DFT() noexcept = default;

            //forward and reverse functions
            ComplexSignal<T> forward(const Signal<T> &input) const{
                std::size_t N = input.size();
                ComplexSignal<T> output(N, input.get_sample_rate());
                
                if(N == 0) return output;

                for(std::size_t k = 0; k < N; ++k){
                    Complex<T> sum(0, 0);
                    for(std::size_t n = 0; n < N; ++n){
                        T angle =
                            -static_cast<T>(2)
                            * std::numbers::pi_v<T>
                            * static_cast<T>(k)
                            * static_cast<T>(n)
                            / static_cast<T>(N);

                        Complex<T> twiddle(std::cos(angle), std::sin(angle));
                        sum += input[n] * twiddle;
                    }

                    output.set(k, sum);
                }

                return output;
            }

            Signal<T> inverse(const ComplexSignal<T> &input) const{
                std::size_t N = input.size();
                Signal<T> output(N);

                if(N == 0) return output;

                for(std::size_t k = 0; k < N; ++k){
                    Complex<T> sum(0, 0);
                    for(std::size_t n = 0; n < N; ++n){
                        T angle =
                            static_cast<T>(2)
                            * std::numbers::pi_v<T>
                            * static_cast<T>(k)
                            * static_cast<T>(n)
                            / static_cast<T>(N);

                        Complex<T> twiddle(std::cos(angle), std::sin(angle));
                        sum += input[n] * twiddle;
                    }
                    output[k] = sum.real() / static_cast<T>(N);
                }

                return output;
            }
    };
}