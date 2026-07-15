#pragma once

#include "Complex.hpp"
#include "Signal.hpp"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <cassert>

namespace fastdsp{
    template<typename T, typename alloc = std::allocator<T>> class ComplexSignal{
        private:
            Signal<T, alloc> real_;
            Signal<T, alloc> imag_;
        public:
            //constructors
            ComplexSignal() = default;
            ComplexSignal(std::size_t n) : real_(n), imag_(n) {}
            ComplexSignal(std::size_t n, double fs) : real_(n, fs), imag_(n, fs) {}

            std::size_t size() const noexcept {
                return real_.size();
            }

            void resize(const std::size_t sz){
                real_.resize(sz);
                imag_.resize(sz);
            }

            Signal<T, alloc>& real() noexcept {
                return real_;
            }

            const Signal<T, alloc>& real() const noexcept {
                return real_;
            }

            Signal<T, alloc>& imag() noexcept {
                return imag_;
            }

            const Signal<T, alloc>& imag() const noexcept {
                return imag_;
            }

            Complex<T> operator[] (std::size_t i) const {
                // if(i > real_.size()) throw std::out_of_range("Out of bounds access");
                assert(i < this->size());
                return Complex<T>(real_[i], imag_[i]);
            }

            void set(std::size_t i, const Complex<T> & val){
                assert(i < this->size());

                real_[i] = val.real();
                imag_[i] = val.imag();
            }

            T* real_data() noexcept { return real_.data(); }
            const T* real_data() const noexcept { return real_.data(); }

            T* imag_data() noexcept { return imag_.data(); }
            const T* imag_data() const noexcept { return imag_.data(); }

            double sample_rate() const noexcept {
                return real_.get_sample_rate();
            }

    };
}
