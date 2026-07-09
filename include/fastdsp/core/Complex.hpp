#pragma once

#include <cmath>
#include <cstdlib>
#include <cstddef>

namespace fastdsp{
    template<typename T> class Complex{
        private:
            T re;
            T im;
        public:
            //constructors
            constexpr Complex() : re(0), im(0) {}
            Complex(T real, T imag) : re(real), im(imag) {}
            static Complex fromPolar(T r, T theta){
                return Complex(r * std::cos(theta), r * std::sin(theta));
            }

            //operator overloading
            constexpr Complex operator+(const Complex& other) const {
                return {re + other.re, im + other.im};
            }

            constexpr Complex operator-(const Complex& other) const {
                return {re - other.re, im - other.im};
            }

            constexpr Complex operator*(const Complex& other) const {
                return {
                    re * other.re - im * other.im,
                    re * other.im + im * other.re
                };
            }

            constexpr Complex operator*(T s) const {
                return {re * s, im * s};
            }

            friend constexpr Complex operator*(T s, Complex c){
                return c * s;
            }

            constexpr Complex operator/(T s) const {
                return {re / s, im / s};
            }

            constexpr Complex operator/(const Complex& other) const {
                T denom = other.re * other.re + other.im * other.im;
                return{
                    (re * other.re + im * other.im) / denom,
                    (im * other.re - re * other.im) / denom
                };
            }

            Complex& operator+=(const Complex& other) {
                re += other.re;
                im += other.im;
                return *this;
            }

            Complex& operator-=(const Complex& other) {
                re -= other.re;
                im -= other.im;
                return *this;
            }

            constexpr Complex operator-() const {
                return {-re, -im};
            }

            Complex& operator=(const Complex& other) {
                re = other.re;
                im = other.im;
                return *this;
            }

            constexpr bool operator==(const Complex&) const = default; 

            //helpers
            constexpr Complex conjugate() const {
                return {re, -im};
            }

            T magnitude() const{
                return std::sqrt(re * re + im * im);
            }

            T phase() const {
                return std::atan2(im, re);
            }

            //getters
            constexpr T real() const { return re; }
            constexpr T imag() const { return im; }
            
            //setters - change name maybe??
            constexpr T real(T x) { return re = x; }
            constexpr T imag(T x) { return im = x; }
    };
}
