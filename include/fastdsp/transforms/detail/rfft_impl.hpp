#pragma once

#include <fastdsp/transforms/detail/fft_impl.hpp>
#include <fastdsp/transforms/detail/fft_common.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <bit>
#include <cassert>
#include <cstddef>

namespace fastdsp::detail{

// Real-input FFT via the half-size complex transform.
//
//   z[n] = x[2n] + i*x[2n+1],  n < M = N/2      (pack)
//   Z    = FFT_M(z)
//   Ze[k] = (Z[k] + conj(Z[M-k])) / 2           DFT of the even samples
//   Zo[k] = (Z[k] - conj(Z[M-k])) / 2i          DFT of the odd samples
//   X[k]  = Ze[k] + exp(-2*pi*i*k/N) * Zo[k]
//
// tw_half: TwiddleTable(M), consumed by fft_impl.
// tw_fix : TwiddleTable(N)  -> exp(-2*pi*i*k/N) for k < N/2, i.e. exactly
//          the k range the fixup loop needs. Two tables are required because
//          fft_impl derives its own index as j*(size/len) and so cannot walk
//          a size-N table with stride 2.
//
// Output is the half spectrum: N/2 + 1 bins. Bins 0 and N/2 are real.

template<class Policy, FloatingPoint T>
ComplexSignal<T> rfft_impl(const Signal<T>& x,
                           const TwiddleTable<T>& tw_half,
                           const TwiddleTable<T>& tw_fix)
{
    const std::size_t N = x.size();
    if(N == 0) return ComplexSignal<T>();

    assert(N >= 2 && std::has_single_bit(N));
    const std::size_t M = N / 2;

    // pack the real input as M interleaved complex samples
    ComplexSignal<T> z(M, x.get_sample_rate());
    {
        T* zr = z.real_data();
        T* zi = z.imag_data();
        for(std::size_t n = 0; n < M; ++n){
            zr[n] = x[2 * n];
            zi[n] = x[2 * n + 1];
        }
    }

    fft_impl<Policy>(z, tw_half);

    ComplexSignal<T> out(M + 1, x.get_sample_rate());

    const T* zr = z.real_data();
    const T* zi = z.imag_data();
    const T* wr = tw_fix.real();
    const T* wi = tw_fix.imag();
    T* Xr = out.real_data();
    T* Xi = out.imag_data();

    const T half = T(0.5);
    const std::size_t mask = M - 1;   // M is a power of two; wraps M-k at k=0

    for(std::size_t k = 0; k < M; ++k){
        const std::size_t m = (M - k) & mask;

        const T a = zr[k], b = zi[k];
        const T c = zr[m], d = zi[m];

        const T evr = half * (a + c);   // Ze[k]
        const T evi = half * (b - d);
        const T odr = half * (b + d);   // Zo[k]
        const T odi = half * (c - a);

        const T tr = wr[k] * odr - wi[k] * odi;
        const T ti = wr[k] * odi + wi[k] * odr;

        Xr[k] = evr + tr;
        Xi[k] = evi + ti;
    }

    // Nyquist: k = M gives Ze = Re(Z[0]), Zo = Im(Z[0]), twiddle = -1
    Xr[M] = zr[0] - zi[0];
    Xi[M] = T(0);

    return out;
}

// Inverse of the above. Input is the half spectrum (M+1 bins), output is 2M reals.
//
//   Ze[k]   = (X[k] + conj(X[M-k])) / 2
//   W*Zo[k] = (X[k] - conj(X[M-k])) / 2   ->  Zo = conj(W) * that
//   z[k]    = Ze[k] + i*Zo[k]
//   x       = deinterleave(IFFT_M(z))

template<class Policy, FloatingPoint T>
Signal<T> irfft_impl(const ComplexSignal<T>& X,
                     const TwiddleTable<T>& tw_half,
                     const TwiddleTable<T>& tw_fix)
{
    const std::size_t bins = X.size();
    if(bins < 2) return Signal<T>();

    const std::size_t M = bins - 1;
    const std::size_t N = 2 * M;
    assert(std::has_single_bit(M));

    ComplexSignal<T> z(M, X.sample_rate());

    const T* Xr = X.real_data();
    const T* Xi = X.imag_data();
    const T* wr = tw_fix.real();
    const T* wi = tw_fix.imag();
    T* zr = z.real_data();
    T* zi = z.imag_data();

    const T half = T(0.5);

    for(std::size_t k = 0; k < M; ++k){
        const std::size_t m = M - k;   // in [1, M], always a valid bin

        const T a = Xr[k], b = Xi[k];
        const T c = Xr[m], d = Xi[m];

        const T evr = half * (a + c);   // Ze[k]
        const T evi = half * (b - d);
        const T pr  = half * (a - c);   // W*Zo[k]
        const T pi  = half * (b + d);

        const T odr = wr[k] * pr + wi[k] * pi;   // Zo = conj(W) * p
        const T odi = wr[k] * pi - wi[k] * pr;

        zr[k] =   evr - odi;
        zi[k] = -(evi + odr);   // negated here: this is the conjugate step of the IFFT
    }

    // IFFT_M(z) = conj(FFT_M(conj(z))) / M; conj(z) was folded into the loop above
    fft_impl<Policy>(z, tw_half);

    const T inv = T(1) / static_cast<T>(M);
    Signal<T> out(N, X.sample_rate());
    for(std::size_t n = 0; n < M; ++n){
        out[2 * n]     =  zr[n] * inv;
        out[2 * n + 1] = -zi[n] * inv;
    }

    return out;
}

}