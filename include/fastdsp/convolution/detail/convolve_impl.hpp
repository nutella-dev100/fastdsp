#pragma once

#include <fastdsp/transforms/rfft_plan.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <bit>
#include <cassert>
#include <cstddef>

namespace fastdsp::detail{

inline std::size_t next_pow2(std::size_t n) noexcept {
    return n <= 1 ? 1 : std::bit_ceil(n);
}

// ---------------------------------------------------------------------------
// The plan type and the three functions below are the only places the
// convolution layer knows how a spectrum is produced or represented. Nothing
// above them knows a spectrum has N/2+1 bins rather than N.
//
// The alias exists so the plan type is named once. Swapping the transform
// backend again should not require editing convolve.hpp or convolver.hpp.
// ---------------------------------------------------------------------------

template<FloatingPoint T, class Policy>
using ConvPlan = RFFTPlan<T, Policy>;

// Smallest usable transform size for a result of `ny` samples.
// RFFTPlan requires N >= 2; next_pow2 alone can return 1.
inline std::size_t plan_size_for(std::size_t ny) noexcept {
    const std::size_t N = next_pow2(ny);
    return N < 2 ? 2 : N;
}

// Zero-pad n real samples to the plan size and transform.
// Result has plan.spectrum_size() == N/2 + 1 bins.
template<class Policy, FloatingPoint T>
void forward_real(const T* src, std::size_t n,
                  const ConvPlan<T, Policy>& plan,
                  ComplexSignal<T>& out)
{
    const std::size_t N = plan.size();
    assert(n <= N);

    Signal<T> padded(N);                                  // zero-initialised
    T* dst = padded.data();
    for(std::size_t i = 0; i < n; ++i) dst[i] = src[i];

    out = plan.forward(padded);
}

// a <- a * b, pointwise. Unchanged by the switch to a half spectrum: the bound
// comes from a.size(), so it follows whatever the spectra actually are.
template<FloatingPoint T>
void multiply_into(ComplexSignal<T>& a, const ComplexSignal<T>& b) noexcept {
    const std::size_t N = a.size();
    assert(b.size() == N);

    T* ar = a.real_data();
    T* ai = a.imag_data();
    const T* br = b.real_data();
    const T* bi = b.imag_data();

    for(std::size_t i = 0; i < N; ++i){
        const T r = ar[i] * br[i] - ai[i] * bi[i];
        ai[i]     = ar[i] * bi[i] + ai[i] * br[i];
        ar[i]     = r;
    }
}

// Back to the time domain: writes plan.size() real samples into out.
// Previously the result was left in spec.real_data(); it is now a real signal
// in its own right, which is why the two call sites changed.
template<class Policy, FloatingPoint T>
void inverse_real(const ComplexSignal<T>& spec,
                  const ConvPlan<T, Policy>& plan,
                  Signal<T>& out)
{
    out = plan.inverse(spec);
}

}