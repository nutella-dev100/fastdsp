#pragma once

#include <fastdsp/convolution/detail/convolve_impl.hpp>
#include <fastdsp/convolution/convolver.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#include <fastdsp/transforms/rfft_plan.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <algorithm>
#include <cstddef>

namespace fastdsp{

// All of these return the *full* linear convolution: nx + nh - 1 samples.
// 'same' and 'valid' are subranges of full, so they belong at the call site
// as slices rather than in the API as modes.

// O(nx * nh). Fastest for short kernels, and the independent oracle the FFT
// paths are tested against.
template<FloatingPoint T>
Signal<T> convolve_direct(const Signal<T>& x, const Signal<T>& h){
    const std::size_t nx = x.size();
    const std::size_t nh = h.size();
    if(nx == 0 || nh == 0) return Signal<T>();

    Signal<T> y(nx + nh - 1, x.get_sample_rate());   // zero-initialised

    // h outer, x inner: x and y are both walked sequentially
    for(std::size_t k = 0; k < nh; ++k){
        const T hk = h[k];
        for(std::size_t n = 0; n < nx; ++n) y[n + k] += hk * x[n];
    }

    return y;
}

// One transform pair over the whole signal, sized next_pow2(nx + nh - 1).
// That bound is >= the output length, so the cyclic convolution the FFT
// actually computes has no wraparound and equals the linear one.
//
// Only competitive when nx and nh are comparable — the FFT size is driven by
// nx even when h is tiny, so this scales badly for long inputs. Convolver
// blocks the work instead and is faster for everything else.
template<FloatingPoint T, class Policy = detail::ScalarPolicy>
Signal<T> convolve_fft(const Signal<T>& x, const Signal<T>& h){
    const std::size_t nx = x.size();
    const std::size_t nh = h.size();
    if(nx == 0 || nh == 0) return Signal<T>();

    const std::size_t ny = nx + nh - 1;
    const detail::ConvPlan<T, Policy> plan(detail::plan_size_for(ny));

    ComplexSignal<T> X, H;
    detail::forward_real<Policy>(x.data(), nx, plan, X);
    detail::forward_real<Policy>(h.data(), nh, plan, H);
    detail::multiply_into(X, H);
    Signal<T> full;
    detail::inverse_real<Policy>(X, plan, full);

    Signal<T> y(ny, x.get_sample_rate());
    for(std::size_t i = 0; i < ny; ++i) y[i] = full[i];

    return y;
}

// --- dispatch thresholds ---------------------------------------------------
// Both were measured (doubles, GCC 13, ScalarPolicy), not estimated, but on a
// different machine than the target. Re-measure before trusting them:
//
//   direct vs overlap-add, nx = 4096 and 65536: direct wins to nh = 32, ties
//   near nh = 64, loses from nh = 128 on.
//
//   one-shot vs overlap-add at nh = 512, 2048, 8192: one-shot wins only at
//   nx == nh, by roughly 25%. At nx = 2*nh overlap-add is already ahead, and
//   by nx = 16*nh it is 4x ahead.
inline constexpr std::size_t kDirectConvolutionMaxTaps = 64;

// Picks direct, one-shot FFT, or overlap-add. Convolution is commutative, so
// the choice depends only on the sorted pair of lengths.
template<FloatingPoint T, class Policy = detail::ScalarPolicy>
Signal<T> convolve(const Signal<T>& x, const Signal<T>& h){
    const std::size_t lo = std::min(x.size(), h.size());
    const std::size_t hi = std::max(x.size(), h.size());

    if(lo == 0) return Signal<T>();
    if(lo <= kDirectConvolutionMaxTaps) return convolve_direct(x, h);
    if(hi < 2 * lo) return convolve_fft<T, Policy>(x, h);

    // long input, shorter kernel: block it
    const bool h_is_kernel = h.size() <= x.size();
    const Signal<T>& kernel = h_is_kernel ? h : x;
    const Signal<T>& input  = h_is_kernel ? x : h;

    return Convolver<T, Policy>(kernel).convolve(input);
}

}