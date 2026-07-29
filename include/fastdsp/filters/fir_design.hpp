#pragma once

#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>

// Windowed-sinc FIR design. No runtime state; everything here is a pure
// function of (kind, window, taps, cutoffs, fs).
//
// Conventions, fixed once here so the tests and the docs agree:
//
//   * Cutoffs are given in Hz together with fs, not as normalised fractions.
//     Internally everything works in fcn = fc/fs, which lies in (0, 0.5).
//
//   * sinc(x) = sin(pi*x)/(pi*x), so the ideal lowpass impulse response is
//     h[n] = 2*fcn * sinc(2*fcn*(n - M)) with M = (N-1)/2. Note M is a
//     half-integer for even N; that is correct, and gives the extra half
//     sample of delay a Type II filter has.
//
//   * Windows are SYMMETRIC (denominator N-1, so w[0] == w[N-1]). The
//     periodic form with denominator N belongs to spectral analysis, not to
//     filter design, and using it here costs you the exact linear phase.
//
//   * Every filter is normalised to unity gain at a reference frequency:
//     DC for lowpass and bandstop, Nyquist for highpass, band centre for
//     bandpass.

namespace fastdsp{

namespace detail{

inline constexpr double kPi = std::numbers::pi_v<double>;

inline double window_value(WindowType type, std::size_t n, std::size_t N) noexcept {
    if(N <= 1) return 1.0;
    const double x = static_cast<double>(n) / static_cast<double>(N - 1);
    switch(type){
        case WindowType::Rectangular: return 1.0;
        case WindowType::Hann:        return 0.5  - 0.5  * std::cos(2.0 * kPi * x);
        case WindowType::Hamming:     return 0.54 - 0.46 * std::cos(2.0 * kPi * x);
        case WindowType::Blackman:    return 0.42 - 0.5  * std::cos(2.0 * kPi * x)
                                                  + 0.08 * std::cos(4.0 * kPi * x);
    }
    return 1.0;
}

inline double sinc(double x) noexcept {
    // Removable singularity at 0. The cutoff is well inside the region where
    // sin(px)/px is still accurate, so no series expansion is needed.
    if(std::abs(x) < 1e-9) return 1.0;
    const double px = kPi * x;
    return std::sin(px) / px;
}

}   // namespace detail

// |H(e^{j 2 pi f / fs})|, evaluated directly in O(N). Used to normalise the
// designs below and to assert on the response in tests. For a full response
// sweep use rfft on the zero-padded coefficients instead; this is for spot
// checks at a handful of frequencies.
template<FloatingPoint T>
double gain_at(const Signal<T>& h, double f, double fs){
    const double w = 2.0 * detail::kPi * f / fs;
    double re = 0.0, im = 0.0;
    for(std::size_t n = 0; n < h.size(); ++n){
        const double a = w * static_cast<double>(n);
        re += static_cast<double>(h[n]) * std::cos(a);
        im -= static_cast<double>(h[n]) * std::sin(a);
    }
    return std::sqrt(re * re + im * im);
}

// Fill a window into a Signal. Exposed on its own because it is useful for
// spectral analysis too -- but see the symmetric/periodic note above before
// reaching for it there.
template<FloatingPoint T>
Signal<T> make_window(WindowType type, std::size_t N, double fs = 44100.0){
    if(N == 0) throw std::invalid_argument("make_window: N must be > 0");
    Signal<T> w(N, fs);
    for(std::size_t n = 0; n < N; ++n) w[n] = static_cast<T>(detail::window_value(type, n, N));
    return w;
}

namespace detail{

// Windowed-sinc lowpass, normalised to unity DC gain.
template<FloatingPoint T>
Signal<T> lowpass_prototype(std::size_t N, double fcn, WindowType win, double fs){
    Signal<T> h(N, fs);
    const double M = 0.5 * static_cast<double>(N - 1);

    double sum = 0.0;
    for(std::size_t n = 0; n < N; ++n){
        const double v = 2.0 * fcn * sinc(2.0 * fcn * (static_cast<double>(n) - M))
                       * window_value(win, n, N);
        h[n] = static_cast<T>(v);
        sum += v;
    }

    if(std::abs(sum) < 1e-12) throw std::invalid_argument("lowpass_prototype: degenerate design");
    for(std::size_t n = 0; n < N; ++n) h[n] = static_cast<T>(static_cast<double>(h[n]) / sum);
    return h;
}

// h <- delta(centre) - h. Turns a lowpass into a highpass, or a bandpass into
// a bandstop. Needs an integer centre tap, hence the odd-length requirement.
template<FloatingPoint T>
void spectral_invert(Signal<T>& h){
    const std::size_t N = h.size();
    for(std::size_t n = 0; n < N; ++n) h[n] = static_cast<T>(-h[n]);
    h[(N - 1) / 2] = static_cast<T>(static_cast<double>(h[(N - 1) / 2]) + 1.0);
}

template<FloatingPoint T>
void normalise_at(Signal<T>& h, double f, double fs){
    const double g = gain_at(h, f, fs);
    if(g < 1e-12) throw std::invalid_argument("normalise_at: zero gain at reference frequency");
    for(std::size_t n = 0; n < h.size(); ++n)
        h[n] = static_cast<T>(static_cast<double>(h[n]) / g);
}

inline void check_cutoff(double f, double fs, const char* what){
    if(!(f > 0.0) || !(f < 0.5 * fs))
        throw std::invalid_argument(std::string("design_fir: ") + what + " must lie in (0, fs/2)");
}

}   // namespace detail

// f_hi is ignored for LowPass and HighPass.
//
// HighPass and BandStop require an ODD tap count. An even-length symmetric FIR
// (Type II) has a forced zero at Nyquist, so it cannot pass the top of the
// band at all -- rejecting the request is better than silently returning a
// filter that does not do what was asked.
template<FloatingPoint T>
Signal<T> design_fir(FilterKind kind, WindowType win, std::size_t taps,
                     double f_lo, double f_hi, double fs)
{
    if(taps == 0)    throw std::invalid_argument("design_fir: taps must be > 0");
    if(!(fs > 0.0))  throw std::invalid_argument("design_fir: fs must be > 0");

    const bool needs_odd = (kind == FilterKind::HighPass || kind == FilterKind::BandStop);
    if(needs_odd && taps % 2 == 0)
        throw std::invalid_argument("design_fir: highpass/bandstop require an odd tap count");

    switch(kind){
        case FilterKind::LowPass: {
            detail::check_cutoff(f_lo, fs, "cutoff");
            return detail::lowpass_prototype<T>(taps, f_lo / fs, win, fs);
        }

        case FilterKind::HighPass: {
            detail::check_cutoff(f_lo, fs, "cutoff");
            Signal<T> h = detail::lowpass_prototype<T>(taps, f_lo / fs, win, fs);
            detail::spectral_invert(h);
            // The inversion is already close to unity at Nyquist; this only
            // removes the residual leakage of the lowpass up there.
            detail::normalise_at(h, 0.5 * fs, fs);
            return h;
        }

        case FilterKind::BandPass: {
            detail::check_cutoff(f_lo, fs, "lower cutoff");
            detail::check_cutoff(f_hi, fs, "upper cutoff");
            if(!(f_lo < f_hi)) throw std::invalid_argument("design_fir: need f_lo < f_hi");

            // Difference of two unity-DC lowpasses: the DC contributions
            // cancel exactly, leaving a bandpass.
            const Signal<T> lo = detail::lowpass_prototype<T>(taps, f_lo / fs, win, fs);
            const Signal<T> hi = detail::lowpass_prototype<T>(taps, f_hi / fs, win, fs);
            Signal<T> h(taps, fs);
            for(std::size_t n = 0; n < taps; ++n) h[n] = static_cast<T>(hi[n] - lo[n]);

            detail::normalise_at(h, 0.5 * (f_lo + f_hi), fs);
            return h;
        }

        case FilterKind::BandStop: {
            Signal<T> h = design_fir<T>(FilterKind::BandPass, win, taps, f_lo, f_hi, fs);
            detail::spectral_invert(h);
            detail::normalise_at(h, 0.0, fs);
            return h;
        }
    }

    throw std::invalid_argument("design_fir: unknown FilterKind");
}

// --- convenience wrappers --------------------------------------------------

template<FloatingPoint T>
Signal<T> design_lowpass(std::size_t taps, double fc, double fs, WindowType win = WindowType::Hamming){
    return design_fir<T>(FilterKind::LowPass, win, taps, fc, 0.0, fs);
}

template<FloatingPoint T>
Signal<T> design_highpass(std::size_t taps, double fc, double fs, WindowType win = WindowType::Hamming){
    return design_fir<T>(FilterKind::HighPass, win, taps, fc, 0.0, fs);
}

template<FloatingPoint T>
Signal<T> design_bandpass(std::size_t taps, double f_lo, double f_hi, double fs, WindowType win = WindowType::Hamming){
    return design_fir<T>(FilterKind::BandPass, win, taps, f_lo, f_hi, fs);
}

template<FloatingPoint T>
Signal<T> design_bandstop(std::size_t taps, double f_lo, double f_hi, double fs, WindowType win = WindowType::Hamming){
    return design_fir<T>(FilterKind::BandStop, win, taps, f_lo, f_hi, fs);
}

}