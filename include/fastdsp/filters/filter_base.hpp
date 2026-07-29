#pragma once

#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <cstddef>

namespace fastdsp{

// Streaming filter interface.
// Kept separate from convolution because convolution return n + taps - 1 samples

template<FloatingPoint T>
class FilterBase{
    public:
        virtual ~FilterBase() = default;

        // n samples in, n samples out. in == out is allowed (in-place);
        // partial overlap is not.
        virtual void process(const T* in, T* out, std::size_t n) = 0;

        // Clear internal state, as if freshly constructed. Coefficients are
        // not touched.
        virtual void reset() = 0;

        // Group delay in samples, for aligning filtered output against the
        // original. Exact only for linear-phase (symmetric) FIR.
        virtual std::size_t latency() const noexcept = 0;

        // --- convenience overloads ---------------------------------------
        // Derived classes must pull these in with
        //     using FilterBase<T>::process;
        // or the pointer overload will hide them.

        void process(const Signal<T>& in, Signal<T>& out){
            if(out.size() != in.size()) out.resize(in.size());
            if(in.size() == 0) return;
            process(in.data(), out.data(), in.size());
        }

        Signal<T> process(const Signal<T>& in){
            Signal<T> out(in.size(), in.get_sample_rate());
            if(in.size()) process(in.data(), out.data(), in.size());
            return out;
        }
};

}