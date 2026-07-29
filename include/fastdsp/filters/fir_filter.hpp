#pragma once

#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/filters/filter_base.hpp>
#include <fastdsp/simd/ops.hpp>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastdsp{

// Direct-form FIR, streaming.
//
// Why this is not just Convolver: Convolver is const, allocates scratch per
// call, and keeps no tail between calls, so it can only do offline block work.
// This keeps taps-1 samples of history alive across process() calls and
// returns exactly as many samples as it consumed.
//
// Inner loop shape. The obvious formulation computes one output as a dot
// product over the taps, which needs a horizontal reduction per sample. This
// does the transpose of that: for each tap k it accumulates that tap's
// contribution across a whole block of outputs at once,
//
//     y[0..m) += h[k] * s[(taps-1-k) .. (taps-1-k+m))
//
// where s is [history | new block]. Every load is contiguous, there is no
// reduction anywhere, and the whole thing is one axpy per tap.
//
// The cost of that shape is that y is re-loaded and re-stored once per tap.
// y is L1-resident so this is cheap, but a fused kernel holding 8 output
// accumulators in registers across all taps is roughly twice as fast; that is
// the obvious next optimisation and a clean before/after for the benchmark
// report.
//
// After construction process() performs no allocation, so it is safe in an
// audio callback.

template<FloatingPoint T, class Ops = simd::DefaultOps>
class FIRFilter : public FilterBase<T>{
    private:
        Signal<T> h_;
        std::vector<T> hist_;      // taps_-1 most recent inputs, oldest first
        std::vector<T> scratch_;   // [ hist_ | up to block_ new samples ]
        std::size_t taps_;
        std::size_t block_;

    public:
        static constexpr std::size_t kDefaultBlock = 512;

        // block bounds the scratch buffer, nothing else: process() accepts any
        // n and chunks internally. Larger blocks amortise the per-tap loop
        // overhead; the default is already past the point of diminishing
        // returns for typical tap counts.
        explicit FIRFilter(const Signal<T>& h, std::size_t block = kDefaultBlock)
            : h_(h),
              taps_(h.size()),
              block_(block ? block : kDefaultBlock)
        {
            if(taps_ == 0) throw std::invalid_argument("FIRFilter: empty coefficient set");
            hist_.assign(taps_ - 1, T(0));
            scratch_.assign(taps_ - 1 + block_, T(0));
        }

        using FilterBase<T>::process;

        void process(const T* in, T* out, std::size_t n) override {
            const std::size_t tail = taps_ - 1;

            for(std::size_t pos = 0; pos < n; pos += block_){
                const std::size_t m = std::min(block_, n - pos);

                // Copying the input into scratch before writing any output is
                // what makes in == out safe: this chunk of `in` is never read
                // again afterwards, and later chunks are not yet written.
                std::copy(hist_.begin(), hist_.end(), scratch_.begin());
                std::copy(in + pos, in + pos + m, scratch_.begin() + tail);

                T* dst = out + pos;
                std::fill(dst, dst + m, T(0));

                // y[i] = sum_k h[k] * s[tail + i - k]
                for(std::size_t k = 0; k < taps_; ++k)
                    Ops::axpy(dst, h_[k], scratch_.data() + (tail - k), m);

                // The valid region of scratch is [0, tail + m); its last
                // `tail` samples become the new history. This is correct even
                // when m < tail, where the new history is part old and part
                // new -- those samples are already adjacent in scratch.
                if(tail) std::copy(scratch_.begin() + m,
                                   scratch_.begin() + m + tail,
                                   hist_.begin());
            }
        }

        void reset() override {
            std::fill(hist_.begin(), hist_.end(), T(0));
        }

        // Assumes a linear-phase (symmetric) coefficient set, which is what
        // everything in fir_design.hpp produces. For an even tap count the
        // true delay is (taps-1)/2 + 0.5 samples and this rounds down.
        std::size_t latency() const noexcept override { return (taps_ - 1) / 2; }

        std::size_t taps() const noexcept { return taps_; }
        std::size_t block_size() const noexcept { return block_; }
        const Signal<T>& coefficients() const noexcept { return h_; }
        static constexpr const char* backend() noexcept { return Ops::name(); }
};

}