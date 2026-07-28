#pragma once

#include <fastdsp/convolution/detail/convolve_impl.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#include <fastdsp/transforms/rfft_plan.hpp>
#include <fastdsp/core/Types.hpp>
#include <fastdsp/core/Signal.hpp>
#include <fastdsp/core/ComplexSignal.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace fastdsp{

// Fixed kernel, many inputs: the kernel spectrum and the twiddle table are
// computed once. Input is processed in blocks by overlap-add.
//
// Holds only immutable state, so like FFTPlan every method is const and one
// instance is safe to share across threads. The cost of that choice is a
// scratch ComplexSignal allocated per convolve() call; a true streaming API
// would keep the block buffer and the overlap tail alive between calls and
// could not be const.

template<FloatingPoint T, class Policy = detail::ScalarPolicy>
class Convolver{
    private:
        std::size_t taps_;
        std::size_t fft_size_;
        std::size_t block_;
        detail::ConvPlan<T, Policy> plan_;
        ComplexSignal<T> kernel_;

        // Each block produces block_ + taps_ - 1 == fft_size_ samples of linear
        // convolution, exactly filling the transform. Correctness therefore only
        // needs fft_size_ >= taps_ (so block_ >= 1). The 2x below is a throughput
        // choice, not a correctness one: at fft_size_ == taps_ the block size
        // collapses to 1 and every sample costs a whole transform.
        static std::size_t choose_fft_size(std::size_t taps, std::size_t requested){
            const std::size_t least = detail::next_pow2(2 * taps);
            return requested >= least ? detail::next_pow2(requested) : least;
        }

    public:
        // fft_size = 0 picks a default of roughly 4x the kernel length, which
        // keeps the per-sample transform cost near its minimum for typical
        // kernels. Pass a size explicitly to trade latency against throughput.
        explicit Convolver(const Signal<T>& h, std::size_t fft_size = 0)
            : taps_(h.size()),
              fft_size_(choose_fft_size(h.size(), fft_size ? fft_size : 4 * h.size())),
              block_(fft_size_ - taps_ + 1),
              plan_(fft_size_)
        {
            assert(taps_ > 0);
            assert(block_ >= 1 && block_ + taps_ - 1 <= fft_size_);   // no wraparound
            detail::forward_real<Policy>(h.data(), taps_, plan_, kernel_);
        }

        std::size_t taps() const noexcept { return taps_; }
        std::size_t fft_size() const noexcept { return fft_size_; }
        std::size_t block_size() const noexcept { return block_; }

        // Full linear convolution, nx + taps - 1 samples.
        Signal<T> convolve(const Signal<T>& x) const {
            const std::size_t nx = x.size();
            if(nx == 0) return Signal<T>();

            const std::size_t ny = nx + taps_ - 1;
            Signal<T> y(ny, x.get_sample_rate());   // zero-initialised; OLA accumulates

            ComplexSignal<T> blk;
            Signal<T> out;
            const T* src = x.data();

            for(std::size_t pos = 0; pos < nx; pos += block_){
                const std::size_t n = std::min(block_, nx - pos);

                detail::forward_real<Policy>(src + pos, n, plan_, blk);
                detail::multiply_into(blk, kernel_);
                detail::inverse_real<Policy>(blk, plan_, out);

                // pos + n <= nx, so pos + span <= ny always — no clamp needed
                const std::size_t span = n + taps_ - 1;
                for(std::size_t i = 0; i < span; ++i) y[pos + i] += out[i];
            }

            return y;
        }
};

}