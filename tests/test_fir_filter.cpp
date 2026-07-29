#include <fastdsp/filters/fir_filter.hpp>
#include <fastdsp/filters/fir_design.hpp>
#include <fastdsp/convolution/convolve.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace fastdsp;

static int g_fail = 0;
#define CHECK(cond) do{ if(!(cond)){ ++g_fail; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n"; } }while(0)
#define CHECK_NEAR(a, b, tol) do{ const double _a=(a), _b=(b); if(!(std::abs(_a-_b) <= (tol))){ ++g_fail; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  |" << _a << " - " << _b \
              << "| > " << (tol) << "\n"; } }while(0)

static Signal<double> random_signal(std::size_t n, unsigned seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Signal<double> x(n, 48000.0);
    for(std::size_t i = 0; i < n; ++i) x[i] = dist(rng);
    return x;
}

// The oracle: convolve_direct is already tested, and the first n samples of
// the full linear convolution are exactly what a streaming filter must emit.
static void test_matches_convolve_direct(){
    for(std::size_t taps : {1u, 2u, 3u, 15u, 64u, 65u, 200u}){
        const auto h = random_signal(taps, 1000u + static_cast<unsigned>(taps));
        const auto x = random_signal(1000, 7u);
        const auto ref = convolve_direct(x, h);

        FIRFilter<double> f(h);
        const auto y = f.process(x);

        CHECK(y.size() == x.size());
        for(std::size_t i = 0; i < y.size(); ++i) CHECK_NEAR(y[i], ref[i], 1e-12);
    }
}

// The real bug-finder: the history handling must be independent of how the
// caller chunks the stream, including chunks shorter than the tap count and
// chunks larger than the internal block.
static void test_chunking_invariance(){
    const auto h = random_signal(97, 3u);
    const auto x = random_signal(4321, 11u);
    const auto ref = convolve_direct(x, h);

    for(std::size_t block : {1u, 2u, 5u, 32u, 96u, 97u, 98u, 512u, 5000u}){
        FIRFilter<double> f(h, 64);   // internal block deliberately != call size
        Signal<double> y(x.size(), x.get_sample_rate());

        for(std::size_t pos = 0; pos < x.size(); pos += block){
            const std::size_t m = std::min<std::size_t>(block, x.size() - pos);
            f.process(x.data() + pos, y.data() + pos, m);
        }
        for(std::size_t i = 0; i < y.size(); ++i) CHECK_NEAR(y[i], ref[i], 1e-12);
    }

    // Irregular chunk sizes, the case a real callback actually produces.
    {
        std::mt19937 rng(99);
        std::uniform_int_distribution<std::size_t> dist(1, 300);
        FIRFilter<double> f(h, 128);
        Signal<double> y(x.size(), x.get_sample_rate());
        std::size_t pos = 0;
        while(pos < x.size()){
            const std::size_t m = std::min(dist(rng), x.size() - pos);
            f.process(x.data() + pos, y.data() + pos, m);
            pos += m;
        }
        for(std::size_t i = 0; i < y.size(); ++i) CHECK_NEAR(y[i], ref[i], 1e-12);
    }
}

static void test_impulse_response(){
    const auto h = design_lowpass<double>(31, 4000.0, 48000.0);
    FIRFilter<double> f(h);

    Signal<double> imp(64, 48000.0);
    imp[0] = 1.0;
    const auto y = f.process(imp);

    for(std::size_t n = 0; n < h.size(); ++n) CHECK_NEAR(y[n], h[n], 1e-15);
    for(std::size_t n = h.size(); n < y.size(); ++n) CHECK_NEAR(y[n], 0.0, 1e-15);
}

static void test_reset(){
    const auto h = random_signal(40, 5u);
    const auto x = random_signal(300, 6u);

    FIRFilter<double> f(h);
    const auto first = f.process(x);
    f.process(x);                 // dirty the history
    f.reset();
    const auto again = f.process(x);

    for(std::size_t i = 0; i < first.size(); ++i) CHECK_NEAR(first[i], again[i], 1e-15);
}

static void test_in_place(){
    const auto h = random_signal(37, 8u);
    const auto x = random_signal(777, 9u);

    FIRFilter<double> a(h), b(h);
    const auto out_of_place = a.process(x);

    Signal<double> buf = x;
    b.process(buf.data(), buf.data(), buf.size());

    for(std::size_t i = 0; i < buf.size(); ++i) CHECK_NEAR(buf[i], out_of_place[i], 1e-15);
}

// A filter behind FilterBase<T>* is the shape a benchmark harness uses to
// swap implementations; make sure the convenience overloads survive it.
static void test_polymorphic_use(){
    const auto h = design_lowpass<double>(33, 3000.0, 48000.0);
    FIRFilter<double> concrete(h);
    FilterBase<double>* f = &concrete;

    const auto x = random_signal(500, 12u);
    Signal<double> y;
    f->process(x, y);

    CHECK(y.size() == x.size());
    CHECK(f->latency() == 16);
    const auto ref = convolve_direct(x, h);
    for(std::size_t i = 0; i < y.size(); ++i) CHECK_NEAR(y[i], ref[i], 1e-12);
}

static void test_dc_and_passband_gain(){
    // End-to-end sanity: a lowpass should pass DC at unity and kill a tone
    // well inside its stopband.
    const double fs = 48000.0;
    const auto h = design_lowpass<double>(129, 2000.0, fs, WindowType::Blackman);
    FIRFilter<double> f(h);

    Signal<double> dc(2000, fs, 1.0);
    const auto y_dc = f.process(dc);
    CHECK_NEAR(y_dc[1500], 1.0, 1e-9);   // past the transient

    f.reset();
    Signal<double> tone(2000, fs);
    for(std::size_t n = 0; n < tone.size(); ++n)
        tone[n] = std::sin(2.0 * M_PI * 12000.0 * static_cast<double>(n) / fs);
    const auto y_tone = f.process(tone);

    double peak = 0.0;
    for(std::size_t n = 500; n < y_tone.size(); ++n) peak = std::max(peak, std::abs(y_tone[n]));
    CHECK(peak < 1e-3);
}

int main(){
    test_matches_convolve_direct();
    test_chunking_invariance();
    test_impulse_response();
    test_reset();
    test_in_place();
    test_polymorphic_use();
    test_dc_and_passband_gain();

    if(g_fail) std::cerr << g_fail << " check(s) failed\n";
    else       std::cout << "test_fir_filter: all checks passed\n";
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}