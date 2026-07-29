#include <fastdsp/filters/fir_design.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace fastdsp;

static int g_fail = 0;
#define CHECK(cond) do{ if(!(cond)){ ++g_fail; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n"; } }while(0)
#define CHECK_NEAR(a, b, tol) do{ const double _a=(a), _b=(b); if(!(std::abs(_a-_b) <= (tol))){ ++g_fail; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  |" << _a << " - " << _b \
              << "| > " << (tol) << "\n"; } }while(0)

static void test_window_shape(){
    for(auto w : {WindowType::Rectangular, WindowType::Hann,
                  WindowType::Hamming, WindowType::Blackman}){
        for(std::size_t N : {2u, 7u, 8u, 33u, 64u}){
            const auto v = make_window<double>(w, N);
            CHECK(v.size() == N);
            // symmetric window: w[n] == w[N-1-n]
            for(std::size_t n = 0; n < N; ++n) CHECK_NEAR(v[n], v[N-1-n], 1e-15);
            for(std::size_t n = 0; n < N; ++n) CHECK(v[n] >= -1e-15 && v[n] <= 1.0 + 1e-15);
        }
    }

    // Closed-form endpoints. These are the values that distinguish the
    // symmetric window from the periodic one, which is the easy mistake here.
    const std::size_t N = 16;
    CHECK_NEAR(make_window<double>(WindowType::Hann,     N)[0], 0.0,  1e-15);
    CHECK_NEAR(make_window<double>(WindowType::Hamming,  N)[0], 0.08, 1e-15);
    CHECK_NEAR(make_window<double>(WindowType::Blackman, N)[0], 0.0,  1e-15);

    // Odd length: the centre tap is exactly 1 for every cosine-sum window.
    const auto hann = make_window<double>(WindowType::Hann, 33);
    CHECK_NEAR(hann[16], 1.0, 1e-15);
}

static void test_linear_phase(){
    // Every design here must come out symmetric, or the phase is not linear
    // and latency() is a lie.
    const double fs = 48000.0;
    const auto lp = design_lowpass<double>(64, 4000.0, fs);
    const auto hp = design_highpass<double>(65, 4000.0, fs);
    const auto bp = design_bandpass<double>(64, 2000.0, 6000.0, fs);
    const auto bs = design_bandstop<double>(65, 2000.0, 6000.0, fs);

    for(const auto* h : {&lp, &hp, &bp, &bs}){
        const std::size_t N = h->size();
        for(std::size_t n = 0; n < N; ++n) CHECK_NEAR((*h)[n], (*h)[N-1-n], 1e-14);
    }
}

static void test_lowpass_response(){
    const double fs = 48000.0, fc = 4000.0;
    const auto h = design_lowpass<double>(129, fc, fs, WindowType::Blackman);

    CHECK_NEAR(gain_at(h, 0.0, fs), 1.0, 1e-12);        // unity DC by construction
    CHECK_NEAR(gain_at(h, fc, fs), 0.5, 0.06);          // ~-6 dB at the cutoff
    CHECK(gain_at(h, 12000.0, fs) < 1e-4);              // deep in the stopband
    CHECK(gain_at(h, 0.5 * fs, fs) < 1e-4);
}

static void test_highpass_response(){
    const double fs = 48000.0, fc = 8000.0;
    const auto h = design_highpass<double>(129, fc, fs, WindowType::Blackman);

    CHECK_NEAR(gain_at(h, 0.5 * fs, fs), 1.0, 1e-12);
    CHECK(gain_at(h, 0.0, fs) < 1e-4);
    CHECK(gain_at(h, 2000.0, fs) < 1e-4);
    CHECK(gain_at(h, 20000.0, fs) > 0.99);
}

static void test_bandpass_response(){
    const double fs = 48000.0, f1 = 4000.0, f2 = 8000.0;
    const auto h = design_bandpass<double>(255, f1, f2, fs, WindowType::Blackman);

    CHECK_NEAR(gain_at(h, 0.5 * (f1 + f2), fs), 1.0, 1e-12);
    CHECK(gain_at(h, 0.0, fs) < 1e-4);
    CHECK(gain_at(h, 0.5 * fs, fs) < 1e-4);
    CHECK(gain_at(h, 1000.0, fs) < 1e-3);
    CHECK(gain_at(h, 20000.0, fs) < 1e-3);
}

static void test_bandstop_response(){
    const double fs = 48000.0, f1 = 4000.0, f2 = 8000.0;
    const auto h = design_bandstop<double>(255, f1, f2, fs, WindowType::Blackman);

    CHECK_NEAR(gain_at(h, 0.0, fs), 1.0, 1e-12);
    CHECK(gain_at(h, 0.5 * (f1 + f2), fs) < 1e-3);
    CHECK(gain_at(h, 1000.0, fs) > 0.99);
    CHECK(gain_at(h, 20000.0, fs) > 0.99);
}

static void test_rejects_bad_arguments(){
    const double fs = 48000.0;
    bool threw;

    // Type II symmetric FIR has a forced zero at Nyquist, so an even-length
    // highpass or bandstop cannot exist.
    threw = false;
    try { design_highpass<double>(64, 4000.0, fs); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);

    threw = false;
    try { design_bandstop<double>(64, 2000.0, 6000.0, fs); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);

    threw = false;
    try { design_lowpass<double>(65, 30000.0, fs); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);   // above Nyquist

    threw = false;
    try { design_lowpass<double>(65, 0.0, fs); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);

    threw = false;
    try { design_bandpass<double>(65, 6000.0, 2000.0, fs); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);   // reversed band

    threw = false;
    try { design_lowpass<double>(0, 4000.0, fs); } catch(const std::invalid_argument&){ threw = true; }
    CHECK(threw);
}

static void test_float_matches_double(){
    const double fs = 48000.0;
    const auto hd = design_lowpass<double>(65, 5000.0, fs, WindowType::Hann);
    const auto hf = design_lowpass<float>(65, 5000.0, fs, WindowType::Hann);
    CHECK(hd.size() == hf.size());
    for(std::size_t n = 0; n < hd.size(); ++n) CHECK_NEAR(hd[n], hf[n], 1e-6);
}

int main(){
    test_window_shape();
    test_linear_phase();
    test_lowpass_response();
    test_highpass_response();
    test_bandpass_response();
    test_bandstop_response();
    test_rejects_bad_arguments();
    test_float_matches_double();

    if(g_fail) std::cerr << g_fail << " check(s) failed\n";
    else       std::cout << "test_fir_design: all checks passed\n";
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}