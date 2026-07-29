#include <fastdsp/simd/ops.hpp>
#include <fastdsp/simd/ops_avx2.hpp>
#include <fastdsp/filters/fir_filter.hpp>
#include <fastdsp/filters/fir_design.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace fastdsp;

static int g_fail = 0;
#define CHECK(cond) do{ if(!(cond)){ ++g_fail; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n"; } }while(0)
#define CHECK_NEAR(a, b, tol) do{ const double _a=(a), _b=(b); if(!(std::abs(_a-_b) <= (tol))){ ++g_fail; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  |" << _a << " - " << _b \
              << "| > " << (tol) << "\n"; } }while(0)

// Not bit-exact by design: FMA keeps the product at full width where the
// scalar path rounds it, so the AVX2 result is often the *more* accurate one.
// Compare against a tolerance, never with ==.
template<typename T> constexpr double tol();
template<> constexpr double tol<float>()  { return 1e-5; }
template<> constexpr double tol<double>() { return 1e-13; }

template<typename T>
static void test_axpy(){
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Lengths around the vector width, so every tail path gets exercised.
    for(std::size_t n : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 63u, 64u, 129u, 1000u}){
        std::vector<T> x(n), y0(n), y1(n);
        for(std::size_t i = 0; i < n; ++i){
            x[i]  = static_cast<T>(dist(rng));
            y0[i] = static_cast<T>(dist(rng));
        }
        y1 = y0;
        const T a = static_cast<T>(dist(rng));

        simd::ScalarOps::axpy(y0.data(), a, x.data(), n);
        simd::Avx2Ops::axpy(y1.data(), a, x.data(), n);

        for(std::size_t i = 0; i < n; ++i) CHECK_NEAR(y0[i], y1[i], tol<T>());
    }
}

template<typename T>
static void test_axpy_aliased(){
    // y and x pointing at the same buffer is legal and happens whenever a
    // caller squares a block; make sure the vector path agrees there too.
    const std::size_t n = 100;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<T> y0(n);
    for(auto& v : y0) v = static_cast<T>(dist(rng));
    std::vector<T> y1 = y0;

    simd::ScalarOps::axpy(y0.data(), T(0.5), y0.data(), n);
    simd::Avx2Ops::axpy(y1.data(), T(0.5), y1.data(), n);
    for(std::size_t i = 0; i < n; ++i) CHECK_NEAR(y0[i], y1[i], tol<T>());
}

template<typename T>
static void test_fir_backends_agree(){
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for(std::size_t taps : {1u, 4u, 8u, 9u, 33u, 128u, 129u}){
        Signal<T> h(taps, 48000.0);
        for(std::size_t i = 0; i < taps; ++i) h[i] = static_cast<T>(dist(rng));

        Signal<T> x(3000, 48000.0);
        for(std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<T>(dist(rng));

        FIRFilter<T, simd::ScalarOps> fs(h);
        FIRFilter<T, simd::Avx2Ops>   fv(h);

        const auto ys = fs.process(x);
        const auto yv = fv.process(x);

        // Scale the tolerance with the tap count: the accumulated error of a
        // taps-long sum grows with it, and for float at 129 taps a fixed 1e-5
        // is too tight to be meaningful.
        const double t = tol<T>() * static_cast<double>(taps);
        for(std::size_t i = 0; i < ys.size(); ++i) CHECK_NEAR(ys[i], yv[i], t);
    }
}

template<typename T>
static void test_fir_backends_agree_streaming(){
    const auto hd = design_lowpass<double>(64, 4000.0, 48000.0);
    Signal<T> h(hd.size(), 48000.0);
    for(std::size_t i = 0; i < hd.size(); ++i) h[i] = static_cast<T>(hd[i]);

    std::mt19937 rng(555);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Signal<T> x(2000, 48000.0);
    for(std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<T>(dist(rng));

    FIRFilter<T, simd::ScalarOps> fs(h, 37);
    FIRFilter<T, simd::Avx2Ops>   fv(h, 37);

    Signal<T> ys(x.size(), 48000.0), yv(x.size(), 48000.0);
    for(std::size_t pos = 0; pos < x.size(); pos += 13){
        const std::size_t m = std::min<std::size_t>(13, x.size() - pos);
        fs.process(x.data() + pos, ys.data() + pos, m);
        fv.process(x.data() + pos, yv.data() + pos, m);
    }
    const double t = tol<T>() * static_cast<double>(h.size());
    for(std::size_t i = 0; i < ys.size(); ++i) CHECK_NEAR(ys[i], yv[i], t);
}

int main(){
    // This binary is only built with -mavx2 -mfma, so DefaultOps must have
    // resolved to the vector path.
    CHECK(std::string(simd::DefaultOps::name()) == "avx2");

    test_axpy<float>();
    test_axpy<double>();
    test_axpy_aliased<float>();
    test_axpy_aliased<double>();
    test_fir_backends_agree<float>();
    test_fir_backends_agree<double>();
    test_fir_backends_agree_streaming<float>();
    test_fir_backends_agree_streaming<double>();

    if(g_fail) std::cerr << g_fail << " check(s) failed\n";
    else       std::cout << "test_fir_simd: all checks passed\n";
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}