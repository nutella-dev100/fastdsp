#include <fastdsp/convolution/convolve.hpp>
#include <fastdsp/convolution/convolver.hpp>
#include <fastdsp/core/Signal.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>

using namespace fastdsp;

template<typename T>
static Signal<T> random_signal(std::size_t n, std::mt19937& rng){
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    Signal<T> s(n);
    for(std::size_t i = 0; i < n; ++i) s[i] = dist(rng);
    return s;
}

// Convolution accumulates, so the error floor is set by the largest sample in
// the result, not by whatever value happens to sit at index i. Normalising
// per-element instead demands absurd precision in the low-magnitude tails of a
// signal with wide dynamic range.
template<typename T>
static T scale_of(const Signal<T>& s){
    T m = T(1);
    for(std::size_t i = 0; i < s.size(); ++i) m = std::max(m, std::abs(s[i]));
    return m;
}

template<typename T>
static void same_signal(const Signal<T>& a, const Signal<T>& b, T tol){
    assert(a.size() == b.size());
    const T scale = scale_of(b);
    for(std::size_t i = 0; i < a.size(); ++i) assert(std::abs(a[i] - b[i]) <= tol * scale);
}

// 1. FFT path against the direct oracle
template<typename T>
static void check_fft_vs_direct(std::size_t nx, std::size_t nh, T tol, std::mt19937& rng){
    const Signal<T> x = random_signal<T>(nx, rng);
    const Signal<T> h = random_signal<T>(nh, rng);

    const Signal<T> ref = convolve_direct(x, h);
    assert(ref.size() == nx + nh - 1);

    same_signal(convolve_fft<T>(x, h), ref, tol);

    // commutative
    same_signal(convolve_fft<T>(h, x), ref, tol);
}

// 2. overlap-add against the same oracle, across block boundaries
template<typename T>
static void check_ola(std::size_t nx, std::size_t nh, std::size_t fft_size,
                      T tol, std::mt19937& rng)
{
    const Signal<T> x = random_signal<T>(nx, rng);
    const Signal<T> h = random_signal<T>(nh, rng);

    const Convolver<T> c(h, fft_size);
    assert(c.block_size() == c.fft_size() - nh + 1);

    same_signal(c.convolve(x), convolve_direct(x, h), tol);
}

// 3. a delta kernel reproduces the input, shifted
template<typename T>
static void check_delta(std::size_t nx, std::size_t nh, std::size_t d, T tol,
                        std::mt19937& rng)
{
    assert(d < nh);
    const Signal<T> x = random_signal<T>(nx, rng);
    Signal<T> h(nh);
    h[d] = T(1);

    Signal<T> expect(nx + nh - 1);          // zero except the shifted copy
    for(std::size_t i = 0; i < nx; ++i) expect[d + i] = x[i];

    same_signal(convolve_fft<T>(x, h), expect, tol);
    same_signal(Convolver<T>(h).convolve(x), expect, tol);
}

// 4. closed form: ones * ones is a triangle peaking at min(nx,nh)
template<typename T>
static void check_triangle(std::size_t nx, std::size_t nh, T tol){
    Signal<T> x(nx), h(nh);
    for(std::size_t i = 0; i < nx; ++i) x[i] = T(1);
    for(std::size_t i = 0; i < nh; ++i) h[i] = T(1);

    const std::size_t ny = nx + nh - 1;
    Signal<T> expect(ny);
    for(std::size_t n = 0; n < ny; ++n){
        std::size_t c = 0;
        for(std::size_t k = 0; k < nh; ++k)
            if(n >= k && n - k < nx) ++c;
        expect[n] = static_cast<T>(c);
    }

    same_signal(convolve_fft<T>(x, h), expect, tol);
}

template<typename T>
static void run_all(T tol, std::mt19937& rng){
    // sizes that land on, just under, and just over a power of two
    check_fft_vs_direct<T>(64, 33, tol, rng);
    check_fft_vs_direct<T>(100, 37, tol, rng);
    check_fft_vs_direct<T>(128, 128, tol, rng);
    check_fft_vs_direct<T>(1000, 64, tol, rng);
    check_fft_vs_direct<T>(1, 1, tol, rng);
    check_fft_vs_direct<T>(1, 64, tol, rng);
    check_fft_vs_direct<T>(4096, 200, tol, rng);

    // nx smaller than / equal to / spanning many blocks, and exact multiples
    const Convolver<T> probe(random_signal<T>(16, rng));
    const std::size_t b = probe.block_size();
    check_ola<T>(1, 16, 0, tol, rng);
    check_ola<T>(b - 1, 16, 0, tol, rng);
    check_ola<T>(b, 16, 0, tol, rng);
    check_ola<T>(b + 1, 16, 0, tol, rng);
    check_ola<T>(3 * b, 16, 0, tol, rng);
    check_ola<T>(1000, 64, 0, tol, rng);
    check_ola<T>(1000, 64, 256, tol, rng);      // explicit fft size
    check_ola<T>(1000, 64, 16, tol, rng);       // too small, must be raised
    check_ola<T>(5000, 3, 0, tol, rng);         // tiny kernel
    check_ola<T>(500, 500, 0, tol, rng);        // kernel as long as input

    check_delta<T>(300, 40, 0, tol, rng);
    check_delta<T>(300, 40, 17, tol, rng);
    check_delta<T>(300, 40, 39, tol, rng);

    check_triangle<T>(100, 40, tol);
    check_triangle<T>(1024, 1024, tol);

    // dispatcher: all three regimes, and both argument orders
    {
        const Signal<T> x = random_signal<T>(2000, rng);
        const Signal<T> tiny = random_signal<T>(8, rng);        // -> direct
        const Signal<T> similar = random_signal<T>(1500, rng);  // -> one-shot fft
        const Signal<T> kernel = random_signal<T>(100, rng);    // -> overlap-add

        for(const Signal<T>* h : {&tiny, &similar, &kernel}){
            same_signal(convolve<T>(x, *h), convolve_direct(x, *h), tol);
            same_signal(convolve<T>(*h, x), convolve_direct(*h, x), tol);
        }
    }

    // empty inputs
    {
        const Signal<T> e;
        const Signal<T> x = random_signal<T>(10, rng);
        assert(convolve_direct(x, e).size() == 0);
        assert(convolve_fft<T>(e, x).size() == 0);
        assert(Convolver<T>(x).convolve(e).size() == 0);
    }
}

int main(){
    std::mt19937 rng(987);
    run_all<double>(1e-10, rng);
    run_all<float>(1e-4f, rng);
    return 0;
}