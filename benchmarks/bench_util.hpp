#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// A small timing harness. Deliberately dependency-free: Google Benchmark would
// be the right call for a library with a build system to spare, but pulling it
// in for three sweeps costs more in setup than it returns, and a harness you
// wrote is a harness you can defend.
//
// Everything here exists to stop one of the four ways a microbenchmark lies:
//
//   1. The compiler deletes the work because nothing consumes the result.
//      -> keep() / clobber()
//   2. The first iteration pays for page faults, cold caches and an untrained
//      branch predictor, and you time that.                    -> warm-up
//   3. The measured interval is short enough that clock resolution and call
//      overhead dominate.                        -> repetition calibration
//   4. Some unrelated process on the machine steals time, and you report the
//      one run that happened to be slow.               -> repeated trials

namespace bench{

// --- keeping the optimiser honest ------------------------------------------
// Without this the whole benchmark can legally compile down to nothing: the
// output buffer is never read, so the filter call has no observable effect.
// The empty asm block claims to read `value` and to clobber memory, which
// forces the compiler to materialise it and to have committed any preceding
// stores first.

#if defined(__GNUC__) || defined(__clang__)
    template<typename T>
    inline void keep(const T& value){ asm volatile("" : : "r,m"(value) : "memory"); }
    inline void clobber(){ asm volatile("" : : : "memory"); }
#else
    namespace detail{ inline volatile char sink = 0; }
    template<typename T>
    inline void keep(const T& value){
        detail::sink = *reinterpret_cast<const volatile char*>(&value);
    }
    inline void clobber(){}
#endif

struct Result{
    double median_s;    // per-iteration
    double min_s;       // per-iteration, best trial
    std::size_t reps;   // iterations per trial
};

// Runs f() enough times to produce a stable per-call time.
//
// Reporting the median rather than the mean, and also the minimum: benchmark
// noise is one-sided (interference can only ever add time), so the minimum is
// the closest thing to the true cost, while the gap between minimum and median
// tells you whether the machine was quiet enough to believe either.
template<typename F>
Result measure(F&& f, double min_trial_s = 0.05, int trials = 5){
    using clock = std::chrono::steady_clock;
    using sec   = std::chrono::duration<double>;

    f();
    f();    // warm-up, never timed

    // Grow the repetition count until one trial is long enough that a
    // microsecond of clock error is irrelevant. Capped growth so a very fast
    // callee converges in a few steps without overshooting to a 10s trial.
    std::size_t reps = 1;
    for(;;){
        const auto t0 = clock::now();
        for(std::size_t i = 0; i < reps; ++i) f();
        const double dt = sec(clock::now() - t0).count();
        if(dt >= min_trial_s || reps >= (std::size_t{1} << 30)) break;
        const double want   = (dt > 0.0) ? (min_trial_s / dt) : 8.0;
        const double factor = std::clamp(want, 2.0, 8.0);
        reps = static_cast<std::size_t>(std::max(static_cast<double>(reps) * factor,
                                                static_cast<double>(reps + 1)));
    }

    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(trials));
    for(int i = 0; i < trials; ++i){
        const auto t0 = clock::now();
        for(std::size_t r = 0; r < reps; ++r) f();
        t.push_back(sec(clock::now() - t0).count() / static_cast<double>(reps));
    }
    std::sort(t.begin(), t.end());
    return { t[t.size() / 2], t.front(), reps };
}

// --- reporting -------------------------------------------------------------

inline std::string cpu_model(){
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while(std::getline(f, line)){
        const auto colon = line.find(':');
        if(colon != std::string::npos && line.compare(0, 10, "model name") == 0)
            return line.substr(colon + 2);
    }
    return "unknown";
}

// Print this at the top of every run. A benchmark number without the machine
// and flags attached is not a result, and this is what makes the table in
// bench_report.md reproducible rather than a claim.
inline void print_environment(){
    std::printf("cpu      : %s\n", cpu_model().c_str());
    std::printf("compiler : %s\n",
    #if defined(__clang__)
        "clang " __clang_version__
    #elif defined(__GNUC__)
        "gcc " __VERSION__
    #else
        "unknown"
    #endif
    );
    std::printf("simd     : %s\n",
    #if defined(__AVX2__) && defined(__FMA__)
        "AVX2 + FMA"
    #elif defined(__AVX2__)
        "AVX2 (no FMA!)"
    #else
        "scalar only -- build with -mavx2 -mfma"
    #endif
    );
#ifndef NDEBUG
    std::printf("WARNING  : assertions enabled (NDEBUG unset) -- build Release\n");
#endif
    std::printf("\n");
}

inline double msamples_per_s(std::size_t n, double seconds){
    return static_cast<double>(n) / seconds / 1e6;
}

// Two flops per multiply-accumulate.
inline double gflops(std::size_t macs, double seconds){
    return 2.0 * static_cast<double>(macs) / seconds / 1e9;
}

}