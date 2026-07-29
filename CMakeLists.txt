cmake_minimum_required(VERSION 3.15)
project(FastDSP VERSION 1.0.0 LANGUAGES CXX)

# Require C++20 standard (needed for 'operator== = default')
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define a header-only (INTERFACE) library target
add_library(fastdsp INTERFACE)

# Expose the current directory so headers can be found via #include "Signal.hpp"
target_include_directories(fastdsp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create a test executable to check for compilation errors
add_executable(fastdsp_test main.cpp)

# Link the test executable to our header-only library
target_link_libraries(fastdsp_test PRIVATE fastdsp)

enable_testing()

set(FASTDSP_TEST_SOURCES
    tests/test_bit_reverse.cpp
    tests/test_twiddle.cpp
    tests/test_fft_vs_dft.cpp
    tests/test_fft_roundtrip.cpp
    tests/test_fft_simd.cpp
    tests/test_convolve.cpp
    tests/test_rfft.cpp
    tests/test_fir_design.cpp
    tests/test_fir_filter.cpp
    tests/test_fir_simd.cpp
)

# Tests that need AVX2. Listing them here instead of hardcoding one target
# means the next SIMD test cannot silently be built without the flags, which
# would make it compile as scalar and quietly test nothing.
set(FASTDSP_SIMD_TESTS
    test_fft_simd
    test_fir_simd
)

foreach(test_src ${FASTDSP_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE fastdsp)
    add_test(NAME ${test_name} COMMAND ${test_name})

    if(${test_name} IN_LIST FASTDSP_SIMD_TESTS)
        target_compile_options(${test_name} PRIVATE -mavx2 -mfma)
        target_compile_definitions(${test_name} PRIVATE FASTDSP_HAS_AVX2)
    endif()
endforeach()

# --- benchmarks ------------------------------------------------------------
# Built at -O3 with NDEBUG regardless of CMAKE_BUILD_TYPE: a benchmark of a
# debug build measures nothing, and the most common way to publish a wrong
# number is to forget which build type was active.
option(FASTDSP_BUILD_BENCHMARKS "Build benchmark executables" ON)

if(FASTDSP_BUILD_BENCHMARKS)
    set(FASTDSP_BENCHMARKS
        benchmarks/bench_fir.cpp
        benchmarks/bench_convolve.cpp
    )

    foreach(bench_src ${FASTDSP_BENCHMARKS})
        get_filename_component(bench_name ${bench_src} NAME_WE)
        add_executable(${bench_name} ${bench_src})
        target_link_libraries(${bench_name} PRIVATE fastdsp)
        target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
        target_compile_options(${bench_name} PRIVATE -O3 -mavx2 -mfma)
        target_compile_definitions(${bench_name} PRIVATE NDEBUG FASTDSP_HAS_AVX2)
    endforeach()

    # Benchmarks are not tests: they measure, they do not assert. They still
    # verify their backends against a reference before timing, but that check
    # exists to stop a wrong number being published, not to replace ctest
endif()