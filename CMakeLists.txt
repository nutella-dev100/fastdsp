cmake_minimum_required(VERSION 3.15)
project(FastDSP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # M_PI etc. aren't defined by <cmath> on MSVC unless this is set first.
    add_compile_definitions(_USE_MATH_DEFINES)

    # /RTC1 (MSVC's default Debug runtime checks) is hard-incompatible with
    # any optimization flag. Benchmarks below force /O2 regardless of the
    # selected configuration, so drop RTC in Debug project-wide.
    string(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

add_library(fastdsp INTERFACE)
target_include_directories(fastdsp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(fastdsp_test main.cpp)
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

set(FASTDSP_SIMD_TESTS
    test_fft_simd
    test_fir_simd
)

if(MSVC)
    set(FASTDSP_AVX2_FLAGS /arch:AVX2)
else()
    set(FASTDSP_AVX2_FLAGS -mavx2 -mfma)
endif()

foreach(test_src ${FASTDSP_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE fastdsp)
    add_test(NAME ${test_name} COMMAND ${test_name})

    if(${test_name} IN_LIST FASTDSP_SIMD_TESTS)
        target_compile_options(${test_name} PRIVATE ${FASTDSP_AVX2_FLAGS})
        target_compile_definitions(${test_name} PRIVATE FASTDSP_HAS_AVX2)
    endif()
endforeach()

option(FASTDSP_BUILD_BENCHMARKS "Build benchmark executables" ON)

if(FASTDSP_BUILD_BENCHMARKS)
    set(FASTDSP_BENCHMARKS
        benchmarks/bench_fir.cpp
        benchmarks/bench_convolve.cpp
        benchmarks/bench_fft.cpp
    )

    if(MSVC)
        set(FASTDSP_OPT_FLAGS /O2)
    else()
        set(FASTDSP_OPT_FLAGS -O3)
    endif()

    foreach(bench_src ${FASTDSP_BENCHMARKS})
        get_filename_component(bench_name ${bench_src} NAME_WE)
        add_executable(${bench_name} ${bench_src})
        target_link_libraries(${bench_name} PRIVATE fastdsp)
        target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
        target_compile_options(${bench_name} PRIVATE ${FASTDSP_OPT_FLAGS} ${FASTDSP_AVX2_FLAGS})
        target_compile_definitions(${bench_name} PRIVATE NDEBUG FASTDSP_HAS_AVX2)
    endforeach()
endif()