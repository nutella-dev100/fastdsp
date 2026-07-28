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
)

foreach(test_src ${FASTDSP_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE fastdsp)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()