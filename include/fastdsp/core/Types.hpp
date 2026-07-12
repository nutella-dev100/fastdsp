#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <concepts>

namespace fastdsp{
    using sample_t = float;
    using index_t = std::size_t;

    //simd alignment constants
    inline constexpr std::size_t kAvx2Alignment    = 32;  
    inline constexpr std::size_t kNeonAlignment    = 16;  
    inline constexpr std::size_t kDefaultAlignment = kAvx2Alignment;
    
    inline constexpr std::size_t kAvx2FloatLanes = 8;  
    inline constexpr std::size_t kNeonFloatLanes = 4;

    //constrain what Signal<T> and simd_ops can get
    template<typename T> 
    concept Numeric = std::is_arithmetic_v<T>;

    template<typename T>
    concept SimdCompatible = std::is_same_v<T, float> || std::is_same_v<T, double>;

    enum class WindowType {
        Rectangular,
        Hamming,
        Hann,
        Blackman
    };
 
    enum class FilterKind {
        LowPass,
        HighPass,
        BandPass,
        BandStop
    };
}