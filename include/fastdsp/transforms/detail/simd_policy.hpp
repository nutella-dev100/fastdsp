#pragma once

#include <fastdsp/core/Types.hpp>
#include <fastdsp/transforms/detail/scalar_policy.hpp>
#include <immintrin.h>
#include <cstddef>
#include <type_traits>

namespace fastdsp::detail{

struct SimdPolicy{
    template<FloatingPoint T>
    static constexpr std::size_t width() noexcept{
        return std::is_same_v<T, float> ? 8 : 4;
    }

    template<FloatingPoint T>
    static void butterfly_block(T* re, T* im,
                                const T* tw_re, const T* tw_im,
                                std::size_t base, std::size_t half, std::size_t stride)
    noexcept{
        //if butterfly block is smaller than 1 vector, fallback to scalar
        if(half < width<T>()){
            ScalarPolicy::butterfly_block(re, im, tw_re, tw_im, base, half, stride);
            return;
        }

        if constexpr (std::is_same_v<T, float>)
            block_f32(re, im, tw_re, tw_im, base, half, stride);    //float
        else
            block_f64(re, im, tw_re, tw_im, base, half, stride);    //double
    }

private:
    static void block_f32(float* re, float* im, 
                        const float* tw_re, const float* tw_im,
                        std::size_t base, std::size_t half, std::size_t stride)
    noexcept{
        // Twiddle index for lane k of iteration j is (j + k) * stride.
        // lane_step holds {0, s, 2s, ..., 7s}; each iteration adds j*stride.
        const __m256i lane_step = _mm256_mullo_epi32(
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            _mm256_set1_epi32(static_cast<int>(stride)));

        // half is a power of two and >= 8 here, so it divides evenly by 8:
        // no scalar tail is possible.
        for(std::size_t j = 0; j < half; j += 8){
            const std::size_t i0 = base + j;
            const std::size_t i1 = i0 + half;

            // Even and odd halves are contiguous -> plain loads.
            // loadu, not load: std::vector gives no 32-byte alignment guarantee.
            const __m256 e_re = _mm256_loadu_ps(re + i0);
            const __m256 e_im = _mm256_loadu_ps(im + i0);
            const __m256 o_re = _mm256_loadu_ps(re + i1);
            const __m256 o_im = _mm256_loadu_ps(im + i1);

            // Twiddles are strided -> gather (or a plain load when stride == 1,
            // which is the whole final stage and the bulk of the work).
            __m256 w_re, w_im;
            if(stride == 1){
                w_re = _mm256_loadu_ps(tw_re + j);
                w_im = _mm256_loadu_ps(tw_im + j);
            } else {
                const __m256i idx = _mm256_add_epi32(
                    lane_step, _mm256_set1_epi32(static_cast<int>(j * stride)));
                w_re = _mm256_i32gather_ps(tw_re, idx, 4); // scale 4 = sizeof(float)
                w_im = _mm256_i32gather_ps(tw_im, idx, 4);
            }

            // p = odd * w   (complex multiply, 8 butterflies at once)
            const __m256 p_re = _mm256_sub_ps(_mm256_mul_ps(o_re, w_re),
                                              _mm256_mul_ps(o_im, w_im));
            const __m256 p_im = _mm256_add_ps(_mm256_mul_ps(o_re, w_im),
                                              _mm256_mul_ps(o_im, w_re));

            _mm256_storeu_ps(re + i1, _mm256_sub_ps(e_re, p_re));
            _mm256_storeu_ps(im + i1, _mm256_sub_ps(e_im, p_im));
            _mm256_storeu_ps(re + i0, _mm256_add_ps(e_re, p_re));
            _mm256_storeu_ps(im + i0, _mm256_add_ps(e_im, p_im));
        }

    }

    static void block_f64(double* re, double* im,
                          const double* tw_re, const double* tw_im,
                          std::size_t base, std::size_t half, std::size_t stride) 
    noexcept{
        // Gather for doubles takes 4 x 32-bit indices in a 128-bit register.
        const __m128i lane_step = _mm_mullo_epi32(
            _mm_setr_epi32(0, 1, 2, 3),
            _mm_set1_epi32(static_cast<int>(stride)));

        for(std::size_t j = 0; j < half; j += 4){
            const std::size_t i0 = base + j;
            const std::size_t i1 = i0 + half;

            const __m256d e_re = _mm256_loadu_pd(re + i0);
            const __m256d e_im = _mm256_loadu_pd(im + i0);
            const __m256d o_re = _mm256_loadu_pd(re + i1);
            const __m256d o_im = _mm256_loadu_pd(im + i1);

            __m256d w_re, w_im;
            if(stride == 1){
                w_re = _mm256_loadu_pd(tw_re + j);
                w_im = _mm256_loadu_pd(tw_im + j);
            } else {
                const __m128i idx = _mm_add_epi32(
                    lane_step, _mm_set1_epi32(static_cast<int>(j * stride)));
                w_re = _mm256_i32gather_pd(tw_re, idx, 8); // scale 8 = sizeof(double)
                w_im = _mm256_i32gather_pd(tw_im, idx, 8);
            }

            const __m256d p_re = _mm256_sub_pd(_mm256_mul_pd(o_re, w_re),
                                               _mm256_mul_pd(o_im, w_im));
            const __m256d p_im = _mm256_add_pd(_mm256_mul_pd(o_re, w_im),
                                               _mm256_mul_pd(o_im, w_re));

            _mm256_storeu_pd(re + i1, _mm256_sub_pd(e_re, p_re));
            _mm256_storeu_pd(im + i1, _mm256_sub_pd(e_im, p_im));
            _mm256_storeu_pd(re + i0, _mm256_add_pd(e_re, p_re));
            _mm256_storeu_pd(im + i0, _mm256_add_pd(e_im, p_im));
        }
    }
};

}