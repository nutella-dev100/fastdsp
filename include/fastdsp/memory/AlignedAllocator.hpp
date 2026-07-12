#pragma once

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <new>
#include "Types.hpp"

namespace fastdsp{
    template<typename T, std::size_t alignment = kDefaultAlignment>
    class AlignedAllocator{
        public:
            using value_type = T;

            //constructors
            AlignedAllocator() noexcept = default;

            template<typename U>
            constexpr AlignedAllocator(const AlignedAllocator<U, alignment>&) noexcept {}

            [[nodiscard]]
            T* allocate(std::size_t n){
                //allocation logic
                if(n == 0) return nullptr;

                static_assert((alignment & (alignment - 1)) == 0, "alignment must be a power of 2");
                std::size_t sz = n * sizeof(T);     // TODO : add overflow check here

                //aligned alloc needs size to be a multiple of allignment
                std::size_t rem = sz % alignment;
                if(rem != 0){
                    sz += (alignment - rem);
                }

                //platform specific aligned alloc
                void *ptr = nullptr;
                #if defined(_MSC_VER)
                    ptr = _aligned_malloc(sz, alignment);
                #else
                    ptr = std::aligned_alloc(alignment, sz);
                #endif

                if(!ptr) throw std::bad_alloc();

                return static_cast<T*>(ptr);
            }

            void deallocate(T* p, std::size_t n) noexcept {
                //deallocation logic
                if(!p) return;

                //os specific dealloc
                #if defined(_MSC_VER)
                    _aligned_free(p);
                #else
                    std::free(p);
                #endif
            }
    };

    template<typename T, typename U, std::size_t alignment>
    bool operator==(const AlignedAllocator<T, alignment>&, const AlignedAllocator<U, alignment>&){
        return true;
    }

    template<typename T, typename U, std::size_t alignment>
    bool operator!=(const AlignedAllocator<T, alignment>&, const AlignedAllocator<U, alignment>&){
        return false;
    }
}