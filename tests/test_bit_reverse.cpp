#include <fastdsp/transforms/detail/fft_common.hpp>
#include <cassert>
#include <cstdio>

using fastdsp::ComplexSignal;
using fastdsp::detail::reverse_bits;
using fastdsp::detail::bit_reverse;

static void test_reverse_bits() {
    // 3 bits: 0..7 reversed
    assert(reverse_bits(0b000, 3) == 0b000);
    assert(reverse_bits(0b001, 3) == 0b100);
    assert(reverse_bits(0b010, 3) == 0b010);
    assert(reverse_bits(0b011, 3) == 0b110);
    assert(reverse_bits(0b100, 3) == 0b001);
    assert(reverse_bits(0b101, 3) == 0b101);
    assert(reverse_bits(0b110, 3) == 0b011);
    assert(reverse_bits(0b111, 3) == 0b111);
    printf("[PASS] reverse_bits\n");
}

static void test_bit_reverse_permutation() {
    // N = 8: expected permutation of index i is reverse_bits(i, 3)
    ComplexSignal<double> x(8);
    for (std::size_t i = 0; i < 8; ++i) x.real_data()[i] = static_cast<double>(i);

    bit_reverse(x);

    for (std::size_t i = 0; i < 8; ++i) {
        const std::size_t expected = reverse_bits(i, 3);
        assert(x.real_data()[i] == static_cast<double>(expected));
    }
    printf("[PASS] bit_reverse permutation (N=8)\n");
}

static void test_bit_reverse_trivial_sizes() {
    // N <= 1 must be a no-op and must not crash (guards against unsigned underflow
    // in bit_width(N) - 1 when N == 0)
    ComplexSignal<double> x0(0);
    bit_reverse(x0);

    ComplexSignal<double> x1(1);
    x1.real_data()[0] = 42.0;
    bit_reverse(x1);
    assert(x1.real_data()[0] == 42.0);

    printf("[PASS] bit_reverse trivial sizes (N=0, N=1)\n");
}

int main() {
    test_reverse_bits();
    test_bit_reverse_permutation();
    test_bit_reverse_trivial_sizes();
    printf("All bit-reverse tests passed.\n");
    return 0;
}
