#include "include/fastdsp/core/ComplexSignal.hpp"
#include <iostream>

int main() {
    // Create a signal of 10 complex numbers
    fastdsp::ComplexSignal<float> sig(10);
    
    // Set a value
    sig.set(0, fastdsp::Complex<float>(3.0f, 4.0f));
    
    // Read a value
    fastdsp::Complex<float> val = sig[0];
    
    std::cout << "Signal[0] = " << val.real() << " + " << val.imag() << "i" << std::endl;
    std::cout << "Magnitude: " << val.magnitude() << std::endl;
    
    return 0;
}