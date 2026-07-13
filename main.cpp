#include "include/fastdsp/core/ComplexSignal.hpp"
#include "include/fastdsp/core/Complex.hpp"
#include "include/fastdsp/core/Signal.hpp"
#include "include/fastdsp/transforms/dft.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // // Create a signal of 10 complex numbers
    // fastdsp::ComplexSignal<float> sig(10);
    
    // // Set a value
    // sig.set(0, fastdsp::Complex<float>(3.0f, 4.0f));
    
    // // Read a value
    // fastdsp::Complex<float> val = sig[0];
    
    // std::cout << "Signal[0] = " << val.real() << " + " << val.imag() << "i" << std::endl;
    // std::cout << "Magnitude: " << val.magnitude() << std::endl;
    std::size_t n = 4;
    fastdsp::Signal<float> sig(n);
    sig[0] = 1.0;
    sig[1] = sig[2] = sig[3] = 0.0;

    std::cout << "Input Signal:\n";
    for (std::size_t i = 0; i < sig.size(); ++i){
        std::cout << sig[i] << ' ';
    }
    std::cout << "\n\n";

    fastdsp::DFT<float> dft;
    auto spectrum = dft.forward(sig);

    std::cout << "DFT Output:\n";
    for (std::size_t i = 0; i < spectrum.size(); ++i)
    {
        auto c = spectrum[i];

        std::cout << std::fixed << std::setprecision(3)
                  << "X[" << i << "] = "
                  << c.real() << " + "
                  << c.imag() << "i\n";
    }

    std::cout << '\n';

    auto reconstructed = dft.inverse(spectrum);

    std::cout << "Reconstructed Signal:\n";
    for (std::size_t i = 0; i < reconstructed.size(); ++i)
    {
        std::cout << reconstructed[i] << ' ';
    }
    std::cout << '\n';
    
    return 0;
}