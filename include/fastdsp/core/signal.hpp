#pragma once

#include <vector>
#include <string>
#include <cstdlib>

template<typename T> class signal{
    private:
        std::vector<T> samples;
        double sample_rate;
    public:

        //constructors
        signal() : sample_rate(44100){}
        signal(double fs) : sample_rate(fs){}
        signal(size_t len, double fs) : samples(len), sample_rate(fs) {}
        signal(size_t len, double fs, T val) : samples(len, val), sample_rate(fs) {}
        signal(const signal &s) : samples(s.samples), sample_rate(s.sample_rate){}
        signal(signal&& other) noexcept : samples(std::move(other.samples)), 
        sample_rate(other.sample_rate){}

        //move assignment
        signal& operator=(signal &&other) noexcept{
            if(this != other){
                samples = std::move(other.samples);
                sample_rate = other.sample_rate;
            }

            return *this;
        }

        //element access -> bound checking??
        T& operator[](size_t idx){
            return samples[idx];
        }

        const T& operator[](size_t idx) const{
            return samples[idx];
        }

        const T& at(size_t idx) const{
            if(idx < 0 || idx >= samples.size()) throw std::out_of_range("Index is out of valid bounds");
        }

        //overloaded operators
        //size(), empty() overloading

};