#pragma once

#include <vector>
#include <string>
#include <cstdlib>

template<typename T> class signal{
    private:
        std::vector<T> samples;
        double sample_rate;
    public:
        signal(){
            this->sample_rate = 44100;
        }

        signal(double fs){
            this->sample_rate = fs;
        }

        signal(size_t len, double fs){
            this->samples = samples(len);
        }

        //constructors
        //overloaded operators
        //move constructor
        //getter methods
        //size(), empty() overloading

};