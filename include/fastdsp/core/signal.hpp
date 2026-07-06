#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include <cstddef>

template<typename T, typename Allocator = std::allocator<T>> class signal{
    private:
        std::vector<T, Allocator> samples;
        double sample_rate;
    public:

        // constructors
        signal() : sample_rate(44100){}
        signal(double fs) : sample_rate(fs){}
        signal(std::size_t len, double fs) : samples(len), sample_rate(fs) {}
        signal(std::size_t len, double fs, T val) : samples(len, val), sample_rate(fs) {}
        signal(const signal &s) : samples(s.samples), sample_rate(s.sample_rate){}
        signal(signal&& other) noexcept : samples(std::move(other.samples)), 
        sample_rate(other.sample_rate){}

        // move assignment
        signal& operator=(signal &&other) noexcept{
            if(this != &other){
                samples = std::move(other.samples);
                sample_rate = other.sample_rate;
            }

            return *this;
        }

        // element access -> bound checking??
        T& operator[](std::size_t idx){
            return samples[idx];
        }

        const T& operator[](std::size_t idx) const{
            return samples[idx];
        }

        T& at(std::size_t idx) {
            if(idx >= samples.size()) throw std::out_of_range("Index is out of valid bounds");
            return samples[idx];
        }

        const T& at(std::size_t idx) const{
            if(idx >= samples.size()) throw std::out_of_range("Index is out of valid bounds");
            return samples[idx];
        }

        T* data() noexcept {
            return samples.data();
        }

        const T* data() const noexcept {
            return samples.data();
        }
        
        // iterators
        // could also use typedef for these
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;
        using reverse_iterator = typename std::vector<T>::reverse_iterator;
        using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

        iterator begin() noexcept { return samples.begin(); }
        iterator end() noexcept { return samples.end(); }

        reverse_iterator rbegin() noexcept { return samples.rbegin(); }
        reverse_iterator rend() noexcept { return samples.rend(); }

        const_iterator cbegin() const noexcept { return samples.cbegin(); }
        const_iterator cend() const noexcept { return samples.cend(); }

        const_reverse_iterator crbegin() const noexcept { return samples.crbegin(); }
        const_reverse_iterator crend() const noexcept { return samples.crend(); }
        
        // modifiers
        void clear() noexcept {
            samples.clear();
        }

        void push_back(const T& item){
            samples.push_back(item);
        }

        void push_back(T&& item){
            samples.push_back(std::move(item));
        }

        void assign(std::size_t sz, const T& val){
            samples.assign(sz, val);
        }

        // TODO : check if this needs the allocator?
        // probably does
        void assign(std::vector<T, Allocator>&& inp){
            samples = std::move(inp);
        }

        void assign(const std::vector<T, Allocator>& inp) {
            samples = inp;
        }
        
        // meta-info
        std::size_t size() const noexcept {
            return samples.size();
        }

        bool empty() const noexcept {
            return samples.empty();
        }

        std::size_t size_bytes() const noexcept {
            return samples.size() * sizeof(T);
        }

        //getters
        double sample_rate() const noexcept{ return sample_rate; }

        void debug_print(std::ostream& os = std::cout) const {
            os << "{";
            for(auto ele : samples){
                os << ele << " ";
            }
            os << "}";
        }

        //alignment helpers
        bool is_aligned(std::size_t alignment) const noexcept {
            // TODO : understand this function
            // Gets base addr of pointer, converts to uintptr_t 
            // then checks if it is divisible by alignment
            return reinterpret_cast<std::uintptr_t>(data()) % alignment == 0;
        }

        static constexpr std::size_t alignment() noexcept {
            return alignof(T);
        }

};