//
// Created by marcuskeil on 18/12/24.
//

#ifndef FFT_BENCH_ABSTRACT_FFT_HPP
#define FFT_BENCH_ABSTRACT_FFT_HPP
#include <chrono>
#include <fstream>
#include <iomanip>
#include "Data_Functions.hpp"

class Abstract_FFT {
public:
    virtual ~Abstract_FFT() = default;

    virtual inline std::string name() = 0;
    virtual inline int get_side() = 0;
    virtual inline size_t get_memory() = 0;
    virtual inline std::complex<double>* get_source() = 0;
    virtual inline int get_element_count() = 0;

    virtual void transform() = 0;
    virtual std::chrono::duration<double, std::milli> time_transform(int runs) = 0;
};


#endif //FFT_BENCH_ABSTRACT_FFT_HPP
