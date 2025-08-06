//
// Created by marcuskeil on 18/12/24.
//
#ifndef FFT_BENCH_ABSTRACT_FFT_HPP
#define FFT_BENCH_ABSTRACT_FFT_HPP
#pragma once
#include <complex>
#include <chrono>
#include <iomanip>

class Abstract_FFT {
public:
    virtual ~Abstract_FFT() = default;

    virtual inline std::string name() = 0;
    virtual inline int get_side() const = 0;
    virtual inline size_t get_memory() = 0;
    virtual inline std::complex<double>* get_source() const = 0;
    virtual inline int get_element_count() const = 0;

    virtual void transform() = 0;
    virtual void cooley_tukey() = 0;
    virtual void partial_transform(std::complex<double>* partial_array, std::size_t size) = 0;
    virtual std::chrono::duration<double, std::milli> time_transform(int runs) = 0;
};
#endif //FFT_BENCH_ABSTRACT_FFT_HPP
