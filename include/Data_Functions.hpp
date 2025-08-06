//
// Created by marcuskeil on 18/12/24.
//
#ifndef FFT_BENCH_DATA_FUNCTIONS_HPP
#define FFT_BENCH_DATA_FUNCTIONS_HPP
#pragma once
#include <cmath>
#include <vector>
#include <complex>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <unistd.h>
#include <algorithm>

#include "Abstract_FFT.hpp"

long long get_sys_mem();

int verify_dimension(int dim);

int possible_vector_size(float memory_size);

void fill_vector(std::complex<double>* v, int element_count);

void CT_transform(Abstract_FFT& fft_obj, int split);

void CT_radix_2(Abstract_FFT& fft_obj);
void CT_radix_3(Abstract_FFT& fft_obj);
void CT_radix_4(Abstract_FFT& fft_obj);
void CT_radix_5(Abstract_FFT& fft_obj);
void CT_radix_7(Abstract_FFT& fft_obj);
void CT_radix_8(Abstract_FFT& fft_obj);

void print_data(std::complex<double>* v, int element_count);

float compare_data(const std::complex<double>* v, int element_count);

#endif //FFT_BENCH_DATA_FUNCTIONS_HPP
