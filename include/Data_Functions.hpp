//
// Created by marcuskeil on 18/12/24.
//
#ifndef FFT_BENCH_DATA_FUNCTIONS_HPP
#define FFT_BENCH_DATA_FUNCTIONS_HPP
#include <cmath>
#include <vector>
#include <complex>
#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unistd.h>
long long get_sys_mem();

int verify_dimension(int dim);

int possible_vector_size(float memory_size);

void fill_vector(std::complex<double>* v, int element_count);

std::vector<float> pre_plot_vector(std::complex<double>* v, int element_count);

std::vector<float> post_plot_vector(std::complex<double>* v, int element_count);

void print_data(std::complex<double>* v, int element_count);

void compare_data(const std::complex<double>* v, int element_count);

#endif //FFT_BENCH_DATA_FUNCTIONS_HPP
