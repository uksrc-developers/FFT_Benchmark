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

#include "Abstract_FFT.hpp"
#if __has_include( "matplotlibcpp.h" )
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#include <map>
#endif

long long get_sys_mem();

int verify_dimension(int dim);

int possible_vector_size(float memory_size);

void fill_vector(std::complex<double>* v, int element_count);

//template <typename T>
//void CT_radix_2(T);

std::vector<float> pre_plot_vector(std::complex<double>* v, int element_count);

std::vector<float> post_plot_vector(std::complex<double>* v, int element_count);

void print_data(std::complex<double>* v, int element_count);

float compare_data(const std::complex<double>* v, int element_count);

#if __has_include( "matplotlibcpp.h" )
void create_preplot(std::complex<double>* source_data, int element_count, const std::string& title, const std::string& file_name);
void create_postplot(std::complex<double>* source_data, int element_count, const std::string& title, const std::string& file_name);
#endif

#endif //FFT_BENCH_DATA_FUNCTIONS_HPP
