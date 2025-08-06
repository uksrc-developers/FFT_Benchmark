//
// Created by marcuskeil on 04/08/25.
//
#ifndef PLOTTING_FUNCTIONS_HPP
#define PLOTTING_FUNCTIONS_HPP
#pragma once

#ifdef PYTHON_PLOTTING
#include <cmath>
#include <complex>
#include "matplotlibcpp.h"

#include "Abstract_FFT.hpp"

namespace plt = matplotlibcpp;
#include <map>

std::vector<float> pre_plot_vector(std::complex<double>* v, int element_count);
std::vector<float> post_plot_vector(std::complex<double>* v, int element_count);

void create_preplot(const Abstract_FFT& fft_obj, const std::string& title, const std::string& file_name);
void create_postplot(const Abstract_FFT& fft_obj, const std::string& title, const std::string& file_name);
#endif

#endif //PLOTTING_FUNCTIONS_HPP
