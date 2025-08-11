//
// Created by marcuskeil on 05/08/25.
//
#ifndef RUN_FUNCTIONS_HPP
#define RUN_FUNCTIONS_HPP
#pragma once
#include <tuple>
#include <vector>
#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <stdexcept>

#include "Abstract_FFT.hpp"
#include "Plotting_Functions.hpp"
#include "Data_Functions.hpp"

std::tuple<bool, bool, std::string, bool, bool, bool, int, int, float> retrieve_arguments(int argc, char **argv);

std::vector<int> get_elements(int run_count);
std::vector<float> linspace(float start, float end, int count);
std::vector<float> get_memories(float start, int count);

inline float memory_retrieve(const int element_count) {
    switch (element_count) {
        case 31360000: return 501.76;
        case 63011844: return 1008.19;
        case 125440000: return 2007.04;
        case 248062500: return 3969;
        case 486202500: return 8028.16;
        case 992250000: return 15876;
        case 1864000000: return 30000; // test only
        case 2007040000: return 32112.6;
        default: throw std::invalid_argument("Unknown element count error");
    }
}

template<class FFT_Class>
void memory_run(std::vector<float> const &memories, int runs, const bool plot, std::ofstream& file){
    for (float mem : memories){
        FFT_Class fft_class(mem);
#ifdef PYTHON_PLOTTING
        if ( plot ){
            fft_class.transform();
            create_postplot(
                fft_class,
                fft_class.name()+" "+std::to_string(mem)+" MB" ,
                fft_class.name() + "_transform_" + std::to_string(mem) + "MB"
                );
        }
#endif
        auto time = fft_class.time_transform(runs).count();
        auto checksum = compare_data(fft_class.get_source(), fft_class.get_element_count());
        file <<
            fft_class.name() << ", " <<
            fft_class.get_memory()/1000000 << ", " <<
            time << ", " <<
            checksum << "\n";
        file.flush();
    }
}

template<class FFT_Class>
void element_run(std::vector<int> const &element_counts, int runs, const bool plot, std::ofstream& file){
    for (int element_count : element_counts){
        FFT_Class fft_class(element_count);
        if (fft_class.transform_fail) {
            std::cout << fft_class.name() << " Unable to perform transform with element count: " << element_count << "\n";
            continue;
        }
#ifdef PYTHON_PLOTTING
        if ( plot ){
            fft_class.transform();
            create_postplot(
                fft_class,
                fft_class.name()+" "+std::to_string(memory_retrieve(element_count))+" MB" ,
                fft_class.name() + "_transform_" + std::to_string(memory_retrieve(element_count)) + "MB"
                );
        }
#endif
        auto time = fft_class.time_transform(runs).count();
        auto checksum = compare_data(fft_class.get_source(), fft_class.get_element_count());
        file <<
            fft_class.name() << ", " <<
            std::to_string(memory_retrieve(element_count)) << ", " <<
            time << ", " <<
            checksum << "\n";
        file.flush();
    }
}

#endif //RUN_FUNCTIONS_HPP
