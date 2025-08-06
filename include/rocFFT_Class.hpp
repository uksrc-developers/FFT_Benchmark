//
// Created by marcuskeil on 31/01/25.
//
#ifndef FFT_BENCH_ROCFFT_CLASS_HPP
#define FFT_BENCH_ROCFFT_CLASS_HPP
#pragma once
#include <chrono>
#include <string>
#include <complex>
#include <rocfft.h>
#include <hip/hip_runtime_api.h>

#include "Data_Functions.hpp"
#include "Abstract_FFT.hpp"

class rocFFT_Class final : public Abstract_FFT{
    std::complex<double> *source_data{};
    std::complex<double> *gpu_source_data{};
    int vector_side;
    int vector_element_count;
    size_t vector_memory_size;
    size_t p_workbuff_size = 0;

    rocfft_plan p = nullptr;
    void* p_workbuff = nullptr;
    rocfft_execution_info p_info = nullptr;
    rocfft_plan_description p_desc = nullptr;

    int split_level = 0;

    public:
        bool transform_fail = false;
        explicit rocFFT_Class(float memory_size); // memory_size given in MB
        explicit rocFFT_Class(int element_count);
        ~rocFFT_Class() override;

        void level_check();

        [[maybe_unused]] inline std::string name() override { return "rocFFT"; };
        [[maybe_unused]] [[nodiscard]] inline int get_side() const override { return vector_side; };
        [[maybe_unused]] [[nodiscard]] inline size_t get_memory() override { return vector_memory_size; };
        [[maybe_unused]] [[nodiscard]] inline std::complex<double>* get_source() const override { return source_data; };
        [[maybe_unused]] [[nodiscard]] inline int get_element_count() const override { return vector_element_count; };

        [[maybe_unused]] void send_data(std::complex<double>* cpu_data, int array_length);
        [[maybe_unused]] void retrieve_data(std::complex<double>* cpu_data, int array_length);

        [[maybe_unused]] void transform() override;
        [[maybe_unused]] void cooley_tukey() override { CT_transform(*this, split_level); };
        [[maybe_unused]] void partial_transform(std::complex<double>* partial_array, int size) override;
        [[maybe_unused]] std::chrono::duration<double, std::milli> time_transform(int runs) override;
};
#endif //FFT_BENCH_ROCFFT_CLASS_HPP
