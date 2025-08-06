//
// Created by marcuskeil on 18/12/24.
//
#ifndef FFT_BENCH_CUFFT_CLASS_HPP
#define FFT_BENCH_CUFFT_CLASS_HPP
#pragma once
#include <cuda_runtime.h>
#include "cufft.h"

#include "Data_Functions.hpp"
#include "Abstract_FFT.hpp"

class cuFFT_Class final : public Abstract_FFT {
    private:
        std::complex<double> *source_data{};
        int vector_side;
        int vector_element_count;
        size_t vector_memory_size;

        cufftHandle p{};
        int split_level = 0;
    public:
        bool transform_fail = false;
        explicit cuFFT_Class(float memory_size); // memory_size given in MB
        explicit cuFFT_Class(int element_count);
        ~cuFFT_Class() override;

        [[maybe_unused]] inline std::string name() override { return "CUDA"; };
        [[maybe_unused]] [[nodiscard]] inline int get_side() const override { return vector_side; };
        [[maybe_unused]] [[nodiscard]] inline size_t get_memory() override { return vector_memory_size; };
        [[maybe_unused]] [[nodiscard]] inline std::complex<double>* get_source() const override { return source_data; };
        [[maybe_unused]] [[nodiscard]] inline int get_element_count() const override { return vector_element_count; };

        [[maybe_unused]] static void allocate_memory(std::complex<double> *data, int element_count);
        [[maybe_unused]] static void free_memory(std::complex<double> *data);

        [[maybe_unused]] void transform() override;
        [[maybe_unused]] void cooley_tukey() override { CT_transform(*this, split_level); };
        [[maybe_unused]] void partial_transform(std::complex<double>* partial_array, std::size_t size) override;
        [[maybe_unused]] static void sync();

        [[maybe_unused]] std::chrono::duration<double, std::milli> time_transform(int runs) override;
};
#endif //FFT_BENCH_CUFFT_CLASS_HPP
