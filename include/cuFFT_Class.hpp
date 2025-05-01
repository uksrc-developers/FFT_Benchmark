//
// Created by marcuskeil on 18/12/24.
//
#ifndef FFT_BENCH_CUFFT_CLASS_HPP
#define FFT_BENCH_CUFFT_CLASS_HPP
#include <cuda_runtime.h>
#include "cufft.h"
#include "Data_Functions.hpp"
#include "Abstract_FFT.hpp"

class cuFFT_Class final : Abstract_FFT {
private:
    std::complex<double> *source_data{};
    int vector_side;
    int vector_element_count;
    size_t vector_memory_size;

    cufftHandle p{};


    bool radix2 = false;
    bool radix4 = false;
    bool radix8 = false;
    int original_split = 0;
    int split_count = 0;
    bool plan = false;

public:
    explicit cuFFT_Class(float memory_size); // memory_size given in MB
    ~cuFFT_Class() override;

    [[maybe_unused]] inline std::string name() override { return "CUDA"; };
    [[maybe_unused]] [[nodiscard]] inline int get_side() override { return vector_side; };
    [[maybe_unused]] [[nodiscard]] inline size_t get_memory() override { return vector_memory_size; };
    [[maybe_unused]] [[nodiscard]] inline std::complex<double>* get_source() override { return source_data; };
    [[maybe_unused]] [[nodiscard]] inline int get_element_count() override { return vector_element_count; };

    [[maybe_unused]] inline void split_fft(std::complex<double> **data, int element_count);
    [[maybe_unused]] static void allocate_memory(std::complex<double> *data, int element_count);
    [[maybe_unused]] static void free_memory(std::complex<double> *data);
    [[maybe_unused]]void make_plan(int element_count);

//    [[maybe_unused]] inline void CT_radix_2();
    [[maybe_unused]] inline void CT_radix_4();
    [[maybe_unused]] inline void CT_radix_8();
    [[maybe_unused]] void transform() override;
    [[maybe_unused]] void transform(std::complex<double> **data) const;
    [[maybe_unused]] static void sync();

    [[maybe_unused]] std::chrono::duration<double, std::milli> time_transform(int runs) override;
};

#endif //FFT_BENCH_CUFFT_CLASS_HPP
