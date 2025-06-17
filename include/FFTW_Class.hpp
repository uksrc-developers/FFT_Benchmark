//
// Created by marcuskeil on 18/12/24.
//
#ifndef FFT_BENCH_FFTW_CLASS_HPP
#define FFT_BENCH_FFTW_CLASS_HPP
#include <fftw3.h>
#include "Data_Functions.hpp"
#include "Abstract_FFT.hpp"
#include <map>

class FFTW_Class final : Abstract_FFT{
    private:
        std::complex<double> *source_data{};
        int vector_side;
        int vector_element_count;
        size_t vector_memory_size;

        fftw_plan p;
    public:
        explicit FFTW_Class(float memory_size); // memory_size given in MB
        ~FFTW_Class() override;

        [[maybe_unused]] inline std::string name() override { return "FFTW"; };
        [[maybe_unused]] [[nodiscard]] inline int get_side() override { return vector_side; };
        [[maybe_unused]] [[nodiscard]] inline size_t get_memory() override { return vector_memory_size; };
        [[maybe_unused]] [[nodiscard]] inline std::complex<double>* get_source() override { return source_data; };
        [[maybe_unused]] [[nodiscard]] inline int get_element_count() override { return vector_element_count; };

        [[maybe_unused]] void transform() override { fftw_execute(p); };
        [[maybe_unused]] std::chrono::duration<double, std::milli> time_transform(int runs) override;
};
#endif //FFT_BENCH_FFTW_CLASS_HPP
