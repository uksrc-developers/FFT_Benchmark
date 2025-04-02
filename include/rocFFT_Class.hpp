//
// Created by marcuskeil on 31/01/25.
//

#ifndef FFT_BENCH_ROCFFT_CLASS_HPP
#define FFT_BENCH_ROCFFT_CLASS_HPP
#include <rocfft.h>
#include "Data_Functions.hpp"
#include "Abstract_FFT.hpp"
#include <hip/hip_runtime_api.h>
#include <map>

class rocFFT_Class final : Abstract_FFT{
    private:
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
        
    public:
        explicit rocFFT_Class(float memory_size); // memory_size given in MB
        ~rocFFT_Class() override;

        [[maybe_unused]] inline std::string name() override { return "rocFFT"; };
        [[maybe_unused]] [[nodiscard]] inline int get_side() override { return vector_side; };
        [[maybe_unused]] [[nodiscard]] inline size_t get_memory() override { return vector_memory_size; };
        [[maybe_unused]] [[nodiscard]] inline std::complex<double>* get_source() override;
        [[maybe_unused]] [[nodiscard]] inline int get_element_count() override { return vector_element_count; };

        [[maybe_unused]] void transform() override;
        [[maybe_unused]] std::chrono::duration<double, std::milli> time_transform(int runs) override;
};
#endif //FFT_BENCH_ROCFFT_CLASS_HPP
