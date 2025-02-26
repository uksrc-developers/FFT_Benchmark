//
// Created by marcuskeil on 31/01/25.
//
#include "../include/rocFFT_Class.hpp"

rocFFT_Class::rocFFT_Class(float memory_size){
    rocfft_setup();
    vector_side = possible_vector_size(memory_size);
    vector_element_count = pow(vector_side, 2);
    vector_memory_size = (vector_element_count*sizeof(std::complex<double>));

    source_data = (std::complex<double> *)malloc(vector_memory_size);
    fill_vector(source_data, vector_element_count);

    rocfft_plan_create(&p, rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward, rocfft_precision_double,
                       1, reinterpret_cast<const size_t *>(&vector_element_count), 1, nullptr);
}

void rocFFT_Class::transform() {
    rocfft_execution_info info = nullptr;
    rocfft_execute(p, (void**) &source_data, nullptr, info);
//#if __has_include( "hip/hip_runtime_api.h" )
//    hipDeviceSynchronize();
//#else
//    cudaDeviceSynchronize();
//#endif
}

std::chrono::duration<double, std::milli> rocFFT_Class::time_transform(int runs) {
    std::chrono::duration<double> times{};
    for ( int i = 0; i < runs ; i++){
        std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
        transform();
        std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();times += (t2 - t1);
    }
    return  times / runs;
}

rocFFT_Class::~rocFFT_Class() {
    rocfft_cleanup();
    free(source_data);
}

