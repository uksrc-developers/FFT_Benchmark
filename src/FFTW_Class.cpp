//
// Created by marcuskeil on 16/12/24.
//
#include "../include/FFTW_Class.hpp"

FFTW_Class::FFTW_Class(float memory_size){
    vector_side = possible_vector_size(memory_size);
    vector_element_count = pow(vector_side, 2);
    vector_memory_size = (vector_element_count*sizeof(std::complex<double>));

    source_data = (std::complex<double> *)malloc(vector_memory_size);
    fill_vector(source_data, vector_element_count);

    p = fftw_plan_dft_2d(vector_side, vector_side,
                         reinterpret_cast<fftw_complex *>(source_data),
                         reinterpret_cast<fftw_complex *>(source_data),
                         FFTW_FORWARD, FFTW_ESTIMATE);
}

std::chrono::duration<double, std::milli> FFTW_Class::time_transform(int runs) {
    std::chrono::duration<double> times{};
    for ( int i = 0; i < runs ; i++){
        std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
        transform();
        std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();
        times += (t2 - t1);
    }
    return  times / runs;
}


FFTW_Class::~FFTW_Class() {
    fftw_destroy_plan(p);
    fftw_cleanup();
    free(source_data);
}
