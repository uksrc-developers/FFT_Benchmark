//
// Created by marcuskeil on 18/12/24.
//
#include "../include/cuFFT_Class.hpp"

const char* cufftGetErrorString(const cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "Unknown CUFFT error";
    }
}

cuFFT_Class::cuFFT_Class(const float memory_size){ // memory_size given in MB
    vector_side = possible_vector_size(memory_size);
    vector_element_count = static_cast<int>(pow(vector_side, 2));
    vector_memory_size = (vector_element_count*sizeof(std::complex<double>));

    cudaMallocManaged(
                &source_data,
                vector_element_count*sizeof(std::complex<double>),
                cudaMemAttachGlobal
        );
    fill_vector(source_data, vector_element_count);

    level_check();
}

cuFFT_Class::cuFFT_Class(const int element_count){ // memory_size given in MB
    vector_side = static_cast<int>(sqrt(element_count));
    vector_element_count = element_count;
    vector_memory_size = (vector_element_count*sizeof(std::complex<double>));

    cudaMallocManaged(
                &source_data,
                vector_element_count*sizeof(std::complex<double>),
                cudaMemAttachGlobal
        );
    fill_vector(source_data, vector_element_count);

    level_check();
}

void cuFFT_Class::level_check() {
    size_t mf, ma;
    cudaMemGetInfo(&mf, &ma);
    const size_t compare_mf = mf*4/7;

    std::vector<int> split_levels = {1,2,3,4,5,7,8};
    bool split_found = false;
    for (int level : split_levels) {
        if (vector_element_count%level == 0) {
            size_t this_level_workEstimate = 0;
            const cufftResult this_level_result = cufftEstimate1d(vector_element_count/level, CUFFT_Z2Z, 1, &this_level_workEstimate);
            if (this_level_result != CUFFT_SUCCESS) {
                continue;
            }
            if (compare_mf > (this_level_workEstimate + vector_memory_size/level)) {
                const cufftResult result = cufftPlan1d(&p, vector_element_count/level, CUFFT_Z2Z, 1);
                if (result != CUFFT_SUCCESS) {
                    throw std::invalid_argument( cufftGetErrorString(result));
                }
                if (level > 1) {
                    split_level = level;
                } else {
                    split_level = 0;
                }
                split_found = true;
                break;
            }
        }
    }
    if (!split_found) {
        transform_fail = true;
    }
}

void cuFFT_Class::transform(){
    if (split_level == 0) {
        const cufftResult result = cufftExecZ2Z(
            p,
            reinterpret_cast<cufftDoubleComplex *>(source_data),
            reinterpret_cast<cufftDoubleComplex *>(source_data),
            CUFFT_FORWARD
            );
        if (result != CUFFT_SUCCESS) {
            std::cerr << "cuFFT error: " << cufftGetErrorString(result) << std::endl;
        }
        cudaDeviceSynchronize();
    } else {
        cooley_tukey();
    }
}

void cuFFT_Class::partial_transform(std::complex<double>* partial_array, int size) {
    const cufftResult result = cufftExecZ2Z(
        p,
        reinterpret_cast<cufftDoubleComplex *>(partial_array),
        reinterpret_cast<cufftDoubleComplex *>(partial_array),
        CUFFT_FORWARD
        );
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: " << cufftGetErrorString(result) << std::endl;
    }
    cudaDeviceSynchronize();
}

std::chrono::duration<double, std::milli> cuFFT_Class::time_transform(const int runs) {
    std::chrono::duration<double> times{};
    for ( int i = 0; i < runs ; i++){
        std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
        transform();
        std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();
        times += (t2 - t1);
    }
    return  times / runs;
}

cuFFT_Class::~cuFFT_Class() {
    cufftDestroy(p);
    cudaFree(source_data);
}
