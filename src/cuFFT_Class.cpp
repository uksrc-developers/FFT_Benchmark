//
// Created by marcuskeil on 18/12/24.
//
#include "../include/cuFFT_Class.hpp"

cuFFT_Class::cuFFT_Class(const float memory_size){ // memory_size given in MB
    vector_side = possible_vector_size(memory_size);
    vector_element_count = static_cast<int>(pow(vector_side, 2));
    vector_memory_size = (vector_element_count*sizeof(std::complex<double>));

    size_t workEstimate = 0;
    cufftEstimate2d(vector_side, vector_side, CUFFT_Z2Z, &workEstimate);

    cudaMallocManaged(
            &source_data,
            vector_element_count*sizeof(std::complex<double>),
            cudaMemAttachGlobal
    );
    fill_vector(source_data, vector_element_count);

    size_t mf, ma;
    cudaMemGetInfo(&mf, &ma);
    size_t compare_mf = mf*3/4;
    if ( compare_mf < ( workEstimate + vector_memory_size ) ){
        // rad_2_est
        cufftEstimate2d(
            static_cast<int>(sqrt(vector_element_count/2)),
            static_cast<int>(sqrt(vector_element_count/2)),
            CUFFT_Z2Z, &workEstimate);
        if( compare_mf < (workEstimate + vector_memory_size/2) ) {
            radix2=true;
        } else {
            cufftEstimate2d(
                static_cast<int>(sqrt(vector_element_count/4)),
                static_cast<int>(sqrt(vector_element_count/4)),
                CUFFT_Z2Z, &workEstimate);
            if( compare_mf < (workEstimate + vector_memory_size/4) ) {
                radix4=true;
            } else {
                cufftEstimate2d(
                    static_cast<int>(sqrt(vector_element_count/4)),
                    static_cast<int>(sqrt(vector_element_count/4)),
                    CUFFT_Z2Z, &workEstimate);
                if( compare_mf < (workEstimate + vector_memory_size/4) ) {
                    radix8=true;
                } else {
                    throw std::invalid_argument( "Work area and Array size too large to perform cuFFT transform." );
                }
            }
        }
    } else {
        cufftPlan2d(&p,vector_side,vector_side,CUFFT_Z2Z);
    }
}

void cuFFT_Class::allocate_memory(std::complex<double> *data, const int element_count) {
    cudaMallocManaged(
            &data,
            element_count*sizeof(std::complex<double>),
            cudaMemAttachGlobal
    );
}

void cuFFT_Class::free_memory(std::complex<double> *data) {
    cudaFree(data);
}

void cuFFT_Class::make_plan(const int element_count) {
    cufftPlan1d(&p, element_count, CUFFT_Z2Z, 1);
}

//void cuFFT_Class::CT_radix_2() {
//    constexpr int radix = 2;
//    const int N_o_R = vector_element_count/radix;
//    std::complex<double> *fft_0;
//    cudaMallocManaged(
//            &fft_0,
//            N_o_R*sizeof(std::complex<double>),
//            cudaMemAttachHost
//    );
//    std::complex<double> *fft_1;
//    cudaMallocManaged(
//            &fft_1,
//            N_o_R*sizeof(std::complex<double>),
//            cudaMemAttachHost
//    );
//    bool toggle = false;
//#pragma omp parallel for
//    for( int i = 0; i < N_o_R; ++i ){
//        fft_0[i] = (source_data)[i*radix];
//        fft_1[i] = (source_data)[i*radix + 1];
//    }
//    cufftPlan1d(&p, N_o_R, CUFFT_Z2Z, 1);
//    cufftExecZ2Z(p,
//                     reinterpret_cast<cufftDoubleComplex *>(fft_0),
//                     reinterpret_cast<cufftDoubleComplex *>(fft_0),
//                     CUFFT_FORWARD);
//    cufftExecZ2Z(p,
//                 reinterpret_cast<cufftDoubleComplex *>(fft_1),
//                 reinterpret_cast<cufftDoubleComplex *>(fft_1),
//                 CUFFT_FORWARD);
//    cudaDeviceSynchronize();
//#pragma omp parallel for
//    for (int i=0; i < N_o_R; i++ ){
//        auto q = std::complex<double>(
//            cos(-((2*M_PI)/vector_element_count)*i),
//            sin(-((2*M_PI)/vector_element_count)*i))*fft_1[i];
//        auto q_1  = std::complex<double>(
//            cos(-(2*M_PI)/radix),
//            sin(-(2*M_PI)/radix));
//        (source_data)[i] = fft_0[i] + q;
//        (source_data)[i+N_o_R] = fft_0[i] + q*q_1;
//    }
//    cudaFree(fft_0);
//    cudaFree(fft_1);
//}

void cuFFT_Class::CT_radix_4() {
    constexpr int radix = 4;
    const int N_o_R = vector_element_count/radix;
    std::complex<double> *fft_0;
    cudaMallocManaged(
            &fft_0,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_1;
    cudaMallocManaged(
            &fft_1,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_2;
    cudaMallocManaged(
            &fft_2,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_3;
    cudaMallocManaged(
            &fft_3,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
            );
#pragma omp parallel for
    for( int i = 0; i < N_o_R; ++i ){
        fft_0[i] = (source_data)[i*radix];
        fft_1[i] = (source_data)[i*radix + 1];
        fft_2[i] = (source_data)[i*radix + 2];
        fft_3[i] = (source_data)[i*radix + 3];
    }
    cufftPlan1d(&p, N_o_R, CUFFT_Z2Z, 1);
    cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(fft_0),
                     reinterpret_cast<cufftDoubleComplex *>(fft_0),
                     CUFFT_FORWARD);
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(fft_1),
                 reinterpret_cast<cufftDoubleComplex *>(fft_1),
                 CUFFT_FORWARD);
    cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(fft_2),
                     reinterpret_cast<cufftDoubleComplex *>(fft_2),
                     CUFFT_FORWARD);
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(fft_3),
                 reinterpret_cast<cufftDoubleComplex *>(fft_3),
                 CUFFT_FORWARD);
    cudaDeviceSynchronize();
#pragma omp parallel for // NEEDS CHANGING
    for (int i=0; i < N_o_R; i++ ){
        auto q = std::complex<double>(
            cos(-((2*M_PI)/vector_element_count)*i),
            sin(-((2*M_PI)/vector_element_count)*i))*fft_1[i];
        auto q_1 = std::complex<double>(
            cos(-(2*M_PI)/radix),
            sin(-(2*M_PI)/radix));
        auto q_2 = std::complex<double>(
            cos(-(2*M_PI*2)/radix),
            sin(-(2*M_PI*2)/radix));
        auto q_3 = std::complex<double>(
            cos(-(2*M_PI*3)/radix),
            sin(-(2*M_PI*3)/radix));
        auto r = std::complex<double>(
            cos(-((2*M_PI*2)/vector_element_count)*i),
            sin(-((2*M_PI*2)/vector_element_count)*i))*fft_2[i];
        auto r_1 = std::complex<double>(
            cos(-(2*M_PI*2)/radix),
            sin(-(2*M_PI*2)/radix));
        auto r_2 = std::complex<double>(
            cos(-(2*M_PI*4)/radix),
            sin(-(2*M_PI*4)/radix));
        auto r_3 = std::complex<double>(
            cos(-(2*M_PI*6)/radix),
            sin(-(2*M_PI*6)/radix));
        auto t = std::complex<double>(
            cos(-((2*M_PI*3)/vector_element_count)*i),
            sin(-((2*M_PI*3)/vector_element_count)*i))*fft_3[i];
        auto t_1 = std::complex<double>(
            cos(-(2*M_PI*3)/radix),
            sin(-(2*M_PI*3)/radix));
        auto t_2 = std::complex<double>(
            cos(-(2*M_PI*6)/radix),
            sin(-(2*M_PI*6)/radix));
        auto t_3 = std::complex<double>(
            cos(-(2*M_PI*9)/radix),
            sin(-(2*M_PI*9)/radix));

        (source_data)[i] = fft_0[i] + q + r + t;
        (source_data)[i+N_o_R] = fft_0[i] + q*q_1 + r*r_1 + t*t_1;
        (source_data)[i+2*N_o_R] = fft_0[i] + q*q_2 + r*r_2 + t*t_2;
        (source_data)[i+3*N_o_R] = fft_0[i] + q*q_3 + r*r_3 + t*t_3;
    }// NEEDS CHANGING
    cudaFree(fft_0);
    cudaFree(fft_1);
    cudaFree(fft_2);
    cudaFree(fft_3);
}

void cuFFT_Class::CT_radix_8() {
    constexpr int radix = 8;
    const int N_o_R = vector_element_count/radix;
    std::complex<double> *fft_0;
    cudaMallocManaged(
            &fft_0,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_1;
    cudaMallocManaged(
            &fft_1,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_2;
    cudaMallocManaged(
            &fft_2,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_3;
    cudaMallocManaged(
            &fft_3,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_4;
    cudaMallocManaged(
            &fft_4,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_5;
    cudaMallocManaged(
            &fft_5,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_6;
    cudaMallocManaged(
            &fft_6,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
    std::complex<double> *fft_7;
    cudaMallocManaged(
            &fft_7,
            N_o_R*sizeof(std::complex<double>),
            cudaMemAttachHost
    );
#pragma omp parallel for
    for( int i = 0; i < N_o_R; ++i ){
        fft_0[i] = (source_data)[i*radix];
        fft_1[i] = (source_data)[i*radix + 1];
        fft_2[i] = (source_data)[i*radix + 2];
        fft_3[i] = (source_data)[i*radix + 3];
        fft_4[i] = (source_data)[i*radix + 4];
        fft_5[i] = (source_data)[i*radix + 5];
        fft_6[i] = (source_data)[i*radix + 6];
        fft_7[i] = (source_data)[i*radix + 7];
    }
    cufftPlan1d(&p, N_o_R, CUFFT_Z2Z, 1);
    cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(fft_0),
                     reinterpret_cast<cufftDoubleComplex *>(fft_0),
                     CUFFT_FORWARD);
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(fft_1),
                 reinterpret_cast<cufftDoubleComplex *>(fft_1),
                 CUFFT_FORWARD);
    cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(fft_2),
                     reinterpret_cast<cufftDoubleComplex *>(fft_2),
                     CUFFT_FORWARD);
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(fft_3),
                 reinterpret_cast<cufftDoubleComplex *>(fft_3),
                 CUFFT_FORWARD);
    cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(fft_4),
                     reinterpret_cast<cufftDoubleComplex *>(fft_4),
                     CUFFT_FORWARD);
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(fft_5),
                 reinterpret_cast<cufftDoubleComplex *>(fft_5),
                 CUFFT_FORWARD);
    cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(fft_6),
                     reinterpret_cast<cufftDoubleComplex *>(fft_6),
                     CUFFT_FORWARD);
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(fft_7),
                 reinterpret_cast<cufftDoubleComplex *>(fft_7),
                 CUFFT_FORWARD);
    cudaDeviceSynchronize();
#pragma omp parallel for // NEEDS CHANGING
    for (int i=0; i < N_o_R; i++ ){
        auto q = std::complex<double>(
                cos(-(((2*M_PI))/(vector_element_count))*i),
                sin(-(((2*M_PI))/(vector_element_count))*i))*fft_1[i];
        (source_data)[i] = fft_0[i] + q;
        (source_data)[i+N_o_R] = fft_0[i] - q;
    }// NEEDS CHANGING
    cudaFree(fft_0);
    cudaFree(fft_1);
    cudaFree(fft_2);
    cudaFree(fft_3);
    cudaFree(fft_4);
    cudaFree(fft_5);
    cudaFree(fft_6);
    cudaFree(fft_7);
}

void cuFFT_Class::split_fft(std::complex<double> **data, const int element_count){
    split_count--;
    assert((void("Transform needed to be split, but was not splittable evenly."), (element_count%2 <= 0) ));
    const int split_element_count = element_count/2;

    std::complex<double> *even;
    cudaMallocManaged(
            &even,
            split_element_count*sizeof(std::complex<double>),
            cudaMemAttachHost
    );

    std::complex<double> *odd;
    cudaMallocManaged(
            &odd,
            split_element_count*sizeof(std::complex<double>),
            cudaMemAttachHost
    );

#pragma omp parallel for
    for( int i = 0; i < split_element_count; ++i ){
        even[i] = (*data)[i*2];
        odd[i] = (*data)[i*2 + 1];
    }

    if ( split_count > 0 ){
        split_fft(&even, split_element_count);
        split_count++;
        split_fft(&odd, split_element_count);
    } else {
        if (!plan){
            cufftPlan1d(&p, split_element_count, CUFFT_Z2Z, 1);
            plan = true;
        }
        cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(even),
                     reinterpret_cast<cufftDoubleComplex *>(even),
                     CUFFT_FORWARD);
        cudaDeviceSynchronize();
        cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(odd),
                     reinterpret_cast<cufftDoubleComplex *>(odd),
                     CUFFT_FORWARD);
        cudaDeviceSynchronize();
    }
#pragma omp parallel for
    for (int i=0; i < split_element_count; i++ ){
        auto q = std::complex<double>(
                cos(-(((2*M_PI))/(element_count))*i),
                sin(-(((2*M_PI))/(element_count))*i))*odd[i];
        (*data)[i] = even[i] + q;
        (*data)[i+split_element_count] = even[i] - q;
    }
    cudaFree(even);
    cudaFree(odd);
}

void cuFFT_Class::transform(){
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(source_data),
                 reinterpret_cast<cufftDoubleComplex *>(source_data), CUFFT_FORWARD);
    cudaDeviceSynchronize();
    //CT_radix_2<cuFFT_Class>(*this);
}

void cuFFT_Class::transform(std::complex<double> **data) const {
    cufftExecZ2Z(p,
                 reinterpret_cast<cufftDoubleComplex *>(*data),
                 reinterpret_cast<cufftDoubleComplex *>(*data), CUFFT_FORWARD);
}

void cuFFT_Class::sync() {
    cudaDeviceSynchronize();
}

std::chrono::duration<double, std::milli> cuFFT_Class::time_transform(int runs) {
    std::chrono::duration<double> times{};
    for ( int i = 0; i < runs ; i++){
        split_count = original_split;
        std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
        transform();
        std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();
        times += (t2 - t1);
    }
    return  times / runs;
}

cuFFT_Class::~cuFFT_Class() {
    cufftDestroy(p);
    cudaFree(source_data);
}
