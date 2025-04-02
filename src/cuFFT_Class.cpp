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
    if ( mf < ( workEstimate + vector_memory_size ) ){
        split = true;
        int split_element_count = vector_element_count/2;
        int mult;
        if ( workEstimate < (split_element_count*sizeof(std::complex<double>))/2 || workEstimate > (split_element_count*sizeof(std::complex<double>))/2 ){
            mult = 7;
        } else {
            mult = 2;
        }
        cufftEstimate2d(static_cast<int>(sqrt(split_element_count)), static_cast<int>(sqrt(split_element_count)), CUFFT_Z2Z, &workEstimate);
        while ( mf*3/4 < ( workEstimate + split_element_count )*mult ){
            split_count++;
            split_element_count = split_element_count/2;
            cufftEstimate2d(static_cast<int>(sqrt(split_element_count)), static_cast<int>(sqrt(split_element_count)), CUFFT_Z2Z, &workEstimate);
            if ( workEstimate < (split_element_count*sizeof(std::complex<double>))/2 || workEstimate > (split_element_count*sizeof(std::complex<double>)) ){
                mult = 7;
            } else {
                mult = 2;
            }
        }
        original_split = split_count;
    } else {
        cufftPlan2d(&p,vector_side,vector_side,CUFFT_Z2Z);
    }
}

void cuFFT_Class::stream_fft(){
    int Nsplits = 2;
    int Npartial = vector_element_count / Nsplits;
    size_t mf, ma;
    cudaMemGetInfo(&mf, &ma);
    while ( pow(vector_side/Nsplits, 2)*sizeof(std::complex<double>) < static_cast<double>(mf)/7 ){
        Nsplits++;
        Npartial = vector_element_count / Nsplits;
    }
    while ( vector_element_count%Nsplits > 0 ){
        Nsplits++;
        Npartial = vector_element_count / Nsplits;
    }

    cufftPlan1d(&p,Npartial,CUFFT_Z2Z, 1);

    auto *out_data = new std::complex<double>[vector_element_count];

    for ( int N = 0; N < Nsplits; N++ ) {
        auto *h_in = new std::complex<double>[Npartial];
        for (int i = 0; i < Npartial; i++) {
            h_in[i] = source_data[i + Npartial * N];
        }
        cudaHostRegister(&h_in, Npartial*sizeof(std::complex<double>), cudaHostRegisterPortable);
        auto *d_in = new std::complex<double>[Npartial];
        cudaMallocManaged((void**)&d_in, vector_element_count*sizeof(std::complex<double>));
        cudaMemset(d_in, 0, vector_element_count*sizeof(float2));

        for ( int i = 0; i < Npartial; i++ ){
            d_in[i + Npartial * N] = h_in[i];
        }

        cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(d_in),
                     reinterpret_cast<cufftDoubleComplex *>(d_in),
                     CUFFT_FORWARD);
        cudaDeviceSynchronize();
        for ( int i = 0; i < vector_element_count; i++ ){
            out_data[i] += d_in[i];
        }
        cudaHostUnregister(h_in);
        cudaFree(d_in);
    }
#pragma omp parallel for
    for ( int i = 0; i < vector_element_count; i++ ){
        source_data[i] = out_data[i];
    }
    free(out_data);
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
    if ( split ){
        split_fft(&source_data, vector_element_count);
    } else {
        cufftExecZ2Z(p,
                     reinterpret_cast<cufftDoubleComplex *>(source_data),
                     reinterpret_cast<cufftDoubleComplex *>(source_data), CUFFT_FORWARD);
        cudaDeviceSynchronize();
    }
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
