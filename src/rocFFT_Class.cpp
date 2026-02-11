//
// Created by marcuskeil on 31/01/25.
//
// Modified on 11/02/26 by Owain Kenway to change malloc behavior to match Cuda -> massive speedup!
// Specifically -> do allocation of source data on the device.
#include "../include/rocFFT_Class.hpp"

rocFFT_Class::rocFFT_Class(const float memory_size){
    assert( rocfft_setup() == rocfft_status_success );
    vector_side = possible_vector_size(memory_size);
    vector_element_count = static_cast<int>(pow(vector_side, 2));

    level_check();
}

rocFFT_Class::rocFFT_Class(const int element_count){
    assert( rocfft_setup() == rocfft_status_success );
    vector_side = static_cast<int>(sqrt(element_count));
    vector_element_count = element_count;

    level_check();
}

void rocFFT_Class::level_check() {
    vector_memory_size = (vector_element_count*sizeof(std::complex<double>));
    assert( hipMalloc(&source_data, vector_memory_size) == hipSuccess);
    fill_vector(source_data, vector_element_count);

    assert( hipSetDevice(0) == hipSuccess );
    assert( rocfft_plan_description_create(&p_desc) == rocfft_status_success );
    assert(
        rocfft_plan_description_set_data_layout( // Alter plan description by setting the following data points
            p_desc, //rocfft_plan_description pointer
            rocfft_array_type_complex_interleaved, // input array type (rocfft_array_type)
            rocfft_array_type_complex_interleaved, // output array type (rocfft_array_type)
            nullptr, // size_t pointer in_offsets
            nullptr, // size_t pointer out_offsets
            1, // input stride length (size_t in_strides_size)
            nullptr, // input stride data pointer (size_t *in_strides)
            0, // input batch distance (size_t in_distance)
            1, // output stride length (size_t out_strides_size)
            nullptr, // output stride data pointer (size_t *out_strides)
            0
            ) == rocfft_status_success
    );

    size_t mf, ma;
    assert( hipMemGetInfo(&mf, &ma) == hipSuccess );
    const size_t compare_mf = mf*4/7;

    std::vector<int> split_levels = {1,2,3,4,5,7,8};
    bool split_found = false;
    for (int level : split_levels) {
        if (vector_element_count%level == 0) {
            size_t this_level_workEstimate = 0;
            std::vector<size_t> length = {static_cast<unsigned long>( vector_element_count/level )};
            assert(
                rocfft_plan_create(
                    &p, // plan from the rocFFT class
                    rocfft_placement_inplace, // result placement (rocfft-result_placement) in this case, into the same array as the input
                    rocfft_transform_type_complex_forward, // direction and type of transform, here forward from complex to complex (rocfft_transform_type)
                    rocfft_precision_double, // precision of transform (rocfft_precision)
                    length.size(), // size_t dimensions of the transform
                    length.data(), // pointer to size_t of the array size to be transformed with this plan
                    1, // size_t of number of transforms to perform
                    p_desc)
                    == rocfft_status_success ); // plan description (rocfft_plan_description)
            // Get the execution info for the fft plan (in particular, work memory requirements):
            assert( rocfft_plan_get_work_buffer_size(p, &this_level_workEstimate) == rocfft_status_success );

            if (compare_mf > (this_level_workEstimate + vector_memory_size/level)) {
                p_workbuff_size = this_level_workEstimate;
                // If the transform requires work memory, allocate a work buffer:
                if(p_workbuff_size > 0)
                {
                    assert( rocfft_execution_info_create(&p_info) == rocfft_status_success );
                    assert( hipMalloc(&p_workbuff, p_workbuff_size) == hipSuccess );
                    assert( rocfft_execution_info_set_work_buffer(p_info, p_workbuff, p_workbuff_size) == rocfft_status_success );
                }
                transform_memory_size = (vector_element_count/level*sizeof(std::complex<double>));
                assert( hipMalloc(&gpu_source_data, vector_memory_size/level) == hipSuccess );
                if (level > 1) {
                    split_level = level;
                } else {
                    split_level = 0;
                }
                split_found = true;
                break;
            }
            assert( rocfft_plan_destroy(p) == rocfft_status_success );
            p = nullptr;
        }
    }
    if (!split_found) {
        transform_fail = true;
    }
}

void rocFFT_Class::send_data(std::complex<double>* cpu_data, const int array_length){
    assert( hipMemcpy(gpu_source_data, cpu_data, transform_memory_size, hipMemcpyHostToDevice) == hipSuccess );
    assert( hipDeviceSynchronize() == hipSuccess );
};

void rocFFT_Class::retrieve_data(std::complex<double>* cpu_data, const int array_length){
	assert( hipMemcpy(cpu_data, gpu_source_data, transform_memory_size, hipMemcpyDeviceToHost) == hipSuccess );
	assert( hipDeviceSynchronize() == hipSuccess );
};

void rocFFT_Class::transform() {
    if (split_level == 0) {
        send_data(source_data, vector_element_count);
        assert(
            rocfft_execute(
                p, // plan
                ( void** )&gpu_source_data, // in_buffer
                nullptr, // out_buffer
                p_info) == rocfft_status_success ); // execution info
        assert( hipDeviceSynchronize() == hipSuccess );
        retrieve_data(source_data, vector_element_count);
    } else {
        cooley_tukey();
    }
}

void rocFFT_Class::partial_transform(std::complex<double>* partial_array, const int size) {
    send_data(partial_array, size);
    assert(
        rocfft_execute(
            p, // plan
            ( void** )&gpu_source_data, // in_buffer
            nullptr, // out_buffer
            p_info) == rocfft_status_success ); // execution info
    assert( hipDeviceSynchronize() == hipSuccess );
    retrieve_data(partial_array, size);
}

std::chrono::duration<double, std::milli> rocFFT_Class::time_transform(const int runs) {
    std::chrono::duration<double> times{};
    for ( int i = 0; i < runs ; i++){
        std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
        transform();
        std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();times += (t2 - t1);
    }
    return  times / runs;
}

rocFFT_Class::~rocFFT_Class() {
    assert( hipFree(source_data) == hipSuccess );
    assert( hipFree(gpu_source_data) == hipSuccess );
    gpu_source_data = nullptr;

    if(p_workbuff_size > 0) {
        assert( hipFree(p_workbuff) == hipSuccess );
        assert( rocfft_execution_info_destroy(p_info) == rocfft_status_success );
        p_workbuff = nullptr;
    }

    assert( rocfft_plan_description_destroy(p_desc) == rocfft_status_success );

    assert( rocfft_plan_destroy(p) == rocfft_status_success );

    assert( rocfft_cleanup() == rocfft_status_success );
}
