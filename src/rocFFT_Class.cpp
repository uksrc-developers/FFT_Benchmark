//
// Created by marcuskeil on 31/01/25.
//
#include "../include/rocFFT_Class.hpp"

rocFFT_Class::rocFFT_Class(float memory_size){
    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");
    std::cout << "Test" << std::endl;
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
    if(hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("hipDeviceSynchronize failed.");
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

#if __has_include( "matplotlibcpp.h" )
void rocFFT_Class::create_preplot(const std::string& file_name){
    matplotlibcpp::figure_size(1200, 780);
    const int colors = 1;
    matplotlibcpp::title(std::to_string(vector_memory_size/1000000) + "[MB] not transformed");
    std::vector<float> plot = pre_plot_vector(source_data, vector_element_count);
    matplotlibcpp::imshow(&(plot[0]),
                          vector_side,
                          vector_side, colors,
                          std::map<std::string, std::string>{{"origin", "lower"}});
    matplotlibcpp::save("Fiducial_outputs/" + file_name + ".png");
}

void rocFFT_Class::create_postplot(const std::string& file_name){
    plt::figure_size(1200, 780);
    const int colors = 1;
    plt::title(std::to_string(vector_memory_size/1000000) + "[MB] " + name() + " transformed");
    std::vector<float> plot = post_plot_vector(source_data, vector_element_count);
    plt::imshow(&(plot[0]),
                vector_side,
                vector_side, colors,
                std::map<std::string, std::string>{{"origin", "lower"}});
    plt::save("Fiducial_outputs/" + file_name + ".png");
}
#endif


rocFFT_Class::~rocFFT_Class() {
    rocfft_cleanup();
    free(source_data);
}

