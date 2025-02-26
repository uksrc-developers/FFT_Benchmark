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

#if __has_include( "matplotlibcpp.h" )
void FFTW_Class::create_preplot(const std::string& file_name){
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

void FFTW_Class::create_postplot(const std::string& file_name){
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

FFTW_Class::~FFTW_Class() {
    fftw_destroy_plan(p);
    fftw_cleanup();
    free(source_data);
}
