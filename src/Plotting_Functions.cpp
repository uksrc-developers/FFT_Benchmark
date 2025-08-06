//
// Created by marcuskeil on 04/08/25.
//
#include "../include/Plotting_Functions.hpp"

#if PYTHON_PLOTTING == 1
std::vector<float> pre_plot_vector(const std::complex<double>* v, const int element_count) {
    std::vector<float> plot_vector;
    plot_vector.resize(element_count);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < element_count; i++)
        plot_vector[i] = static_cast<float>(abs(v[i]));
    return plot_vector;
}

std::vector<float> post_plot_vector(const std::complex<double>* v, const int element_count){
    std::vector<float> plot_vector;
    plot_vector.resize(element_count);
    int dim = int(sqrt(element_count));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < element_count; i++)
        plot_vector[i] = static_cast<float>(abs(v[i]));
    if ( dim%2 == 0){
        std::rotate(plot_vector.rbegin(), plot_vector.rbegin() + int(element_count/2), plot_vector.rend());
    } else {
        int rotate = int(element_count*(3/2) - dim*((dim+1)/2));
        std::rotate(plot_vector.rbegin(), plot_vector.rbegin() + rotate, plot_vector.rend());
    }
#pragma omp parallel for
    for (int j = 0; j < dim; j++){
        std::rotate(plot_vector.rbegin()+((dim)*j),
                    plot_vector.rbegin()+((dim)*j + int((dim)/2)),
                    plot_vector.rbegin()+((dim)*j + (dim)));
    }
    return plot_vector;
}

void create_preplot(const Abstract_FFT& fft_obj, const std::string& title, const std::string& file_name){
    const std::complex<double>* source_data = fft_obj.get_source();
    auto N = fft_obj.get_element_count();
    auto vector_side = fft_obj.get_side();

    plt::figure_size(1200, 780);
    constexpr int colors = 1;
    plt::title(title);
    const std::vector<float> plot = pre_plot_vector(source_data, N);
    plt::imshow(&(plot[0]),
                          vector_side,
                          vector_side, colors,
                          std::map<std::string, std::string>{{"origin", "lower"}});
    plt::save("Plots/" + file_name + ".png");
}

void create_postplot(const Abstract_FFT& fft_obj, const std::string& title, const std::string& file_name){
    const std::complex<double>* source_data = fft_obj.get_source();
    const int N = fft_obj.get_element_count();
    const int  vector_side = fft_obj.get_side();

    plt::figure_size(1200, 780);
    constexpr int colors = 1;
    plt::title(title);
    const std::vector<float> plot = post_plot_vector(source_data, N);
    plt::imshow(&(plot[0]),
                vector_side,
                vector_side, colors,
                std::map<std::string, std::string>{{"origin", "lower"}});
    plt::save("Plots/" + file_name + ".png");
}
#endif