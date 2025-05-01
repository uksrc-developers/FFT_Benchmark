//
// Created by marcuskeil on 13/12/24.
//
#include "../include/Data_Functions.hpp"

#include "../include/Abstract_FFT.hpp"

double epsilon = 1e-10;

int compare_length = 1000;

long long get_sys_mem() {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long mem_size = (pages * page_size);
    return mem_size;
}

int verify_dimension(int dim){
    while ( std::fmod(pow(dim, 2), 8) > 0 ){
        dim--;
    }
    return dim;
}

int possible_vector_size(float memory_size){
    long long memory_limit = get_sys_mem();
    double requested_bytes = 0;
    int dimension = 0;
    dimension = int(std::sqrt( (memory_size * 1000000) / sizeof(std::complex<double>) ));
    dimension = verify_dimension(dimension);
    assert((void("Dimensions of created array were too small"), dimension >= 128));
    requested_bytes = (pow(dimension,2))*sizeof(std::complex<double>) + 24;
    assert((void("Requested too much memory"), requested_bytes < memory_limit));
    return dimension;
}

void fill_vector(std::complex<double>* v, int element_count){
//    generate(v.begin(), v.end(), fill_complex);
    std::vector<int> mids;
    int dim = int(sqrt(element_count));
    if ( dim%2 != 0 ){
        mids.resize(9);
        mids = std::vector<int>({
                                        int(dim/2 - 1) + int(dim*(dim/2 - 1)),
                                        int(dim/2 - 1) + int(dim*(dim/2 )),
                                        int(dim/2 - 1) + int(dim*(dim/2 + 1)),

                                        int(dim/2) + int(dim*(dim/2 - 1)),
                                        int(dim/2) + int(dim*(dim/2)),
                                        int(dim/2) + int(dim*(dim/2 + 1 )),

                                        int(dim/2 + 1) + int(dim*(dim/2 - 1)),
                                        int(dim/2 + 1) + int(dim*(dim/2 )),
                                        int(dim/2 + 1) + int(dim*(dim/2 + 1)),
                                });
    } else {
        mids.resize(4);
        mids = std::vector<int>({

                                        int(dim/2) + int(dim*(dim/2)),
                                        int(dim/2) + int(dim*(dim/2 + 1)),

                                        int(dim/2 + 1) + int(dim*(dim/2)),
                                        int(dim/2 + 1) + int(dim*(dim/2 + 1)),
                                });
    }
#pragma omp parallel for
    for ( int i : mids) {
        v[i] = std::complex<double>(1, 0);
    }
}

//void CT_radix_2<Abstract_FFT>(Abstract_FFT FFT_OBJECT) {
//    std::cout << "using radix 2" << std::endl;
//    constexpr int radix = 2;
//    int vec = FFT_OBJECT.get_element_count();
//    const int N_o_R = vec/radix;
//    std::complex<double> *fft_0;
//    FFT_OBJECT.allocate_memory(fft_0, N_o_R);
//    std::complex<double> *fft_1;
//    FFT_OBJECT.allocate_memory(fft_1, N_o_R);
//    bool toggle = false;
//#pragma omp parallel for
//    for( int i = 0; i < N_o_R; ++i ){
//        fft_0[i] = (FFT_OBJECT.get_source())[i*radix];
//        fft_1[i] = (FFT_OBJECT.get_source())[i*radix + 1];
//    }
//    FFT_OBJECT.make_plan(N_o_R);
//    FFT_OBJECT.transform(fft_0);
//    FFT_OBJECT.transform(fft_1);
//    FFT_OBJECT.sync();
//#pragma omp parallel for
//    for (int i=0; i < N_o_R; i++ ){
//        auto q = std::complex<double>(
//            cos(-((2*M_PI)/vec)*i),
//            sin(-((2*M_PI)/vec)*i))*fft_1[i];
//        auto q_1  = std::complex<double>(
//            cos(-(2*M_PI)/radix),
//            sin(-(2*M_PI)/radix));
//        (FFT_OBJECT.get_source())[i] = fft_0[i] + q;
//        (FFT_OBJECT.get_source())[i+N_o_R] = fft_0[i] + q*q_1;
//    }
//    FFT_OBJECT.free_memory(fft_0);
//    FFT_OBJECT.free_memory(fft_1);
//}


std::vector<float> pre_plot_vector(std::complex<double>* v, const int element_count) {
    std::vector<float> plot_vector;
    plot_vector.resize(element_count);
#pragma omp parallel for
    for (int i = 0; i < element_count; i++)
        plot_vector[i] = static_cast<float>(abs(v[i]));
    return plot_vector;
}

std::vector<float> post_plot_vector(std::complex<double>* v, int element_count){
    std::vector<float> plot_vector;
    plot_vector.resize(element_count);
    int dim = int(sqrt(element_count));
#pragma omp parallel for
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

void print_data(std::complex<double>* v, const int element_count){
    int count = 0;
    for (int i = 0; i < element_count; i++) {
        std::cout << v[i];
        count++;
        if ( count >= int(sqrt(element_count)) ){
            std::cout << "\n";
            count = 0;
        }
    }
    std::cout<<"\n";
}

float compare_data(const std::complex<double>* v, const int element_count){
    std::complex<double> sum = {0, 0};
    int sum_length = compare_length;
    if (sum_length > element_count)
        sum_length = element_count;
#pragma omp parallel for
    for (int i = 0; i < sum_length; i++){
        sum += std::complex<double>(std::abs(v[i].real()), std::abs(v[i].imag()));
    }
    if( static_cast<float>(abs(sum)) == 0.0 ) {
        std::cout << "Sum value = " << sum << std::endl;
        throw std::runtime_error("Transform not performed");
    }
    return static_cast<float>(abs(sum));
}

#if __has_include( "matplotlibcpp.h" )
void create_preplot(std::complex<double>* source_data, int element_count, const std::string& title, const std::string& file_name){
    const int vector_side = sqrt(element_count);
    plt::figure_size(1200, 780);
    constexpr int colors = 1;
    plt::title(title);
    std::vector<float> plot = pre_plot_vector(source_data, element_count);
    plt::imshow(&(plot[0]),
                          vector_side,
                          vector_side, colors,
                          std::map<std::string, std::string>{{"origin", "lower"}});
    plt::save("Plots/" + file_name + ".png");
}

void create_postplot(std::complex<double>* source_data, int element_count, const std::string& title, const std::string& file_name){
    const int vector_side = sqrt(element_count);
    plt::figure_size(1200, 780);
    constexpr int colors = 1;
    plt::title(title);
    std::vector<float> plot = post_plot_vector(source_data, element_count);
    plt::imshow(&(plot[0]),
                vector_side,
                vector_side, colors,
                std::map<std::string, std::string>{{"origin", "lower"}});
    plt::save("Plots/" + file_name + ".png");
}
#endif

