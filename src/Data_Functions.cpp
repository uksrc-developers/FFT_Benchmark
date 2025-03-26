//
// Created by marcuskeil on 13/12/24.
//
#include "../include/Data_Functions.hpp"

double epsilon = 1e-10;

int compare_length = 1000;

long int get_sys_mem() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long mem_size = (pages * page_size);
    std::cout << mem_size << std::endl;
    return mem_size;
}

int verify_dimension(int dim){
    while ( std::fmod(pow(dim, 2), 8) > 0 ){
        dim--;
    }
    return dim;
}

int possible_vector_size(float memory_size){
    long int memory_limit = get_sys_mem();
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

void compare_data(const std::complex<double>* v, const int element_count){
    std::complex<double> sum = {0, 0};
    for (int i = 0; i < compare_length; i++){
        sum += std::complex<double>(std::abs(v[i].real()), std::abs(v[i].imag()));
    }
    //if (sum.real() < epsilon && sum.imag() < epsilon) {
    //    std::cout << "Transform for element_count " << element_count << " had a sum of " << sum << " for the first "
    //              << compare_length << " entries.\n";
    //}
}
