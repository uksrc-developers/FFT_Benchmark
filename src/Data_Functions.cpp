//
// Created by marcuskeil on 13/12/24.
//
#include "../include/Data_Functions.hpp"

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
#pragma omp parallel for schedule(static)
    for ( int i : mids) {
        v[i] = std::complex<double>(1, 0);
    }
}

void CT_transform(Abstract_FFT& fft_obj, const int split) {
    switch (split) {
        case 2: return CT_radix_2(fft_obj);
        case 3: return CT_radix_3(fft_obj);
        case 4: return CT_radix_4(fft_obj);
        case 5: return CT_radix_5(fft_obj);
        case 7: return CT_radix_7(fft_obj);
        case 8: return CT_radix_8(fft_obj);
        default: throw std::invalid_argument("Unknown split count error");
    }
}

inline std::complex<double> WKN(int k, int N) {
    return {
        cos(-((2*M_PI*(k))/N)),
        sin(-((2*M_PI*(k))/N))
    };
}

void CT_radix_2(Abstract_FFT& fft_obj) {
    std::complex<double>* v = fft_obj.get_source();
    const std::size_t N = fft_obj.get_element_count();
    const int N_int = static_cast<int>(N);
    const std::size_t half = N / 2;

    std::vector<std::complex<double>> temp(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < half; ++i) {
        temp[i] = v[2 * i];
        temp[i + half] = v[2 * i + 1];
    }
    std::copy(temp.begin(), temp.end(), v);

    std::complex<double>* v_e = v;
    std::complex<double>* v_o = v + half;
    //<<< FFT CALLS >>>
    fft_obj.partial_transform(v_e, half);
    fft_obj.partial_transform(v_o, half);
    //<<< FFT CALLS >>>

#pragma omp parallel for
    for (int k = 0; k < half; k++) {
        auto w = WKN(static_cast<int>(k), N_int);
        std::complex<double> u = v_e[k];
        std::complex<double> t = v_o[k];
        v[k] = u + t * w;
        v[k + half] = u - t * w;
    }
}

void CT_radix_3(Abstract_FFT& fft_obj) {
    int radix = 3;
    std::complex<double>* v = fft_obj.get_source();
    const std::size_t N = fft_obj.get_element_count();
    const int N_int = static_cast<int>(N);

    const std::size_t third = N / radix;
    // Split into 3 sub-arrays: v0, v1, v2 interleaved
    std::vector<std::complex<double>> temp(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < third; ++i) {
        temp[i] = v[radix * i];       // v0
        temp[i + third] = v[radix * i + 1]; // v1
        temp[i + 2 * third] = v[radix * i + 2]; // v2
    }
    std::copy(temp.begin(), temp.end(), v);

    // Views into split sections
    std::complex<double>* v0 = v;
    std::complex<double>* v1 = v + third;
    std::complex<double>* v2 = v + 2 * third;

    // Recursively transform each third
    fft_obj.partial_transform(v0, third);
    fft_obj.partial_transform(v1, third);
    fft_obj.partial_transform(v2, third);


    // Butterfly recombination
#pragma omp parallel for
    for (std::size_t k = 0; k < third; ++k) { // W^k_N = {cos( -2 * pi * i * k / N), sin( -2 * pi * i * k / N)}
        auto wk = WKN(static_cast<int>(k), N_int);
        auto w2k = WKN(static_cast<int>(k+third), N_int);
        auto w3k = WKN(static_cast<int>(k+2*third), N_int);

        std::complex<double> a = v0[k];
        std::complex<double> b = v1[k];
        std::complex<double> c = v2[k];
        
        v[k] = a + b *wk + c * (wk*wk);
        v[k + third] = a + b * w2k + c * (w2k*w2k);
        v[k + 2*third] = a + b * w3k + c * (w3k*w3k);
    }
}

void CT_radix_4(Abstract_FFT& fft_obj) {
    int radix = 4;
    std::complex<double>* v = fft_obj.get_source();
    const std::size_t N = fft_obj.get_element_count();
    const int N_int = static_cast<int>(N);
    const std::size_t fourth = N / radix;

    std::vector<std::complex<double>> temp(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < fourth; ++i) {
        temp[i] = v[radix * i]; // v0
        temp[i + fourth] = v[radix * i + 1]; // v1
        temp[i + 2 * fourth] = v[radix * i + 2]; // v2
        temp[i + 3 * fourth] = v[radix * i + 3]; // v3
    }
    std::copy(temp.begin(), temp.end(), v);

    std::complex<double>* v0 = v;
    std::complex<double>* v1 = v + fourth;
    std::complex<double>* v2 = v + 2 * fourth;
    std::complex<double>* v3 = v + 3 * fourth;


    fft_obj.partial_transform(v0, fourth);
    fft_obj.partial_transform(v1, fourth);
    fft_obj.partial_transform(v2, fourth);
    fft_obj.partial_transform(v3, fourth);
#pragma omp parallel for
    for (int k = 0; k < fourth; k++) {
        auto w = WKN(static_cast<int>(k), N_int);
        std::complex<double> a = v0[k];
        std::complex<double> b = v1[k];
        std::complex<double> c = v2[k];
        std::complex<double> d = v3[k];
        auto wk = WKN(static_cast<int>(k), N_int);
        auto w2k = WKN(static_cast<int>(k+fourth), N_int);
        auto w3k = WKN(static_cast<int>(k+2*fourth), N_int);
        auto w4k = WKN(static_cast<int>(k+3*fourth), N_int);
        v[k] = a + b*wk + c*(wk*wk) + d*(wk*wk*wk);
        v[k+fourth] = a + b*w2k + c*(w2k*w2k) + d*(w2k*w2k*w2k);
        v[k+2*fourth] = a + b*w3k + c*(w3k*w3k) + d*(w3k*w3k*w3k);
        v[k+3*fourth] = a + b*w4k + c*(w4k*w4k) + d*(w4k*w4k*w4k);
    }
}

void CT_radix_5(Abstract_FFT& fft_obj) {
    int radix = 5;
    std::complex<double>* v = fft_obj.get_source();
    const std::size_t N = fft_obj.get_element_count();
    const int N_int = static_cast<int>(N);
    const std::size_t fifth = N / radix;

    std::vector<std::complex<double>> temp(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < fifth; ++i) {
        temp[i] = v[radix * i]; // v0
        temp[i + fifth] = v[radix * i + 1]; // v1
        temp[i + 2 * fifth] = v[radix * i + 2]; // v2
        temp[i + 3 * fifth] = v[radix * i + 3]; // v3
        temp[i + 4 * fifth] = v[radix * i + 4]; // v4
    }
    std::copy(temp.begin(), temp.end(), v);

    std::complex<double>* v0 = v;
    std::complex<double>* v1 = v + fifth;
    std::complex<double>* v2 = v + 2 * fifth;
    std::complex<double>* v3 = v + 3 * fifth;
    std::complex<double>* v4 = v + 4 * fifth;

    fft_obj.partial_transform(v0, fifth);
    fft_obj.partial_transform(v1, fifth);
    fft_obj.partial_transform(v2, fifth);
    fft_obj.partial_transform(v3, fifth);
    fft_obj.partial_transform(v4, fifth);

#pragma omp parallel for
    for (int k = 0; k < fifth; k++) {
        auto w = WKN(static_cast<int>(k), N_int);
        std::complex<double> a = v0[k];
        std::complex<double> b = v1[k];
        std::complex<double> c = v2[k];
        std::complex<double> d = v3[k];
        std::complex<double> e = v4[k];

        auto wk = WKN(static_cast<int>(k), N_int);
        auto w2k = WKN(static_cast<int>(k+fifth), N_int);
        auto w3k = WKN(static_cast<int>(k+2*fifth), N_int);
        auto w4k = WKN(static_cast<int>(k+3*fifth), N_int);
        auto w5k = WKN(static_cast<int>(k+4*fifth), N_int);

        v[k] = a + b*wk + c*(wk*wk) + d*(wk*wk*wk) + e*(wk*wk*wk*wk);
        v[k+fifth] = a + b*w2k + c*(w2k*w2k) + d*(w2k*w2k*w2k) + e*(w2k*w2k*w2k*w2k);
        v[k+2*fifth] = a + b*w3k + c*(w3k*w3k) + d*(w3k*w3k*w3k) + e*(w3k*w3k*w3k*w3k);
        v[k+3*fifth] = a + b*w4k + c*(w4k*w4k) + d*(w4k*w4k*w4k) + e*(w4k*w4k*w4k*w4k);
        v[k+4*fifth] = a + b*w5k + c*(w5k*w5k) + d*(w5k*w5k*w5k) + e*(w5k*w5k*w5k*w5k);
    }
}

void CT_radix_7(Abstract_FFT& fft_obj) {
    int radix = 7;
    std::complex<double>* v = fft_obj.get_source();
    const std::size_t N = fft_obj.get_element_count();
    const int N_int = static_cast<int>(N);
    const std::size_t seventh = N / radix;

    std::vector<std::complex<double>> temp(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < seventh; ++i) {
        temp[i] = v[radix * i]; // v0
        temp[i + seventh] = v[radix * i + 1]; // v1
        temp[i + 2 * seventh] = v[radix * i + 2]; // v2
        temp[i + 3 * seventh] = v[radix * i + 3]; // v3
        temp[i + 4 * seventh] = v[radix * i + 4]; // v4
        temp[i + 5 * seventh] = v[radix * i + 5]; // v4
        temp[i + 6 * seventh] = v[radix * i + 6]; // v6
    }
    std::copy(temp.begin(), temp.end(), v);

    std::complex<double>* v0 = v;
    std::complex<double>* v1 = v + seventh;
    std::complex<double>* v2 = v + 2 * seventh;
    std::complex<double>* v3 = v + 3 * seventh;
    std::complex<double>* v4 = v + 4 * seventh;
    std::complex<double>* v5 = v + 5 * seventh;
    std::complex<double>* v6 = v + 6 * seventh;

    fft_obj.partial_transform(v0, seventh);
    fft_obj.partial_transform(v1, seventh);
    fft_obj.partial_transform(v2, seventh);
    fft_obj.partial_transform(v3, seventh);
    fft_obj.partial_transform(v4, seventh);
    fft_obj.partial_transform(v5, seventh);
    fft_obj.partial_transform(v6, seventh);

#pragma omp parallel for
    for (int k = 0; k < seventh; k++) {
        auto w = WKN(static_cast<int>(k), N_int);
        std::complex<double> a = v0[k];
        std::complex<double> b = v1[k];
        std::complex<double> c = v2[k];
        std::complex<double> d = v3[k];
        std::complex<double> e = v4[k];
        std::complex<double> f = v5[k];
        std::complex<double> g = v6[k];

        auto wk = WKN(static_cast<int>(k), N_int);
        auto w2k = WKN(static_cast<int>(k+seventh), N_int);
        auto w3k = WKN(static_cast<int>(k+2*seventh), N_int);
        auto w4k = WKN(static_cast<int>(k+3*seventh), N_int);
        auto w5k = WKN(static_cast<int>(k+4*seventh), N_int);
        auto w6k = WKN(static_cast<int>(k+5*seventh), N_int);
        auto w7k = WKN(static_cast<int>(k+6*seventh), N_int);

        v[k] = a + b*wk + c*(wk*wk) + d*(wk*wk*wk) + e*(wk*wk*wk*wk) + f*(wk*wk*wk*wk*wk) + g*(wk*wk*wk*wk*wk*wk);
        v[k+seventh] = a + b*w2k + c*(w2k*w2k) + d*(w2k*w2k*w2k) + e*(w2k*w2k*w2k*w2k) + f*(w2k*w2k*w2k*w2k*w2k) + g*(w2k*w2k*w2k*w2k*w2k*w2k);
        v[k+2*seventh] = a + b*w3k + c*(w3k*w3k) + d*(w3k*w3k*w3k) + e*(w3k*w3k*w3k*w3k) + f*(w3k*w3k*w3k*w3k*w3k) + g*(w3k*w3k*w3k*w3k*w3k*w3k);
        v[k+3*seventh] = a + b*w4k + c*(w4k*w4k) + d*(w4k*w4k*w4k) + e*(w4k*w4k*w4k*w4k) + f*(w4k*w4k*w4k*w4k*w4k) + g*(w4k*w4k*w4k*w4k*w4k*w4k);
        v[k+4*seventh] = a + b*w5k + c*(w5k*w5k) + d*(w5k*w5k*w5k) + e*(w5k*w5k*w5k*w5k) + f*(w5k*w5k*w5k*w5k*w5k) + g*(w5k*w5k*w5k*w5k*w5k*w5k);
        v[k+5*seventh] = a + b*w6k + c*(w6k*w6k) + d*(w6k*w6k*w6k) + e*(w6k*w6k*w6k*w6k) + f*(w6k*w6k*w6k*w6k*w6k) + g*(w6k*w6k*w6k*w6k*w6k*w6k);
        v[k+6*seventh] = a + b*w7k + c*(w7k*w7k) + d*(w7k*w7k*w7k) + e*(w7k*w7k*w7k*w7k) + f*(w7k*w7k*w7k*w7k*w7k) + g*(w7k*w7k*w7k*w7k*w7k*w7k);
    }
}

void CT_radix_8(Abstract_FFT& fft_obj) {
    int radix = 8;
    std::complex<double>* v = fft_obj.get_source();
    const std::size_t N = fft_obj.get_element_count();
    const int N_int = static_cast<int>(N);
    const std::size_t eighth = N / radix;

    std::vector<std::complex<double>> temp(N);
#pragma omp parallel for
    for (std::size_t i = 0; i < eighth; ++i) {
        temp[i] = v[radix * i]; // v0
        temp[i + eighth] = v[radix * i + 1]; // v1
        temp[i + 2 * eighth] = v[radix * i + 2]; // v2
        temp[i + 3 * eighth] = v[radix * i + 3]; // v3
        temp[i + 4 * eighth] = v[radix * i + 4]; // v4
        temp[i + 5 * eighth] = v[radix * i + 5]; // v4
        temp[i + 6 * eighth] = v[radix * i + 6]; // v6
        temp[i + 7 * eighth] = v[radix * i + 7]; // v7
    }
    std::copy(temp.begin(), temp.end(), v);

    std::complex<double>* v0 = v;
    std::complex<double>* v1 = v + eighth;
    std::complex<double>* v2 = v + 2 * eighth;
    std::complex<double>* v3 = v + 3 * eighth;
    std::complex<double>* v4 = v + 4 * eighth;
    std::complex<double>* v5 = v + 5 * eighth;
    std::complex<double>* v6 = v + 6 * eighth;
    std::complex<double>* v7 = v + 7 * eighth;

    fft_obj.partial_transform(v0, static_cast<int>(eighth));
    fft_obj.partial_transform(v1, static_cast<int>(eighth));
    fft_obj.partial_transform(v2, static_cast<int>(eighth));
    fft_obj.partial_transform(v3, static_cast<int>(eighth));
    fft_obj.partial_transform(v4, static_cast<int>(eighth));
    fft_obj.partial_transform(v5, static_cast<int>(eighth));
    fft_obj.partial_transform(v6, static_cast<int>(eighth));
    fft_obj.partial_transform(v7, static_cast<int>(eighth));

#pragma omp parallel for
    for (int k = 0; k < eighth; k++) {
        auto w = WKN(static_cast<int>(k), N_int);
        std::complex<double> a = v0[k];
        std::complex<double> b = v1[k];
        std::complex<double> c = v2[k];
        std::complex<double> d = v3[k];
        std::complex<double> e = v4[k];
        std::complex<double> f = v5[k];
        std::complex<double> g = v6[k];
        std::complex<double> h = v7[k];

        auto wk = WKN(static_cast<int>(k), N_int);
        auto w2k = WKN(static_cast<int>(k+eighth), N_int);
        auto w3k = WKN(static_cast<int>(k+2*eighth), N_int);
        auto w4k = WKN(static_cast<int>(k+3*eighth), N_int);
        auto w5k = WKN(static_cast<int>(k+4*eighth), N_int);
        auto w6k = WKN(static_cast<int>(k+5*eighth), N_int);
        auto w7k = WKN(static_cast<int>(k+6*eighth), N_int);
        auto w8k = WKN(static_cast<int>(k+7*eighth), N_int);

        v[k] = a + b*wk + c*(wk*wk) + d*(wk*wk*wk) + e*(wk*wk*wk*wk) + f*(wk*wk*wk*wk*wk) + g*(wk*wk*wk*wk*wk*wk) + h*(wk*wk*wk*wk*wk*wk*wk);
        v[k+eighth] = a + b*w2k + c*(w2k*w2k) + d*(w2k*w2k*w2k) + e*(w2k*w2k*w2k*w2k) + f*(w2k*w2k*w2k*w2k*w2k) + g*(w2k*w2k*w2k*w2k*w2k*w2k) + h*(w2k*w2k*w2k*w2k*w2k*w2k*w2k);
        v[k+2*eighth] = a + b*w3k + c*(w3k*w3k) + d*(w3k*w3k*w3k) + e*(w3k*w3k*w3k*w3k) + f*(w3k*w3k*w3k*w3k*w3k) + g*(w3k*w3k*w3k*w3k*w3k*w3k) + h*(w3k*w3k*w3k*w3k*w3k*w3k*w3k);
        v[k+3*eighth] = a + b*w4k + c*(w4k*w4k) + d*(w4k*w4k*w4k) + e*(w4k*w4k*w4k*w4k) + f*(w4k*w4k*w4k*w4k*w4k) + g*(w4k*w4k*w4k*w4k*w4k*w4k) + h*(w4k*w4k*w4k*w4k*w4k*w4k*w4k);
        v[k+4*eighth] = a + b*w5k + c*(w5k*w5k) + d*(w5k*w5k*w5k) + e*(w5k*w5k*w5k*w5k) + f*(w5k*w5k*w5k*w5k*w5k) + g*(w5k*w5k*w5k*w5k*w5k*w5k) + h*(w5k*w5k*w5k*w5k*w5k*w5k*w5k);
        v[k+5*eighth] = a + b*w6k + c*(w6k*w6k) + d*(w6k*w6k*w6k) + e*(w6k*w6k*w6k*w6k) + f*(w6k*w6k*w6k*w6k*w6k) + g*(w6k*w6k*w6k*w6k*w6k*w6k) + h*(w6k*w6k*w6k*w6k*w6k*w6k*w6k);
        v[k+6*eighth] = a + b*w7k + c*(w7k*w7k) + d*(w7k*w7k*w7k) + e*(w7k*w7k*w7k*w7k) + f*(w7k*w7k*w7k*w7k*w7k) + g*(w7k*w7k*w7k*w7k*w7k*w7k) + h*(w7k*w7k*w7k*w7k*w7k*w7k*w7k);
        v[k+7*eighth] = a + b*w8k + c*(w8k*w8k) + d*(w8k*w8k*w8k) + e*(w8k*w8k*w8k*w8k) + f*(w8k*w8k*w8k*w8k*w8k) + g*(w8k*w8k*w8k*w8k*w8k*w8k) + h*(w8k*w8k*w8k*w8k*w8k*w8k*w8k);
    }
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
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < sum_length; i++){
        sum += std::complex<double>(std::abs(v[i].real()), std::abs(v[i].imag()));
    }
    auto abs_sum = static_cast<float>(std::sqrt(sum.real()*sum.real() + sum.imag()*sum.imag()));
    if( static_cast<float>(abs_sum) == 0.0 ) {
        std::cout << "Sum value = " << sum << std::endl;
        throw std::runtime_error("Transform not performed");
    }
    return abs_sum;
}

