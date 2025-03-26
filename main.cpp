#include <cmath>
#include <iostream>
#include <getopt.h>

#include "include/Abstract_FFT.hpp"
#include "include/Data_Functions.hpp"
#include "include/FFTW_Class.hpp"

#if __has_include( "cufft.h")
#include "include/cuFFT_Class.hpp"
#endif

#if __has_include( <rocfft.h> )
#include "include/rocFFT_Class.hpp"
#endif

using namespace std;

template<class FFT_Class>
void memory_run(std::vector<float> const &memories, int runs = 5, bool plot = false){
    for (float mem : memories){
        FFT_Class fft_class(mem);
        if ( plot){
//#if __has_include( "matplotlibcpp.h" )
//            fft_class.transform();
//            fft_class.create_postplot(fft_class.name() + "_transform_"s + std::to_string(mem) + "MB");
//#endif
        }
        auto time = fft_class.time_transform(runs).count();
        std::cout << fft_class.name() << ", " <<
                     fft_class.get_element_count() << ", " <<
                     fft_class.get_memory() << ", " <<
                     time << "\n";
        compare_data(fft_class.get_source(),
                     fft_class.get_element_count());
    }
}

std::vector<float> linspace(const float start, const float end, const int count){
    std::vector<float> linspace_vec;
    linspace_vec.push_back(start);
    if (count == 1){
        return linspace_vec;
    } else {
        const float delta = (end - start) / static_cast<float>(count - 1);
        for(int i=1; i < count-1; i++){
            linspace_vec.push_back(start + delta*static_cast<float>(i));
        }
        linspace_vec.push_back(end);
        return linspace_vec;
    }
}

std::vector<float> double_space(float start, int count){
    std::vector<float> double_pace_vec;
    float current = start;
    double_pace_vec.push_back(start);
    for(int i=2; i < count+2; i++){
        current = current*2;
        double_pace_vec.push_back(current);
    }
    return double_pace_vec;
}

int main(int argc, char **argv) {
    //
    /// START ############# Get Commandline Options #############
    //
    float mem_start = 1000; // in [MB]
    int mem_count = 10;
    int run_count = 5;
    bool Plot = false;
    bool FFTW = false;
    bool CUDA = false;
    bool rocFFT = false;
    opterr = 0;

    while ( true ) {
        static struct option long_options[] =
                {
            /* These options don’t set a flag.
               We distinguish them by their indices. */
            {"mem_start",optional_argument, nullptr, 's'},
            {"mem_count",optional_argument,nullptr, 'm'},
            {"run_count",optional_argument,nullptr, 'n'},
            {"Plot",no_argument, nullptr,'p'},
            {"FFTW",no_argument, nullptr,'f'},
            {"CUDA",no_argument, nullptr,'c'},
            {"rocFFT",no_argument, nullptr,'r'},
            {nullptr, 0, nullptr, 0}
                };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        const int option = getopt_long(argc, argv, "s:m:n:pfcr", long_options, &option_index);

        /* Detect the end of the options. */
        if (option == -1)
            break;
        switch (option) {
            case 's':
                mem_start = float(strtod(optarg, nullptr));
                break;
            case 'm':
                mem_count = int(strtod(optarg, nullptr));
                break;
            case 'n':
                run_count = int(strtod(optarg, nullptr));
                assert(run_count>=1);
                break;
            case 'p':
                Plot = true;
                break;
            case 'f':
                FFTW = true;
                break;
            case 'c':
                CUDA = true;
                break;
            case 'r':
                rocFFT = true;
                break;
            case '?':
                if (optopt == 'c')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
            return 1;
            default:
                abort();
        }
    }

    std::cout << sizeof(std::complex<double>) << "\n";

    std::vector<float> memory_sizes = double_space(mem_start, mem_count);
    for (const float i : memory_sizes ){
        std::cout << i << ", ";
    }
    std::cout << "\n";

    //    std::vector<float> memory_sizes = {2000};
    std::cout << "Run_Count: " << run_count << "\n";
    std::cout << "FFT_Code, Element_Count, Memory_Size [B], Avg_time [ms]\n";

    if (FFTW){
        memory_run<FFTW_Class>(memory_sizes, run_count, Plot);
    }
#if __has_include( "cufft.h")
    if (CUDA){
        memory_run<cuFFT_Class>(memory_sizes, run_count, Plot);
    }
#endif
#if __has_include( <rocfft.h> )
    if (rocFFT){
        memory_run<rocFFT_Class>(memory_sizes, run_count, Plot);
    }
#endif

//    if ( true ){
//        std::cout << "Single call Test\n";
//        FFTW_Class fftw_test(fftw_mem);
//        std::cout << fftw_test.name() << ", " <<
//                     fftw_test.get_element_count() << ", " <<
//                     fftw_test.get_memory() << ", " <<
//                     fftw_test.time_transform(5).count() << "\n";
//        fftw_test.create_postplot("FFTW_post_transform");
//    }

    //
    /// FFTW END
    /// \n
    /// \n
    /// START CUDA
    //

//    if ( true ){
//        std::cout << "Single call Test\n";
//        cuFFT_Class cuda_test(0.3);
//        std::cout << cuda_test.name() << ", " <<
//                     cuda_test.get_element_count() << ", " <<
//                     cuda_test.get_memory() << ", " <<
//                     cuda_test.time_transform(1).count() << "\n";
//        cuda_test.create_postplot("CUDA_post_transform");
//        cuda_test.store_output();
//        std::cout << compare_data(cuda_test.get_source(), cuda_test.get_element_count(), "CUDA_128.txt")<< " = sum dif.\n";
//    }

    //
    /// CUDA END
    /// \n
    /// ############# Individual Run Example Section ############# END
    //
//    float memory_size = 12000;
//    std::cout << "Asking for " << memory_size/1000 << "[GB] RAM\n";
//    int vector_side = possible_vector_size(memory_size, false);
//    int vector_element_count = pow(vector_side, 2);
//    size_t vector_memory_size = vector_element_count*sizeof(std::complex<double>);
//    std::cout << "vector_side, vector_element_count, vector_memory_size [B]\n";
//    std::cout << vector_side << ", " <<
//                 vector_element_count << ", " <<
//                 vector_memory_size/1000000000.0 << "[GB]\n";
//    size_t workEstimate = 0;
//    cufftEstimate2d(vector_side, vector_side, CUFFT_Z2Z, &workEstimate);
//    std::complex<double> *source_data;
//    cudaMallocManaged(
//            &source_data,
//            vector_element_count*sizeof(std::complex<double>),
//            cudaMemAttachHost
//    );
//    fill_vector(source_data, vector_element_count);
//    split_fft(&source_data, vector_element_count, false);
//
//    plt::figure_size(1200, 780);
//    const int colors = 1;
//    plt::title(std::to_string(vector_memory_size/1000000) + "[MB] transformed");
//    std::vector<float> plot_vector = post_plot_vector(source_data, vector_element_count);
//    plt::imshow(&(plot_vector[0]),
//                vector_side,
//                vector_side, colors,
//                std::map<std::string, std::string>{{"origin", "lower"}});
//    plt::save("New_Split_DeallocateOld.png");

    return 0;
}
