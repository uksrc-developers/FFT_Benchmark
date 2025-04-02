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
#if __has_include( "matplotlibcpp.h" )
        if ( plot){
            fft_class.transform();
            create_postplot(
                fft_class.get_source(),
                fft_class.get_element_count(),
                fft_class.name()+" "+std::to_string(mem)+" MB" ,
                fft_class.name() + "_transform_" + std::to_string(mem) + "MB");
        }
#endif
        auto time = fft_class.time_transform(runs).count();
        auto checksum = compare_data(fft_class.get_source(), fft_class.get_element_count());
        std::cout << fft_class.name() << ", \t\t" <<
                     fft_class.get_memory()/1000000 << ", \t\t" <<
                     time << ", \t" <<
                     checksum << "\n";
        ;
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
    std::cout << "FFT_Code,\tMem_Size[MB],\tAvg_time[ms],\tCheck_Value\n";

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
    return 0;
}
