#include <tuple>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <iostream>

#include "include/Run_Functions.hpp"
#include "include/FFTW_Class.hpp"
#ifdef PYTHON_PLOTTING
#include "include/Plotting_Functions.hpp"
#endif
#ifdef CUDA_FFT
#include "include/cuFFT_Class.hpp"
#endif
#ifdef ROC_FFT
#include "include/rocFFT_Class.hpp"
#endif

using namespace std;

ofstream output_file;

void atexit_handler_1() {
    output_file << "Premature close" << endl;
    output_file.close();
}

int main(int argc, char **argv) {
    auto [Mem_Run, Plot, file_name,  FFTW, cuFFT, rocFFT,
        run_count, repeat_count, mem_start] = retrieve_arguments(argc, argv);
    output_file.open(file_name, ios::out | ios::app);
    const int exiting = std::atexit(atexit_handler_1);
    if (exiting)
    {
        std::cerr << "Registration failed!\n";
        return EXIT_FAILURE;
    }

    if (Mem_Run) {
        output_file << "Repeating runs " << repeat_count << " times; With Memory Size\n";
        output_file << "FFT_Code,\tMem_Size[MB],\tAvg_time[ms],\tCheck_Value\n";
        vector<float> memories = get_memories(mem_start, run_count);
        if (FFTW){
            memory_run<FFTW_Class>(memories, run_count, Plot, output_file);
        }
        #ifdef CUDA_FFT
        if (cuFFT){
            memory_run<cuFFT_Class>(memories, run_count, Plot, output_file);
        }
        #endif
        #ifdef ROC_FFT
        if (rocFFT){
            memory_run<rocFFT_Class>(memories, run_count, Plot, output_file);
        }
        #endif
    } else {
        output_file << "Repeating runs " << repeat_count << " times; With elements.\n";
        output_file << "FFT_Code,\tMem_Size[MB],\tAvg_time[ms],\tCheck_Value\n";
        vector<int> elements = get_elements(run_count);
        if (FFTW){
            element_run<FFTW_Class>(elements, run_count, Plot, output_file);
        }
        #ifdef CUDA_FFT
        if (cuFFT){
            element_run<cuFFT_Class>(elements, run_count, Plot, output_file);
        }
        #endif
        #ifdef ROC_FFT
        if (rocFFT){
            element_run<rocFFT_Class>(elements, run_count, Plot, output_file);
        }
        #endif
    }
    output_file << "Run_Finished\n";
    output_file.close();
    return 0;
}
