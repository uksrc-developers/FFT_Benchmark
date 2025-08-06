#include <tuple>
#include <vector>
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


int main(int argc, char **argv) {
    auto [Mem_Run, Plot, FFTW, cuFFT, rocFFT,
        run_count, repeat_count, mem_start] = retrieve_arguments(argc, argv);
    if (Mem_Run) {
        cout << "Run_Count: " << run_count << "; With Memory Size\n";
        cout << "FFT_Code,\tMem_Size[MB],\tAvg_time[ms],\tCheck_Value\n";
        vector<float> memories = get_memories(mem_start, run_count);
        if (FFTW){
            memory_run<FFTW_Class>(memories, run_count, Plot);
        }
#ifdef CUDA_FFT
        if (cuFFT){
            memory_run<cuFFT_Class>(memories, run_count, Plot);
        }
#endif
#ifdef ROC_FFT
        if (rocFFT){
            memory_run<rocFFT_Class>(memories, run_count, Plot);
        }
#endif
    } else {
        cout << "Run_Count: " << run_count << "; With elements.\n";
        cout << "FFT_Code,\tMem_Size[MB],\tAvg_time[ms],\tCheck_Value\n";
        vector<int> elements = get_elements(run_count);
        if (FFTW){
            element_run<FFTW_Class>(elements, run_count, Plot);
        }
#ifdef CUDA_FFT
        if (cuFFT){
            element_run<cuFFT_Class>(elements, run_count, Plot);
        }
#endif
#ifdef ROC_FFT
        if (rocFFT){
            element_run<rocFFT_Class>(elements, run_count, Plot);
        }
#endif
    }
    cout << "Run_Finished\n";
    return 0;
}
