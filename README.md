# FFT Benchmark

FFT Benchmark aimes to be a unified method for running FFT libraries in a standardised approach to test hardware and library
performance across different hardware architectures.

# Description

FFT Benchmark standardises the creation of a 2D square matrix into a designated size of memory, which can then be used with various FFT libraries for the sake of timing and profiling individual libraries. This allows for the direct comparison between FFT libraries when running multiple, or the direct comparison between systems when running the same transforms on different systems. For example, it would allow for comparing different accelerators, such as NVidia or AMD GPUs, by using the recommended libraries for each accelarator.
## Component details

<details>
<summary>Data Functions</summary>
The *Data_Functions* script provides several functions that are used to create the
2D matricies that are used for the FFT. A brief overview of each functions use is
detailed in the table below.

|                         Function                          |                                                                          Description                                                                           |
|:---------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                      `get_sys_mem()`                      |                                              returns the ammount of RAM currently available for use by the system                                              |
|                `verify_dimension(int dim)`                |         verifies that the current dimensions are divisible by sixteen in order to facilitate splitting the 2D matrix when it is too large for GPU VRAM         |
|         `possible_vector_size(float memory_size)`         | Uses the given memory_size float in order to create a matrix which uses as much memory as possible within those limits and those imposed by verify_dimension() |
|   `fill_vector(complex<double>* v, int element_count)`    |                                  Fills the 2D matrix with pointer *v with a rectanle of value 1 at the centre of the matrix.                                   |
| `pre_plot_vector(complex<double>* v, int element_count)`  |                    Prepares the 2D matrix with pointer *v to be plotted using python matplotlib prior to transforming for user inpsection.                     |
| `post_plot_vector(complex<double>* v, int element_count)` |                      Prepares the 2D matrix with pointer *v to be plotted using python matplotlib post transforming for user inpsection.                       |
|    `print_data(complex<double>* v, int element_count)`    |                                                   Prints the 2D matrix with pointer *v for user inspection.                                                    |
|   `compare_data(complex<double>* v, int element_count)`   |          Prints the sum of the values of the 2D matrix with pointer *v in order to verify that the transformation returned values other than just 0.           |

</details>

<details>
<summary>FFT Classes</summary>

The basic `Abstract_FFT` class represents the top level class all other FFT classes inherit from. Each class must have
the following functions

|          Function          |                                                               Description                                                               |
|:--------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
|          `name()`          |                                        Returns the name of the library being used for this class                                        |
|        `get_side()`        |                                       Returns the side length of the 2D matrix to be transformed                                        |
|       `get_memory()`       |                                            Returns the size of the matrix on the RAM in MB.                                             |
|       `get_source()`       | Returns the 2D matrix, if prior to transform it will return the original matrix. Post transform it will return the fourier space matrix |
|   `get_element_count()`    |                                        Returns the count of elements in the matrix. i.e. side^2                                         |
|       `transform()`        |                                                     Performs the fourier transform.                                                     |
| `time_transform(int runs)` |                 Performs the fourier transform `runs` amount of times and returns the average time the transform took.                  |

Classes that inherit from `Abstract_FFT` may contain other functions, but should be limited to internal use only.

</details>

## Installation
<details>
<summary>Installation</summary>



In order to build this benchmark, we use cmake. We recommend setting the following options.

|        Option        |       Value(s)       |                                                       Description                                                       |
|:--------------------:|:--------------------:|:-----------------------------------------------------------------------------------------------------------------------:|
|  `CMAKE_C_COMPILER`  |      Ex: clang       |                                         C compiler to user. Tested with clang.                                          |
| `CMAKE_CXX_COMPILER` |     Ex: clang++      |                                        C++ compiler to use. Tested with clang++.                                        |
|       `ONEAPI`       |        ON/OFF        |                    Option to turn on using oneApi mkl instead of default FFTW kernel. (default OFF)                     |
|     `ONEAPI_DIR`     | <\Path\To\oneAPImkl> | Option to pass path to fftw3, if it is not in default installation directory. (default `/opt/intel/oneapi/mkl/2025.1`)  |
|     `FFTW3_DIR`      |   <\Path\To\FFTW3>   |          Option to pass path to fftw3, if it is not in default installation directory. (default `/usr/local`)           |
|  `PYTHON_PLOTTING`   |        ON/OFF        |                      Option to turn on the Python library for plotting FFT results. (default OFF)                       |
|      `CUDA_FFT`      |        ON/OFF        |                                   Option to turn on the cuFFT library. (default OFF)                                    |
|      `CUDA_DIR`      |   <\Path\To\CUDA>    |  Option to pass path to the CUDA libraries, if it is not in default installation directory.(default `/usr/local/cuda`)  |
|      `ROC_FFT`       |        ON/OFF        |                                   Option to turn on the ROCm library.  (default OFF)                                    |
|      `ROCM_DIR`      |   <\Path\To\ROCm>    |        Option to pass path to ROCm library, if it is not in default installation directory.(default `/opt/rocm`)        |
|      `HIP_DIR`       |    <Path\To\hip>     | Option to pass path to hip library, if it is not in default installation directory.(default `/opt/rocm/lib/cmake/hip`)  |

</details>

##  In- and Outputs

### Inputs
<details>
<summary>Inputs</summary>
Once compiled and make has been run, calling this code uses the following inputs

| Option |           Value           |                                                                  Description                                                                   |
|:------:|:-------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
|   -e   |           None            |                                 Flag that toggles running by memory size instead of element counts (Optional)                                  |
|   -p   |           None            |                                        Flag to plot each FFT result of different sizes once (Optional)                                         |
|   -o   | <\Path\to\output\file.txt |                                             Output file path. (Optional, Default "./default.txt")                                              |
|   -f   |           None            |                                                Flag to toggle running FFTs with FFTW (Optional)                                                |
|   -n   |           None            |                                               Flag to toggle running FFTs with cuFFT (Optional)                                                |
|   -a   |           None            |                                              Flag to toggle running FFTs with rocFFT (Optional)                                                |
|   -r   |            Int            |                          Int on how many different size runs to perform (min 1, if -e is not set, max 7, else no max)                          |
|   -c   |            Int            |                                      Int on how often to repeat transforms for averaging results (min 1)                                       |
|   -s   |           float           | If -e is set, this determines the starting size in memory to use. Units are in MB, all subsequent runs are double the size of the previous run |

</details>


### Outputs
<details>
<summary>Outputs</summary>

The output of FFT_Benchmark is a text file with a small table. The table summarises the results. The columns are:
FFT_Code, Mem_Size [MB], Avg_time[ms], and Check_Value. FFT_Code indicates which library was used to run the transform.
Mem_Size is the size of the array that was transformed in [MB]. Avg_time[ms] is the sum of time each transform took divided by
the number of runs that were performed. Check_Value is an output meant to assist in determining if the transform was computed.
This last column may be removed in the future when better methods to catch failed transformations are implemented across 
libraries.

</details>

## Example usage

<details>
<summary>Example</summary>
</details>

## Software Requirements

<details>
<summary>Software Requirements</summary>
</details>

## Hardware Requirements

<details>
<summary>Hardware Requirements</summary>
</details>

## Directory Structure

<details>
<summary>File Structure Diagram</summary>
```md
FFT_Bench
├── include
│   ├── Abstract_FFT.hpp
│   ├── Run_Functions.hpp
│   ├── Data_Functions.hpp
│   ├── Plotting_Functions.hpp
│   ├── FFTW_Class.hpp
│   ├── cuFFT_Class.hpp
│   └── rocFFT_Class.hpp
├── src
│   ├── Run_Functions.cpp
│   ├── Data_Functions.cpp
│   ├── Plotting_Functions.cpp
│   ├── FFTW_Class.cpp
│   ├── cuFFT_Class.cpp
│   └── rocFFT_Class.cpp
├── CMakeLists.txt
├── main.cpp
└── README.md
```
</details>

# To Do


# UKSRC related Links

### Developers and Contributors

Developers:
- Keil, Marcus (UCL)
