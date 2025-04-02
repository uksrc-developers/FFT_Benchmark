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

In order to build this benchmark, we use cmake. We recommend setting the following options.

|          Option          |     Value(s)     |                                                       Description                                                        |
|:------------------------:|:----------------:|:------------------------------------------------------------------------------------------------------------------------:|
|   `CMAKE_CXX_COMPILER`   |   Ex: clang++    |                                        C++ compiler to use. Tested with clang++.                                         |
| `CMAKE_EXE_LINKER_FLAGS` |   Ex: -fopenmp   | Options to pass to the linker tool. We recommend using some version of openmp to parallalise the for loops in this code. |
|       `FFTW3_DIR`        | <\Path\To\FFTW3> |           Option to pass path to fftw3, if it is not in default installation directory. (default `/usr/local`)           |
|        `CUDA_FFT`        |      ON/OFF      |                                    Option to turn on the cuFFT library. (default OFF)                                    |
|        `CUDA_DIR`        | <\Path\To\CUDA>  |  Option to pass path to the CUDA libraries, if it is not in default installation directory.(default `/usr/local/cuda`)   |
|        `ROC_FFT`         |      ON/OFF      |                                    Option to turn on the ROCm library.  (default OFF)                                    |
|        `ROCM_DIR`        | <\Path\To\ROCm>  |        Option to pass path to ROCm library, if it is not in default installation directory.(default `/opt/rocm`)         |
|                          |                  |                                                                                                                          |

-DROC_FFT=ON -S ./ -B ./debug-cmake


##  In- and Outputs

### Inputs
<details>
<summary>Inputs</summary>

<details>
<summary></summary>
</details>

</details>


### Outputs
<details>
<summary>Outputs</summary>

<details>
<summary></summary>
</details>

</details>

## Example usage

<details>
<summary>Example</summary>

<details>
<summary></summary>
</details>

</details>

## Software Requirements

<details>
<summary>Software Requirements</summary>

<details>
<summary></summary>
</details>

</details>

## Hardware Requirements

<details>
<summary>Hardware Requirements</summary>

<details>
<summary></summary>
</details>

</details>

## Directory Structure

<details>
<summary>File Structure Diagram</summary>
```md
FFT_Bench
├── include
│   ├── Abstract_FFT.hpp
│   ├── Data_Functions.hpp
│   ├── FFTW_Class.hpp
│   ├── cuFFT_Class.hpp
│   └── rocFFT_Class.hpp
├── src
│   ├── Data_Functions.cpp
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