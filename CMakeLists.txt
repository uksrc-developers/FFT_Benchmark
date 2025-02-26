cmake_minimum_required(VERSION 3.22)
project(FFT_Bench LANGUAGES CXX CUDA HIP)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "-DCMAKE_C_FLAGS=-fopenmp -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -D__HIP_PLATFORM_NVIDIA__")

find_package( PythonLibs REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkitINCLUDE_DIRS})

set(hip_DIR "/opt/rocm/lib/cmake/hip")
set(CMAKE_MODULE_PATH "/opt/rocm/include" ${CMAKE_MODULE_PATH})
find_package(rocfft REQUIRED)
include_directories(${ROCFFT_INCLUDE_DIRS})

add_executable( FFT_Bench
        main.cpp
        src/Data_Functions.cpp
        src/cuFFT_Class.cpp
        src/FFTW_Class.cpp
        src/rocFFT_Class.cpp
)

target_include_directories( FFT_Bench PUBLIC include )
target_link_libraries( FFT_Bench ${PYTHON_LIBRARIES}
        fftw3
        CUDA::cudart CUDA::cufft
        roc::rocfft
)
# Build Flags on local
# -G Ninja
# -D CMAKE_BUILD_TYPE=Debug
# -D CMAKE_MAKE_PROGRAM=/usr/bin/ninja
# -D CMAKE_C_COMPILER=/usr/bin/gcc
# -D CMAKE_CXX_COMPILER=/usr/bin/g++
# -D CMAKE_C_FLAGS=-fopenmp
# -D CMAKE_EXE_LINKER_FLAGS=-fopenmp
# -D __HIP_PLATFORM_NVIDIA__
# -D GPU_TARGETS=
# -S /home/marcuskeil/Projects/RSE/Benchmarks_Hardware/FFT/FFT_Bench
# -B /home/marcuskeil/Projects/RSE/Benchmarks_Hardware/FFT/FFT_Bench/cmake-build-debug