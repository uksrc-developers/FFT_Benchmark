cmake_minimum_required( VERSION 3.22 )
project( FFT_Bench LANGUAGES CXX )
set( CMAKE_CXX_STANDARD 23 )

set( LIBRARIES fftw3 )
set( SOURCES main.cpp src/Data_Functions.cpp src/FFTW_Class.cpp )

find_package( PythonLibs )
if( PythonLibs_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    list( APPEND LIBRARIES ${PYTHON_LIBRARIES} )
endif( PythonLibs_FOUND )

set( CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc )
find_package( CUDAToolkit )
if( CUDAToolkit_FOUND )
    include_directories( ${CUDAToolkitINCLUDE_DIRS} )
    set( CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_NVIDIA__" )
    list( APPEND SOURCES FFT_Bench src/cuFFT_Class.cpp )
    list( APPEND LIBRARIES CUDA::cudart CUDA::cufft )
endif( CUDAToolkit_FOUND )


set( hip_DIR "/opt/rocm/bin" )
#if( DEFINED ${__HIP_PLATFORM_NVIDIA__} )
set( CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_NVIDIA__" )
#elseif( DEFINED ${__HIP_PLATFORM_AMD__} )
#    set( CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__" )
#endif()
set( CMAKE_MODULE_PATH "/opt/rocm" ${CMAKE_MODULE_PATH})
set( CMAKE_MODULE_PATH "/opt/rocm/include" ${CMAKE_MODULE_PATH} )
find_package( rocfft )
if( rocfft_FOUND )
    include_directories( ${rocfftINCLUDE_DIRS} )
    list( APPEND SOURCES src/rocFFT_Class.cpp )
    list( APPEND LIBRARIES roc::rocfft )
    if( NOT CUDAToolkit_FOUND )
        set( CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__" )
    endif( NOT CUDAToolkit_FOUND )
endif( rocfft_FOUND )

add_executable( FFT_Bench ${SOURCES} )
target_include_directories( FFT_Bench PUBLIC include )
target_link_libraries( FFT_Bench ${LIBRARIES} )
# Build Flags on local
# -G Ninja
# -D CMAKE_MAKE_PROGRAM=/usr/bin/ninja
# -D CMAKE_BUILD_TYPE=Debug
# -D CMAKE_C_COMPILER=/usr/bin/gcc
# -D CMAKE_CXX_COMPILER=/usr/bin/g++
# -D CMAKE_C_FLAGS=-fopenmp
# -D CMAKE_EXE_LINKER_FLAGS=-fopenmp
# -D __HIP_PLATFORM_NVIDIA__
# -D GPU_TARGETS=
# -S /home/marcuskeil/Projects/RSE/Benchmarks_Hardware/FFT/FFT_Bench
# -B /home/marcuskeil/Projects/RSE/Benchmarks_Hardware/FFT/FFT_Bench/cmake-build-debug

# -G Ninja
# -D CMAKE_MAKE_PROGRAM=/usr/bin/ninja
# -D CMAKE_BUILD_TYPE=Debug
# -D CMAKE_C_COMPILER=/usr/bin/gcc
# -D CMAKE_CXX_COMPILER=/usr/bin/g++
# -D CMAKE_C_FLAGS=-fopenmp
# -D CMAKE_EXE_LINKER_FLAGS=-fopenmp
# -S /home/marcuskeil/Projects/RSE/Benchmarks_Hardware/FFT/FFT_Bench
# -B /home/marcuskeil/Projects/RSE/Benchmarks_Hardware/FFT/FFT_Bench/cmake-build-debug