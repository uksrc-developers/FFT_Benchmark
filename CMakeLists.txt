cmake_minimum_required( VERSION 3.22 )
project( FFT_Bench LANGUAGES CXX )
set( CMAKE_CXX_STANDARD 23 )

# Inclusion of FFTW3, not optional
set(FFTW3_DIR "/usr/local" CACHE PATH "Root directory of fftw3 installation")
find_library(FFTW3_LIBRARY
  NAMES fftw3
  PATHS ${FFTW3_DIR}/lib /usr/lib /usr/local/lib
  NO_DEFAULT_PATH)
find_path(FFTW3_INCLUDE_DIR
  NAMES fftw3.h
  PATHS ${FFTW3_DIR}/include /usr/include
  NO_DEFAULT_PATH)

if(NOT FFTW3_LIBRARY AND NOT FFTW3_INCLUDE_DIR)
  message(FATAL_ERROR "fftw3 not found. Please specify FFTW3_DIR if it is in a non-standard location.")
endif(NOT FFTW3_LIBRARY AND NOT FFTW3_INCLUDE_DIR)
include_directories(${FFTW3_INCLUDE_DIR})
set( LIBRARIES ${FFTW3_LIBRARY})
set( SOURCES include/Abstract_FFT.hpp src/Data_Functions.cpp include/Data_Functions.hpp src/FFTW_Class.cpp include/FFTW_Class.hpp )

option( PYTHON_PLOTTING "Enable matplotlib plotting of fft outputs." OFF )
if( PYTHON_PLOTTING )
    find_package( PythonLibs )
    if( NOT PythonLibs_FOUND )
        message(FATAL_ERROR "PYTHON_PLOTTING set to ON. Unable to find PythonLibs.")
    endif( NOT PythonLibs_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    list( APPEND LIBRARIES ${PYTHON_LIBRARIES} )
endif( PYTHON_PLOTTING )

option( CUDA_FFT "Enable cuFFT." OFF )
if ( CUDA_FFT )
    find_package( CUDAToolkit )
    if( NOT CUDAToolkit_FOUND )
        message(FATAL_ERROR "CUDA_FFT set to ON. Unable to find CUDAToolkit")
    endif( NOT CUDAToolkit_FOUND )
    include_directories( ${CUDAToolkitINCLUDE_DIRS} )
    list( APPEND SOURCES src/cuFFT_Class.cpp include/cuFFT_Class.hpp )
    list( APPEND LIBRARIES CUDA::cudart CUDA::cufft )
endif( CUDA_FFT )

option( ROC_FFT "Enable rocFFT." OFF )
if( ROC_FFT )
    set(ROCM_DIR "/opt/rocm" CACHE PATH "Root directory of rocm installation")
    find_library(ROCFFT_LIBRARY
            NAMES rocfft
            PATHS ${ROCM_DIR}/lib
            NO_DEFAULT_PATH)
    find_path(ROCFFT_INCLUDE_DIR
            NAMES rocfft.h
            PATHS ${ROCM_DIR}/include/rocfft
            NO_DEFAULT_PATH)
    if( NOT ROCFFT_LIBRARY OR NOT ROCFFT_INCLUDE_DIR )
        message(FATAL_ERROR "ROC_FFT set to ON. Unable to find rocfft")
    endif()
    if( NOT CUDAToolikit_FOUND )
        set( CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__" )
    else()
        set( CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_NVIDIA__" )
    endif()
    find_package(hip REQUIRED HINTS "${ROCM_DIR}/lib/cmake/hip")
    include_directories( ${ROCFFT_INCLUDE_DIR} ${HIP_INCLUDE_DIR} )
    list( APPEND SOURCES src/rocFFT_Class.cpp include/rocFFT_Class.hpp )
    list( APPEND LIBRARIES ${ROCFFT_LIBRARY} ${HIP_LIBRARY}  )
endif( ROC_FFT )

message(LIBRARIES="${LIBRARIES}")

add_executable( FFT_Bench main.cpp ${SOURCES} )
target_link_libraries( FFT_Bench ${LIBRARIES} )