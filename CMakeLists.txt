cmake_minimum_required(VERSION 3.12)
project(BatchMatmulBench CXX CUDA) # Enable C++ and CUDA languages

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Find MKL ---
# This part might need adjustment based on your MKL installation method (e.g., environment variables, specific paths)
# Option 1: Using environment variable MKLROOT (common)
if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT "$ENV{MKLROOT}")
    message(STATUS "Found MKLROOT: ${MKL_ROOT}")
    find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT}/include)

    # Determine MKL library path (depends on architecture and interface layer)
    # Assuming 64-bit architecture and lp64 interface (adjust if needed)
    # Common paths: ${MKL_ROOT}/lib/intel64, ${MKL_ROOT}/lib
    find_library(MKL_CORE_LIB NAMES mkl_core PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64)
    find_library(MKL_INTEL_LP64_LIB NAMES mkl_intel_lp64 PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64)
    find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64) # Or mkl_tbb_thread for threaded
    find_library(MKL_THREADING_LIB NAMES mkl_sequential PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64) # Defaulting to sequential, change to mkl_intel_thread or mkl_gnu_thread if needed

    # Find OpenMP library (often needed by MKL) - adjust based on your compiler
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB} ${OpenMP_CXX_LIBRARIES}) # Order can matter
         message(STATUS "Found OpenMP, linking with MKL: ${MKL_LIBRARIES}")
    else()
         # Might need iomp5 library explicitly if OpenMP find_package fails
         find_library(MKL_IOMP5_LIB NAMES libiomp5md.dll iomp5 PATHS ${MKL_ROOT}/../compiler/lib/intel64_win ${MKL_ROOT}/../compiler/lib/intel64 ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 /opt/intel/compilers_and_libraries/linux/lib/intel64)
         if(MKL_IOMP5_LIB)
             set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB} ${MKL_IOMP5_LIB})
             message(STATUS "Linking MKL with explicit iomp5: ${MKL_LIBRARIES}")
         else()
             message(WARNING "OpenMP library not found automatically, MKL linking might be incomplete.")
             set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB})
         endif()
    endif()

else()
    message(FATAL_ERROR "MKLROOT environment variable not set. Cannot find MKL.")
endif()

if(NOT MKL_INCLUDE_DIR OR NOT MKL_CORE_LIB OR NOT MKL_INTEL_LP64_LIB OR NOT MKL_SEQUENTIAL_LIB)
    message(FATAL_ERROR "Could not find all required MKL components (include dir, core, lp64, sequential libs). Check MKLROOT and library paths.")
endif()

# --- Find CUDA ---
# CMake's FindCUDA module handles finding CUDA toolkit components
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found.")
else()
     message(STATUS "Found CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
     message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
     message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
     # Find cuBLAS specifically
     find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64) # Adjust path if needed
     if(NOT CUBLAS_LIBRARY)
         message(FATAL_ERROR "cuBLAS library not found in CUDA toolkit.")
     else()
         message(STATUS "Found cuBLAS: ${CUBLAS_LIBRARY}")
     endif()
endif()

# --- Add Executable ---
add_executable(batch_matmul_benchmark batch_matmul_benchmark.cpp)

# --- Link Libraries ---
target_include_directories(batch_matmul_benchmark PRIVATE
    ${MKL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(batch_matmul_benchmark PRIVATE
    ${MKL_LIBRARIES}
    ${CUDA_LIBRARIES} # Includes cuda_runtime
    ${CUBLAS_LIBRARY}
    pthread # Often needed with MKL/CUDA on Linux
    dl      # Often needed on Linux
)

# Optional: Set compiler flags if needed (e.g., optimization)
# target_compile_options(batch_matmul_benchmark PRIVATE -O3)

# Optional: Specify CUDA architecture if needed (e.g., for specific GPU features)
# set(CMAKE_CUDA_ARCHITECTURES 75) # Example: Turing architecture

message(STATUS "Executable target: batch_matmul_benchmark")
message(STATUS "Linking against MKL: ${MKL_LIBRARIES}")
message(STATUS "Linking against CUDA/cuBLAS: ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY}")
