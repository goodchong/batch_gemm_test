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
    # find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64) # No longer needed for threaded
    find_library(MKL_THREADING_LIB NAMES mkl_intel_thread PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64) # Use mkl_intel_thread for OpenMP threading

    # Find OpenMP library (often needed by MKL threaded layer) - adjust based on your compiler
    find_package(OpenMP REQUIRED) # Make OpenMP required when using threaded MKL
    if(OpenMP_FOUND)
        # Order matters: Interface layer, Threading layer, Core library, OpenMP runtime
        set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIB} ${MKL_THREADING_LIB} ${MKL_CORE_LIB} ${OpenMP_CXX_LIBRARIES})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

         message(STATUS "Found OpenMP, linking with threaded MKL: ${MKL_LIBRARIES}")
    else()
         # This block should ideally not be reached if OpenMP is REQUIRED, but kept for robustness
         # Might need iomp5 library explicitly if OpenMP find_package fails (less common with modern CMake)
         find_library(MKL_IOMP5_LIB NAMES libiomp5md.dll iomp5 PATHS ${MKL_ROOT}/../compiler/lib/intel64_win ${MKL_ROOT}/../compiler/lib/intel64 ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 /opt/intel/compilers_and_libraries/linux/lib/intel64)
         if(MKL_IOMP5_LIB)
             # Order matters: Interface layer, Threading layer, Core library, OpenMP runtime
             set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIB} ${MKL_THREADING_LIB} ${MKL_CORE_LIB} ${MKL_IOMP5_LIB})
             message(STATUS "Linking threaded MKL with explicit iomp5: ${MKL_LIBRARIES}")
         else()
             message(FATAL_ERROR "OpenMP library (required for mkl_intel_thread) not found automatically or explicitly. MKL linking failed.")
             # set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIB} ${MKL_THREADING_LIB} ${MKL_CORE_LIB}) # This would likely fail at runtime
         endif()
    endif()

else()
    message(FATAL_ERROR "MKLROOT environment variable not set. Cannot find MKL.")
endif()

if(NOT MKL_INCLUDE_DIR OR NOT MKL_CORE_LIB OR NOT MKL_INTEL_LP64_LIB OR NOT MKL_THREADING_LIB)
    message(FATAL_ERROR "Could not find all required MKL components (include dir, core, lp64, threading libs). Check MKLROOT and library paths.")
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
# List all source files needed for the executable
add_executable(batch_matmul_benchmark
    main.cpp
    utils.cpp
    # Headers are included, not listed here unless using specific CMake features
)

# --- Link Libraries ---
# Target the executable directly
target_include_directories(batch_matmul_benchmark PRIVATE
    ${MKL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(batch_matmul_benchmark PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(batch_matmul_benchmark PRIVATE
    ${MKL_LIBRARIES}      # Contains MKL interface, threading, core, and potentially OpenMP already
    ${CUDA_LIBRARIES}     # Includes cuda_runtime
    ${CUBLAS_LIBRARY}
    ${OpenMP_CXX_LIBRARIES} # Explicitly add OpenMP again to help linker resolve symbols
    pthread               # Often needed with MKL/CUDA on Linux
    dl                    # Often needed on Linux
    iomp5                 # Explicitly link iomp5 if needed (rare)
)

# Optional: Set compiler flags if needed (e.g., optimization)
# target_compile_options(batch_matmul_benchmark PRIVATE -O3)

# Optional: Specify CUDA architecture if needed (e.g., for specific GPU features)
# set(CMAKE_CUDA_ARCHITECTURES 75) # Example: Turing architecture

message(STATUS "Executable target: batch_matmul_benchmark")
message(STATUS "Linking against MKL: ${MKL_LIBRARIES}")
message(STATUS "Linking against CUDA/cuBLAS: ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY}")
